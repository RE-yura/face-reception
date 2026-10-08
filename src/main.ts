import './layout.css';
import './theme.css';
import { cameraProblemMessage, classifyCameraError, initFailureMessage } from './errors.ts';
import { createMemoryPeopleStore, openPeopleStore, type PeopleStore, type Person } from './store.ts';
import { byId } from './ui/dom.ts';
import { EnrollPanel } from './ui/enroll-panel.ts';
import { ReceptionPanel } from './ui/reception-panel.ts';
import { Stage } from './ui/stage.ts';
import { StartScreen } from './ui/start-screen.ts';
import { InitError, VisionClient } from './vision-client.ts';

type Tab = 'reception' | 'enroll';
const TABS: Tab[] = ['reception', 'enroll'];

const startScreen = new StartScreen();
const mainScreen = byId<HTMLElement>('main-screen');
const tabButtons: Record<Tab, HTMLButtonElement> = { reception: byId('tab-reception'), enroll: byId('tab-enroll') };
const tabPanels: Record<Tab, HTMLElement> = { reception: byId('panel-reception'), enroll: byId('panel-enroll') };

const client = new VisionClient();
const stage = new Stage(byId('stage'), byId('camera'), byId('overlay'), client);
const storeReady = openStore();

let people: Person[] = [];
let modelsReady = false;
let booting = false;
/** The main screen is up. Whether the camera and detection run then follows page visibility. */
let running = false;
/** The camera is open, the detection loop runs and the active tab's panel is listening. */
let active = false;
/** Bumped by every suspend and resume, so a resume still waiting for the camera knows it is stale. */
let resumeToken = 0;
/** After a camera denial, iOS Safari keeps refusing until the page reloads, even once the setting is changed. */
let retryReloads = false;
let panels: Record<Tab, ReceptionPanel | EnrollPanel> | undefined;
let activeTab: Tab | undefined;

async function openStore(): Promise<{ store: PeopleStore; volatile: boolean }> {
  try {
    return { store: await openPeopleStore(), volatile: false };
  } catch (error) {
    console.error(error);
    return { store: createMemoryPeopleStore(), volatile: true };
  }
}

function showCameraError(error: unknown): void {
  const problem = classifyCameraError(error, window.isSecureContext);
  retryReloads = problem === 'denied';
  startScreen.showError(cameraProblemMessage(problem), true);
}

async function boot(): Promise<void> {
  if (booting) return;
  booting = true;
  startScreen.showLoading();
  const [camera, models] = await Promise.allSettled([
    stage.cameraLive() ? Promise.resolve(true) : stage.openCamera(),
    modelsReady ? Promise.resolve() : client.init((ratio) => startScreen.setProgress(ratio)),
  ]);
  booting = false;
  if (models.status === 'fulfilled') {
    modelsReady = true;
    document.body.dataset.models = 'ready';
  }
  if (models.status === 'rejected') {
    const reason = models.reason instanceof InitError ? models.reason.reason : 'unsupported';
    retryReloads = false;
    startScreen.showError(initFailureMessage(reason), reason === 'network');
    return;
  }
  if (camera.status === 'rejected') {
    showCameraError(camera.reason);
    return;
  }
  if (!panels) panels = await createPanels();
  startScreen.hide();
  mainScreen.hidden = false;
  running = true;
  if (!activeTab) selectTab(people.length === 0 ? 'enroll' : 'reception');
  await resume();
}

async function createPanels(): Promise<Record<Tab, ReceptionPanel | EnrollPanel>> {
  const { store, volatile } = await storeReady;
  people = await store.listPeople().catch(() => []);
  const enroll = new EnrollPanel(stage, store, volatile, (list) => {
    people = list;
    enroll.renderPeople(list);
  });
  enroll.renderPeople(people);
  const reception = new ReceptionPanel(stage, () => people, () => selectTab('enroll'));
  for (const tab of TABS) tabButtons[tab].addEventListener('click', () => selectTab(tab));
  return { reception, enroll };
}

function selectTab(tab: Tab): void {
  if (!panels || activeTab === tab) return;
  if (active && activeTab) panels[activeTab].deactivate();
  activeTab = tab;
  mainScreen.dataset.tab = tab;
  for (const t of TABS) {
    tabButtons[t].setAttribute('aria-selected', String(t === tab));
    tabPanels[t].hidden = t !== tab;
  }
  if (active) panels[tab].activate();
}

/** Releases the camera and stops detection, e.g. when the page goes to the background. */
function suspend(): void {
  resumeToken++;
  if (active && panels && activeTab) panels[activeTab].deactivate();
  active = false;
  stage.stop();
}

/** Opens the camera and starts detection again, if the main screen is up and the page is visible. */
async function resume(): Promise<void> {
  if (!running || active || document.hidden) return;
  const token = ++resumeToken;
  let opened: boolean;
  try {
    opened = stage.cameraLive() || (await stage.openCamera());
  } catch (error) {
    if (token !== resumeToken) return;
    running = false;
    mainScreen.hidden = true;
    showCameraError(error);
    return;
  }
  if (!opened || token !== resumeToken || document.hidden) return;
  active = true;
  stage.start();
  if (panels && activeTab) panels[activeTab].activate();
}

// iOS stops the camera in the background; release it ourselves and reopen it when the page comes back.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) suspend();
  else void resume();
});

startScreen.onStart(() => void boot());
startScreen.onRetry(() => (retryReloads ? location.reload() : void boot()));
