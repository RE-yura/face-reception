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
let cameraReady = false;
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

async function boot(): Promise<void> {
  startScreen.showLoading();
  const [camera, models] = await Promise.allSettled([
    cameraReady ? Promise.resolve() : stage.openCamera(),
    modelsReady ? Promise.resolve() : client.init((ratio) => startScreen.setProgress(ratio)),
  ]);
  if (camera.status === 'fulfilled') cameraReady = true;
  if (models.status === 'fulfilled') {
    modelsReady = true;
    document.body.dataset.models = 'ready';
  }
  if (models.status === 'rejected') {
    const reason = models.reason instanceof InitError ? models.reason.reason : 'unsupported';
    startScreen.showError(initFailureMessage(reason), reason === 'network');
    return;
  }
  if (camera.status === 'rejected') {
    startScreen.showError(cameraProblemMessage(classifyCameraError(camera.reason, window.isSecureContext)), true);
    return;
  }
  if (!panels) panels = await createPanels();
  startScreen.hide();
  mainScreen.hidden = false;
  stage.start();
  if (activeTab) panels[activeTab].activate();
  else selectTab(people.length === 0 ? 'enroll' : 'reception');
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
  if (activeTab) panels[activeTab].deactivate();
  activeTab = tab;
  mainScreen.dataset.tab = tab;
  for (const t of TABS) {
    tabButtons[t].setAttribute('aria-selected', String(t === tab));
    tabPanels[t].hidden = t !== tab;
  }
  panels[tab].activate();
}

// iOS stops the camera in the background; release it ourselves and reopen when the page comes back.
document.addEventListener('visibilitychange', () => {
  if (!panels || !activeTab || mainScreen.hidden) return;
  if (document.hidden) {
    panels[activeTab].deactivate();
    stage.stop();
    cameraReady = false;
    return;
  }
  stage.openCamera().then(
    () => {
      cameraReady = true;
      stage.start();
      if (panels && activeTab) panels[activeTab].activate();
    },
    (error: unknown) => {
      mainScreen.hidden = true;
      startScreen.showError(cameraProblemMessage(classifyCameraError(error, window.isSecureContext)), true);
    },
  );
});

startScreen.onStart(() => void boot());
startScreen.onRetry(() => void boot());
