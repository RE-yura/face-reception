import './layout.css';
import './theme.css';
import { cameraProblemMessage, classifyCameraError, initFailureMessage } from './errors.ts';
import { byId } from './ui/dom.ts';
import { Stage } from './ui/stage.ts';
import { StartScreen } from './ui/start-screen.ts';
import { InitError, VisionClient } from './vision-client.ts';

const startScreen = new StartScreen();
const mainScreen = byId<HTMLElement>('main-screen');
const client = new VisionClient();
const stage = new Stage(byId('stage'), byId('camera'), byId('overlay'), client);

let modelsReady = false;
let cameraReady = false;

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
  startScreen.hide();
  mainScreen.hidden = false;
  stage.start();
}

startScreen.onStart(() => void boot());
startScreen.onRetry(() => void boot());
