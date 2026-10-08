import { byId } from './dom.ts';

export class StartScreen {
  private readonly root = byId<HTMLElement>('start-screen');
  private readonly startButton = byId<HTMLButtonElement>('start-button');
  private readonly status = byId<HTMLElement>('load-status');
  private readonly progress = byId<HTMLProgressElement>('load-progress');
  private readonly message = byId<HTMLElement>('load-message');
  private readonly error = byId<HTMLElement>('start-error');
  private readonly errorMessage = byId<HTMLElement>('start-error-message');
  private readonly retryButton = byId<HTMLButtonElement>('retry-button');

  onStart(handler: () => void): void {
    this.startButton.addEventListener('click', handler);
  }

  onRetry(handler: () => void): void {
    this.retryButton.addEventListener('click', handler);
  }

  showLoading(): void {
    this.root.hidden = false;
    this.startButton.hidden = true;
    this.error.hidden = true;
    this.status.hidden = false;
  }

  setProgress(ratio: number): void {
    this.progress.value = ratio;
    // After the download, the inference engine may still be loading and compiling.
    this.message.textContent = ratio < 1 ? `モデルを読み込んでいます… ${Math.floor(ratio * 100)}%` : '準備しています…';
  }

  showError(message: string, canRetry: boolean, retryLabel = 'もう一度試す'): void {
    this.root.hidden = false;
    this.startButton.hidden = true;
    this.status.hidden = true;
    this.error.hidden = false;
    this.errorMessage.textContent = message;
    this.retryButton.hidden = !canRetry;
    this.retryButton.textContent = retryLabel;
  }

  hide(): void {
    this.root.hidden = true;
  }
}
