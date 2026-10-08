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
    this.message.textContent = `モデルを読み込んでいます… ${Math.floor(ratio * 100)}%`;
  }

  showError(message: string, canRetry: boolean): void {
    this.root.hidden = false;
    this.startButton.hidden = true;
    this.status.hidden = true;
    this.error.hidden = false;
    this.errorMessage.textContent = message;
    this.retryButton.hidden = !canRetry;
  }

  hide(): void {
    this.root.hidden = true;
  }
}
