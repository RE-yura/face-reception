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

  /** Shows the progress of the models loading in the background, in the space kept for it under the start button. */
  showPreloading(): void {
    this.status.classList.remove('load-status-idle');
  }

  /** The background load is over. Until the start button is pressed, its progress goes away but keeps its space. */
  endPreloading(): void {
    if (!this.startButton.hidden && !this.startButton.disabled) this.status.classList.add('load-status-idle');
  }

  /** Keeps the start button in place, pressed and disabled, so nothing on the screen moves. */
  showLoading(): void {
    this.root.hidden = false;
    this.startButton.hidden = false;
    this.startButton.disabled = true;
    this.error.hidden = true;
    this.status.hidden = false;
    this.status.classList.remove('load-status-idle');
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
