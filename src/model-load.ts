import { InitError } from './vision-client.ts';

type Load = (onProgress: (ratio: number) => void) => Promise<void>;

/**
 * Loads the models once, possibly before they are needed: the page starts loading when it opens, and the start
 * button joins the load under way. After a network failure the next start() tries again; any other failure stays.
 */
export class ModelLoad {
  private readonly load: Load;
  private readonly onProgress: (ratio: number) => void;
  private attempt: Promise<void> | undefined;

  constructor(load: Load, onProgress: (ratio: number) => void) {
    this.load = load;
    this.onProgress = onProgress;
  }

  /** The load under way or done. Starts one, with its progress from 0, when there is none or the last one failed for the network. */
  start(): Promise<void> {
    if (this.attempt) return this.attempt;
    this.onProgress(0);
    const attempt = this.load(this.onProgress);
    this.attempt = attempt;
    attempt.catch((error: unknown) => {
      if (error instanceof InitError && error.reason === 'network') this.attempt = undefined;
    });
    return attempt;
  }
}
