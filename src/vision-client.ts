import type { Analysis, InitFailure, WorkerRequest, WorkerResponse } from './worker-protocol.ts';

export class InitError extends Error {
  readonly reason: InitFailure;

  constructor(reason: InitFailure, message: string) {
    super(message);
    this.reason = reason;
  }
}

interface Pending {
  resolve: (analysis: Analysis) => void;
  reject: (error: Error) => void;
}

interface InitWaiter {
  resolve: () => void;
  reject: (error: Error) => void;
  onProgress: (ratio: number) => void;
}

/** Main-thread handle to the inference worker. */
export class VisionClient {
  private readonly worker: Worker;
  private readonly pending = new Map<number, Pending>();
  private nextId = 1;
  private initWaiter: InitWaiter | undefined;

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handle(event.data);
    this.worker.onerror = (event) => {
      const error = new InitError('unsupported', event.message || 'worker failed');
      this.initWaiter?.reject(error);
      this.initWaiter = undefined;
      for (const p of this.pending.values()) p.reject(error);
      this.pending.clear();
    };
  }

  /** Downloads the models and creates the ONNX sessions. Safe to call again after a failure. */
  init(onProgress: (ratio: number) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      this.initWaiter = { resolve, reject, onProgress };
      this.post({ type: 'init', modelBaseUrl: `${import.meta.env.BASE_URL}models/` });
    });
  }

  /** Detects faces in `frame` (ownership moves to the worker). With `embed`, also embeds the largest face. */
  analyze(frame: ImageBitmap, embed: boolean): Promise<Analysis> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.post({ type: 'analyze', id, frame, embed }, [frame]);
    });
  }

  private post(message: WorkerRequest, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, transfer);
  }

  private handle(message: WorkerResponse): void {
    switch (message.type) {
      case 'progress':
        this.initWaiter?.onProgress(message.loaded / message.total);
        break;
      case 'ready':
        this.initWaiter?.resolve();
        this.initWaiter = undefined;
        break;
      case 'init-error':
        this.initWaiter?.reject(new InitError(message.reason, message.message));
        this.initWaiter = undefined;
        break;
      case 'analysis':
        this.pending.get(message.id)?.resolve(message.analysis);
        this.pending.delete(message.id);
        break;
      case 'analyze-error':
        this.pending.get(message.id)?.reject(new Error(message.message));
        this.pending.delete(message.id);
        break;
    }
  }
}
