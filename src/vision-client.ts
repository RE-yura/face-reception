import { ANALYZE_TIMEOUT_MS } from './config.ts';
import type { Analysis, InitFailure, WorkerRequest, WorkerResponse } from './worker-protocol.ts';

export class InitError extends Error {
  readonly reason: InitFailure;

  constructor(reason: InitFailure, message: string) {
    super(message);
    this.reason = reason;
  }
}

/** The worker stopped answering, or crashed after start-up. Reloading the page is the way out. */
export class AnalysisStalled extends Error {}

/** The part of a Worker the client uses, so tests can stand in for it. */
export interface WorkerPort {
  postMessage(message: WorkerRequest, transfer: Transferable[]): void;
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
}

interface Pending {
  resolve: (analysis: Analysis) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

interface InitWaiter {
  resolve: () => void;
  reject: (error: Error) => void;
  onProgress: (ratio: number) => void;
}

/** Main-thread handle to the inference worker. */
export class VisionClient {
  private readonly worker: WorkerPort;
  private readonly analyzeTimeoutMs: number;
  private readonly pending = new Map<number, Pending>();
  private nextId = 1;
  private initWaiter: InitWaiter | undefined;

  constructor(
    worker: WorkerPort = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }),
    analyzeTimeoutMs = ANALYZE_TIMEOUT_MS,
  ) {
    this.worker = worker;
    this.analyzeTimeoutMs = analyzeTimeoutMs;
    this.worker.onmessage = (event) => this.handle(event.data);
    this.worker.onerror = (event) => {
      const message = event.message || 'worker failed';
      // A worker that cannot even start means an unsupported browser; a crash later is a stall.
      this.initWaiter?.reject(new InitError('unsupported', message));
      this.initWaiter = undefined;
      for (const id of [...this.pending.keys()]) this.settle(id)?.reject(new AnalysisStalled(message));
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
      const timer = setTimeout(
        () => this.settle(id)?.reject(new AnalysisStalled(`no answer in ${this.analyzeTimeoutMs} ms`)),
        this.analyzeTimeoutMs,
      );
      this.pending.set(id, { resolve, reject, timer });
      this.post({ type: 'analyze', id, frame, embed }, [frame]);
    });
  }

  private post(message: WorkerRequest, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, transfer);
  }

  /** Removes a pending request and returns it, or undefined when it was already answered or gave up. */
  private settle(id: number): Pending | undefined {
    const p = this.pending.get(id);
    if (!p) return undefined;
    clearTimeout(p.timer);
    this.pending.delete(id);
    return p;
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
        this.settle(message.id)?.resolve(message.analysis);
        break;
      case 'analyze-error':
        this.settle(message.id)?.reject(new Error(message.message));
        break;
    }
  }
}
