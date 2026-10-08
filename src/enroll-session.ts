export interface EnrollSample<T> {
  faceCount: number;
  embedding?: Float32Array;
  aligned?: T;
}

export type EnrollStatus<T> =
  | { kind: 'collecting'; count: number; total: number; hint: 'ok' | 'one-face' }
  | { kind: 'done'; embeddings: Float32Array[]; thumbnailSource: T }
  | { kind: 'timeout'; count: number; total: number };

/** Collects `shots` embeddings from frames that show exactly one face, within `timeoutMs`. */
export class EnrollSession<T> {
  private readonly embeddings: Float32Array[] = [];
  private first: T | undefined;
  private readonly shots: number;
  private readonly deadline: number;

  constructor(opts: { shots: number; timeoutMs: number; startedAt: number }) {
    this.shots = opts.shots;
    this.deadline = opts.startedAt + opts.timeoutMs;
  }

  accept(sample: EnrollSample<T>, now: number): EnrollStatus<T> {
    const count = this.embeddings.length;
    if (now > this.deadline) return { kind: 'timeout', count, total: this.shots };
    if (sample.faceCount !== 1) return { kind: 'collecting', count, total: this.shots, hint: 'one-face' };
    if (sample.embedding && sample.aligned !== undefined) {
      this.embeddings.push(sample.embedding);
      this.first ??= sample.aligned;
      if (this.embeddings.length >= this.shots) {
        return { kind: 'done', embeddings: [...this.embeddings], thumbnailSource: this.first };
      }
    }
    return { kind: 'collecting', count: this.embeddings.length, total: this.shots, hint: 'ok' };
  }
}
