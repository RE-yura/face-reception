import type { Match } from './match.ts';

export type ReceptionView =
  | { kind: 'no-people' }
  | { kind: 'no-face' }
  | { kind: 'checking' }
  | { kind: 'matched'; name: string; score: number }
  | { kind: 'unknown'; score: number };

export interface ReceptionInput {
  faceCount: number;
  peopleCount: number;
  /** Present only on frames where an embedding was computed. */
  match?: Match | null;
}

/** Turns per-frame analyses into what the reception tab shows. Keeps the last result while the face stays in view. */
export class ReceptionState {
  private last: Match | null = null;

  update(input: ReceptionInput): ReceptionView {
    if (input.peopleCount === 0) {
      this.last = null;
      return { kind: 'no-people' };
    }
    if (input.faceCount === 0) {
      this.last = null;
      return { kind: 'no-face' };
    }
    if (input.match) this.last = input.match;
    if (!this.last) return { kind: 'checking' };
    return this.last.kind === 'matched'
      ? { kind: 'matched', name: this.last.name, score: this.last.score }
      : { kind: 'unknown', score: this.last.score };
  }

  reset(): void {
    this.last = null;
  }
}
