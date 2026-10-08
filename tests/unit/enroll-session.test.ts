import { describe, expect, it } from 'vitest';
import { EnrollSession } from '../../src/enroll-session.ts';

const emb = (v: number) => new Float32Array([v]);
const session = () => new EnrollSession<string>({ shots: 3, timeoutMs: 10_000, startedAt: 1_000 });

describe('EnrollSession', () => {
  it('collects the requested number of embeddings and keeps the first crop as the thumbnail', () => {
    const s = session();
    expect(s.accept({ faceCount: 1, embedding: emb(1), aligned: 'first' }, 1_100)).toEqual({ kind: 'collecting', count: 1, total: 3, hint: 'ok' });
    expect(s.accept({ faceCount: 1, embedding: emb(2), aligned: 'second' }, 1_500)).toMatchObject({ kind: 'collecting', count: 2 });
    const done = s.accept({ faceCount: 1, embedding: emb(3), aligned: 'third' }, 1_900);
    expect(done).toEqual({ kind: 'done', embeddings: [emb(1), emb(2), emb(3)], thumbnailSource: 'first' });
  });

  it('does not count frames without an embedding', () => {
    expect(session().accept({ faceCount: 1 }, 1_100)).toEqual({ kind: 'collecting', count: 0, total: 3, hint: 'ok' });
  });

  it('does not count frames with no face or several faces, and asks for one face', () => {
    const s = session();
    expect(s.accept({ faceCount: 0, embedding: emb(1), aligned: 'x' }, 1_100)).toEqual({ kind: 'collecting', count: 0, total: 3, hint: 'one-face' });
    expect(s.accept({ faceCount: 2, embedding: emb(1), aligned: 'x' }, 1_200)).toEqual({ kind: 'collecting', count: 0, total: 3, hint: 'one-face' });
  });

  it('times out after the deadline', () => {
    const s = session();
    s.accept({ faceCount: 1, embedding: emb(1), aligned: 'x' }, 1_100);
    expect(s.accept({ faceCount: 1, embedding: emb(2), aligned: 'y' }, 11_001)).toEqual({ kind: 'timeout', count: 1, total: 3 });
  });
});
