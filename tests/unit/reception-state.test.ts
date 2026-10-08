import { describe, expect, it } from 'vitest';
import { ReceptionState } from '../../src/reception-state.ts';

const matched = { kind: 'matched', id: 'a', name: 'あおやま', score: 0.7 } as const;

describe('ReceptionState', () => {
  it('asks to enroll first when nobody is enrolled', () => {
    expect(new ReceptionState().update({ faceCount: 1, peopleCount: 0 })).toEqual({ kind: 'no-people' });
  });

  it('asks for a face when none is visible', () => {
    expect(new ReceptionState().update({ faceCount: 0, peopleCount: 2 })).toEqual({ kind: 'no-face' });
  });

  it('shows checking until the first embedding arrives', () => {
    expect(new ReceptionState().update({ faceCount: 1, peopleCount: 2 })).toEqual({ kind: 'checking' });
  });

  it('keeps the last result on frames without an embedding, and clears it when the face leaves', () => {
    const state = new ReceptionState();
    expect(state.update({ faceCount: 1, peopleCount: 2, match: matched })).toEqual({ kind: 'matched', name: 'あおやま', score: 0.7 });
    expect(state.update({ faceCount: 1, peopleCount: 2 })).toEqual({ kind: 'matched', name: 'あおやま', score: 0.7 });
    expect(state.update({ faceCount: 0, peopleCount: 2 })).toEqual({ kind: 'no-face' });
    expect(state.update({ faceCount: 1, peopleCount: 2 })).toEqual({ kind: 'checking' });
  });

  it('shows unknown with the best score', () => {
    const view = new ReceptionState().update({ faceCount: 1, peopleCount: 2, match: { kind: 'unknown', score: 0.12 } });
    expect(view).toEqual({ kind: 'unknown', score: 0.12 });
  });

  it('forgets the last result on reset', () => {
    const state = new ReceptionState();
    state.update({ faceCount: 1, peopleCount: 2, match: matched });
    state.reset();
    expect(state.update({ faceCount: 1, peopleCount: 2 })).toEqual({ kind: 'checking' });
  });
});
