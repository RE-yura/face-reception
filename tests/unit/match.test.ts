import { describe, expect, it } from 'vitest';
import { dot, matchFace, type EnrolledPerson } from '../../src/match.ts';

const unit = (deg: number) => [Math.cos((deg * Math.PI) / 180), Math.sin((deg * Math.PI) / 180)];

const people: EnrolledPerson[] = [
  { id: 'a', name: 'あおやま', embeddings: [unit(0), unit(80)] },
  { id: 'b', name: 'たなか', embeddings: [unit(40)] },
];

describe('dot', () => {
  it('is the cosine similarity of unit vectors', () => {
    expect(dot(unit(0), unit(60))).toBeCloseTo(0.5, 9);
  });
});

describe('matchFace', () => {
  it('returns null when nobody is enrolled', () => {
    expect(matchFace(unit(0), [], 0.363)).toBeNull();
    expect(matchFace(unit(0), [{ id: 'x', name: 'x', embeddings: [] }], 0.363)).toBeNull();
  });

  it("scores each person by their closest embedding and picks the best person", () => {
    const m = matchFace(unit(75), people, 0.363);
    expect(m).toMatchObject({ kind: 'matched', id: 'a', name: 'あおやま' });
    expect(m?.score).toBeCloseTo(Math.cos((5 * Math.PI) / 180), 9);
  });

  it('accepts a score exactly at the threshold', () => {
    const threshold = dot(unit(30), unit(40));
    expect(matchFace(unit(30), [people[1]], threshold)?.kind).toBe('matched');
  });

  it('reports unknown with the best score when nobody reaches the threshold', () => {
    const m = matchFace(unit(-100), people, 0.363);
    expect(m?.kind).toBe('unknown');
    expect(m?.score).toBeLessThan(0.363);
  });
});
