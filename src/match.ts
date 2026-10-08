export interface EnrolledPerson {
  id: string;
  name: string;
  embeddings: ArrayLike<number>[];
}

export type Match =
  | { kind: 'matched'; id: string; name: string; score: number }
  | { kind: 'unknown'; score: number };

/** Dot product. Embeddings are unit length, so this is their cosine similarity. */
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/** Scores each person by their most similar embedding. Returns null when nobody has any embedding. */
export function matchFace(query: ArrayLike<number>, people: EnrolledPerson[], threshold: number): Match | null {
  let best: { person: EnrolledPerson; score: number } | null = null;
  for (const person of people) {
    for (const embedding of person.embeddings) {
      const score = dot(query, embedding);
      if (!best || score > best.score) best = { person, score };
    }
  }
  if (!best) return null;
  return best.score >= threshold
    ? { kind: 'matched', id: best.person.id, name: best.person.name, score: best.score }
    : { kind: 'unknown', score: best.score };
}
