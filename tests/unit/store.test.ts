import 'fake-indexeddb/auto';
import { describe, expect, it } from 'vitest';
import { createMemoryPeopleStore, openPeopleStore, withMemoryFallback, type PeopleStore } from '../../src/store.ts';

const thumb = (byte: number) => new Blob([new Uint8Array([byte])], { type: 'image/jpeg' });
const vec = (...v: number[]) => new Float32Array(v);

describe.each([
  ['IndexedDB', () => openPeopleStore(`test-${crypto.randomUUID()}`)],
  ['memory', async () => createMemoryPeopleStore()],
  ['IndexedDB with memory fallback', async () => withMemoryFallback(await openPeopleStore(`test-${crypto.randomUUID()}`), () => {})],
] as const)('%s store', (_label, open: () => Promise<PeopleStore>) => {
  it('adds a person with their embeddings and thumbnail', async () => {
    const store = await open();
    const added = await store.addEnrollment('あおやま', [vec(0.5, 0.25), vec(1, 0)], thumb(1));
    const [person, ...rest] = await store.listPeople();
    expect(rest).toEqual([]);
    expect(person.id).toBe(added.id);
    expect(person.name).toBe('あおやま');
    expect(person.embeddings).toEqual([[0.5, 0.25], [1, 0]]);
    expect(person.thumbnail).toBeInstanceOf(Blob);
    expect(person.thumbnail.size).toBe(1);
  });

  it('appends embeddings when the same name (ignoring surrounding spaces) enrolls again', async () => {
    const store = await open();
    const first = await store.addEnrollment('たなか', [vec(1, 0)], thumb(1));
    await store.addEnrollment('  たなか ', [vec(0, 1)], thumb(2));
    const people = await store.listPeople();
    expect(people).toHaveLength(1);
    expect(people[0].id).toBe(first.id);
    expect(people[0].embeddings).toEqual([[1, 0], [0, 1]]);
    expect(new Uint8Array(await people[0].thumbnail.arrayBuffer())).toEqual(new Uint8Array([1]));
  });

  it('rejects an empty name', async () => {
    const store = await open();
    await expect(store.addEnrollment('   ', [vec(1, 0)], thumb(1))).rejects.toThrow('name is empty');
  });

  it('deletes one person or everyone', async () => {
    const store = await open();
    const a = await store.addEnrollment('a', [vec(1, 0)], thumb(1));
    await store.addEnrollment('b', [vec(0, 1)], thumb(2));
    await store.addEnrollment('c', [vec(1, 1)], thumb(3));
    await store.deletePerson(a.id);
    expect((await store.listPeople()).map((p) => p.name).sort()).toEqual(['b', 'c']);
    await store.deleteAll();
    expect(await store.listPeople()).toEqual([]);
  });
});

describe('IndexedDB store persistence', () => {
  it('keeps people across reopening the database', async () => {
    const name = `test-${crypto.randomUUID()}`;
    await (await openPeopleStore(name)).addEnrollment('すずき', [vec(1, 0)], thumb(1));
    const reopened = await openPeopleStore(name);
    expect((await reopened.listPeople()).map((p) => p.name)).toEqual(['すずき']);
  });
});

/** Wraps a store so that, once broken, chosen methods fail the way a full or unusable IndexedDB does. */
function breakable(store: PeopleStore) {
  const broken = new Set<keyof PeopleStore>();
  const quota = () => Promise.reject(new DOMException('The quota has been exceeded.', 'QuotaExceededError'));
  const wrapped: PeopleStore = {
    listPeople: () => (broken.has('listPeople') ? quota() : store.listPeople()),
    addEnrollment: (...args) => (broken.has('addEnrollment') ? quota() : store.addEnrollment(...args)),
    deletePerson: (id) => (broken.has('deletePerson') ? quota() : store.deletePerson(id)),
    deleteAll: () => (broken.has('deleteAll') ? quota() : store.deleteAll()),
  };
  return { store: wrapped, breakMethods: (...methods: (keyof PeopleStore)[]) => methods.forEach((m) => broken.add(m)) };
}

describe('withMemoryFallback', () => {
  const saved = async (...names: string[]) => {
    const store = await openPeopleStore(`test-${crypto.randomUUID()}`);
    for (const [i, name] of names.entries()) await store.addEnrollment(name, [vec(i, 1)], thumb(i));
    return store;
  };

  it('keeps a new enrollment, and everyone saved before, in memory when saving fails', async () => {
    const primary = breakable(await saved('すずき'));
    primary.breakMethods('addEnrollment');
    const fallbacks: unknown[] = [];
    const store = withMemoryFallback(primary.store, (error) => fallbacks.push(error));
    const added = await store.addEnrollment('たなか', [vec(0, 1)], thumb(2));
    expect(added.name).toBe('たなか');
    expect((await store.listPeople()).map((p) => p.name)).toEqual(['すずき', 'たなか']);
    await store.addEnrollment('さとう', [vec(1, 1)], thumb(3));
    expect((await store.listPeople()).map((p) => p.name)).toEqual(['すずき', 'たなか', 'さとう']);
    expect(fallbacks).toHaveLength(1);
    expect(fallbacks[0]).toBeInstanceOf(DOMException);
  });

  it('starts from the people it last listed when the database cannot be read either', async () => {
    const primary = breakable(await saved('すずき'));
    const store = withMemoryFallback(primary.store, () => {});
    await store.listPeople();
    primary.breakMethods('listPeople', 'addEnrollment');
    await store.addEnrollment('たなか', [vec(0, 1)], thumb(2));
    expect((await store.listPeople()).map((p) => p.name)).toEqual(['すずき', 'たなか']);
  });

  it('deletes in memory when deleting fails', async () => {
    const primary = breakable(await saved('a', 'b', 'c'));
    primary.breakMethods('deletePerson', 'deleteAll');
    let fellBack = 0;
    const store = withMemoryFallback(primary.store, () => fellBack++);
    const a = (await store.listPeople()).find((p) => p.name === 'a')!;
    await store.deletePerson(a.id);
    expect((await store.listPeople()).map((p) => p.name).sort()).toEqual(['b', 'c']);
    await store.deleteAll();
    expect(await store.listPeople()).toEqual([]);
    expect(fellBack).toBe(1);
  });

  it('does not fall back for an empty name', async () => {
    const primary = breakable(await saved());
    primary.breakMethods('addEnrollment');
    let fellBack = 0;
    const store = withMemoryFallback(primary.store, () => fellBack++);
    await expect(store.addEnrollment('  ', [vec(1, 0)], thumb(1))).rejects.toThrow('name is empty');
    expect(fellBack).toBe(0);
  });
});
