import 'fake-indexeddb/auto';
import { describe, expect, it } from 'vitest';
import { createMemoryPeopleStore, openPeopleStore, type PeopleStore } from '../../src/store.ts';

const thumb = (byte: number) => new Blob([new Uint8Array([byte])], { type: 'image/jpeg' });
const vec = (...v: number[]) => new Float32Array(v);

describe.each([
  ['IndexedDB', () => openPeopleStore(`test-${crypto.randomUUID()}`)],
  ['memory', async () => createMemoryPeopleStore()],
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
