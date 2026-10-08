import { openDB, type IDBPDatabase } from 'idb';

export interface Person {
  id: string;
  name: string;
  embeddings: number[][];
  thumbnail: Blob;
  createdAt: number;
}

export interface PeopleStore {
  listPeople(): Promise<Person[]>;
  /** Adds a new person, or appends the embeddings to the person who already has this name. */
  addEnrollment(name: string, embeddings: ArrayLike<number>[], thumbnail: Blob): Promise<Person>;
  deletePerson(id: string): Promise<void>;
  deleteAll(): Promise<void>;
}

const STORE = 'people';

/** The person to write for an enrollment, given everyone currently stored. */
export function mergeEnrollment(
  people: Person[],
  name: string,
  embeddings: ArrayLike<number>[],
  thumbnail: Blob,
): Person {
  const trimmed = name.trim();
  if (!trimmed) throw new Error('name is empty');
  const vectors = embeddings.map((e) => Array.from(e));
  const existing = people.find((p) => p.name === trimmed);
  if (existing) return { ...existing, embeddings: [...existing.embeddings, ...vectors] };
  return { id: crypto.randomUUID(), name: trimmed, embeddings: vectors, thumbnail, createdAt: Date.now() };
}

const byCreatedAt = (a: Person, b: Person) => a.createdAt - b.createdAt;

class IdbPeopleStore implements PeopleStore {
  private readonly db: IDBPDatabase;

  constructor(db: IDBPDatabase) {
    this.db = db;
  }

  async listPeople(): Promise<Person[]> {
    return ((await this.db.getAll(STORE)) as Person[]).sort(byCreatedAt);
  }

  async addEnrollment(name: string, embeddings: ArrayLike<number>[], thumbnail: Blob): Promise<Person> {
    const tx = this.db.transaction(STORE, 'readwrite');
    const person = mergeEnrollment((await tx.store.getAll()) as Person[], name, embeddings, thumbnail);
    await tx.store.put(person);
    await tx.done;
    return person;
  }

  async deletePerson(id: string): Promise<void> {
    await this.db.delete(STORE, id);
  }

  async deleteAll(): Promise<void> {
    await this.db.clear(STORE);
  }
}

export async function openPeopleStore(dbName = 'face-reception'): Promise<PeopleStore> {
  const db = await openDB(dbName, 1, {
    upgrade(database) {
      database.createObjectStore(STORE, { keyPath: 'id' });
    },
  });
  return new IdbPeopleStore(db);
}

/** Fallback when IndexedDB is unavailable: same behavior, but nothing survives a reload. */
export function createMemoryPeopleStore(initial: Person[] = []): PeopleStore {
  const people = new Map(initial.map((p) => [p.id, p]));
  return {
    async listPeople() {
      return [...people.values()].sort(byCreatedAt);
    },
    async addEnrollment(name, embeddings, thumbnail) {
      const person = mergeEnrollment([...people.values()], name, embeddings, thumbnail);
      people.set(person.id, person);
      return person;
    },
    async deletePerson(id) {
      people.delete(id);
    },
    async deleteAll() {
      people.clear();
    },
  };
}

/**
 * Uses `primary` until one of its operations fails (a full disk, or a browser refusing storage), then switches for
 * good to an in-memory store that starts with everyone known so far, and calls `onFallback` once. Enrollment and
 * matching keep working; what changes after that is lost on reload.
 */
export function withMemoryFallback(primary: PeopleStore, onFallback: (error: unknown) => void): PeopleStore {
  let known: Person[] = [];
  let memory: Promise<PeopleStore> | undefined;
  const fallBack = (error: unknown) =>
    (memory ??= (async () => {
      const seed = await primary.listPeople().catch(() => known);
      onFallback(error);
      return createMemoryPeopleStore(seed);
    })());
  const run = async <T>(op: (store: PeopleStore) => Promise<T>): Promise<T> => {
    if (memory) return op(await memory);
    try {
      return await op(primary);
    } catch (error) {
      return op(await fallBack(error));
    }
  };
  return {
    async listPeople() {
      known = await run((store) => store.listPeople());
      return known;
    },
    async addEnrollment(name, embeddings, thumbnail) {
      // A bad name is the caller's mistake, not a storage failure.
      if (!name.trim()) throw new Error('name is empty');
      return run((store) => store.addEnrollment(name, embeddings, thumbnail));
    },
    deletePerson: (id) => run((store) => store.deletePerson(id)),
    deleteAll: () => run((store) => store.deleteAll()),
  };
}
