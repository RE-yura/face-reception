import { ENROLL_INTERVAL_MS, ENROLL_SHOTS, ENROLL_TIMEOUT_MS } from '../config.ts';
import { EnrollSession, type EnrollStatus } from '../enroll-session.ts';
import type { PeopleStore, Person } from '../store.ts';
import type { RgbaImage } from '../vision/types.ts';
import { byId } from './dom.ts';
import type { Stage } from './stage.ts';
import { toJpegBlob } from './thumbnail.ts';

export class EnrollPanel {
  private readonly form = byId<HTMLFormElement>('enroll-form');
  private readonly nameInput = byId<HTMLInputElement>('enroll-name');
  private readonly button = byId<HTMLButtonElement>('enroll-button');
  private readonly status = byId<HTMLElement>('enroll-status');
  private readonly volatileNote = byId<HTMLElement>('volatile-note');
  private readonly list = byId<HTMLUListElement>('people-list');
  private readonly empty = byId<HTMLElement>('people-empty');
  private readonly deleteAllButton = byId<HTMLButtonElement>('delete-all');
  private readonly stage: Stage;
  private readonly store: PeopleStore;
  private readonly onPeopleChanged: (people: Person[]) => void;
  private objectUrls: string[] = [];
  private cancelCapture: (() => void) | undefined;

  constructor(stage: Stage, store: PeopleStore, volatile: boolean, onPeopleChanged: (people: Person[]) => void) {
    this.stage = stage;
    this.store = store;
    this.onPeopleChanged = onPeopleChanged;
    this.volatileNote.hidden = !volatile;
    this.nameInput.addEventListener('input', () => this.updateButton());
    this.form.addEventListener('submit', (event) => {
      event.preventDefault();
      this.startCapture(this.nameInput.value.trim());
    });
    this.deleteAllButton.addEventListener('click', () => void this.deleteAll());
  }

  activate(): void {
    this.setStatus('');
    this.updateButton();
  }

  deactivate(): void {
    this.cancelCapture?.();
  }

  renderPeople(people: Person[]): void {
    for (const url of this.objectUrls) URL.revokeObjectURL(url);
    this.objectUrls = [];
    this.list.replaceChildren(...people.map((person) => this.personItem(person)));
    this.empty.hidden = people.length > 0;
    this.deleteAllButton.hidden = people.length === 0;
  }

  private personItem(person: Person): HTMLLIElement {
    const item = document.createElement('li');
    item.className = 'person';
    const thumb = document.createElement('img');
    thumb.className = 'person-thumb';
    thumb.alt = '';
    thumb.width = 56;
    thumb.height = 56;
    const url = URL.createObjectURL(person.thumbnail);
    this.objectUrls.push(url);
    thumb.src = url;
    const name = document.createElement('span');
    name.className = 'person-name';
    name.textContent = person.name;
    const count = document.createElement('span');
    count.className = 'person-count';
    count.textContent = `${person.embeddings.length}枚`;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'button person-delete';
    remove.textContent = '削除';
    remove.setAttribute('aria-label', `${person.name} さんを削除`);
    remove.addEventListener('click', () => void this.deletePerson(person.id));
    item.append(thumb, name, count, remove);
    return item;
  }

  /** #enroll-status is a live region; rewriting it with the same text would make screen readers repeat it. */
  private setStatus(text: string): void {
    if (this.status.textContent !== text) this.status.textContent = text;
  }

  private updateButton(): void {
    this.button.disabled = this.cancelCapture !== undefined || this.nameInput.value.trim() === '';
  }

  private startCapture(name: string): void {
    if (!name || this.cancelCapture) return;
    const session = new EnrollSession<RgbaImage>({ shots: ENROLL_SHOTS, timeoutMs: ENROLL_TIMEOUT_MS, startedAt: performance.now() });
    const handle = (status: EnrollStatus<RgbaImage>) => {
      switch (status.kind) {
        case 'collecting':
          this.setStatus(
            status.hint === 'one-face'
              ? `1人だけ映ってください（${status.count} / ${status.total}）`
              : `少しずつ顔の向きを変えてください（${status.count} / ${status.total}）`,
          );
          break;
        case 'timeout':
          stop();
          this.setStatus(`時間内に撮影できませんでした（${status.count} / ${status.total}）。もう一度試してください。`);
          break;
        case 'done':
          stop();
          void this.save(name, status.embeddings, status.thumbnailSource);
          break;
      }
    };
    const unsubscribe = this.stage.onAnalysis((analysis) =>
      handle(
        session.accept(
          { faceCount: analysis.faces.length, embedding: analysis.largest?.embedding, aligned: analysis.largest?.aligned },
          performance.now(),
        ),
      ),
    );
    const timer = setInterval(() => this.stage.requestEmbedding(), ENROLL_INTERVAL_MS);
    const timeout = setTimeout(() => handle(session.accept({ faceCount: 0 }, performance.now())), ENROLL_TIMEOUT_MS + 50);
    const stop = () => {
      unsubscribe();
      clearInterval(timer);
      clearTimeout(timeout);
      this.cancelCapture = undefined;
      this.updateButton();
    };
    this.cancelCapture = () => {
      stop();
      this.setStatus('撮影を中止しました。');
    };
    this.setStatus(`少しずつ顔の向きを変えてください（0 / ${ENROLL_SHOTS}）`);
    this.stage.requestEmbedding();
    this.updateButton();
  }

  private async save(name: string, embeddings: Float32Array[], aligned: RgbaImage): Promise<void> {
    try {
      const thumbnail = await toJpegBlob(aligned);
      await this.store.addEnrollment(name, embeddings, thumbnail);
      this.onPeopleChanged(await this.store.listPeople());
      this.nameInput.value = '';
      this.updateButton();
      this.setStatus(`${name} さんを登録しました。`);
    } catch (error) {
      console.error(error);
      this.setStatus('保存できませんでした。');
    }
  }

  private async deletePerson(id: string): Promise<void> {
    await this.store.deletePerson(id);
    this.onPeopleChanged(await this.store.listPeople());
  }

  private async deleteAll(): Promise<void> {
    if (!window.confirm('登録した人をすべて削除しますか？')) return;
    await this.store.deleteAll();
    this.onPeopleChanged(await this.store.listPeople());
  }
}
