import { MATCH_THRESHOLD, RECOGNIZE_INTERVAL_MS } from '../config.ts';
import { matchFace } from '../match.ts';
import { ReceptionState, type ReceptionView } from '../reception-state.ts';
import type { Person } from '../store.ts';
import type { Analysis } from '../worker-protocol.ts';
import { byId } from './dom.ts';
import type { Stage } from './stage.ts';

export class ReceptionPanel {
  private readonly message = byId<HTMLElement>('reception-message');
  private readonly score = byId<HTMLElement>('reception-score');
  private readonly goEnroll = byId<HTMLButtonElement>('go-enroll');
  private readonly state = new ReceptionState();
  private readonly stage: Stage;
  private readonly getPeople: () => Person[];
  private timer: ReturnType<typeof setInterval> | undefined;
  private unsubscribe: (() => void) | undefined;
  /** What the message currently says, so the live region changes only when that does. */
  private shown = '';

  constructor(stage: Stage, getPeople: () => Person[], onGoEnroll: () => void) {
    this.stage = stage;
    this.getPeople = getPeople;
    this.goEnroll.addEventListener('click', onGoEnroll);
  }

  activate(): void {
    this.deactivate();
    this.state.reset();
    this.shown = '';
    this.render(this.state.update({ faceCount: 0, peopleCount: this.getPeople().length }));
    this.unsubscribe = this.stage.onAnalysis((analysis) => this.handle(analysis));
    this.stage.requestEmbedding();
    this.timer = setInterval(() => this.stage.requestEmbedding(), RECOGNIZE_INTERVAL_MS);
  }

  deactivate(): void {
    clearInterval(this.timer);
    this.timer = undefined;
    this.unsubscribe?.();
    this.unsubscribe = undefined;
    this.stage.setLabel(null);
  }

  private handle(analysis: Analysis): void {
    const people = this.getPeople();
    const match = analysis.largest ? matchFace(analysis.largest.embedding, people, MATCH_THRESHOLD) : undefined;
    this.render(this.state.update({ faceCount: analysis.faces.length, peopleCount: people.length, match }));
  }

  private render(view: ReceptionView): void {
    const score = view.kind === 'matched' || view.kind === 'unknown' ? `類似度 ${view.score.toFixed(2)}` : '';
    if (this.score.textContent !== score) this.score.textContent = score;
    // #reception-message is a live region: screen readers announce every change, so touch it only when it changes.
    const key = view.kind === 'matched' ? `matched:${view.name}` : view.kind;
    if (key === this.shown) return;
    this.shown = key;
    this.goEnroll.hidden = view.kind !== 'no-people';
    switch (view.kind) {
      case 'no-people':
        this.message.textContent = 'まずは登録してください';
        this.stage.setLabel(null);
        break;
      case 'no-face':
        this.message.textContent = 'カメラに顔を映してください';
        this.stage.setLabel(null);
        break;
      case 'checking':
        this.message.textContent = '確認しています…';
        this.stage.setLabel(null);
        break;
      case 'matched': {
        const name = document.createElement('strong');
        name.className = 'reception-name';
        name.textContent = view.name;
        this.message.replaceChildren('あなたは ', name, ' さんですね?');
        this.stage.setLabel(`${view.name} さん`);
        break;
      }
      case 'unknown':
        this.message.textContent = '登録されていません';
        this.stage.setLabel('未登録');
        break;
    }
  }
}
