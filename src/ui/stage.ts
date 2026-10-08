import { AnalysisHealth } from '../analysis-health.ts';
import { cameraIsLive, closeCamera, openCamera } from '../camera.ts';
import { ANALYZE_MAX_FAILURES } from '../config.ts';
import type { VisionClient } from '../vision-client.ts';
import { largestFace } from '../vision/detector.ts';
import type { Analysis } from '../worker-protocol.ts';
import { toViewBox } from './geometry.ts';

type Listener = (analysis: Analysis) => void;

const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** Camera preview with face boxes drawn on top. Runs the detection loop and hands each analysis to listeners. */
export class Stage {
  private readonly root: HTMLElement;
  private readonly video: HTMLVideoElement;
  private readonly overlay: HTMLCanvasElement;
  private readonly client: VisionClient;
  private readonly listeners = new Set<Listener>();
  private running = false;
  private generation = 0;
  private embedRequested = false;
  private label: string | null = null;
  private last: Analysis | null = null;
  private health = new AnalysisHealth(ANALYZE_MAX_FAILURES);
  private fatalHandler: (() => void) | undefined;

  constructor(root: HTMLElement, video: HTMLVideoElement, overlay: HTMLCanvasElement, client: VisionClient) {
    this.root = root;
    this.video = video;
    this.overlay = overlay;
    this.client = client;
  }

  /** Resolves false when the page went to the background before the camera opened. */
  openCamera(): Promise<boolean> {
    return openCamera(this.video);
  }

  cameraLive(): boolean {
    return cameraIsLive(this.video);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.health = new AnalysisHealth(ANALYZE_MAX_FAILURES);
    void this.loop(++this.generation);
  }

  /** Stops the loop and the camera. */
  stop(): void {
    this.running = false;
    this.generation++;
    closeCamera(this.video);
    this.last = null;
    this.root.dataset.faceCount = '0';
    this.draw();
  }

  /** The next analyzed frame will include the embedding of its largest face. */
  requestEmbedding(): void {
    this.embedRequested = true;
  }

  /** Text drawn above the largest face, or null for none. */
  setLabel(label: string | null): void {
    this.label = label;
    this.draw();
  }

  onAnalysis(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Called after the stage stopped because the worker stalled or kept failing. */
  onFatal(handler: () => void): void {
    this.fatalHandler = handler;
  }

  private async loop(generation: number): Promise<void> {
    while (generation === this.generation) {
      if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || this.video.videoWidth === 0) {
        await nextFrame();
        continue;
      }
      const embed = this.embedRequested;
      this.embedRequested = false;
      try {
        const analysis = await this.client.analyze(await createImageBitmap(this.video), embed);
        if (generation !== this.generation) break;
        this.health.succeeded();
        this.last = analysis;
        this.root.dataset.faceCount = String(analysis.faces.length);
        this.draw();
        for (const listener of this.listeners) listener(analysis);
      } catch (error) {
        if (generation !== this.generation) break;
        console.error(error);
        if (this.health.failed(error)) {
          this.stop();
          this.fatalHandler?.();
          break;
        }
        if (embed) this.embedRequested = true;
        await sleep(200);
      }
      await nextFrame();
    }
  }

  private draw(): void {
    const canvas = this.overlay;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const a = this.last;
    if (!a) return;
    const style = getComputedStyle(canvas);
    const color = style.getPropertyValue('--box').trim() || '#ffffff';
    const accent = style.getPropertyValue('--box-accent').trim() || '#ffd400';
    const labelColor = style.getPropertyValue('--label-fg').trim() || '#000000';
    const largest = largestFace(a.faces);
    for (const face of a.faces) {
      const b = toViewBox(face.box, a.frameWidth, a.frameHeight, width, height, true);
      const isLargest = face === largest;
      ctx.strokeStyle = isLargest ? accent : color;
      ctx.lineWidth = isLargest ? 3 : 2;
      ctx.strokeRect(b.x, b.y, b.width, b.height);
      if (isLargest && this.label) {
        ctx.font = style.getPropertyValue('--label-font').trim() || 'bold 16px sans-serif';
        const padding = 6;
        const labelHeight = 26;
        const labelWidth = ctx.measureText(this.label).width + padding * 2;
        const top = Math.max(0, b.y - labelHeight - 4);
        ctx.fillStyle = accent;
        ctx.fillRect(b.x, top, labelWidth, labelHeight);
        ctx.fillStyle = labelColor;
        ctx.textBaseline = 'middle';
        ctx.fillText(this.label, b.x + padding, top + labelHeight / 2);
      }
    }
  }
}
