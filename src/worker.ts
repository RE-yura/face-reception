import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { DETECT_INPUT_LONG_SIDE, DETECT_SCORE_THRESHOLD, MODELS, NMS_IOU_THRESHOLD } from './config.ts';
import { detectorLayout } from './vision/detector.ts';
import { FacePipeline } from './vision/pipeline.ts';
import type { Analysis, WorkerRequest, WorkerResponse } from './worker-protocol.ts';

// Serve the wasm binary from our own build output; GitHub Pages cannot send COOP/COEP, so stay single-threaded.
ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';

let pipeline: FacePipeline | undefined;
let inputCanvas: OffscreenCanvas | undefined;
let frameCanvas: OffscreenCanvas | undefined;

function send(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, { transfer });
}

async function fetchModels(baseUrl: string): Promise<[Uint8Array, Uint8Array]> {
  const total = MODELS.yunet.bytes + MODELS.sface.bytes;
  let loaded = 0;
  const fetchOne = async (file: string): Promise<Uint8Array> => {
    const res = await fetch(`${baseUrl}${file}`);
    if (!res.ok || !res.body) throw new Error(`${file}: HTTP ${res.status}`);
    const chunks: Uint8Array[] = [];
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      send({ type: 'progress', loaded: Math.min(loaded, total), total });
    }
    const bytes = new Uint8Array(chunks.reduce((n, c) => n + c.byteLength, 0));
    let offset = 0;
    for (const c of chunks) {
      bytes.set(c, offset);
      offset += c.byteLength;
    }
    return bytes;
  };
  return Promise.all([fetchOne(MODELS.yunet.file), fetchOne(MODELS.sface.file)]);
}

async function init(modelBaseUrl: string): Promise<void> {
  if (pipeline) {
    send({ type: 'ready' });
    return;
  }
  if (typeof OffscreenCanvas === 'undefined') {
    send({ type: 'init-error', reason: 'unsupported', message: 'OffscreenCanvas is unavailable' });
    return;
  }
  let models: [Uint8Array, Uint8Array];
  try {
    models = await fetchModels(modelBaseUrl);
  } catch (error) {
    send({ type: 'init-error', reason: 'network', message: String(error) });
    return;
  }
  try {
    pipeline = await FacePipeline.create(models[0], models[1]);
  } catch (error) {
    send({ type: 'init-error', reason: 'unsupported', message: String(error) });
    return;
  }
  send({ type: 'ready' });
}

function sized(canvas: OffscreenCanvas | undefined, width: number, height: number): OffscreenCanvas {
  if (!canvas) return new OffscreenCanvas(width, height);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return canvas;
}

function context2d(canvas: OffscreenCanvas): OffscreenCanvasRenderingContext2D {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D canvas is unavailable');
  return ctx;
}

async function analyze(frame: ImageBitmap, embed: boolean): Promise<{ analysis: Analysis; transfer: Transferable[] }> {
  try {
    if (!pipeline) throw new Error('models are not loaded');
    const { width, height } = frame;
    const layout = detectorLayout(width, height, DETECT_INPUT_LONG_SIDE);
    inputCanvas = sized(inputCanvas, layout.width, layout.height);
    const inputContext = context2d(inputCanvas);
    inputContext.clearRect(0, 0, layout.width, layout.height);
    inputContext.drawImage(frame, 0, 0, width * layout.scale, height * layout.scale);
    const opts = { scoreThreshold: DETECT_SCORE_THRESHOLD, iouThreshold: NMS_IOU_THRESHOLD };
    const fullFrame = () => {
      frameCanvas = sized(frameCanvas, width, height);
      const full = context2d(frameCanvas);
      full.drawImage(frame, 0, 0);
      return full.getImageData(0, 0, width, height);
    };
    const input = inputContext.getImageData(0, 0, layout.width, layout.height);
    const { faces, largest } = await pipeline.analyze(input, layout, opts, embed ? fullFrame : undefined);
    const analysis: Analysis = { faces, largest, frameWidth: width, frameHeight: height };
    const transfer: Transferable[] = largest ? [largest.embedding.buffer as ArrayBuffer, largest.aligned.data.buffer as ArrayBuffer] : [];
    return { analysis, transfer };
  } finally {
    frame.close();
  }
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  if (message.type === 'init') {
    void init(message.modelBaseUrl);
    return;
  }
  analyze(message.frame, message.embed).then(
    ({ analysis, transfer }) => send({ type: 'analysis', id: message.id, analysis }, transfer),
    (error: unknown) => send({ type: 'analyze-error', id: message.id, message: String(error) }),
  );
};
