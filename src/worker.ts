import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { DETECT_INPUT_LONG_SIDE, DETECT_SCORE_THRESHOLD, DOWNLOAD_IDLE_TIMEOUT_MS, MODELS, NMS_IOU_THRESHOLD, ORT_WASM_BYTES } from './config.ts';
import { DownloadError, downloadFiles } from './model-download.ts';
import { detectorLayout } from './vision/detector.ts';
import { FacePipeline } from './vision/pipeline.ts';
import type { Analysis, WorkerRequest, WorkerResponse } from './worker-protocol.ts';

// The wasm binary comes from our own build output, downloaded in init() and handed over as wasmBinary.
// GitHub Pages cannot send COOP/COEP, so stay single-threaded.
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';

let pipeline: FacePipeline | undefined;
let inputCanvas: OffscreenCanvas | undefined;
let frameCanvas: OffscreenCanvas | undefined;

function send(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, { transfer });
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
  // The wasm goes through the same download as the models, so a slow or broken connection while it loads
  // gets the idle timeout, the size check and a retry, instead of looking like an unsupported browser.
  const stop = new AbortController();
  const [yunet, wasm, sface] = downloadFiles(
    [
      { url: `${modelBaseUrl}${MODELS.yunet.file}`, bytes: MODELS.yunet.bytes },
      { url: wasmUrl, bytes: ORT_WASM_BYTES },
      { url: `${modelBaseUrl}${MODELS.sface.file}`, bytes: MODELS.sface.bytes },
    ],
    {
      onProgress: (loaded, total) => send({ type: 'progress', loaded, total }),
      idleTimeoutMs: DOWNLOAD_IDLE_TIMEOUT_MS,
      signal: stop.signal,
    },
  );
  const yunetWithRuntime = Promise.all([yunet, wasm]).then(([model, binary]) => {
    ort.env.wasm.wasmBinary = binary;
    return model;
  });
  try {
    pipeline = await FacePipeline.create(yunetWithRuntime, sface);
  } catch (error) {
    // When YuNet cannot be loaded, SFace (about 36MB) may still be on its way; it is no use now.
    stop.abort();
    send({ type: 'init-error', reason: error instanceof DownloadError ? 'network' : 'unsupported', message: String(error) });
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
