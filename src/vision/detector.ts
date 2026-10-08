import type { Face, Point, RgbaImage } from './types.ts';

export const YUNET_STRIDES = [8, 16, 32] as const;

/** Size of the YuNet input: the frame scaled so its long side is `longSide`, then padded right/bottom to multiples of 32. */
export interface DetectorLayout {
  width: number;
  height: number;
  /** input pixels per frame pixel */
  scale: number;
}

export function detectorLayout(frameWidth: number, frameHeight: number, longSide: number): DetectorLayout {
  const scale = longSide / Math.max(frameWidth, frameHeight);
  return {
    width: Math.ceil((frameWidth * scale) / 32) * 32,
    height: Math.ceil((frameHeight * scale) / 32) * 32,
    scale,
  };
}

/** RGBA → YuNet input tensor data: BGR, 0–255, NCHW. */
export function toYuNetInput(img: RgbaImage): Float32Array {
  const plane = img.width * img.height;
  const out = new Float32Array(plane * 3);
  const d = img.data;
  for (let i = 0; i < plane; i++) {
    out[i] = d[i * 4 + 2];
    out[plane + i] = d[i * 4 + 1];
    out[plane * 2 + i] = d[i * 4];
  }
  return out;
}

/** YuNet outputs keyed by name: cls_8, obj_8, bbox_8, kps_8, … for strides 8/16/32. */
export type YuNetOutputs = Record<string, { data: unknown }>;

const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);

/** Decodes raw YuNet outputs into faces in input-pixel coordinates (same formulas as OpenCV's FaceDetectorYN). */
export function decodeYuNet(outputs: YuNetOutputs, inputWidth: number, inputHeight: number, scoreThreshold: number): Face[] {
  const faces: Face[] = [];
  for (const stride of YUNET_STRIDES) {
    const cols = inputWidth / stride;
    const rows = inputHeight / stride;
    const cls = outputs[`cls_${stride}`].data as Float32Array;
    const obj = outputs[`obj_${stride}`].data as Float32Array;
    const bbox = outputs[`bbox_${stride}`].data as Float32Array;
    const kps = outputs[`kps_${stride}`].data as Float32Array;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const score = Math.sqrt(clamp01(cls[idx]) * clamp01(obj[idx]));
        if (score < scoreThreshold) continue;
        const cx = (c + bbox[idx * 4]) * stride;
        const cy = (r + bbox[idx * 4 + 1]) * stride;
        const w = Math.exp(bbox[idx * 4 + 2]) * stride;
        const h = Math.exp(bbox[idx * 4 + 3]) * stride;
        const landmarks: Point[] = [];
        for (let n = 0; n < 5; n++) {
          landmarks.push([(kps[idx * 10 + n * 2] + c) * stride, (kps[idx * 10 + n * 2 + 1] + r) * stride]);
        }
        faces.push({ box: { x: cx - w / 2, y: cy - h / 2, width: w, height: h }, landmarks, score });
      }
    }
  }
  return faces;
}

function iou(a: Face['box'], b: Face['box']): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.width * a.height + b.width * b.height - inter);
}

/** Greedy non-maximum suppression: keeps the highest-scoring face of each overlapping group, sorted by score. */
export function nms(faces: Face[], iouThreshold: number): Face[] {
  const sorted = [...faces].sort((p, q) => q.score - p.score);
  const kept: Face[] = [];
  for (const f of sorted) {
    if (kept.every((k) => iou(k.box, f.box) <= iouThreshold)) kept.push(f);
  }
  return kept;
}

/** The face with the biggest box (the first one on ties), or undefined for none. */
export function largestFace(faces: Face[]): Face | undefined {
  let best: Face | undefined;
  for (const f of faces) if (!best || f.box.width * f.box.height > best.box.width * best.box.height) best = f;
  return best;
}

/** Maps a face from input-pixel coordinates back to frame coordinates. */
export function toFrameCoords(face: Face, layout: DetectorLayout): Face {
  const s = 1 / layout.scale;
  return {
    box: { x: face.box.x * s, y: face.box.y * s, width: face.box.width * s, height: face.box.height * s },
    landmarks: face.landmarks.map(([x, y]) => [x * s, y * s] as Point),
    score: face.score,
  };
}

export function postprocessYuNet(
  outputs: YuNetOutputs,
  layout: DetectorLayout,
  scoreThreshold: number,
  iouThreshold: number,
): Face[] {
  return nms(decodeYuNet(outputs, layout.width, layout.height, scoreThreshold), iouThreshold).map((f) =>
    toFrameCoords(f, layout),
  );
}
