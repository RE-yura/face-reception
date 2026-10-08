import type { RgbaImage } from './types.ts';

/** Aligned 112×112 RGBA → SFace input tensor data: RGB, 0–255, NCHW (SFace normalizes internally). */
export function toSFaceInput(aligned: RgbaImage): Float32Array {
  const plane = aligned.width * aligned.height;
  const out = new Float32Array(plane * 3);
  const d = aligned.data;
  for (let i = 0; i < plane; i++) {
    out[i] = d[i * 4];
    out[plane + i] = d[i * 4 + 1];
    out[plane * 2 + i] = d[i * 4 + 2];
  }
  return out;
}

export function l2normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (const x of v) sum += x * x;
  const norm = Math.sqrt(sum);
  return v.map((x) => x / norm);
}
