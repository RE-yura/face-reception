import jpeg from 'jpeg-js';
import { readFileSync } from 'node:fs';
import { warpAffine } from '../../src/vision/align.ts';
import { detectorLayout, type DetectorLayout } from '../../src/vision/detector.ts';
import type { RgbaImage } from '../../src/vision/types.ts';

export const FACE_NAMES = ['meir-a', 'meir-b', 'kim'] as const;
export type FaceName = (typeof FACE_NAMES)[number];

export function loadFace(name: FaceName): RgbaImage {
  const file = new URL(`../fixtures/faces/${name}.jpg`, import.meta.url);
  return jpeg.decode(readFileSync(file), { useTArray: true, formatAsRGBA: true });
}

/** Resizes and pads a frame for YuNet the way the worker's canvas does (pixel-centre aligned bilinear). */
export function prepareDetectorInput(frame: RgbaImage, longSide: number): { input: RgbaImage; layout: DetectorLayout } {
  const layout = detectorLayout(frame.width, frame.height, longSide);
  const s = layout.scale;
  const input = warpAffine(frame, [s, 0, 0.5 * s - 0.5, 0, s, 0.5 * s - 0.5], layout.width, layout.height);
  return { input, layout };
}

export interface Placement {
  name: FaceName;
  /** Region of the fixture to copy: [x, y, width, height] */
  crop: [number, number, number, number];
  scale: number;
  /** Top-left corner of the copy in the composed frame */
  at: [number, number];
  /** Multiplies the pixel values, to make a face harder to detect without hiding it */
  gain?: number;
}

/** Pastes scaled crops of the face fixtures onto a grey frame, e.g. to put two people in one picture. */
export function composeFaces(width: number, height: number, placements: Placement[]): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) data.set([128, 128, 128, 255], i);
  for (const { name, crop, scale: s, at, gain = 1 } of placements) {
    const [cx, cy, cw, ch] = crop;
    const w = Math.round(cw * s);
    const h = Math.round(ch * s);
    const piece = warpAffine(loadFace(name), [s, 0, 0.5 * s - 0.5 - cx * s, 0, s, 0.5 * s - 0.5 - cy * s], w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const tx = at[0] + x;
        const ty = at[1] + y;
        if (tx < 0 || ty < 0 || tx >= width || ty >= height) continue;
        const o = (ty * width + tx) * 4;
        const p = (y * w + x) * 4;
        for (let c = 0; c < 3; c++) data[o + c] = piece.data[p + c] * gain;
      }
    }
  }
  return { width, height, data };
}
