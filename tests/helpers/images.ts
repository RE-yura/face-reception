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
