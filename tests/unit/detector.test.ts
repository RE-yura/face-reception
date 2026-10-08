import { describe, expect, it } from 'vitest';
import { decodeYuNet, detectorLayout, largestFace, nms, toFrameCoords, toYuNetInput, YUNET_STRIDES } from '../../src/vision/detector.ts';
import type { Face } from '../../src/vision/types.ts';

function emptyOutputs(width: number, height: number) {
  const out: Record<string, { data: Float32Array }> = {};
  for (const s of YUNET_STRIDES) {
    const cells = (width / s) * (height / s);
    out[`cls_${s}`] = { data: new Float32Array(cells) };
    out[`obj_${s}`] = { data: new Float32Array(cells) };
    out[`bbox_${s}`] = { data: new Float32Array(cells * 4) };
    out[`kps_${s}`] = { data: new Float32Array(cells * 10) };
  }
  return out;
}

const face = (x: number, y: number, size: number, score: number): Face => ({
  box: { x, y, width: size, height: size },
  landmarks: [[x, y], [x, y], [x, y], [x, y], [x, y]],
  score,
});

describe('detectorLayout', () => {
  it('scales the long side to 320 and pads to multiples of 32', () => {
    expect(detectorLayout(640, 480, 320)).toEqual({ width: 320, height: 256, scale: 0.5 });
    expect(detectorLayout(480, 640, 320)).toEqual({ width: 256, height: 320, scale: 0.5 });
    expect(detectorLayout(1280, 720, 320)).toEqual({ width: 320, height: 192, scale: 0.25 });
  });
});

describe('toYuNetInput', () => {
  it('converts RGBA to planar BGR in 0–255', () => {
    const img = { width: 2, height: 1, data: new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]) };
    expect(Array.from(toYuNetInput(img))).toEqual([30, 60, 20, 50, 10, 40]);
  });
});

describe('decodeYuNet', () => {
  it('decodes box, landmarks and score of a cell', () => {
    const out = emptyOutputs(64, 32);
    // stride 16 grid is 4×2; cell (row 1, col 2) has index 6
    out.cls_16.data[6] = 0.81;
    out.obj_16.data[6] = 1;
    out.bbox_16.data.set([0.5, 0.25, Math.log(2), Math.log(1.5)], 6 * 4);
    out.kps_16.data.set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 6 * 10);
    const faces = decodeYuNet(out, 64, 32, 0.5);
    expect(faces).toHaveLength(1);
    const [f] = faces;
    expect(f.score).toBeCloseTo(0.9, 6);
    expect(f.box.x).toBeCloseTo(24, 4);
    expect(f.box.y).toBeCloseTo(8, 4);
    expect(f.box.width).toBeCloseTo(32, 4);
    expect(f.box.height).toBeCloseTo(24, 4);
    expect(f.landmarks[0][0]).toBeCloseTo(33.6, 4);
    expect(f.landmarks[0][1]).toBeCloseTo(19.2, 4);
    expect(f.landmarks[4][0]).toBeCloseTo(46.4, 4);
    expect(f.landmarks[4][1]).toBeCloseTo(32, 4);
  });

  it('drops cells below the score threshold and clamps scores above 1', () => {
    const out = emptyOutputs(64, 32);
    out.cls_8.data[0] = 0.25;
    out.obj_8.data[0] = 1; // score 0.5
    out.cls_32.data[1] = 1.4;
    out.obj_32.data[1] = 1; // clamped to 1
    const faces = decodeYuNet(out, 64, 32, 0.8);
    expect(faces.map((f) => f.score)).toEqual([1]);
  });
});

describe('nms', () => {
  it('keeps the best of overlapping faces and every separate face, best first', () => {
    const kept = nms([face(0, 0, 100, 0.85), face(10, 10, 100, 0.95), face(300, 0, 100, 0.9)], 0.3);
    expect(kept.map((f) => f.score)).toEqual([0.95, 0.9]);
  });

  it('keeps overlapping faces whose IoU is at or below the threshold', () => {
    const kept = nms([face(0, 0, 100, 0.9), face(80, 0, 100, 0.8)], 0.3);
    expect(kept).toHaveLength(2);
  });
});

describe('toFrameCoords', () => {
  it('undoes the detector scaling', () => {
    const f = toFrameCoords(face(10, 20, 30, 0.9), { width: 320, height: 256, scale: 0.5 });
    expect(f.box).toEqual({ x: 20, y: 40, width: 60, height: 60 });
    expect(f.landmarks[0]).toEqual([20, 40]);
    expect(f.score).toBe(0.9);
  });
});

describe('largestFace', () => {
  it('picks the face with the biggest box, so only the person nearest the camera is recognized', () => {
    expect(largestFace([face(0, 0, 50, 0.99), face(100, 0, 120, 0.85), face(300, 0, 80, 0.9)])?.score).toBe(0.85);
  });

  it('returns undefined when there is no face', () => {
    expect(largestFace([])).toBeUndefined();
  });
});
