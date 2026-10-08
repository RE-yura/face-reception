import { describe, expect, it } from 'vitest';
import { l2normalize, toSFaceInput } from '../../src/vision/embedder.ts';

describe('toSFaceInput', () => {
  it('converts RGBA to planar RGB in 0–255', () => {
    const img = { width: 2, height: 1, data: new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]) };
    expect(Array.from(toSFaceInput(img))).toEqual([10, 40, 20, 50, 30, 60]);
  });
});

describe('l2normalize', () => {
  it('scales the vector to unit length', () => {
    const v = l2normalize(new Float32Array([3, 4]));
    expect(v[0]).toBeCloseTo(0.6, 6);
    expect(v[1]).toBeCloseTo(0.8, 6);
  });
});
