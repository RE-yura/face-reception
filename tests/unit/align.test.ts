import { describe, expect, it } from 'vitest';
import { ALIGNED_SIZE, ARCFACE_TEMPLATE, alignFace, estimateSimilarity, invertAffine, warpAffine, type Affine } from '../../src/vision/align.ts';
import type { Point, RgbaImage } from '../../src/vision/types.ts';

const apply = (m: Affine, [x, y]: Point): Point => [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];

function similarity(scale: number, angle: number, tx: number, ty: number): Affine {
  const a = scale * Math.cos(angle);
  const b = scale * Math.sin(angle);
  return [a, -b, tx, b, a, ty];
}

/** R = 10·x, G = 10·y, B = 0 */
function gradient(width: number, height: number): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      data[o] = x * 10;
      data[o + 1] = y * 10;
      data[o + 3] = 255;
    }
  }
  return { width, height, data };
}

const pixel = (img: RgbaImage, x: number, y: number) => Array.from(img.data.slice((y * img.width + x) * 4, (y * img.width + x) * 4 + 4));

describe('estimateSimilarity', () => {
  it('recovers a known rotation, scale and translation', () => {
    const truth = similarity(1.7, 0.4, 12, -30);
    const src: Point[] = ARCFACE_TEMPLATE.map(([x, y]) => [x * 3 + 5, y * 2 - 1]);
    const m = estimateSimilarity(src, src.map((p) => apply(truth, p)));
    m.forEach((v, i) => expect(v).toBeCloseTo(truth[i], 6));
  });

  it('returns the identity for identical point sets', () => {
    const m = estimateSimilarity(ARCFACE_TEMPLATE, ARCFACE_TEMPLATE);
    [1, 0, 0, 0, 1, 0].forEach((v, i) => expect(m[i]).toBeCloseTo(v, 9));
  });
});

describe('invertAffine', () => {
  it('undoes the transform', () => {
    const m = similarity(0.5, -1.1, 40, 7);
    const back = apply(invertAffine(m), apply(m, [13, 29]));
    expect(back[0]).toBeCloseTo(13, 9);
    expect(back[1]).toBeCloseTo(29, 9);
  });
});

describe('warpAffine', () => {
  it('moves pixels under a translation', () => {
    const out = warpAffine(gradient(10, 10), [1, 0, 2, 0, 1, 3], 10, 10);
    expect(pixel(out, 5, 5)).toEqual([30, 20, 0, 255]);
  });

  it('interpolates bilinearly between source pixels', () => {
    const out = warpAffine(gradient(10, 10), [1, 0, 0.5, 0, 1, 0], 10, 10);
    expect(pixel(out, 5, 5)).toEqual([45, 50, 0, 255]);
  });

  it('fills pixels that fall outside the source with opaque black', () => {
    const out = warpAffine(gradient(10, 10), [1, 0, 5, 0, 1, 5], 10, 10);
    expect(pixel(out, 0, 0)).toEqual([0, 0, 0, 255]);
  });
});

describe('alignFace', () => {
  it('moves the five landmarks onto the ArcFace template in a 112×112 crop', () => {
    const toImage = similarity(2.2, 0.3, 90, 40);
    const landmarks = ARCFACE_TEMPLATE.map((p) => apply(toImage, p));
    const width = 400;
    const height = 400;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 3; i < data.length; i += 4) data[i] = 255;
    for (const [lx, ly] of landmarks) {
      for (let y = Math.round(ly) - 3; y <= Math.round(ly) + 3; y++) {
        for (let x = Math.round(lx) - 3; x <= Math.round(lx) + 3; x++) data.fill(255, (y * width + x) * 4, (y * width + x) * 4 + 3);
      }
    }
    const aligned = alignFace({ width, height, data }, landmarks);
    expect([aligned.width, aligned.height]).toEqual([ALIGNED_SIZE, ALIGNED_SIZE]);
    for (const [tx, ty] of ARCFACE_TEMPLATE) expect(pixel(aligned, Math.round(tx), Math.round(ty))[0]).toBeGreaterThan(200);
    expect(pixel(aligned, 56, 20)[0]).toBe(0);
  });
});
