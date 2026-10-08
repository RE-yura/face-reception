import type { Point, RgbaImage } from './types.ts';

export const ALIGNED_SIZE = 112;

/** Five-point ArcFace template for a 112×112 crop (same values as OpenCV's FaceRecognizerSF). */
export const ARCFACE_TEMPLATE: Point[] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

/** 2×3 affine matrix in OpenCV order: x' = m[0]x + m[1]y + m[2], y' = m[3]x + m[4]y + m[5]. */
export type Affine = [number, number, number, number, number, number];

/** Least-squares similarity transform (rotation + uniform scale + translation) mapping src onto dst. */
export function estimateSimilarity(src: Point[], dst: Point[]): Affine {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    sx += src[i][0]; sy += src[i][1];
    dx += dst[i][0]; dy += dst[i][1];
  }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let num = 0, numCross = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const ax = src[i][0] - sx, ay = src[i][1] - sy;
    const bx = dst[i][0] - dx, by = dst[i][1] - dy;
    num += ax * bx + ay * by;
    numCross += ax * by - ay * bx;
    den += ax * ax + ay * ay;
  }
  const a = num / den;
  const b = numCross / den;
  return [a, -b, dx - (a * sx - b * sy), b, a, dy - (b * sx + a * sy)];
}

export function invertAffine(m: Affine): Affine {
  const [a, b, c, d, e, f] = m;
  const det = a * e - b * d;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  return [ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)];
}

/** Warps `src` by `m` (src → dst coordinates) into a width×height image with bilinear sampling; outside pixels are black. */
export function warpAffine(src: RgbaImage, m: Affine, width: number, height: number): RgbaImage {
  const inv = invertAffine(m);
  const out = new Uint8ClampedArray(width * height * 4);
  const sw = src.width, sh = src.height, sd = src.data;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const fx = inv[0] * x + inv[1] * y + inv[2];
      const fy = inv[3] * x + inv[4] * y + inv[5];
      const x0 = Math.floor(fx), y0 = Math.floor(fy);
      const wx = fx - x0, wy = fy - y0;
      const o = (y * width + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        let v = 0;
        for (let j = 0; j < 2; j++) {
          const yy = y0 + j;
          if (yy < 0 || yy >= sh) continue;
          const wyj = j === 0 ? 1 - wy : wy;
          for (let i = 0; i < 2; i++) {
            const xx = x0 + i;
            if (xx < 0 || xx >= sw) continue;
            const wxi = i === 0 ? 1 - wx : wx;
            v += sd[(yy * sw + xx) * 4 + ch] * wxi * wyj;
          }
        }
        out[o + ch] = v;
      }
      out[o + 3] = 255;
    }
  }
  return { width, height, data: out };
}

/** Crops the face to 112×112 so that its five landmarks land on the ArcFace template. */
export function alignFace(src: RgbaImage, landmarks: Point[]): RgbaImage {
  return warpAffine(src, estimateSimilarity(landmarks, ARCFACE_TEMPLATE), ALIGNED_SIZE, ALIGNED_SIZE);
}
