export type Point = [number, number];

/** RGBA pixels, row-major. Structurally compatible with the DOM ImageData. */
export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}

export interface Face {
  box: { x: number; y: number; width: number; height: number };
  /** right eye, left eye, nose tip, right mouth corner, left mouth corner (image coordinates) */
  landmarks: Point[];
  score: number;
}
