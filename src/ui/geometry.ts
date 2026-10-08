import type { Face } from '../vision/types.ts';

export type Box = Face['box'];

/**
 * Maps a box from camera-frame pixels to the on-screen video element, which shows the frame with
 * `object-fit: cover` and, when `mirrored`, flipped horizontally.
 */
export function toViewBox(box: Box, frameWidth: number, frameHeight: number, viewWidth: number, viewHeight: number, mirrored: boolean): Box {
  const scale = Math.max(viewWidth / frameWidth, viewHeight / frameHeight);
  const offsetX = (viewWidth - frameWidth * scale) / 2;
  const offsetY = (viewHeight - frameHeight * scale) / 2;
  const width = box.width * scale;
  const left = box.x * scale + offsetX;
  return {
    x: mirrored ? viewWidth - left - width : left,
    y: box.y * scale + offsetY,
    width,
    height: box.height * scale,
  };
}
