import { describe, expect, it } from 'vitest';
import { toViewBox } from '../../src/ui/geometry.ts';

const box = { x: 200, y: 100, width: 100, height: 50 };

describe('toViewBox', () => {
  it('crops the sides when a landscape frame fills a portrait view', () => {
    expect(toViewBox(box, 640, 480, 320, 480, false)).toEqual({ x: 40, y: 100, width: 100, height: 50 });
  });

  it('flips horizontally for a mirrored view', () => {
    expect(toViewBox(box, 640, 480, 320, 480, true)).toEqual({ x: 180, y: 100, width: 100, height: 50 });
  });

  it('handles the portrait frames an iPhone front camera delivers', () => {
    // 480×640 frame in a 390×300 view: scale 0.8125, frame shown 390×520, 110px cropped top and bottom
    expect(toViewBox(box, 480, 640, 390, 300, true)).toEqual({ x: 146.25, y: -28.75, width: 81.25, height: 40.625 });
  });

  it('scales when the view is smaller than the frame', () => {
    expect(toViewBox(box, 640, 480, 320, 240, false)).toEqual({ x: 100, y: 50, width: 50, height: 25 });
  });
});
