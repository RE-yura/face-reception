import * as ort from 'onnxruntime-web/wasm';
import { readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { DETECT_INPUT_LONG_SIDE, DETECT_SCORE_THRESHOLD, MATCH_THRESHOLD, MODELS, NMS_IOU_THRESHOLD } from '../../src/config.ts';
import { dot } from '../../src/match.ts';
import { FacePipeline } from '../../src/vision/pipeline.ts';
import { FACE_NAMES, loadFace, prepareDetectorInput, type FaceName } from '../helpers/images.ts';
import { readModel } from '../helpers/models.ts';

interface Reference {
  box: [number, number, number, number];
  landmarks: [number, number][];
  embedding: number[];
}
const reference: Record<FaceName, Reference> = JSON.parse(
  readFileSync(new URL('../fixtures/opencv-reference.json', import.meta.url), 'utf8'),
);
const opts = { scoreThreshold: DETECT_SCORE_THRESHOLD, iouThreshold: NMS_IOU_THRESHOLD };

ort.env.wasm.numThreads = 1;
let pipeline: FacePipeline;
beforeAll(async () => {
  pipeline = await FacePipeline.create(readModel(MODELS.yunet.file), readModel(MODELS.sface.file));
});

describe('FacePipeline against OpenCV (full resolution)', () => {
  it.each(FACE_NAMES)('%s: same face box, landmarks and embedding as OpenCV', async (name) => {
    const frame = loadFace(name);
    const { input, layout } = prepareDetectorInput(frame, Math.max(frame.width, frame.height));
    const faces = await pipeline.detect(input, layout, opts);
    expect(faces).toHaveLength(1);
    const ref = reference[name];
    const { box, landmarks } = faces[0];
    [box.x, box.y, box.width, box.height].forEach((v, i) => expect(Math.abs(v - ref.box[i])).toBeLessThan(1));
    landmarks.forEach(([x, y], i) => {
      expect(Math.abs(x - ref.landmarks[i][0])).toBeLessThan(1);
      expect(Math.abs(y - ref.landmarks[i][1])).toBeLessThan(1);
    });
    const { embedding, aligned } = await pipeline.embed(frame, faces[0]);
    expect([aligned.width, aligned.height]).toEqual([112, 112]);
    expect(dot(embedding, ref.embedding)).toBeGreaterThan(0.99);
  });
});

describe('FacePipeline at the app setting (long side 320)', () => {
  const embeddings = {} as Record<FaceName, Float32Array>;
  beforeAll(async () => {
    for (const name of FACE_NAMES) {
      const frame = loadFace(name);
      const { input, layout } = prepareDetectorInput(frame, DETECT_INPUT_LONG_SIDE);
      const faces = await pipeline.detect(input, layout, opts);
      expect(faces).toHaveLength(1);
      embeddings[name] = (await pipeline.embed(frame, faces[0])).embedding;
    }
  });

  it('matches two different photos of the same person', () => {
    expect(dot(embeddings['meir-a'], embeddings['meir-b'])).toBeGreaterThanOrEqual(MATCH_THRESHOLD);
  });

  it('does not match different people', () => {
    expect(dot(embeddings['meir-a'], embeddings.kim)).toBeLessThan(MATCH_THRESHOLD);
    expect(dot(embeddings['meir-b'], embeddings.kim)).toBeLessThan(MATCH_THRESHOLD);
  });

  it('finds no face in a blank frame', async () => {
    const blank = { width: 640, height: 480, data: new Uint8ClampedArray(640 * 480 * 4) };
    const { input, layout } = prepareDetectorInput(blank, DETECT_INPUT_LONG_SIDE);
    expect(await pipeline.detect(input, layout, opts)).toEqual([]);
  });
});
