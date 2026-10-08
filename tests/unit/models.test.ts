import * as ort from 'onnxruntime-web/wasm';
import { statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { describe, expect, it } from 'vitest';
import { MODELS, ORT_WASM_BYTES } from '../../src/config.ts';
import { readModel } from '../helpers/models.ts';

ort.env.wasm.numThreads = 1;
const options: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], logSeverityLevel: 3 };

describe('model files', () => {
  it('have the sizes the loading progress bar expects', () => {
    expect(readModel(MODELS.yunet.file).byteLength).toBe(MODELS.yunet.bytes);
    expect(readModel(MODELS.sface.file).byteLength).toBe(MODELS.sface.bytes);
  });

  it('include the ONNX Runtime wasm, whose size the download checks too', () => {
    const wasm = createRequire(import.meta.url).resolve('onnxruntime-web/ort-wasm-simd-threaded.wasm');
    expect(statSync(wasm).size).toBe(ORT_WASM_BYTES);
  });

  it('YuNet takes a 320×256 BGR image and returns cls/obj/bbox/kps per stride', async () => {
    const session = await ort.InferenceSession.create(readModel(MODELS.yunet.file), options);
    expect(session.inputNames).toEqual(['input']);
    const out = await session.run({ input: new ort.Tensor('float32', new Float32Array(3 * 256 * 320), [1, 3, 256, 320]) });
    for (const [stride, cells] of [[8, 1280], [16, 320], [32, 80]] as const) {
      expect(out[`cls_${stride}`].dims).toEqual([1, cells, 1]);
      expect(out[`obj_${stride}`].dims).toEqual([1, cells, 1]);
      expect(out[`bbox_${stride}`].dims).toEqual([1, cells, 4]);
      expect(out[`kps_${stride}`].dims).toEqual([1, cells, 10]);
    }
  });

  it('SFace takes a 112×112 RGB face and returns a 128-d feature', async () => {
    const session = await ort.InferenceSession.create(readModel(MODELS.sface.file), options);
    expect(session.inputNames).toEqual(['data']);
    const out = await session.run({ data: new ort.Tensor('float32', new Float32Array(3 * 112 * 112).fill(128), [1, 3, 112, 112]) });
    expect(out.fc1.dims).toEqual([1, 128]);
  });
});
