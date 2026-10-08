import * as ort from 'onnxruntime-web/wasm';
import { alignFace } from './align.ts';
import { largestFace, postprocessYuNet, toYuNetInput, type DetectorLayout } from './detector.ts';
import { l2normalize, toSFaceInput } from './embedder.ts';
import type { Face, RgbaImage } from './types.ts';

export interface DetectOptions {
  scoreThreshold: number;
  iouThreshold: number;
}

export interface FrameAnalysis {
  faces: Face[];
  /** Present only when a frame was given for embedding and at least one face was found. */
  largest?: { embedding: Float32Array; aligned: RgbaImage };
}

/** Runs YuNet and SFace. Environment-agnostic: used by the Web Worker and by Node tests. */
export class FacePipeline {
  private readonly yunet: ort.InferenceSession;
  private readonly sface: ort.InferenceSession;

  private constructor(yunet: ort.InferenceSession, sface: ort.InferenceSession) {
    this.yunet = yunet;
    this.sface = sface;
  }

  static async create(yunetModel: Uint8Array, sfaceModel: Uint8Array): Promise<FacePipeline> {
    const opts: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], logSeverityLevel: 3 };
    const yunet = await ort.InferenceSession.create(yunetModel, opts);
    const sface = await ort.InferenceSession.create(sfaceModel, opts);
    return new FacePipeline(yunet, sface);
  }

  /** `input` is the frame already resized and padded to layout.width × layout.height. Returns faces in frame coordinates. */
  async detect(input: RgbaImage, layout: DetectorLayout, opts: DetectOptions): Promise<Face[]> {
    const tensor = new ort.Tensor('float32', toYuNetInput(input), [1, 3, layout.height, layout.width]);
    const outputs = await this.yunet.run({ input: tensor });
    return postprocessYuNet(outputs, layout, opts.scoreThreshold, opts.iouThreshold);
  }

  /** `frame` is the full-resolution frame the face was detected in. */
  async embed(frame: RgbaImage, face: Face): Promise<{ embedding: Float32Array; aligned: RgbaImage }> {
    const aligned = alignFace(frame, face.landmarks);
    const tensor = new ort.Tensor('float32', toSFaceInput(aligned), [1, 3, aligned.height, aligned.width]);
    const outputs = await this.sface.run({ data: tensor });
    return { embedding: l2normalize(outputs.fc1.data as Float32Array), aligned };
  }

  /**
   * Detects faces and, when `frame` is given, embeds the largest one. `frame` returns the full-resolution
   * frame and is called only when there is a face to embed.
   */
  async analyze(input: RgbaImage, layout: DetectorLayout, opts: DetectOptions, frame: (() => RgbaImage) | undefined): Promise<FrameAnalysis> {
    const faces = await this.detect(input, layout, opts);
    const largest = largestFace(faces);
    if (!frame || !largest) return { faces };
    return { faces, largest: await this.embed(frame(), largest) };
  }
}
