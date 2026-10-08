import type { Face, RgbaImage } from './vision/types.ts';

export interface LargestFace {
  embedding: Float32Array;
  /** 112×112 aligned crop, used as the enrollment thumbnail */
  aligned: RgbaImage;
}

export interface Analysis {
  faces: Face[];
  /** Present only when the request asked for an embedding and at least one face was found. */
  largest?: LargestFace;
  frameWidth: number;
  frameHeight: number;
}

export type InitFailure = 'network' | 'unsupported';

export type WorkerRequest =
  | { type: 'init'; modelBaseUrl: string }
  | { type: 'analyze'; id: number; frame: ImageBitmap; embed: boolean };

export type WorkerResponse =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'ready' }
  | { type: 'init-error'; reason: InitFailure; message: string }
  | { type: 'analysis'; id: number; analysis: Analysis }
  | { type: 'analyze-error'; id: number; message: string };
