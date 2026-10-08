import type { InitFailure } from './worker-protocol.ts';

export type CameraProblem = 'insecure' | 'denied' | 'not-found' | 'other';

export function classifyCameraError(error: unknown, secureContext: boolean): CameraProblem {
  if (!secureContext) return 'insecure';
  const name = error instanceof DOMException || error instanceof Error ? error.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') return 'denied';
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'not-found';
  return 'other';
}

export function cameraProblemMessage(problem: CameraProblem): string {
  switch (problem) {
    case 'insecure':
      return 'カメラは https:// で始まるページでしか使えません。';
    case 'denied':
      return 'カメラの使用が許可されていません。iPhone では「設定」の Safari の項目でカメラを「確認」か「許可」にしてから、もう一度試してください。';
    case 'not-found':
      return 'カメラが見つかりませんでした。';
    case 'other':
      return 'カメラを起動できませんでした。ほかのアプリがカメラを使っていないか確認してください。';
  }
}

export function initFailureMessage(reason: InitFailure): string {
  return reason === 'network'
    ? 'モデルの読み込みに失敗しました。通信環境を確認して、もう一度試してください。'
    : 'このブラウザには対応していません。最新の Safari か Chrome で開いてください。';
}
