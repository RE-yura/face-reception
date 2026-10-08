import { describe, expect, it } from 'vitest';
import { cameraProblemMessage, classifyCameraError, initFailureMessage } from '../../src/errors.ts';

describe('classifyCameraError', () => {
  it('blames the page origin when the context is not secure', () => {
    expect(classifyCameraError(new DOMException('x', 'NotAllowedError'), false)).toBe('insecure');
  });

  it('maps getUserMedia errors', () => {
    expect(classifyCameraError(new DOMException('x', 'NotAllowedError'), true)).toBe('denied');
    expect(classifyCameraError(new DOMException('x', 'SecurityError'), true)).toBe('denied');
    expect(classifyCameraError(new DOMException('x', 'NotFoundError'), true)).toBe('not-found');
    expect(classifyCameraError(new DOMException('x', 'OverconstrainedError'), true)).toBe('not-found');
    expect(classifyCameraError(new DOMException('x', 'NotReadableError'), true)).toBe('other');
    expect(classifyCameraError('weird', true)).toBe('other');
  });
});

describe('messages', () => {
  it('explains how to allow the camera on iPhone when denied', () => {
    expect(cameraProblemMessage('denied')).toContain('設定');
  });

  it('tells network failures apart from unsupported browsers', () => {
    expect(initFailureMessage('network')).toContain('もう一度');
    expect(initFailureMessage('unsupported')).toContain('対応していません');
  });
});
