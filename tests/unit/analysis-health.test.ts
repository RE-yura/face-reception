import { describe, expect, it } from 'vitest';
import { AnalysisHealth } from '../../src/analysis-health.ts';
import { AnalysisStalled } from '../../src/vision-client.ts';

describe('AnalysisHealth', () => {
  it('keeps retrying until the limit of failures in a row', () => {
    const health = new AnalysisHealth(5);
    for (let i = 0; i < 4; i++) expect(health.failed(new Error('run failed'))).toBe(false);
    expect(health.failed(new Error('run failed'))).toBe(true);
  });

  it('starts counting again after a success', () => {
    const health = new AnalysisHealth(5);
    for (let i = 0; i < 4; i++) health.failed(new Error('run failed'));
    health.succeeded();
    for (let i = 0; i < 4; i++) expect(health.failed(new Error('run failed'))).toBe(false);
  });

  it('gives up at once when the worker stalled', () => {
    expect(new AnalysisHealth(5).failed(new AnalysisStalled('no answer'))).toBe(true);
  });
});
