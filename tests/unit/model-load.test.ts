import { describe, expect, it } from 'vitest';
import { ModelLoad } from '../../src/model-load.ts';
import { InitError } from '../../src/vision-client.ts';

describe('ModelLoad', () => {
  it('loads once however often it is started, and reports progress from 0', async () => {
    const progress: number[] = [];
    let calls = 0;
    let finish = () => {};
    const models = new ModelLoad(
      (onProgress) => {
        calls++;
        onProgress(0.5);
        return new Promise<void>((resolve) => (finish = resolve));
      },
      (ratio) => progress.push(ratio),
    );
    const first = models.start();
    expect(models.start()).toBe(first);
    finish();
    await first;
    expect(models.start()).toBe(first);
    expect(calls).toBe(1);
    expect(progress).toEqual([0, 0.5]);
  });

  it('tries again after a network failure', async () => {
    const attempts = [() => Promise.reject(new InitError('network', 'offline')), () => Promise.resolve()];
    let calls = 0;
    const models = new ModelLoad(() => attempts[calls++](), () => {});
    await expect(models.start()).rejects.toBeInstanceOf(InitError);
    await expect(models.start()).resolves.toBeUndefined();
    expect(calls).toBe(2);
  });

  it('does not try again once the browser turned out unsupported', async () => {
    let calls = 0;
    const models = new ModelLoad(() => {
      calls++;
      return Promise.reject(new InitError('unsupported', 'no wasm'));
    }, () => {});
    const first = models.start();
    await expect(first).rejects.toBeInstanceOf(InitError);
    expect(models.start()).toBe(first);
    expect(calls).toBe(1);
  });
});
