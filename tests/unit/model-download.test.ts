import { afterEach, describe, expect, it, vi } from 'vitest';
import { DownloadError, downloadFiles } from '../../src/model-download.ts';

interface FakeFile {
  body?: Uint8Array;
  status?: number;
  /** Sends nothing until the request is aborted. */
  stall?: boolean;
  /** Waits this long before each chunk. */
  delayMs?: number;
  /** Sends the first chunk, then the rest once this resolves. */
  restAfter?: Promise<void>;
  /** The response arrives only once this resolves, even if the request was aborted meanwhile. */
  respondAfter?: Promise<void>;
  /** The body never reports an abort, unlike a real fetch's. */
  ignoreAbort?: boolean;
}

const CHUNK = 1000;
const bytes = (n: number, seed = 1) => Uint8Array.from({ length: n }, (_, i) => (i * seed) & 255);
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** A fetch serving files by name, which honors the abort signal like the real one. */
function fakeFetch(files: Record<string, FakeFile>) {
  const signals: Record<string, AbortSignal> = {};
  const fetch = async (url: string, init?: RequestInit) => {
    const name = url.split('/').pop()!;
    const file = files[name];
    const signal = init!.signal!;
    signals[name] = signal;
    if (signal.aborted) throw signal.reason;
    if (file.respondAfter) await file.respondAfter;
    if (file.status) return new Response('not found', { status: file.status });
    let offset = 0;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        if (!file.ignoreAbort) signal.addEventListener('abort', () => controller.error(signal.reason));
      },
      async pull(controller) {
        if (file.stall) return new Promise<void>(() => {});
        if (file.delayMs) await sleep(file.delayMs);
        if (file.restAfter && offset > 0) await file.restAfter;
        if (offset >= file.body!.length) return controller.close();
        controller.enqueue(file.body!.slice(offset, offset + CHUNK));
        offset += CHUNK;
      },
    });
    return new Response(stream);
  };
  return { fetch: fetch as typeof globalThis.fetch, signals };
}

const noProgress = () => {};

/** Timers the download uses. Promises and streams keep running for real. */
const fakeTimers = () => vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });

afterEach(() => {
  vi.useRealTimers();
});

describe('downloadFiles', () => {
  it('downloads each file and reports progress against their total size', async () => {
    const a = bytes(2500);
    const b = bytes(4000, 7);
    const { fetch } = fakeFetch({ 'a.onnx': { body: a }, 'b.onnx': { body: b } });
    const progress: number[] = [];
    const [gotA, gotB] = downloadFiles(
      [
        { url: '/models/a.onnx', bytes: 2500 },
        { url: '/models/b.onnx', bytes: 4000 },
      ],
      {
        fetch,
        idleTimeoutMs: 1000,
        onProgress: (loaded, total) => {
          expect(total).toBe(6500);
          progress.push(loaded);
        },
      },
    );
    expect(await gotA).toEqual(a);
    expect(await gotB).toEqual(b);
    expect(progress.at(-1)).toBe(6500);
    expect(progress).toEqual([...progress].sort((p, q) => p - q));
  });

  it('treats a body of the wrong size, such as a Wi-Fi login page, as a failed download', async () => {
    const page = new TextEncoder().encode('<!doctype html><title>Wi-Fi login</title>');
    const { fetch } = fakeFetch({ 'a.onnx': { body: page }, 'b.onnx': { body: bytes(3000) } });
    const [shorter, longer] = downloadFiles(
      [
        { url: '/models/a.onnx', bytes: 2500 },
        { url: '/models/b.onnx', bytes: 2500 },
      ],
      { fetch, idleTimeoutMs: 1000, onProgress: noProgress },
    );
    await expect(shorter).rejects.toBeInstanceOf(DownloadError);
    await expect(longer).rejects.toBeInstanceOf(DownloadError);
  });

  it('fails on an HTTP error', async () => {
    const { fetch } = fakeFetch({ 'a.onnx': { status: 404 } });
    const [a] = downloadFiles([{ url: '/models/a.onnx', bytes: 10 }], { fetch, idleTimeoutMs: 1000, onProgress: noProgress });
    await expect(a).rejects.toBeInstanceOf(DownloadError);
  });

  it('fails when the request itself fails', async () => {
    const fetch = (async () => {
      throw new TypeError('Failed to fetch');
    }) as typeof globalThis.fetch;
    const [a] = downloadFiles([{ url: '/models/a.onnx', bytes: 10 }], { fetch, idleTimeoutMs: 1000, onProgress: noProgress });
    await expect(a).rejects.toBeInstanceOf(DownloadError);
  });

  it('gives up when no data arrives for the idle timeout, and not before', async () => {
    fakeTimers();
    const { fetch } = fakeFetch({ 'a.onnx': { stall: true } });
    const [a] = downloadFiles([{ url: '/models/a.onnx', bytes: 10 }], { fetch, idleTimeoutMs: 20_000, onProgress: noProgress });
    let settled = false;
    a.catch(() => {}).finally(() => (settled = true));
    await vi.advanceTimersByTimeAsync(19_000);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1_500);
    await expect(a).rejects.toBeInstanceOf(DownloadError);
  });

  it('keeps going while data keeps arriving, however slowly', async () => {
    fakeTimers();
    const body = bytes(8 * CHUNK);
    const { fetch } = fakeFetch({ 'a.onnx': { body, delayMs: 15_000 } });
    const [a] = downloadFiles([{ url: '/models/a.onnx', bytes: body.length }], { fetch, idleTimeoutMs: 20_000, onProgress: noProgress });
    await vi.advanceTimersByTimeAsync(10 * 15_000);
    expect(await a).toEqual(body);
  });

  it('does not count time the page spent asleep, such as Safari in the background, as a silence', async () => {
    fakeTimers();
    let wake = () => {};
    const woken = new Promise<void>((resolve) => (wake = resolve));
    const body = bytes(2 * CHUNK);
    const { fetch } = fakeFetch({ 'a.onnx': { body, restAfter: woken } });
    const [a] = downloadFiles([{ url: '/models/a.onnx', bytes: body.length }], { fetch, idleTimeoutMs: 20_000, onProgress: noProgress });
    // The first chunk arrives, then the page goes to sleep.
    await vi.advanceTimersByTimeAsync(1_000);
    // Asleep for a minute, the clock moves on but no timer runs. On waking, each pending timer fires once,
    // before the data that arrived meanwhile is read.
    vi.setSystemTime(Date.now() + 60_000);
    vi.runOnlyPendingTimers();
    wake();
    expect(await a).toEqual(body);
  });

  it('stops the other downloads when one fails', async () => {
    const { fetch, signals } = fakeFetch({ 'a.onnx': { status: 404 }, 'b.onnx': { stall: true } });
    const [a, b] = downloadFiles(
      [
        { url: '/models/a.onnx', bytes: 10 },
        { url: '/models/b.onnx', bytes: 10 },
      ],
      { fetch, idleTimeoutMs: 10_000, onProgress: noProgress },
    );
    await expect(a).rejects.toBeInstanceOf(DownloadError);
    await expect(b).rejects.toBeInstanceOf(DownloadError);
    expect(signals['b.onnx'].aborted).toBe(true);
  });

  it('stops every download when the caller aborts', async () => {
    const { fetch, signals } = fakeFetch({ 'a.onnx': { stall: true }, 'b.onnx': { stall: true } });
    const stop = new AbortController();
    const [a, b] = downloadFiles(
      [
        { url: '/models/a.onnx', bytes: 10 },
        { url: '/models/b.onnx', bytes: 10 },
      ],
      { fetch, idleTimeoutMs: 600_000, onProgress: noProgress, signal: stop.signal },
    );
    stop.abort();
    const outcome = await Promise.race([Promise.allSettled([a, b]), sleep(1_000).then(() => 'still waiting')]);
    expect(outcome).toEqual([
      { status: 'rejected', reason: expect.any(DownloadError) },
      { status: 'rejected', reason: expect.any(DownloadError) },
    ]);
    expect(signals['a.onnx'].aborted).toBe(true);
    expect(signals['b.onnx'].aborted).toBe(true);
  });

  it('stops a download whose response arrived only after the abort', async () => {
    let respond = () => {};
    const responded = new Promise<void>((resolve) => (respond = resolve));
    const { fetch } = fakeFetch({
      'a.onnx': { status: 404 },
      'b.onnx': { stall: true, ignoreAbort: true, respondAfter: responded },
    });
    const [a, b] = downloadFiles(
      [
        { url: '/models/a.onnx', bytes: 10 },
        { url: '/models/b.onnx', bytes: 10 },
      ],
      { fetch, idleTimeoutMs: 10_000, onProgress: noProgress },
    );
    await expect(a).rejects.toBeInstanceOf(DownloadError);
    respond();
    const outcome = await Promise.race([b.then(() => 'resolved', () => 'rejected'), sleep(1_000).then(() => 'still waiting')]);
    expect(outcome).toBe('rejected');
  });
});
