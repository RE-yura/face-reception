export interface DownloadFile {
  url: string;
  /** The exact size; a body of any other size is a failed download. */
  bytes: number;
}

/** A file did not arrive intact. Unlike a model that arrived but cannot be loaded, trying again may help. */
export class DownloadError extends Error {}

export interface DownloadOptions {
  onProgress: (loaded: number, total: number) => void;
  /** Gives up when no data arrives for this long. */
  idleTimeoutMs: number;
  /** Aborting it stops every download. */
  signal?: AbortSignal;
  fetch?: typeof fetch;
}

/**
 * The idle timeout is counted in checks rather than measured on the clock. While the page sleeps (Safari in the
 * background), no check runs, so the time asleep counts as one check at most instead of a long silence.
 */
const IDLE_CHECKS = 20;

/**
 * Downloads the files in parallel and checks each one's size. Returns one promise per file, so a caller can use a
 * small file before the large ones arrive. When any download fails, or the caller aborts `signal`, the others are
 * aborted, and every promise rejects with a DownloadError.
 */
export function downloadFiles(files: readonly DownloadFile[], options: DownloadOptions): Promise<Uint8Array>[] {
  const { onProgress, idleTimeoutMs, signal: stopSignal, fetch: fetchFile = fetch } = options;
  const controller = new AbortController();
  const stop = () => controller.abort(stopSignal?.reason);
  if (stopSignal?.aborted) stop();
  else stopSignal?.addEventListener('abort', stop);
  const total = files.reduce((n, f) => n + f.bytes, 0);
  let loaded = 0;
  let quietChecks = 0;
  const idleCheck = setInterval(() => {
    if (++quietChecks >= IDLE_CHECKS) controller.abort(new DownloadError(`no data for ${idleTimeoutMs} ms`));
  }, idleTimeoutMs / IDLE_CHECKS);
  const onChunk = (n: number) => {
    quietChecks = 0;
    loaded += n;
    onProgress(loaded, total);
  };
  const downloads = files.map((f) =>
    download(f.url, f.bytes, controller.signal, fetchFile, onChunk).catch((error: unknown) => {
      const failure = error instanceof DownloadError ? error : new DownloadError(`${f.url}: ${String(error)}`);
      controller.abort(failure);
      throw failure;
    }),
  );
  void Promise.allSettled(downloads).then(() => {
    clearInterval(idleCheck);
    stopSignal?.removeEventListener('abort', stop);
  });
  return downloads;
}

async function download(
  url: string,
  expected: number,
  signal: AbortSignal,
  fetchFile: typeof fetch,
  onChunk: (n: number) => void,
): Promise<Uint8Array> {
  const res = await fetchFile(url, { signal });
  // An abort that came while the request was starting would never reach the listener below.
  signal.throwIfAborted();
  if (!res.ok || !res.body) throw new DownloadError(`${url}: HTTP ${res.status}`);
  const bytes = new Uint8Array(expected);
  let offset = 0;
  const reader = res.body.getReader();
  const cancel = () => void reader.cancel().catch(() => {});
  signal.addEventListener('abort', cancel);
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (signal.aborted) throw signal.reason;
      if (done) break;
      if (offset + value.byteLength > expected) throw new DownloadError(`${url}: larger than ${expected} bytes`);
      bytes.set(value, offset);
      offset += value.byteLength;
      onChunk(value.byteLength);
    }
  } finally {
    signal.removeEventListener('abort', cancel);
  }
  if (offset !== expected) throw new DownloadError(`${url}: ${offset} of ${expected} bytes`);
  return bytes;
}
