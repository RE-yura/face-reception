export interface ModelFile {
  file: string;
  bytes: number;
}

/** A model did not arrive intact. Unlike a model that arrived but cannot be loaded, trying again may help. */
export class DownloadError extends Error {}

export interface DownloadOptions {
  onProgress: (loaded: number, total: number) => void;
  /** Gives up when no data arrives for this long. */
  idleTimeoutMs: number;
  fetch?: typeof fetch;
}

/**
 * Downloads the files in parallel and checks each one's size. Returns one promise per file, so a caller can use a
 * small file before the large ones arrive. When any download fails, the others are aborted, and every promise
 * rejects with a DownloadError.
 */
export function downloadModels(baseUrl: string, files: readonly ModelFile[], options: DownloadOptions): Promise<Uint8Array>[] {
  const { onProgress, idleTimeoutMs, fetch: fetchFile = fetch } = options;
  const controller = new AbortController();
  const total = files.reduce((n, f) => n + f.bytes, 0);
  let loaded = 0;
  let idleTimer: ReturnType<typeof setTimeout> | undefined;
  const keepAlive = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => controller.abort(new DownloadError(`no data for ${idleTimeoutMs} ms`)), idleTimeoutMs);
  };
  keepAlive();
  const onChunk = (n: number) => {
    keepAlive();
    loaded += n;
    onProgress(loaded, total);
  };
  const downloads = files.map((f) =>
    download(`${baseUrl}${f.file}`, f.bytes, controller.signal, fetchFile, onChunk).catch((error: unknown) => {
      const failure = error instanceof DownloadError ? error : new DownloadError(`${f.file}: ${String(error)}`);
      controller.abort(failure);
      throw failure;
    }),
  );
  void Promise.allSettled(downloads).then(() => clearTimeout(idleTimer));
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
