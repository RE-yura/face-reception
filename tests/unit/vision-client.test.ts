import { describe, expect, it } from 'vitest';
import { AnalysisStalled, InitError, VisionClient, type WorkerPort } from '../../src/vision-client.ts';
import type { Analysis, WorkerRequest, WorkerResponse } from '../../src/worker-protocol.ts';

class FakeWorker implements WorkerPort {
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  readonly posted: WorkerRequest[] = [];

  postMessage(message: WorkerRequest): void {
    this.posted.push(message);
  }

  reply(message: WorkerResponse): void {
    this.onmessage?.({ data: message } as MessageEvent<WorkerResponse>);
  }

  crash(): void {
    this.onerror?.({ message: 'boom' } as ErrorEvent);
  }
}

const frame = {} as ImageBitmap;
const analysis: Analysis = { faces: [], frameWidth: 640, frameHeight: 480 };
const lastId = (worker: FakeWorker) => (worker.posted.at(-1) as { id: number }).id;

async function readyClient(worker: FakeWorker, analyzeTimeoutMs: number): Promise<VisionClient> {
  const client = new VisionClient(worker, analyzeTimeoutMs);
  const init = client.init(() => {});
  worker.reply({ type: 'ready' });
  await init;
  return client;
}

describe('VisionClient', () => {
  it("resolves with the worker's answer", async () => {
    const worker = new FakeWorker();
    const client = await readyClient(worker, 1000);
    const result = client.analyze(frame, false);
    worker.reply({ type: 'analysis', id: lastId(worker), analysis });
    expect(await result).toBe(analysis);
  });

  it('gives up with AnalysisStalled when the worker does not answer in time', async () => {
    const worker = new FakeWorker();
    const client = await readyClient(worker, 50);
    const result = client.analyze(frame, false);
    await expect(result).rejects.toBeInstanceOf(AnalysisStalled);
    // A late answer is ignored.
    worker.reply({ type: 'analysis', id: lastId(worker), analysis });
  });

  it('reports a worker crash after start-up as AnalysisStalled, not as an unsupported browser', async () => {
    const worker = new FakeWorker();
    const client = await readyClient(worker, 10_000);
    const result = client.analyze(frame, false);
    worker.crash();
    await expect(result).rejects.toBeInstanceOf(AnalysisStalled);
  });

  it('still reports a worker that fails during start-up as unsupported', async () => {
    const worker = new FakeWorker();
    const init = new VisionClient(worker, 10_000).init(() => {});
    worker.crash();
    await expect(init).rejects.toBeInstanceOf(InitError);
    await expect(init).rejects.toMatchObject({ reason: 'unsupported' });
  });
});
