import { AnalysisStalled } from './vision-client.ts';

/** Decides when the detection loop should stop retrying and give up on the worker. */
export class AnalysisHealth {
  private readonly maxFailures: number;
  private failures = 0;

  constructor(maxFailures: number) {
    this.maxFailures = maxFailures;
  }

  succeeded(): void {
    this.failures = 0;
  }

  /** Records a failed analysis. True when detection should stop: the worker stalled, or too many failed in a row. */
  failed(error: unknown): boolean {
    this.failures++;
    return error instanceof AnalysisStalled || this.failures >= this.maxFailures;
  }
}
