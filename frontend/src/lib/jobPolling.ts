import { ApiError } from "../api/client";
import type { JobResponse } from "../api/types";

export interface PollOptions {
  getJob: (jobId: string, signal?: AbortSignal) => Promise<JobResponse>;
  onUpdate: (job: JobResponse) => void;
  intervalMs?: number;
  /** Consecutive transient errors tolerated before giving up. */
  maxConsecutiveErrors?: number;
  signal?: AbortSignal;
}

export const isTerminal = (status: JobResponse["status"]): boolean =>
  status === "completed" || status === "failed";

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Polls GET /jobs/{id} until the job reaches a terminal state.
 * Network / 5xx hiccups are retried; a 404 (unknown job) fails immediately.
 * Resolves with the final job; rejects with ApiError or AbortError.
 */
export async function pollJob(jobId: string, opts: PollOptions): Promise<JobResponse> {
  const { getJob, onUpdate, intervalMs = 1500, maxConsecutiveErrors = 5, signal } = opts;
  let errors = 0;
  for (;;) {
    try {
      const job = await getJob(jobId, signal);
      errors = 0;
      onUpdate(job);
      if (isTerminal(job.status)) return job;
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") throw err;
      const status = err instanceof ApiError ? err.status : 0;
      if (status === 404 || status === 422) throw err;
      errors += 1;
      if (errors >= maxConsecutiveErrors) throw err;
    }
    await sleep(intervalMs, signal);
  }
}
