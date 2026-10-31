import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api as defaultApi, type Api } from "../api/client";
import type { JobStatus } from "../api/types";
import { MAX_UPLOAD_MB, POLL_INTERVAL_MS } from "../lib/config";
import { newId } from "../lib/format";
import { pollJob } from "../lib/jobPolling";

export type UploadPhase = "uploading" | JobStatus;

export interface UploadItem {
  localId: string;
  filename: string;
  size: number;
  phase: UploadPhase;
  progress: number;
  message: string | null;
  error: string | null;
  jobId: string | null;
  documentId: string | null;
  pageCount: number | null;
  chunkCount: number | null;
  imageCount: number | null;
}

/** Client-side validation; returns an error message or null. */
export function validatePdf(file: File, maxMb = MAX_UPLOAD_MB): string | null {
  const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  if (!isPdf) return "Solo se admiten archivos PDF.";
  if (file.size === 0) return "El archivo está vacío.";
  if (file.size > maxMb * 1024 * 1024) return `El archivo supera el límite de ${maxMb} MB.`;
  return null;
}

interface Options {
  client?: Pick<Api, "uploadDocument" | "getJob">;
  pollIntervalMs?: number;
  onCompleted?: (documentId: string) => void;
}

export function useUploads({ client = defaultApi, pollIntervalMs = POLL_INTERVAL_MS, onCompleted }: Options = {}) {
  const [items, setItems] = useState<UploadItem[]>([]);
  const controllers = useRef(new Map<string, AbortController>());
  const onCompletedRef = useRef(onCompleted);
  onCompletedRef.current = onCompleted;

  useEffect(() => {
    const map = controllers.current;
    return () => {
      for (const c of map.values()) c.abort();
      map.clear();
    };
  }, []);

  const patch = useCallback((localId: string, changes: Partial<UploadItem>) => {
    setItems((prev) => prev.map((it) => (it.localId === localId ? { ...it, ...changes } : it)));
  }, []);

  const startOne = useCallback(
    async (file: File) => {
      const localId = newId();
      const base: UploadItem = {
        localId,
        filename: file.name,
        size: file.size,
        phase: "uploading",
        progress: 0,
        message: null,
        error: null,
        jobId: null,
        documentId: null,
        pageCount: null,
        chunkCount: null,
        imageCount: null,
      };
      const validationError = validatePdf(file);
      if (validationError) {
        setItems((prev) => [{ ...base, phase: "failed", error: validationError }, ...prev]);
        return;
      }
      setItems((prev) => [base, ...prev]);
      const controller = new AbortController();
      controllers.current.set(localId, controller);
      try {
        const res = await client.uploadDocument(file, controller.signal);
        patch(localId, { jobId: res.job_id, documentId: res.document_id, phase: res.status });
        const final = await pollJob(res.job_id, {
          getJob: client.getJob,
          intervalMs: pollIntervalMs,
          signal: controller.signal,
          onUpdate: (job) =>
            patch(localId, {
              phase: job.status,
              progress: Math.max(0, Math.min(100, Math.round(job.progress ?? 0))),
              message: job.message || null,
              error: job.error,
              pageCount: job.page_count,
              chunkCount: job.chunk_count,
              imageCount: job.image_count,
            }),
        });
        if (final.status === "completed") {
          patch(localId, { progress: 100 });
          onCompletedRef.current?.(final.document_id);
        } else if (!final.error) {
          patch(localId, { error: "El procesamiento del documento falló." });
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        const msg = err instanceof ApiError ? err.message : "Error inesperado al subir el archivo.";
        patch(localId, { phase: "failed", error: msg });
      } finally {
        controllers.current.delete(localId);
      }
    },
    [client, patch, pollIntervalMs],
  );

  const upload = useCallback(
    (files: FileList | File[]) => {
      for (const f of Array.from(files)) void startOne(f);
    },
    [startOne],
  );

  const dismiss = useCallback((localId: string) => {
    controllers.current.get(localId)?.abort();
    controllers.current.delete(localId);
    setItems((prev) => prev.filter((it) => it.localId !== localId));
  }, []);

  const clearFinished = useCallback(() => {
    setItems((prev) => prev.filter((it) => it.phase !== "completed" && it.phase !== "failed"));
  }, []);

  return { items, upload, dismiss, clearFinished };
}
