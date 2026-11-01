import type {
  ApiErrorBody,
  ChatRequest,
  ChatResponse,
  DocumentSummary,
  JobResponse,
  UploadResponse,
} from "./types";
import { SSEParser, type SSEEvent } from "./sse";

/** Base URL of the versioned API. Overridable at build time with VITE_API_URL. */
export const API_BASE_URL: string = (
  (import.meta.env.VITE_API_URL as string | undefined) ?? "/api/v1"
).replace(/\/+$/, "");

/** Error raised for any non-2xx response or network failure. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;

  constructor(message: string, status: number, code: string, requestId: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== "object" || value === null) return false;
  const err = (value as { error?: unknown }).error;
  return typeof err === "object" && err !== null && typeof (err as { message?: unknown }).message === "string";
}

const STATUS_MESSAGES: Record<number, string> = {
  404: "Recurso no encontrado.",
  413: "El archivo es demasiado grande.",
  422: "La solicitud no es válida.",
  429: "Demasiadas solicitudes. Inténtalo de nuevo en unos segundos.",
  502: "Un servicio externo (modelo de lenguaje) no respondió correctamente.",
  503: "El servicio no está disponible temporalmente. Inténtalo de nuevo en unos segundos.",
  504: "El servidor tardó demasiado en responder.",
};

/** Turns a failed Response into an ApiError, parsing `{"error": {...}}` when present. */
export async function parseErrorResponse(res: Response): Promise<ApiError> {
  const headerRequestId = res.headers.get("X-Request-ID");
  let body: unknown = null;
  try {
    const text = await res.text();
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (isApiErrorBody(body)) {
    const { code, message, request_id } = body.error;
    return new ApiError(message, res.status, code ?? "error", request_id ?? headerRequestId);
  }
  // FastAPI default validation error: {"detail": [...]} or {"detail": "..."}
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    const msg = typeof detail === "string" ? detail : STATUS_MESSAGES[422];
    return new ApiError(msg, res.status, "validation_error", headerRequestId);
  }
  const fallback = STATUS_MESSAGES[res.status] ?? `Error inesperado del servidor (HTTP ${res.status}).`;
  return new ApiError(fallback, res.status, `http_${res.status}`, headerRequestId);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(
      "No se pudo conectar con el servidor. Comprueba tu conexión o que la API esté en marcha.",
      0,
      "network_error",
    );
  }
  if (!res.ok) throw await parseErrorResponse(res);
  return (await res.json()) as T;
}

/** Resolves an API-provided path (e.g. "/api/v1/documents/..") to a usable URL. */
export function resolveAssetUrl(url: string): string {
  if (/^(https?:|data:|blob:)/i.test(url)) return url;
  // If the API lives on another origin, prefix image paths with that origin.
  if (/^https?:/i.test(API_BASE_URL) && url.startsWith("/")) {
    return new URL(url, API_BASE_URL).toString();
  }
  return url;
}

export const api = {
  listDocuments(signal?: AbortSignal): Promise<DocumentSummary[]> {
    return request<DocumentSummary[]>("/documents", { signal });
  },

  uploadDocument(file: File, signal?: AbortSignal): Promise<UploadResponse> {
    const form = new FormData();
    form.append("file", file, file.name);
    return request<UploadResponse>("/documents", { method: "POST", body: form, signal });
  },

  getJob(jobId: string, signal?: AbortSignal): Promise<JobResponse> {
    return request<JobResponse>(`/jobs/${encodeURIComponent(jobId)}`, { signal });
  },

  chat(payload: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
    return request<ChatResponse>("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
  },
};

export type Api = typeof api;

// ---------------------------------------------------------------------------
// Streaming chat (Server-Sent Events over fetch)
// ---------------------------------------------------------------------------

export interface StreamChatHandlers {
  /** A text delta of the answer (never contains the [[NO_CONTEXT]] marker). */
  onToken: (text: string) => void;
  /** The complete answer; same shape as the POST /chat response. Called at most once. */
  onFinal: (response: ChatResponse) => void;
  /**
   * Terminal failure (HTTP error, server `error` event, network drop, stream
   * ended without `final`). Called at most once, never after `onFinal`.
   * Not called on abort: the returned promise rejects with an AbortError instead.
   */
  onError: (error: ApiError) => void;
}

/** Error codes produced by `streamChat` itself (besides HTTP / server codes). */
export const STREAM_ERROR_CODES = {
  network: "network_error",
  unsupported: "stream_unsupported",
  interrupted: "stream_interrupted",
  incomplete: "stream_incomplete",
  badEvent: "stream_bad_event",
} as const;

const NETWORK_MESSAGE = "No se pudo conectar con el servidor. Comprueba tu conexión o que la API esté en marcha.";
const INTERRUPTED_MESSAGE = "Se perdió la conexión mientras se generaba la respuesta.";

function isAbort(err: unknown, signal?: AbortSignal): boolean {
  return (err instanceof DOMException && err.name === "AbortError") || signal?.aborted === true;
}

function abortError(): DOMException {
  return new DOMException("The operation was aborted.", "AbortError");
}

/**
 * POST /chat/stream and dispatch SSE events (`token`, `final`, `error`) to the handlers.
 * Resolves once a terminal handler (`onFinal` / `onError`) has been called; rejects
 * with an AbortError if `signal` aborts.
 */
export async function streamChat(
  payload: ChatRequest,
  handlers: StreamChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal,
      cache: "no-store",
    });
  } catch (err) {
    if (isAbort(err, signal)) throw abortError();
    handlers.onError(new ApiError(NETWORK_MESSAGE, 0, STREAM_ERROR_CODES.network));
    return;
  }
  if (!res.ok) {
    handlers.onError(await parseErrorResponse(res));
    return;
  }
  const contentType = res.headers.get("Content-Type") ?? "";
  if (!res.body || !contentType.includes("text/event-stream")) {
    void res.body?.cancel().catch(() => undefined);
    handlers.onError(
      new ApiError("El servidor no admite respuestas en streaming.", res.status, STREAM_ERROR_CODES.unsupported),
    );
    return;
  }

  const requestId = res.headers.get("X-Request-ID");
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const parser = new SSEParser();
  let done = false;

  const handle = (ev: SSEEvent): void => {
    if (done) return;
    let data: unknown;
    try {
      data = JSON.parse(ev.data);
    } catch {
      if (ev.event === "token" || ev.event === "message") return; // tolerate a garbled delta
      done = true;
      handlers.onError(new ApiError("Respuesta de streaming no válida.", res.status, STREAM_ERROR_CODES.badEvent, requestId));
      return;
    }
    if (ev.event === "token") {
      const text = (data as { text?: unknown } | null)?.text;
      if (typeof text === "string" && text) handlers.onToken(text);
    } else if (ev.event === "final") {
      const r = data as Partial<ChatResponse> | null;
      if (!r || typeof r.answer !== "string") {
        done = true;
        handlers.onError(new ApiError("Respuesta final no válida.", res.status, STREAM_ERROR_CODES.badEvent, requestId));
        return;
      }
      done = true;
      handlers.onFinal({
        answer: r.answer,
        has_sufficient_context: r.has_sufficient_context !== false,
        sources: Array.isArray(r.sources) ? r.sources : [],
        images: Array.isArray(r.images) ? r.images : [],
      });
    } else if (ev.event === "error") {
      done = true;
      if (isApiErrorBody(data)) {
        const { code, message, request_id } = data.error;
        handlers.onError(new ApiError(message, res.status, code ?? "error", request_id ?? requestId));
      } else {
        handlers.onError(new ApiError("Error durante la generación de la respuesta.", res.status, "stream_error", requestId));
      }
    }
    // Unknown event names are ignored for forward compatibility.
  };

  try {
    while (!done) {
      const { value, done: eof } = await reader.read();
      if (eof) {
        for (const ev of parser.feed(decoder.decode())) handle(ev);
        for (const ev of parser.flush()) handle(ev);
        break;
      }
      for (const ev of parser.feed(decoder.decode(value, { stream: true }))) handle(ev);
    }
  } catch (err) {
    if (isAbort(err, signal)) throw abortError();
    if (!done) {
      done = true;
      handlers.onError(new ApiError(INTERRUPTED_MESSAGE, 0, STREAM_ERROR_CODES.interrupted, requestId));
    }
    return;
  } finally {
    if (done) void reader.cancel().catch(() => undefined);
  }
  if (signal?.aborted) throw abortError();
  if (!done) {
    done = true;
    handlers.onError(new ApiError(INTERRUPTED_MESSAGE, 0, STREAM_ERROR_CODES.incomplete, requestId));
  }
}
