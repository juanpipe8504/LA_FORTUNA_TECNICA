import type {
  ApiErrorBody,
  ChatRequest,
  ChatResponse,
  DocumentSummary,
  JobResponse,
  UploadResponse,
} from "./types";

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
