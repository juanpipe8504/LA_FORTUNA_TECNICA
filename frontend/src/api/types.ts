/**
 * Types mirroring the backend HTTP contract (docs/CONTRACTS.md, section C).
 * Keep in sync with backend/app/api/schemas.py.
 */

export type JobStatus = "pending" | "processing" | "completed" | "failed";

/** [x0, y0, x1, y1] in PDF points. */
export type BBox = [number, number, number, number];

export interface UploadResponse {
  job_id: string;
  document_id: string;
  filename: string;
  status: JobStatus;
}

export interface JobResponse {
  job_id: string;
  document_id: string;
  filename: string;
  status: JobStatus;
  progress: number;
  message: string | null;
  error: string | null;
  page_count: number | null;
  chunk_count: number | null;
  image_count: number | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentSummary {
  document_id: string;
  filename: string;
  page_count: number | null;
  chunk_count: number | null;
  image_count: number | null;
  created_at: string;
}

export type ChatRole = "user" | "assistant";

export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export interface ChatRequest {
  question: string;
  history: ChatTurn[];
  document_ids: string[] | null;
}

export interface SourceCitation {
  chunk_id: string;
  document_id: string;
  document_name: string;
  page: number;
  score: number;
  snippet: string;
  bbox: BBox | null;
}

export interface ImageCitation {
  image_id: string;
  document_id: string;
  document_name: string;
  page: number;
  caption: string;
  bbox: BBox | null;
  related_chunk_id: string;
  relevance: number;
  url: string;
}

export interface ChatResponse {
  answer: string;
  has_sufficient_context: boolean;
  sources: SourceCitation[];
  images: ImageCitation[];
}

export interface ApiErrorBody {
  error: { code: string; message: string; request_id?: string | null };
}
