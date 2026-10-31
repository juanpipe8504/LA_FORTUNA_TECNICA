const parsed = Number.parseInt(String(import.meta.env.VITE_MAX_UPLOAD_MB ?? ""), 10);
/** Must match the backend's max_upload_mb (default 50). */
export const MAX_UPLOAD_MB = Number.isFinite(parsed) && parsed > 0 ? parsed : 50;
export const POLL_INTERVAL_MS = 1500;
/** Messages (user + assistant) sent as history: last 6 turns. */
export const HISTORY_MAX_MESSAGES = 12;
