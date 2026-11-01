import type { ChatResponse } from "../api/types";

export const tokenEvent = (text: string) => `event: token\ndata: ${JSON.stringify({ text })}\n\n`;
export const finalEvent = (r: ChatResponse) => `event: final\ndata: ${JSON.stringify(r)}\n\n`;
export const errorEvent = (code: string, message: string, request_id = "rid-1") =>
  `event: error\ndata: ${JSON.stringify({ error: { code, message, request_id } })}\n\n`;

const SSE_HEADERS = { "Content-Type": "text/event-stream; charset=utf-8" };

/** A complete SSE response from a list of raw string chunks. */
export function sseResponse(chunks: string[]): Response {
  const enc = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: SSE_HEADERS });
}

/**
 * An SSE response the test drives by hand. Aborting `signal` errors the body
 * with an AbortError, like a real fetch does.
 */
export function controlledSSE(signal?: AbortSignal | null) {
  const enc = new TextEncoder();
  let ctrl!: ReadableStreamDefaultController<Uint8Array>;
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      ctrl = c;
    },
  });
  signal?.addEventListener("abort", () => {
    try {
      ctrl.error(new DOMException("The operation was aborted.", "AbortError"));
    } catch {
      /* already closed */
    }
  });
  return {
    response: new Response(stream, { status: 200, headers: SSE_HEADERS }),
    push: (text: string) => ctrl.enqueue(enc.encode(text)),
    close: () => ctrl.close(),
    fail: (err: unknown = new TypeError("network error")) => ctrl.error(err),
  };
}
