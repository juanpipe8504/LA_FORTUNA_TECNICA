import { describe, expect, it, vi } from "vitest";
import { makeResponse } from "../test/fixtures";
import { controlledSSE, errorEvent, finalEvent, sseResponse, tokenEvent } from "../test/sse";
import { ApiError, streamChat, type StreamChatHandlers } from "./client";

const payload = { question: "hola", history: [], document_ids: null };

function recorder() {
  const tokens: string[] = [];
  const handlers = {
    onToken: vi.fn((t: string) => tokens.push(t)),
    onFinal: vi.fn(),
    onError: vi.fn(),
  } satisfies StreamChatHandlers;
  return { tokens, handlers };
}

describe("streamChat", () => {
  it("posts to /chat/stream and dispatches tokens then final", async () => {
    const final = makeResponse();
    const body = ": ping\n\n" + tokenEvent("Hola ") + tokenEvent("mundo") + finalEvent(final);
    // Split the raw stream at awkward places (mid-event, mid-JSON).
    const fetchMock = vi.fn().mockResolvedValue(sseResponse([body.slice(0, 7), body.slice(7, 40), body.slice(40)]));
    vi.stubGlobal("fetch", fetchMock);
    const { tokens, handlers } = recorder();

    await streamChat(payload, handlers);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/chat/stream");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(payload);
    expect(init.headers).toMatchObject({ "Content-Type": "application/json", Accept: "text/event-stream" });
    expect(tokens).toEqual(["Hola ", "mundo"]);
    expect(handlers.onFinal).toHaveBeenCalledWith(final);
    expect(handlers.onError).not.toHaveBeenCalled();
  });

  it("reports server error events with the backend envelope", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse([tokenEvent("a"), errorEvent("llm_timeout", "El LLM tardó demasiado", "r-7")])));
    const { handlers } = recorder();
    await streamChat(payload, handlers);
    expect(handlers.onFinal).not.toHaveBeenCalled();
    const err = handlers.onError.mock.calls[0][0];
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ code: "llm_timeout", message: "El LLM tardó demasiado", requestId: "r-7" });
  });

  it("maps HTTP errors, network failures, non-SSE responses and truncated streams", async () => {
    const cases: Array<[Response | Error, Partial<ApiError>]> = [
      [new Response("Not Found", { status: 404 }), { status: 404 }],
      [new TypeError("Failed to fetch"), { status: 0, code: "network_error" }],
      [new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }), { code: "stream_unsupported" }],
      [sseResponse([tokenEvent("parcial")]), { code: "stream_incomplete" }],
    ];
    for (const [result, expected] of cases) {
      vi.stubGlobal("fetch", result instanceof Error ? vi.fn().mockRejectedValue(result) : vi.fn().mockResolvedValue(result));
      const { handlers } = recorder();
      await streamChat(payload, handlers);
      expect(handlers.onError).toHaveBeenCalledTimes(1);
      expect(handlers.onError.mock.calls[0][0]).toMatchObject(expected);
    }
  });

  it("reports a connection dropped mid-stream as interrupted", async () => {
    let sse!: ReturnType<typeof controlledSSE>;
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => {
      sse = controlledSSE();
      return Promise.resolve(sse.response);
    }));
    const { tokens, handlers } = recorder();
    const done = streamChat(payload, handlers);
    await vi.waitFor(() => expect(sse).toBeDefined());
    sse.push(tokenEvent("uno "));
    await vi.waitFor(() => expect(tokens).toEqual(["uno "]));
    sse.fail();
    await done;
    expect(handlers.onError.mock.calls[0][0]).toMatchObject({ code: "stream_interrupted" });
  });

  it("rejects with AbortError when aborted and calls no terminal handler", async () => {
    const controller = new AbortController();
    let sse!: ReturnType<typeof controlledSSE>;
    vi.stubGlobal("fetch", vi.fn().mockImplementation((_url: string, init: RequestInit) => {
      sse = controlledSSE(init.signal);
      return Promise.resolve(sse.response);
    }));
    const { tokens, handlers } = recorder();
    const done = streamChat(payload, handlers, controller.signal);
    await vi.waitFor(() => expect(sse).toBeDefined());
    sse.push(tokenEvent("a"));
    await vi.waitFor(() => expect(tokens).toEqual(["a"]));
    controller.abort();
    await expect(done).rejects.toMatchObject({ name: "AbortError" });
    expect(handlers.onError).not.toHaveBeenCalled();
    expect(handlers.onFinal).not.toHaveBeenCalled();
  });
});
