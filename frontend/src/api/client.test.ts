import { describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./client";

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });
}

describe("api client", () => {
  it("parses the backend error envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ error: { code: "circuit_open", message: "LLM no disponible", request_id: "req-1" } }, 503),
      ),
    );
    const err = await api.chat({ question: "hola", history: [], document_ids: null }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 503, code: "circuit_open", message: "LLM no disponible", requestId: "req-1" });
  });

  it("falls back to a friendly message on non-JSON errors and network failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>bad gateway</html>", { status: 502 })));
    await expect(api.listDocuments()).rejects.toMatchObject({ status: 502, code: "http_502" });

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.listDocuments()).rejects.toMatchObject({ status: 0, code: "network_error" });
  });

  it("posts the chat payload as JSON to /api/v1/chat", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ answer: "ok", has_sufficient_context: true, sources: [], images: [] }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await api.chat({ question: "q", history: [{ role: "user", content: "a" }], document_ids: ["d1"] });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/chat");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ question: "q", history: [{ role: "user", content: "a" }], document_ids: ["d1"] });
  });
});
