import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatPanel } from "../components/ChatPanel";
import { makeResponse } from "../test/fixtures";
import { controlledSSE, finalEvent, tokenEvent } from "../test/sse";
import { useChat } from "./useChat";

function Harness() {
  const chat = useChat();
  return (
    <ChatPanel
      messages={chat.messages}
      pending={chat.pending}
      phase={chat.phase}
      error={chat.error}
      scopeLabel="Todos"
      hasDocuments
      onSend={(q) => void chat.send(q, null)}
      onRetry={chat.retry}
      onDismissError={chat.dismissError}
      onStop={chat.stop}
    />
  );
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

/** fetch mock: /chat/stream gets a hand-driven SSE stream; /chat gets `chatImpl`. */
function setupFetch(chatImpl: () => Promise<Response> = () => Promise.resolve(json(makeResponse()))) {
  const streams: Array<ReturnType<typeof controlledSSE>> = [];
  const calls: string[] = [];
  const fetchMock = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
    calls.push(url);
    if (url === "/api/v1/chat/stream") {
      const s = controlledSSE(init?.signal);
      streams.push(s);
      return Promise.resolve(s.response);
    }
    if (url === "/api/v1/chat") return chatImpl();
    return Promise.reject(new Error(`unexpected ${url}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return { streams, calls, fetchMock, current: () => streams[streams.length - 1] };
}

async function ask(text: string) {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Escribe tu pregunta"), `${text}{Enter}`);
  return user;
}

const answer = () => screen.getByRole("article", { name: "Respuesta del asistente" });

describe("useChat streaming", () => {
  it("shows typing until the first token, streams text, then replaces it with the final answer", async () => {
    const net = setupFetch();
    render(<Harness />);
    await ask("¿Arquitectura?");

    await waitFor(() => expect(net.streams).toHaveLength(1));
    expect(screen.getByText(/Buscando en los documentos/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Detener la generación" })).toBeInTheDocument();

    net.current().push(tokenEvent("La arquitectura usa "));
    expect(await screen.findByText(/La arquitectura usa/)).toBeInTheDocument();
    // Typing indicator gives way to the streaming status.
    expect(screen.queryByText(/Buscando en los documentos/)).not.toBeInTheDocument();
    expect(screen.getByText(/Generando…/)).toBeInTheDocument();

    net.current().push(tokenEvent("una cola [3]."));
    await waitFor(() => expect(answer()).toHaveTextContent("La arquitectura usa una cola [3]."));
    // Citations are not clickable and no sources/figures while streaming.
    expect(within(answer()).queryByRole("button", { name: /Ver Fuente/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("complementary", { name: "Contexto visual" })).not.toBeInTheDocument();
    // A half-streamed answer is never persisted as complete.
    const stored = JSON.parse(window.localStorage.getItem("rag.conversation.v1") ?? "[]");
    expect(stored.find((m: { role: string }) => m.role === "assistant")?.status).toBe("interrupted");

    const final = makeResponse({ answer: "[[NO_CONTEXT]]La arquitectura usa una cola [1] y Qdrant [2]." });
    net.current().push(finalEvent(final));
    net.current().close();

    expect(await screen.findByRole("button", { name: /Ver Fuente 1: manual\.pdf, página 3/ })).toBeInTheDocument();
    expect(answer()).not.toHaveTextContent("[3]");
    expect(answer()).not.toHaveTextContent("NO_CONTEXT");
    expect(screen.getByRole("complementary", { name: "Contexto visual" })).toBeInTheDocument();
    expect(screen.queryByText(/Generando…/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enviar pregunta" })).toBeInTheDocument();
    expect(net.calls).not.toContain("/api/v1/chat");
    await waitFor(() => {
      const saved = JSON.parse(window.localStorage.getItem("rag.conversation.v1") ?? "[]");
      expect(saved[1]).toMatchObject({ role: "assistant", content: final.answer });
      expect(saved[1].status).toBeUndefined();
    });
  });

  it("falls back to POST /chat when the stream endpoint returns 404, and remembers it", async () => {
    const calls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        calls.push(url);
        if (url === "/api/v1/chat/stream") return Promise.resolve(json({ detail: "Not Found" }, 404));
        return Promise.resolve(json(makeResponse()));
      }),
    );
    render(<Harness />);
    const user = await ask("Primera");
    expect(await screen.findByRole("complementary", { name: "Contexto visual" })).toBeInTheDocument();
    expect(calls).toEqual(["/api/v1/chat/stream", "/api/v1/chat"]);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    await user.type(screen.getByLabelText("Escribe tu pregunta"), "Segunda{Enter}");
    await waitFor(() => expect(screen.getAllByRole("complementary", { name: "Contexto visual" })).toHaveLength(2));
    expect(calls).toEqual(["/api/v1/chat/stream", "/api/v1/chat", "/api/v1/chat"]);
  });

  it("falls back to POST /chat when the stream fails with a network error before any token", async () => {
    const net = setupFetch();
    render(<Harness />);
    await ask("Hola");
    await waitFor(() => expect(net.streams).toHaveLength(1));
    net.current().fail(new TypeError("network error"));
    expect(await screen.findByRole("complementary", { name: "Contexto visual" })).toBeInTheDocument();
    expect(net.calls).toEqual(["/api/v1/chat/stream", "/api/v1/chat"]);
  });

  it("keeps the partial text and offers retry when the stream fails after tokens", async () => {
    const net = setupFetch();
    render(<Harness />);
    const user = await ask("Hola");
    await waitFor(() => expect(net.streams).toHaveLength(1));
    net.current().push(tokenEvent("Texto parcial"));
    await screen.findByText(/Texto parcial/);
    net.current().fail(new TypeError("network error"));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Se perdió la conexión");
    expect(answer()).toHaveTextContent("Texto parcial");
    expect(screen.getByText(/Respuesta incompleta/)).toBeInTheDocument();
    expect(net.calls).not.toContain("/api/v1/chat"); // no silent fallback after tokens

    await user.click(screen.getByRole("button", { name: "Reintentar" }));
    await waitFor(() => expect(net.streams).toHaveLength(2));
    // The partial answer is dropped; the question is not duplicated.
    expect(screen.queryByText(/Texto parcial/)).not.toBeInTheDocument();
    expect(screen.getAllByText("Hola")).toHaveLength(1);
    net.current().push(tokenEvent("Bien") + finalEvent(makeResponse()));
    net.current().close();
    expect(await screen.findByRole("complementary", { name: "Contexto visual" })).toBeInTheDocument();
    expect(screen.getAllByRole("article", { name: "Respuesta del asistente" })).toHaveLength(1);
  });

  it("shows server error events without falling back", async () => {
    const net = setupFetch();
    render(<Harness />);
    await ask("Hola");
    await waitFor(() => expect(net.streams).toHaveLength(1));
    net.current().push('event: error\ndata: {"error": {"code": "llm_error", "message": "El LLM falló", "request_id": "r-2"}}\n\n');
    net.current().close();
    expect(await screen.findByText("El LLM falló")).toBeInTheDocument();
    expect(screen.getByText("ID de solicitud: r-2")).toBeInTheDocument();
    expect(net.calls).not.toContain("/api/v1/chat");
  });

  it("stops generation with 'Detener', keeping the partial answer marked as interrupted", async () => {
    const net = setupFetch();
    render(<Harness />);
    const user = await ask("Hola");
    await waitFor(() => expect(net.streams).toHaveLength(1));
    net.current().push(tokenEvent("Respuesta a medias"));
    await screen.findByText(/Respuesta a medias/);

    await user.click(screen.getByRole("button", { name: "Detener la generación" }));

    expect(await screen.findByText("Respuesta interrumpida")).toBeInTheDocument();
    expect(answer()).toHaveTextContent("Respuesta a medias");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enviar pregunta" })).toBeInTheDocument();
    const signal = (net.fetchMock.mock.calls[0][1] as RequestInit).signal;
    expect(signal?.aborted).toBe(true);
    await waitFor(() => {
      const saved = JSON.parse(window.localStorage.getItem("rag.conversation.v1") ?? "[]");
      expect(saved[1]).toMatchObject({ content: "Respuesta a medias", status: "interrupted" });
    });
  });

  it("stopping before any token leaves an empty interrupted message and no fallback", async () => {
    const net = setupFetch();
    render(<Harness />);
    const user = await ask("Hola");
    await waitFor(() => expect(net.streams).toHaveLength(1));
    await user.click(screen.getByRole("button", { name: "Detener la generación" }));
    expect(await screen.findByText("Respuesta interrumpida")).toBeInTheDocument();
    expect(screen.getByText(/Generación detenida/)).toBeInTheDocument();
    expect(net.calls).toEqual(["/api/v1/chat/stream"]);
  });
});
