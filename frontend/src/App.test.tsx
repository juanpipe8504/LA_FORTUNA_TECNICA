import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "./App";
import { makeResponse } from "./test/fixtures";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("chat flow", () => {
  it("sends on Enter with history and document scope, then retries after an error", async () => {
    const chatBodies: unknown[] = [];
    let chatCalls = 0;
    const fetchMock = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url === "/api/v1/documents") {
        return Promise.resolve(
          json([{ document_id: "doc-a", filename: "manual.pdf", page_count: 9, chunk_count: 20, image_count: 2, created_at: "2026-01-01T00:00:00Z" }]),
        );
      }
      if (url === "/api/v1/chat") {
        chatBodies.push(JSON.parse(String(init?.body)));
        chatCalls += 1;
        if (chatCalls === 2) {
          return Promise.resolve(json({ error: { code: "external_service", message: "El LLM falló", request_id: "r-9" } }, 502));
        }
        return Promise.resolve(json(makeResponse()));
      }
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByLabelText(/manual\.pdf/));
    const input = screen.getByLabelText("Escribe tu pregunta");

    // Shift+Enter inserts a newline instead of sending.
    await user.type(input, "Primera{Shift>}{Enter}{/Shift}línea");
    expect(input).toHaveValue("Primera\nlínea");
    await user.keyboard("{Enter}");

    expect(await screen.findByRole("complementary", { name: "Contexto visual" })).toBeInTheDocument();
    expect(chatBodies[0]).toEqual({ question: "Primera\nlínea", history: [], document_ids: ["doc-a"] });

    await user.type(input, "¿Y la segunda?{Enter}");
    expect(await screen.findByText("El LLM falló")).toBeInTheDocument();
    expect(screen.getByText("ID de solicitud: r-9")).toBeInTheDocument();
    expect(chatBodies[1]).toMatchObject({
      question: "¿Y la segunda?",
      history: [
        { role: "user", content: "Primera\nlínea" },
        { role: "assistant", content: makeResponse().answer },
      ],
    });

    await user.click(screen.getByRole("button", { name: "Reintentar" }));
    await waitFor(() => expect(screen.getAllByRole("complementary", { name: "Contexto visual" })).toHaveLength(2));
    // Retry re-sends the same question without duplicating it in the history.
    expect(chatBodies[2]).toEqual(chatBodies[1]);
    expect(screen.getAllByText("¿Y la segunda?")).toHaveLength(1);

    // Conversation is persisted and can be reset.
    expect(window.localStorage.getItem("rag.conversation.v1")).toContain("¿Y la segunda?");
    await user.click(screen.getByRole("button", { name: /Nueva conversación/ }));
    expect(screen.getByText("Pregunta a tus documentos")).toBeInTheDocument();
  });
});
