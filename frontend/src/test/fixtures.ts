import type { ChatResponse } from "../api/types";
import type { AssistantMessage } from "../hooks/useChat";

export function makeResponse(overrides: Partial<ChatResponse> = {}): ChatResponse {
  return {
    answer:
      "La arquitectura usa una cola de trabajos [1] y un almacén vectorial híbrido [2].\n\n| Componente | Puerto |\n|---|---|\n| API | 8000 |",
    has_sufficient_context: true,
    sources: [
      {
        chunk_id: "c-1",
        document_id: "doc-a",
        document_name: "manual.pdf",
        page: 3,
        score: 0.82,
        snippet: "Los documentos se procesan de forma asíncrona mediante una cola.",
        bbox: null,
      },
      {
        chunk_id: "c-2",
        document_id: "doc-a",
        document_name: "manual.pdf",
        page: 5,
        score: 0.41,
        snippet: "Qdrant combina vectores densos y dispersos.",
        bbox: [0, 0, 100, 100],
      },
    ],
    images: [
      {
        image_id: "doc-a_p5_i0",
        document_id: "doc-a",
        document_name: "manual.pdf",
        page: 5,
        caption: "Figura 2: Diagrama de arquitectura",
        bbox: [10, 10, 200, 200],
        related_chunk_id: "c-2",
        relevance: 0.4,
        url: "/api/v1/documents/doc-a/images/doc-a_p5_i0",
      },
    ],
    ...overrides,
  };
}

export function makeAssistantMessage(overrides: Partial<ChatResponse> = {}): AssistantMessage {
  const response = makeResponse(overrides);
  return { id: "m1", role: "assistant", content: response.answer, createdAt: new Date().toISOString(), response };
}
