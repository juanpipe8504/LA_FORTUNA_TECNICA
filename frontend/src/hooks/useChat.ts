import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api/client";
import type { ChatResponse, ChatTurn } from "../api/types";
import { HISTORY_MAX_MESSAGES } from "../lib/config";
import { newId } from "../lib/format";
import { loadJSON, removeKey, saveJSON } from "../lib/storage";

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
  createdAt: string;
  /** Document scope used for this question (null = all documents). */
  documentIds: string[] | null;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  createdAt: string;
  response: ChatResponse;
}

export type Message = UserMessage | AssistantMessage;

export interface ChatError {
  message: string;
  requestId: string | null;
  /** The question that failed, so it can be retried. */
  question: string;
  documentIds: string[] | null;
}

const STORAGE_KEY = "rag.conversation.v1";

function isMessageArray(v: unknown): v is Message[] {
  return (
    Array.isArray(v) &&
    v.every(
      (m) =>
        typeof m === "object" &&
        m !== null &&
        typeof (m as Message).id === "string" &&
        typeof (m as Message).content === "string" &&
        ((m as Message).role === "user" ||
          ((m as Message).role === "assistant" && typeof (m as AssistantMessage).response === "object")),
    )
  );
}

/** Last N user/assistant turns sent to the backend as conversational memory. */
export function buildHistory(messages: Message[], max = HISTORY_MAX_MESSAGES): ChatTurn[] {
  return messages.slice(-max).map((m) => ({ role: m.role, content: m.content }));
}

export function useChat(chatFn: typeof api.chat = api.chat) {
  const [messages, setMessages] = useState<Message[]>(() => loadJSON(STORAGE_KEY, [], isMessageArray));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<ChatError | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  useEffect(() => saveJSON(STORAGE_KEY, messages), [messages]);
  useEffect(() => () => controllerRef.current?.abort(), []);

  const ask = useCallback(
    async (question: string, documentIds: string[] | null, { appendUser = true } = {}) => {
      const q = question.trim();
      if (!q || controllerRef.current) return;
      const history = buildHistory(
        appendUser ? messagesRef.current : messagesRef.current.slice(0, -1), // on retry the question is already the last message
      );
      if (appendUser) {
        const userMsg: UserMessage = {
          id: newId(),
          role: "user",
          content: q,
          createdAt: new Date().toISOString(),
          documentIds,
        };
        setMessages((prev) => [...prev, userMsg]);
      }
      setError(null);
      setPending(true);
      const controller = new AbortController();
      controllerRef.current = controller;
      try {
        const response = await chatFn({ question: q, history, document_ids: documentIds }, controller.signal);
        const assistant: AssistantMessage = {
          id: newId(),
          role: "assistant",
          content: response.answer,
          createdAt: new Date().toISOString(),
          response: { ...response, sources: response.sources ?? [], images: response.images ?? [] },
        };
        setMessages((prev) => [...prev, assistant]);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError({
          message: err instanceof ApiError ? err.message : "Ha ocurrido un error inesperado.",
          requestId: err instanceof ApiError ? err.requestId : null,
          question: q,
          documentIds,
        });
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null;
        setPending(false);
      }
    },
    [chatFn],
  );

  const send = useCallback((question: string, documentIds: string[] | null) => ask(question, documentIds), [ask]);

  const retry = useCallback(() => {
    if (!error) return;
    void ask(error.question, error.documentIds, { appendUser: false });
  }, [ask, error]);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setPending(false);
    setError(null);
    setMessages([]);
    removeKey(STORAGE_KEY);
  }, []);

  return { messages, pending, error, send, retry, reset, dismissError: () => setError(null) };
}
