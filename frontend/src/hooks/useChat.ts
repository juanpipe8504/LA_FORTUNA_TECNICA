import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, STREAM_ERROR_CODES, api, streamChat } from "../api/client";
import type { ChatRequest, ChatResponse, ChatTurn } from "../api/types";
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

/**
 * Lifecycle of an assistant message:
 * - `streaming`: tokens are arriving; `content` is partial, `response` is a placeholder.
 * - `interrupted`: the user stopped generation; `content` is the partial text.
 * - `error`: the stream failed after some text arrived; `content` is the partial text.
 * - undefined / `complete`: `content` and `response` come from the final answer.
 */
export type AssistantStatus = "streaming" | "interrupted" | "error" | "complete";

export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  createdAt: string;
  response: ChatResponse;
  status?: AssistantStatus;
}

export type Message = UserMessage | AssistantMessage;

export interface ChatError {
  message: string;
  requestId: string | null;
  /** The question that failed, so it can be retried. */
  question: string;
  documentIds: string[] | null;
  /** Partial assistant message left by a failed stream (removed on retry). */
  partialMessageId?: string | null;
}

/** `waiting`: request in flight, no token yet. `streaming`: tokens arriving. */
export type ChatPhase = "idle" | "waiting" | "streaming";

export type StreamFn = typeof streamChat;

const STORAGE_KEY = "rag.conversation.v1";

const PLACEHOLDER_RESPONSE: ChatResponse = { answer: "", has_sufficient_context: true, sources: [], images: [] };

export function isFinalMessage(m: AssistantMessage): boolean {
  return m.status === undefined || m.status === "complete";
}

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

/** Never persist (or restore) a half-streamed message as if it were complete. */
function forStorage(messages: Message[]): Message[] {
  return messages.map((m) =>
    m.role === "assistant" && m.status === "streaming" ? { ...m, status: "interrupted" as const } : m,
  );
}

function loadMessages(): Message[] {
  return forStorage(loadJSON<Message[]>(STORAGE_KEY, [], isMessageArray));
}

/** Last N user/assistant turns sent to the backend as conversational memory. */
export function buildHistory(messages: Message[], max = HISTORY_MAX_MESSAGES): ChatTurn[] {
  return messages
    .filter((m) => !(m.role === "assistant" && (m.status === "streaming" || !m.content.trim())))
    .slice(-max)
    .map((m) => ({ role: m.role, content: m.content }));
}

function errorInfo(err: unknown): { message: string; requestId: string | null } {
  return {
    message: err instanceof ApiError ? err.message : "Ha ocurrido un error inesperado.",
    requestId: err instanceof ApiError ? err.requestId : null,
  };
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

/** Stream failures (before any token) after which retrying with the classic POST /chat makes sense. */
function shouldFallback(err: ApiError): boolean {
  if (err.status === 404 || err.status === 405) return true;
  return (
    err.code === STREAM_ERROR_CODES.network ||
    err.code === STREAM_ERROR_CODES.unsupported ||
    err.code === STREAM_ERROR_CODES.interrupted ||
    err.code === STREAM_ERROR_CODES.incomplete
  );
}

/** Endpoint missing (old backend / proxy): stop trying to stream for this session. */
function streamUnsupported(err: ApiError): boolean {
  return err.status === 404 || err.status === 405 || err.code === STREAM_ERROR_CODES.unsupported;
}

/**
 * Chat state. Streams answers from POST /chat/stream by default, transparently
 * falling back to POST /chat when streaming is unavailable. Pass `streamFn = null`
 * to disable streaming.
 */
export function useChat(chatFn: typeof api.chat = api.chat, streamFn: StreamFn | null = streamChat) {
  const [messages, setMessages] = useState<Message[]>(loadMessages);
  const [phase, setPhase] = useState<ChatPhase>("idle");
  const [error, setError] = useState<ChatError | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const streamDisabledRef = useRef(false);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  useEffect(() => saveJSON(STORAGE_KEY, forStorage(messages)), [messages]);
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
      setPhase("waiting");
      const controller = new AbortController();
      controllerRef.current = controller;
      const isCurrent = () => controllerRef.current === controller;

      const payload: ChatRequest = { question: q, history, document_ids: documentIds };
      const assistantId = newId();
      const createdAt = new Date().toISOString();
      let partial = "";
      let started = false;

      const upsertAssistant = (patch: Partial<AssistantMessage>) => {
        setMessages((prev) => {
          const idx = prev.findIndex((m) => m.id === assistantId);
          if (idx === -1) {
            const msg: AssistantMessage = {
              id: assistantId,
              role: "assistant",
              content: "",
              createdAt,
              response: PLACEHOLDER_RESPONSE,
              ...patch,
            };
            return [...prev, msg];
          }
          const next = prev.slice();
          next[idx] = { ...(prev[idx] as AssistantMessage), ...patch };
          return next;
        });
      };

      const finish = (response: ChatResponse) => {
        upsertAssistant({
          // The final answer replaces the streamed text: citations are renumbered server-side.
          content: response.answer,
          status: undefined,
          response: { ...response, sources: response.sources ?? [], images: response.images ?? [] },
        });
      };

      const fail = (err: unknown) => {
        if (started) upsertAssistant({ content: partial, status: "error" });
        setError({ ...errorInfo(err), question: q, documentIds, partialMessageId: started ? assistantId : null });
      };

      try {
        if (streamFn && !streamDisabledRef.current) {
          let streamError: ApiError | null = null;
          let finalReceived = false;
          await streamFn(
            payload,
            {
              onToken: (text) => {
                if (!isCurrent()) return;
                partial += text;
                if (!started) {
                  started = true;
                  setPhase("streaming");
                }
                upsertAssistant({ content: partial, status: "streaming" });
              },
              onFinal: (response) => {
                if (!isCurrent()) return;
                finalReceived = true;
                finish(response);
              },
              onError: (err) => {
                streamError = err;
              },
            },
            controller.signal,
          );
          if (!isCurrent() || finalReceived) return;
          const err = streamError as ApiError | null;
          if (err && (started || !shouldFallback(err))) {
            fail(err);
            return;
          }
          if (err && streamUnsupported(err)) streamDisabledRef.current = true;
        }
        // Classic request: no stream support, or the stream failed before any token arrived.
        const response = await chatFn(payload, controller.signal);
        if (!isCurrent()) return;
        finish(response);
      } catch (err) {
        if (!isCurrent()) return; // reset / unmount
        if (isAbortError(err) || controller.signal.aborted) {
          // Stopped by the user: keep whatever was generated, marked as interrupted.
          upsertAssistant({ content: partial, status: "interrupted" });
          return;
        }
        fail(err);
      } finally {
        if (isCurrent()) {
          controllerRef.current = null;
          setPhase("idle");
        }
      }
    },
    [chatFn, streamFn],
  );

  const send = useCallback((question: string, documentIds: string[] | null) => ask(question, documentIds), [ask]);

  const retry = useCallback(() => {
    if (!error) return;
    if (error.partialMessageId) {
      // Drop the incomplete answer so the question is again the last message.
      const id = error.partialMessageId;
      const next = messagesRef.current.filter((m) => m.id !== id);
      messagesRef.current = next;
      setMessages(next);
    }
    void ask(error.question, error.documentIds, { appendUser: false });
  }, [ask, error]);

  /** Aborts the in-flight answer, keeping the partial text marked as interrupted. */
  const stop = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    const controller = controllerRef.current;
    controllerRef.current = null;
    controller?.abort();
    setPhase("idle");
    setError(null);
    setMessages([]);
    removeKey(STORAGE_KEY);
  }, []);

  const dismissError = useCallback(() => setError(null), []);

  return { messages, pending: phase !== "idle", phase, error, send, retry, stop, reset, dismissError };
}
