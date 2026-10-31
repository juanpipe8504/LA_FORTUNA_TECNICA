import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import type { ChatError, Message } from "../hooks/useChat";
import { AssistantMessageView } from "./AssistantMessageView";

interface Props {
  messages: Message[];
  pending: boolean;
  error: ChatError | null;
  scopeLabel: string;
  hasDocuments: boolean;
  onSend: (question: string) => void;
  onRetry: () => void;
  onDismissError: () => void;
}

const SUGGESTIONS = [
  "¿De qué trata el documento? Resume los puntos clave.",
  "¿Qué muestra el diagrama de arquitectura?",
  "Haz una tabla con las cifras principales mencionadas.",
];

function TypingIndicator() {
  return (
    <div className="message message--assistant" role="status" aria-live="polite">
      <div className="avatar avatar--assistant" aria-hidden="true">
        IA
      </div>
      <div className="bubble bubble--assistant typing">
        <span className="typing__dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span className="typing__text">Buscando en los documentos y redactando la respuesta…</span>
      </div>
    </div>
  );
}

export function ChatPanel({ messages, pending, error, scopeLabel, hasDocuments, onSend, onRetry, onDismissError }: Props) {
  const [draft, setDraft] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll on new messages / pending state / errors.
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages.length, pending, error]);

  // Auto-grow textarea.
  useLayoutEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [draft]);

  const canSend = draft.trim().length > 0 && !pending;

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (!canSend) return;
    onSend(draft.trim());
    setDraft("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <section className="chat" aria-label="Conversación">
      <div className="chat__messages" ref={listRef} aria-live="polite" aria-busy={pending}>
        {messages.length === 0 && !pending && (
          <div className="empty-state">
            <h2>Pregunta a tus documentos</h2>
            <p>
              Las respuestas citan la fuente exacta <span className="cite cite--static">1</span> y, cuando una figura es
              relevante, la muestran junto a su documento y página.
            </p>
            {!hasDocuments && <p className="empty-state__hint">Empieza subiendo un PDF desde el panel de la biblioteca.</p>}
            <ul className="suggestions" aria-label="Preguntas de ejemplo">
              {SUGGESTIONS.map((s) => (
                <li key={s}>
                  <button
                    type="button"
                    className="suggestion"
                    onClick={() => {
                      setDraft(s);
                      textareaRef.current?.focus();
                    }}
                  >
                    {s}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {messages.map((m) =>
          m.role === "user" ? (
            <article key={m.id} className="message message--user" aria-label="Tu pregunta">
              <div className="bubble bubble--user">{m.content}</div>
            </article>
          ) : (
            <AssistantMessageView key={m.id} message={m} />
          ),
        )}

        {pending && <TypingIndicator />}

        {error && !pending && (
          <div className="error-banner" role="alert">
            <span aria-hidden="true">⛔</span>
            <div className="error-banner__body">
              <strong>No se pudo obtener la respuesta</strong>
              <p>{error.message}</p>
              {error.requestId && <p className="error-banner__rid">ID de solicitud: {error.requestId}</p>}
            </div>
            <div className="error-banner__actions">
              <button type="button" className="button button--primary" onClick={onRetry}>
                Reintentar
              </button>
              <button type="button" className="button button--ghost" onClick={onDismissError} aria-label="Descartar error">
                ✕
              </button>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={submit}>
        <div className="composer__scope" aria-live="polite">
          <span aria-hidden="true">🔎</span> {scopeLabel}
        </div>
        <div className="composer__row">
          <label htmlFor="question" className="visually-hidden">
            Escribe tu pregunta
          </label>
          <textarea
            id="question"
            ref={textareaRef}
            className="composer__input"
            rows={1}
            placeholder="Escribe tu pregunta…  (Enter para enviar, Mayús+Enter para nueva línea)"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            maxLength={4000}
          />
          <button type="submit" className="button button--primary composer__send" disabled={!canSend} aria-label="Enviar pregunta">
            {pending ? <span className="spinner" aria-hidden="true" /> : <span aria-hidden="true">➤</span>}
            <span className="composer__send-text">{pending ? "Enviando" : "Enviar"}</span>
          </button>
        </div>
      </form>
    </section>
  );
}
