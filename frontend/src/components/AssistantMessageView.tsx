import type { ImageCitation, SourceCitation } from "../api/types";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { isFinalMessage, type AssistantMessage } from "../hooks/useChat";
import { buildCitationMap, cleanAnswer } from "../lib/citations";
import { plural } from "../lib/format";
import { FigureCard } from "./FigureCard";
import { Lightbox } from "./Lightbox";
import { MarkdownAnswer } from "./MarkdownAnswer";
import { SourcesList, sourceDomId } from "./SourcesList";

const NO_SOURCES: SourceCitation[] = [];
const NO_IMAGES: ImageCitation[] = [];

interface Props {
  message: AssistantMessage;
}

export function AssistantMessageView({ message }: Props) {
  const { response } = message;
  const final = isFinalMessage(message);
  const streaming = message.status === "streaming";
  // Sources / figures / citation links only exist once the final answer has arrived.
  const sources = useMemo(() => (final ? response.sources ?? [] : NO_SOURCES), [final, response.sources]);
  // Visual context is only meaningful when the answer is grounded.
  const images = useMemo(
    () => (final && response.has_sufficient_context ? response.images ?? [] : NO_IMAGES),
    [final, response.has_sufficient_context, response.images],
  );
  const content = useMemo(() => cleanAnswer(message.content), [message.content]);
  const { toSource, toCitation } = useMemo(() => buildCitationMap(content, sources.length), [content, sources.length]);

  const [sourcesOpen, setSourcesOpen] = useState(true);
  const [highlighted, setHighlighted] = useState<number | null>(null);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);
  const timer = useRef<number>();
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const focusSource = useCallback(
    (idx: number) => {
      setSourcesOpen(true);
      setHighlighted(idx);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setHighlighted(null), 2400);
      requestAnimationFrame(() => {
        const el = document.getElementById(sourceDomId(message.id, idx));
        el?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
        el?.focus({ preventScroll: true });
      });
    },
    [message.id],
  );

  const insufficient = final && !response.has_sufficient_context;
  const withVisual = images.length > 0;

  return (
    <article
      className={`message message--assistant${insufficient ? " message--insufficient" : ""}${
        streaming ? " message--streaming" : ""
      }`}
      aria-label="Respuesta del asistente"
      aria-busy={streaming || undefined}
    >
      <div className="avatar avatar--assistant" aria-hidden="true">
        IA
      </div>
      <div className="bubble bubble--assistant">
        {insufficient && (
          <div className="insufficient-banner" role="note">
            <span aria-hidden="true">⚠️</span>
            <div>
              <strong>Información insuficiente</strong>
              <p>Los documentos disponibles no contienen información suficiente para responder con seguridad.</p>
            </div>
          </div>
        )}

        <div className={`answer${withVisual ? " answer--with-visual" : ""}`}>
          <div className="answer__text">
            {content ? (
              <MarkdownAnswer
                content={content}
                sources={sources}
                citationToSource={toSource}
                onCite={focusSource}
                linkCitations={final}
              />
            ) : (
              message.status === "interrupted" && <p className="answer__empty">Generación detenida antes de recibir texto.</p>
            )}
          </div>

          {withVisual && (
            <aside className="answer__visual" aria-label="Contexto visual">
              <h3 className="answer__visual-title">
                Contexto visual <span className="count-pill">{plural(images.length, "figura", "figuras")}</span>
              </h3>
              {images.map((img, i) => {
                const srcIdx = sources.findIndex((s) => s.chunk_id === img.related_chunk_id);
                return (
                  <FigureCard
                    key={`${img.image_id}-${i}`}
                    image={img}
                    relatedCitation={srcIdx >= 0 ? toCitation.get(srcIdx) ?? srcIdx + 1 : null}
                    onOpen={() => setLightboxIndex(i)}
                    onShowSource={srcIdx >= 0 ? () => focusSource(srcIdx) : null}
                  />
                );
              })}
            </aside>
          )}
        </div>

        {streaming && (
          <div className="stream-status" role="status">
            <span className="stream-status__pulse" aria-hidden="true" />
            Generando… <span className="stream-status__hint">Las fuentes y figuras aparecerán al terminar.</span>
          </div>
        )}
        {message.status === "interrupted" && (
          <div className="stream-status stream-status--stopped" role="note">
            <span aria-hidden="true">■</span> Respuesta interrumpida
          </div>
        )}
        {message.status === "error" && (
          <div className="stream-status stream-status--error" role="note">
            <span aria-hidden="true">⚠️</span> Respuesta incompleta: la generación falló.
          </div>
        )}

        <SourcesList
          messageId={message.id}
          sources={sources}
          images={images}
          sourceToCitation={toCitation}
          highlighted={highlighted}
          open={sourcesOpen}
          onToggle={setSourcesOpen}
        />
      </div>

      {lightboxIndex !== null && images[lightboxIndex] && (
        <Lightbox
          images={images}
          index={lightboxIndex}
          onIndexChange={setLightboxIndex}
          onClose={() => setLightboxIndex(null)}
        />
      )}
    </article>
  );
}
