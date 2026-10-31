import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AssistantMessage } from "../hooks/useChat";
import { buildCitationMap, cleanAnswer } from "../lib/citations";
import { plural } from "../lib/format";
import { FigureCard } from "./FigureCard";
import { Lightbox } from "./Lightbox";
import { MarkdownAnswer } from "./MarkdownAnswer";
import { SourcesList, sourceDomId } from "./SourcesList";

interface Props {
  message: AssistantMessage;
}

export function AssistantMessageView({ message }: Props) {
  const { response } = message;
  const sources = useMemo(() => response.sources ?? [], [response.sources]);
  // Visual context is only meaningful when the answer is grounded.
  const images = useMemo(
    () => (response.has_sufficient_context ? response.images ?? [] : []),
    [response.has_sufficient_context, response.images],
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

  const insufficient = !response.has_sufficient_context;
  const withVisual = images.length > 0;

  return (
    <article
      className={`message message--assistant${insufficient ? " message--insufficient" : ""}`}
      aria-label="Respuesta del asistente"
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
            <MarkdownAnswer content={content} sources={sources} citationToSource={toSource} onCite={focusSource} />
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
