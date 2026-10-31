import type { ImageCitation, SourceCitation } from "../api/types";

interface Props {
  messageId: string;
  sources: SourceCitation[];
  images: ImageCitation[];
  /** source index → citation number */
  sourceToCitation: Map<number, number>;
  highlighted: number | null;
  open: boolean;
  onToggle: (open: boolean) => void;
}

export const sourceDomId = (messageId: string, idx: number) => `src-${messageId}-${idx}`;

export function SourcesList({ messageId, sources, images, sourceToCitation, highlighted, open, onToggle }: Props) {
  if (sources.length === 0) return null;
  const maxScore = Math.max(...sources.map((s) => s.score), Number.EPSILON);
  // Show sources sorted by their citation number so [1], [2]... read top to bottom.
  const order = sources
    .map((_, i) => i)
    .sort((a, b) => (sourceToCitation.get(a) ?? a + 1) - (sourceToCitation.get(b) ?? b + 1));

  return (
    <details className="sources" open={open} onToggle={(e) => onToggle((e.currentTarget as HTMLDetailsElement).open)}>
      <summary className="sources__summary">
        Fuentes <span className="count-pill">{sources.length}</span>
      </summary>
      <ol className="sources__list">
        {order.map((i) => {
          const s = sources[i];
          const n = sourceToCitation.get(i) ?? i + 1;
          const relative = Math.max(0.04, Math.min(1, s.score / maxScore));
          const hasFigure = images.some((img) => img.related_chunk_id === s.chunk_id);
          return (
            <li
              key={`${s.chunk_id}-${i}`}
              id={sourceDomId(messageId, i)}
              className={`source-card${highlighted === i ? " source-card--highlight" : ""}`}
              tabIndex={-1}
            >
              <div className="source-card__head">
                <span className="source-card__num" aria-label={`Cita ${n}`}>
                  {n}
                </span>
                <span className="source-card__title">
                  Fuente: <strong>{s.document_name}</strong>, Página {s.page}
                </span>
                {hasFigure && (
                  <span className="source-card__figure" title="Esta fuente tiene una figura asociada">
                    🖼️ <span className="visually-hidden">Tiene figura asociada</span>
                  </span>
                )}
              </div>
              {s.snippet && <blockquote className="source-card__snippet">{s.snippet}</blockquote>}
              <div className="score" title={`Puntuación de recuperación: ${s.score.toFixed(3)}`}>
                <span className="score__label">Relevancia</span>
                <span
                  className="score__bar"
                  role="meter"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(relative * 100)}
                  aria-label="Relevancia relativa"
                >
                  <span className="score__fill" style={{ width: `${relative * 100}%` }} />
                </span>
                <span className="score__value">{s.score.toFixed(3)}</span>
              </div>
            </li>
          );
        })}
      </ol>
    </details>
  );
}
