import type { UploadItem, UploadPhase } from "../hooks/useUploads";
import { formatBytes, plural } from "../lib/format";

const PHASE_LABEL: Record<UploadPhase, string> = {
  uploading: "Subiendo",
  pending: "Pendiente",
  processing: "Procesando",
  completed: "Completado",
  failed: "Fallido",
};

interface Props {
  items: UploadItem[];
  onDismiss: (localId: string) => void;
  onClearFinished: () => void;
}

export function UploadList({ items, onDismiss, onClearFinished }: Props) {
  if (items.length === 0) return null;
  const anyFinished = items.some((i) => i.phase === "completed" || i.phase === "failed");
  return (
    <section className="uploads" aria-label="Ingestas en curso">
      <div className="section-head">
        <h2 className="section-title">Ingestas</h2>
        {anyFinished && (
          <button type="button" className="link-button" onClick={onClearFinished}>
            Limpiar terminadas
          </button>
        )}
      </div>
      <ul className="uploads__list">
        {items.map((it) => {
          const active = it.phase !== "completed" && it.phase !== "failed";
          const indeterminate = it.phase === "uploading" || (it.phase === "pending" && it.progress === 0);
          return (
            <li key={it.localId} className={`upload upload--${it.phase}`}>
              <div className="upload__head">
                <span className="upload__name" title={it.filename}>
                  {it.filename}
                </span>
                <span className={`chip chip--${it.phase}`}>{PHASE_LABEL[it.phase]}</span>
                <button
                  type="button"
                  className="icon-button icon-button--sm"
                  onClick={() => onDismiss(it.localId)}
                  aria-label={active ? `Dejar de seguir ${it.filename}` : `Quitar ${it.filename} de la lista`}
                >
                  ✕
                </button>
              </div>
              {it.phase !== "failed" && (
                <div
                  className={`progress${indeterminate ? " progress--indeterminate" : ""}`}
                  role="progressbar"
                  aria-label={`Progreso de ${it.filename}`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={indeterminate ? undefined : it.progress}
                >
                  <span className="progress__fill" style={{ width: indeterminate ? undefined : `${it.progress}%` }} />
                </div>
              )}
              <p className="upload__meta">
                {it.phase === "failed" ? (
                  <span className="upload__error" role="alert">
                    {it.error ?? "Error desconocido"}
                  </span>
                ) : it.phase === "completed" ? (
                  <>
                    {[
                      it.pageCount != null && plural(it.pageCount, "página", "páginas"),
                      it.chunkCount != null && plural(it.chunkCount, "fragmento", "fragmentos"),
                      it.imageCount != null && plural(it.imageCount, "imagen", "imágenes"),
                    ]
                      .filter(Boolean)
                      .join(" · ") || "Listo para consultar"}
                  </>
                ) : (
                  <>
                    {it.message ?? (it.phase === "uploading" ? "Enviando archivo…" : "En cola…")}
                    {!indeterminate && <span className="upload__pct"> · {it.progress}%</span>}
                    <span className="upload__size"> · {formatBytes(it.size)}</span>
                  </>
                )}
              </p>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
