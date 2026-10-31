import type { DocumentSummary } from "../api/types";
import { formatDate, plural } from "../lib/format";

interface Props {
  documents: DocumentSummary[];
  loading: boolean;
  error: string | null;
  selected: string[];
  onToggle: (id: string) => void;
  onClearSelection: () => void;
  onRefresh: () => void;
}

export function DocumentList({ documents, loading, error, selected, onToggle, onClearSelection, onRefresh }: Props) {
  return (
    <section className="library" aria-label="Biblioteca de documentos">
      <div className="section-head">
        <h2 className="section-title">
          Biblioteca <span className="count-pill">{documents.length}</span>
        </h2>
        <button type="button" className="icon-button icon-button--sm" onClick={onRefresh} aria-label="Actualizar biblioteca" title="Actualizar">
          ⟳
        </button>
      </div>
      <p className="library__hint">
        {selected.length === 0
          ? "Sin selección: se busca en todos los documentos."
          : `${plural(selected.length, "documento seleccionado", "documentos seleccionados")}.`}
        {selected.length > 0 && (
          <>
            {" "}
            <button type="button" className="link-button" onClick={onClearSelection}>
              Usar todos
            </button>
          </>
        )}
      </p>

      {error && (
        <p className="inline-error" role="alert">
          {error}{" "}
          <button type="button" className="link-button" onClick={onRefresh}>
            Reintentar
          </button>
        </p>
      )}

      {loading && documents.length === 0 && !error && (
        <ul className="library__list" aria-hidden="true">
          {[0, 1, 2].map((i) => (
            <li key={i} className="doc doc--skeleton" />
          ))}
        </ul>
      )}

      {!loading && documents.length === 0 && !error && <p className="library__empty">Aún no hay documentos procesados.</p>}

      {documents.length > 0 && (
        <ul className="library__list">
          {documents.map((d) => {
            const checked = selected.includes(d.document_id);
            const inputId = `doc-${d.document_id}`;
            return (
              <li key={d.document_id} className={`doc${checked ? " doc--selected" : ""}`}>
                <input id={inputId} type="checkbox" checked={checked} onChange={() => onToggle(d.document_id)} />
                <label htmlFor={inputId} className="doc__label">
                  <span className="doc__name" title={d.filename}>
                    {d.filename}
                  </span>
                  <span className="doc__meta">
                    {[
                      d.page_count != null && plural(d.page_count, "pág.", "págs."),
                      d.image_count != null && plural(d.image_count, "imagen", "imágenes"),
                      formatDate(d.created_at),
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </label>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
