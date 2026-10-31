import { useState } from "react";
import { resolveAssetUrl } from "../api/client";
import type { ImageCitation } from "../api/types";
import { PageBadge } from "./PageBadge";

interface Props {
  image: ImageCitation;
  /** Citation number of the source chunk this figure is linked to, if any. */
  relatedCitation: number | null;
  onOpen: () => void;
  onShowSource: (() => void) | null;
}

export function FigureCard({ image, relatedCitation, onOpen, onShowSource }: Props) {
  const [status, setStatus] = useState<"loading" | "loaded" | "error">("loading");
  const caption = image.caption?.trim() || "Figura sin pie de imagen";
  const where = `${image.document_name}, página ${image.page}`;

  return (
    <figure className="figure-card">
      <button
        type="button"
        className={`figure-card__thumb figure-card__thumb--${status}`}
        onClick={onOpen}
        disabled={status === "error"}
        aria-label={`Ampliar imagen: ${caption} (${where})`}
      >
        {status !== "error" ? (
          <img
            src={resolveAssetUrl(image.url)}
            alt={caption}
            loading="lazy"
            decoding="async"
            onLoad={() => setStatus("loaded")}
            onError={() => setStatus("error")}
          />
        ) : (
          <span className="figure-card__fallback" role="img" aria-label="Imagen no disponible">
            <span aria-hidden="true">🖼️</span>
            No se pudo cargar la imagen
          </span>
        )}
        {status === "loaded" && (
          <span className="figure-card__zoom" aria-hidden="true">
            ⤢
          </span>
        )}
      </button>
      <figcaption className="figure-card__caption">
        <PageBadge documentName={image.document_name} page={image.page} />
        <span className="figure-card__text">{caption}</span>
        {relatedCitation !== null && onShowSource && (
          <button type="button" className="link-button" onClick={onShowSource}>
            Ver fuente [{relatedCitation}]
          </button>
        )}
      </figcaption>
    </figure>
  );
}
