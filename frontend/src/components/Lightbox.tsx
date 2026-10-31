import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { resolveAssetUrl } from "../api/client";
import type { ImageCitation } from "../api/types";
import { PageBadge } from "./PageBadge";

interface Props {
  images: ImageCitation[];
  index: number;
  onIndexChange: (i: number) => void;
  onClose: () => void;
}

export function Lightbox({ images, index, onIndexChange, onClose }: Props) {
  const image = images[index];
  const closeRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);
  const hasMany = images.length > 1;

  useEffect(() => setFailed(false), [index]);

  // Focus management: focus close button, restore focus on unmount, lock body scroll.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prevOverflow;
      previouslyFocused?.focus?.();
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (hasMany && e.key === "ArrowRight") {
        onIndexChange((index + 1) % images.length);
      } else if (hasMany && e.key === "ArrowLeft") {
        onIndexChange((index - 1 + images.length) % images.length);
      } else if (e.key === "Tab" && dialogRef.current) {
        // Simple focus trap.
        const focusables = dialogRef.current.querySelectorAll<HTMLElement>("button, a[href]");
        if (focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [hasMany, images.length, index, onClose, onIndexChange]);

  if (!image) return null;
  const caption = image.caption?.trim() || "Figura sin pie de imagen";
  const src = resolveAssetUrl(image.url);

  return createPortal(
    <div className="lightbox" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        ref={dialogRef}
        className="lightbox__dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="lightbox-title"
        aria-describedby="lightbox-caption"
      >
        <header className="lightbox__header">
          <h2 id="lightbox-title" className="lightbox__title">
            <PageBadge documentName={image.document_name} page={image.page} />
            {hasMany && (
              <span className="lightbox__counter">
                {index + 1} / {images.length}
              </span>
            )}
          </h2>
          <button ref={closeRef} type="button" className="icon-button" onClick={onClose} aria-label="Cerrar (Esc)">
            ✕
          </button>
        </header>
        <div className="lightbox__body">
          {hasMany && (
            <button
              type="button"
              className="icon-button lightbox__nav lightbox__nav--prev"
              onClick={() => onIndexChange((index - 1 + images.length) % images.length)}
              aria-label="Imagen anterior"
            >
              ‹
            </button>
          )}
          {failed ? (
            <p className="lightbox__error" role="alert">
              No se pudo cargar la imagen.
            </p>
          ) : (
            <img key={src} src={src} alt={caption} className="lightbox__img" onError={() => setFailed(true)} />
          )}
          {hasMany && (
            <button
              type="button"
              className="icon-button lightbox__nav lightbox__nav--next"
              onClick={() => onIndexChange((index + 1) % images.length)}
              aria-label="Imagen siguiente"
            >
              ›
            </button>
          )}
        </div>
        <footer className="lightbox__footer">
          <p id="lightbox-caption" className="lightbox__caption">
            {caption}
          </p>
          <a className="link-button" href={src} target="_blank" rel="noopener noreferrer">
            Abrir en pestaña nueva ↗
          </a>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
