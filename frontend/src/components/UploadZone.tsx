import { useRef, useState, type DragEvent } from "react";
import { MAX_UPLOAD_MB } from "../lib/config";

interface Props {
  onFiles: (files: File[]) => void;
}

export function UploadZone({ onFiles }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length) onFiles(Array.from(e.dataTransfer.files));
  };

  return (
    <div
      className={`dropzone${dragging ? " dropzone--active" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false);
      }}
      onDrop={onDrop}
    >
      <span className="dropzone__icon" aria-hidden="true">
        ⇪
      </span>
      <p className="dropzone__text">
        Arrastra aquí tus PDF o{" "}
        <button type="button" className="link-button" onClick={() => inputRef.current?.click()}>
          selecciónalos
        </button>
      </p>
      <p className="dropzone__hint">Solo PDF · máx. {MAX_UPLOAD_MB} MB por archivo</p>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        aria-label="Seleccionar archivos PDF"
        data-testid="file-input"
        onChange={(e) => {
          if (e.target.files?.length) onFiles(Array.from(e.target.files));
          e.target.value = "";
        }}
      />
    </div>
  );
}
