import { useCallback, useEffect, useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { DocumentList } from "./components/DocumentList";
import { UploadList } from "./components/UploadList";
import { UploadZone } from "./components/UploadZone";
import { useChat } from "./hooks/useChat";
import { useDocuments } from "./hooks/useDocuments";
import { useUploads } from "./hooks/useUploads";
import { plural } from "./lib/format";

export default function App() {
  const docs = useDocuments();
  const chat = useChat();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { refresh } = docs;

  const uploads = useUploads({ onCompleted: useCallback(() => void refresh(), [refresh]) });

  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawerOpen(false);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  const scopeIds = docs.selected.length > 0 ? docs.selected : null;
  const scopeLabel =
    scopeIds === null
      ? "Buscando en todos los documentos"
      : scopeIds.length === 1
        ? `Buscando en: ${docs.documents.find((d) => d.document_id === scopeIds[0])?.filename ?? "1 documento"}`
        : `Buscando en ${plural(scopeIds.length, "documento", "documentos")}`;

  return (
    <div className="app">
      <header className="topbar">
        <button
          type="button"
          className="icon-button topbar__menu"
          aria-label={drawerOpen ? "Cerrar biblioteca" : "Abrir biblioteca"}
          aria-expanded={drawerOpen}
          aria-controls="sidebar"
          onClick={() => setDrawerOpen((o) => !o)}
        >
          ☰
        </button>
        <h1 className="topbar__title">
          <span className="topbar__logo" aria-hidden="true">
            ◈
          </span>
          Asistente documental
        </h1>
        <button
          type="button"
          className="button button--ghost"
          onClick={chat.reset}
          disabled={chat.messages.length === 0 && !chat.pending && !chat.error}
        >
          ＋ Nueva conversación
        </button>
      </header>

      <div className="layout">
        <aside id="sidebar" className={`sidebar${drawerOpen ? " sidebar--open" : ""}`} aria-label="Documentos">
          <UploadZone onFiles={uploads.upload} />
          <UploadList items={uploads.items} onDismiss={uploads.dismiss} onClearFinished={uploads.clearFinished} />
          <DocumentList
            documents={docs.documents}
            loading={docs.loading}
            error={docs.error}
            selected={docs.selected}
            onToggle={docs.toggle}
            onClearSelection={docs.clearSelection}
            onRefresh={() => void docs.refresh()}
          />
        </aside>
        {drawerOpen && <div className="scrim" onClick={() => setDrawerOpen(false)} aria-hidden="true" />}

        <main className="main">
          <ChatPanel
            messages={chat.messages}
            pending={chat.pending}
            phase={chat.phase}
            error={chat.error}
            scopeLabel={scopeLabel}
            hasDocuments={docs.documents.length > 0}
            onSend={(q) => void chat.send(q, scopeIds)}
            onRetry={chat.retry}
            onDismissError={chat.dismissError}
            onStop={chat.stop}
          />
        </main>
      </div>
    </div>
  );
}
