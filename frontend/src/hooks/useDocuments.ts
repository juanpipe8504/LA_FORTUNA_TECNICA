import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import type { DocumentSummary } from "../api/types";
import { loadJSON, saveJSON } from "../lib/storage";

const SELECTION_KEY = "rag.selectedDocuments.v1";
const isStringArray = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === "string");

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>(() => loadJSON(SELECTION_KEY, [], isStringArray));

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    try {
      const docs = await api.listDocuments(signal);
      setDocuments(docs);
      setError(null);
      // Drop selections pointing to documents that no longer exist.
      setSelected((prev) => prev.filter((id) => docs.some((d) => d.document_id === id)));
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof ApiError ? err.message : "No se pudo cargar la biblioteca.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const c = new AbortController();
    void refresh(c.signal);
    return () => c.abort();
  }, [refresh]);

  useEffect(() => saveJSON(SELECTION_KEY, selected), [selected]);

  const toggle = useCallback((id: string) => {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }, []);
  const clearSelection = useCallback(() => setSelected([]), []);

  return { documents, loading, error, refresh, selected, toggle, clearSelection };
}
