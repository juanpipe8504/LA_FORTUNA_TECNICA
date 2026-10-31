# ADR 0003 – Chunking consciente del layout (estructural/semántico) en lugar de tamaño fijo

- **Estado:** Aceptada
- **Fecha:** 2026-10-31

## Contexto

Los PDF técnicos tienen estructura fuerte: secciones jerárquicas, tablas de especificaciones,
procedimientos numerados y figuras con pie. Cortar cada N caracteres parte tablas por la mitad,
mezcla secciones distintas y separa el texto de su figura, degradando tanto la recuperación como la
calidad de la respuesta.

## Decisión

1. **Extracción con PyMuPDF** (`PyMuPDFParser`, `app/infrastructure/parsing/`): bloques de texto con
   tamaño de fuente y negrita, tablas con `find_tables()` convertidas a Markdown, imágenes con su
   *bounding box* (`BoundingBox` en puntos PDF, origen arriba-izquierda) y pies de figura.
2. **`LayoutAwareChunker`** (`app/infrastructure/chunking/layout_chunker.py`):
   - Detecta encabezados por heurística de tamaño de fuente / negrita y mantiene `section_path`
     (p. ej. `["5. Mantenimiento", "5.1 Procedimiento de relubricación de rodamientos"]`), que aporta
     contexto al texto embebido.
   - Empaqueta párrafos de una misma sección hasta `CHUNK_MAX_TOKENS`, cortando solo en límites de
     oración, con solapamiento de `CHUNK_OVERLAP_SENTENCES` oraciones.
   - **Tablas → un único chunk `TABLE`** en Markdown (nunca se dividen).
   - **Imágenes → un chunk `IMAGE`** con pie + descripción del LLM de visión + contexto de sección, de
     forma que una pregunta sobre un diagrama pueda recuperarlo directamente.
   - Cada chunk conserva página(s) y `bbox`, lo que permite citar "Página N" y enlazar imágenes
     (ADR 0007).

## Alternativas consideradas

- **Tamaño fijo con solapamiento:** simple, pero rompe tablas y procedimientos.
- **Chunking semántico por embeddings (cortes por caída de similitud):** caro (embebe cada frase) e
  ignora la estructura visual que el PDF ya ofrece.
- **Unstructured / docling / LayoutLM:** mejor detección de layout y OCR en PDFs escaneados, pero
  dependencias pesadas y menos control; quedan como evolución (ver README, "Limitaciones y próximos pasos").

## Consecuencias

- (+) Chunks autocontenidos y citables; tablas íntegras; figuras recuperables por su descripción.
- (+) Heurísticas deterministas y testeables con documentos sintéticos.
- (−) Las heurísticas de encabezado pueden fallar con maquetaciones exóticas; PDFs escaneados requieren OCR.
- (−) Tablas muy grandes producen chunks largos (aceptado: es mejor que tablas partidas).
