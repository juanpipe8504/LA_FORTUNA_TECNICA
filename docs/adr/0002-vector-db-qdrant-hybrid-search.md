# ADR 0002 – Qdrant como base vectorial con búsqueda híbrida (densa + BM25) fusionada con RRF

- **Estado:** Aceptada
- **Fecha:** 2026-10-31

## Contexto

Los manuales técnicos mezclan lenguaje natural ("¿por qué se calienta el motor?") con términos exactos
(códigos de rodamiento `6313-2Z/C3`, "IP55", bornes `U1/W2`, números de pieza). La búsqueda puramente
semántica falla con identificadores exactos; la puramente léxica falla con paráfrasis. El reto pide
explícitamente recuperación híbrida.

## Decisión

- Usar **Qdrant** (`qdrant/qdrant` en docker-compose) a través de `QdrantVectorStore`
  (`app/infrastructure/vectorstore/qdrant_store.py`), que implementa `VectorStorePort`.
- Cada punto tiene dos *named vectors*: `dense` (embedding del proveedor configurado) y `bm25`
  (vector *sparse* calculado con FastEmbed `Qdrant/bm25`, con modificador IDF aplicado por Qdrant).
- `hybrid_search` usa la **Query API** de Qdrant: dos `prefetch` (denso y sparse) fusionados con
  **Reciprocal Rank Fusion (RRF)** en el servidor, en una única llamada.
- El payload guarda el chunk completo (texto, páginas, `bbox`, `section_path`, `chunk_type`, `images`)
  para no necesitar otra base de datos al responder.
- Índice de payload sobre `document_id` para filtrar por documento de forma eficiente
  (`document_ids` en `POST /chat`).

## Alternativas consideradas

- **pgvector + `tsvector`:** muy válido si ya existe Postgres; la fusión híbrida hay que hacerla a mano
  en SQL y el ranking de Postgres no es BM25 real.
- **Elasticsearch / OpenSearch:** excelente BM25, pero mucho más pesado de operar para este alcance.
- **Chroma / FAISS:** sin búsqueda híbrida nativa ni filtrado/escalado de servidor comparable.
- **Weaviate:** también tiene híbrido; Qdrant ofrece mejor ergonomía de vectores *sparse* + RRF y una
  imagen ligera.
- **Fusión ponderada (α·denso + (1−α)·bm25):** requiere normalizar scores de escalas distintas; RRF
  solo usa rangos y no necesita calibración.

## Consecuencias

- (+) Términos exactos y paráfrasis se recuperan bien; una sola llamada de red por consulta.
- (+) Filtrado por documento nativo; escalado horizontal (sharding / réplicas) disponible.
- (−) Los scores RRF no son similitudes coseno: `RETRIEVAL_MIN_SCORE` debe interpretarse en esa escala.
- (−) Cambiar de modelo de embeddings cambia la dimensión → nueva colección y re-ingesta.
