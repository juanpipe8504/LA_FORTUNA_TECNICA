# ADR 0008 – FastEmbed (local, CPU) como proveedor de embeddings por defecto

- **Estado:** Aceptada
- **Fecha:** 2026-10-31

## Contexto

Los embeddings se calculan para cada chunk en la ingesta y para cada pregunta. Depender de una API de
pago para esto encarece la ingesta masiva, añade latencia y un punto de fallo, y obliga a tener clave
incluso para probar el sistema. Además el BM25 *sparse* (ADR 0002) ya usa FastEmbed.

## Decisión

- `EMBEDDING_PROVIDER=fastembed` por defecto, con `BAAI/bge-small-en-v1.5` (384 dimensiones, ONNX,
  CPU) mediante `FastEmbedEmbeddings` (carga perezosa del modelo).
- Los modelos `BAAI/bge-small-en-v1.5` y `Qdrant/bm25` se **descargan en la build de la imagen
  Docker** (`FASTEMBED_CACHE_PATH=/opt/models/fastembed`): el contenedor arranca sin depender de
  Hugging Face y la primera ingesta no paga la descarga.
- Alternativas intercambiables por configuración vía `EmbeddingPort`: `openai`
  (`text-embedding-3-small`) y `ollama` (`nomic-embed-text`). Todas pasan por `ResilientEmbeddings`.
- El LLM (generación y visión) es independiente: `LLM_PROVIDER=openai|ollama`.

## Alternativas consideradas

- **OpenAI embeddings por defecto:** mejor calidad multilingüe, pero coste por token y requiere clave.
- **Modelo multilingüe (`intfloat/multilingual-e5-*`, `bge-m3`):** mejor para español; más pesados en
  imagen y CPU. Es el primer candidato de mejora (cambio de una variable + re-ingesta).
- **sentence-transformers + PyTorch:** imagen de varios GB; FastEmbed (ONNX Runtime) es mucho más ligero.

## Consecuencias

- (+) El sistema funciona sin coste de embeddings y, con Ollama, queda 100 % local.
- (+) Arranque reproducible: modelos incluidos en la imagen.
- (−) `bge-small-en` está optimizado para inglés; con manuales en español la parte léxica (BM25) de la
  búsqueda híbrida compensa parcialmente. En producción se recomienda un modelo multilingüe.
- (−) Cambiar de modelo cambia la dimensión → nueva colección (`QDRANT_COLLECTION`) y re-ingesta.
