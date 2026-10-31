# ADR 0008 – FastEmbed (local, CPU) como proveedor de embeddings por defecto

- **Estado:** Aceptada
- **Fecha:** 2026-10-31

## Contexto

Los embeddings se calculan para cada chunk en la ingesta y para cada pregunta. Depender de una API de
pago para esto encarece la ingesta masiva, añade latencia y un punto de fallo, y obliga a tener clave
incluso para probar el sistema. Además el BM25 *sparse* (ADR 0002) ya usa FastEmbed.

## Decisión

- `EMBEDDING_PROVIDER=fastembed` por defecto, con el modelo **multilingüe**
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones, ONNX cuantizado,
  ~0,22 GB, CPU) mediante `FastEmbedEmbeddings` (carga perezosa del modelo).
- El BM25 *sparse* usa el stemmer Snowball y las *stop-words* del idioma configurado
  (`BM25_LANGUAGE=spanish` por defecto): así «relubricar» y «Relubricación» comparten raíz y
  «de/la/el…» no contaminan la búsqueda léxica.
- Los modelos (`FASTEMBED_MODEL`, recibido como *build arg* desde `.env`, y `Qdrant/bm25`) se
  **descargan en la build de la imagen Docker** (`FASTEMBED_CACHE_PATH=/opt/models/fastembed`): el
  contenedor arranca sin depender de Hugging Face y la primera ingesta no paga la descarga.
- Alternativas intercambiables por configuración vía `EmbeddingPort`: `openai`
  (`text-embedding-3-small`) y `ollama` (`nomic-embed-text`). Todas pasan por `ResilientEmbeddings`.
- El LLM (generación y visión) es independiente: `LLM_PROVIDER=openai|ollama`.

## Medición (por qué este modelo)

Benchmark sobre `docs/sample/Manual_Motor.pdf` (PyMuPDFParser + LayoutAwareChunker → 47 chunks,
Qdrant en memoria, búsqueda híbrida RRF real de `QdrantVectorStore`) con 10 preguntas en español
(intervalo de relubricación, grasa NLGI 3, conexión en estrella, corriente nominal, medición de
aislamiento, vibraciones, temperatura ambiente, rodamiento 6313, garantía, cantidad de grasa).
MRR = media de 1/rango del chunk esperado; tiempos en CPU de portátil, por chunk / por pregunta.

| Modelo denso | Dim | Tamaño | MRR híbrido (BM25 en / **es**) | MRR solo denso | Embed chunk | Embed pregunta |
|---|---|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` (anterior) | 384 | 0,07 GB | 0,61 / **0,64** | 0,59 | ~100 ms* | ~10 ms |
| **`paraphrase-multilingual-MiniLM-L12-v2`** | 384 | 0,22 GB | 0,71 / **0,83** | 0,62 | ~39 ms | ~8 ms |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 1,0 GB | 0,77 / **0,79** | 0,64 | ~220 ms | ~22 ms |
| `intfloat/multilingual-e5-large` | 1024 | 2,2 GB | 0,80 / **0,86** | 0,88 | ~880 ms | ~90–240 ms |

\* incluye calentamiento del primer lote. BM25 solo: MRR 0,55 con `english` → 0,62 con `spanish`.
Los prefijos `query: ` / `passage: ` de e5 (FastEmbed no los añade) no cambiaron el resultado en este
conjunto (MRR denso idéntico), por eso el adaptador no los incluye.

- El BM25 en español mejora todas las combinaciones (+0,02 a +0,12 de MRR).
- `multilingual-e5-large` es el mejor denso, pero en híbrido solo supera a MiniLM por una pregunta
  (0,86 vs 0,83), es ~20× más lento por chunk, pesa 10× más en la imagen y carga ~2,2 GB de pesos en RAM
  **en cada proceso** que lo carga (API y worker), incompatible con un host de 8 GB que además
  ejecuta Ollama `gemma3:4b`.
- MiniLM multilingüe ofrece casi la misma calidad híbrida con el menor coste de CPU/RAM/imagen.

## Alternativas consideradas

- **OpenAI embeddings por defecto:** buena calidad multilingüe, pero coste por token y requiere clave.
- **`intfloat/multilingual-e5-large` / `bge-m3`:** mejor recuperación densa; recomendable con GPU o
  más RAM (cambio de `FASTEMBED_MODEL` + nueva colección + re-ingesta).
- **`BAAI/bge-small-en-v1.5`:** solo inglés; el más pequeño, pero el peor en español (MRR 0,64).
- **sentence-transformers + PyTorch:** imagen de varios GB; FastEmbed (ONNX Runtime) es mucho más ligero.

## Consecuencias

- (+) El sistema funciona sin coste de embeddings y, con Ollama, queda 100 % local.
- (+) Arranque reproducible: modelos incluidos en la imagen.
- (+) Recuperación adecuada para manuales en español (denso multilingüe + BM25 con stemmer español).
- (−) MiniLM trunca a 128 *tokens* para el vector denso; el resto del chunk solo se recupera por BM25
  (que indexa el texto completo). Chunks más cortos o e5 mitigan esta limitación.
- (−) Cambiar de modelo o de `BM25_LANGUAGE` invalida los vectores (aunque la dimensión coincida,
  p. ej. 384 → 384): nueva colección (`QDRANT_COLLECTION`) o borrar la existente, y re-ingesta.
