# ADR 0001 – Arquitectura hexagonal (puertos y adaptadores) con inyección de dependencias

- **Estado:** Aceptada
- **Fecha:** 2026-10-31

## Contexto

El reto exige poder cambiar de proveedor de IA (OpenAI ↔ Ollama) "sin tocar la lógica de negocio",
configurar el sistema por entorno y testear la lógica con mocks. Además el sistema tiene dos procesos
de entrada (API HTTP y worker Celery) que deben ejecutar exactamente la misma lógica de ingesta.

## Decisión

Adoptar arquitectura hexagonal en cuatro capas dentro del paquete `app`:

| Capa | Paquete | Responsabilidad | Depende de |
|---|---|---|---|
| Dominio | `app/domain` | Entidades (`Chunk`, `BoundingBox`, `IngestionJob`, `Answer`...), puertos abstractos (`ports.py`) y excepciones | nada (Python puro) |
| Aplicación | `app/application` | Casos de uso: `SubmitDocumentUseCase`, `ProcessDocumentUseCase`, `AnswerQuestionUseCase`, `ListDocumentsUseCase`; prompts | dominio |
| Infraestructura | `app/infrastructure` | Adaptadores *driven*: PyMuPDF, chunker, Qdrant, OpenAI, Ollama, FastEmbed, Redis, Celery, almacenamiento local, resiliencia | dominio + librerías |
| Entrada / composición | `app/api`, `app/infrastructure/queue/tasks.py`, `app/bootstrap` | Adaptadores *driving* (FastAPI, tarea Celery) y *composition root* | todo |

- Los casos de uso reciben **puertos** por constructor (`LLMPort`, `EmbeddingPort`, `VectorStorePort`,
  `DocumentParserPort`, `ChunkerPort`, `BlobStoragePort`, `JobRepositoryPort`, `TaskQueuePort`).
- `app/bootstrap/container.py::build_container(settings)` es el único lugar que conoce las clases
  concretas; elige adaptadores según `LLM_PROVIDER` / `EMBEDDING_PROVIDER` y envuelve LLM y embeddings
  con los decoradores de resiliencia (ADR 0005).
- Configuración tipada con `pydantic-settings` (`app/core/config.py`): `.env` + overlay `.env.<APP_ENV>`
  + variables de entorno reales (máxima prioridad).
- DI manual (dataclass `Container`) en lugar de un framework de DI: explícita, sin magia y suficiente
  para este tamaño. FastAPI obtiene el contenedor mediante una dependencia, lo que permite inyectar uno
  de test en `create_app(container)`.

## Alternativas consideradas

- **LangChain / LlamaIndex como columna vertebral:** aceleran el prototipo pero acoplan el dominio a
  sus abstracciones, dificultan controlar el chunking por layout, el vínculo imagen-texto y la
  resiliencia, y ocultan justo lo que el reto quiere evaluar.
- **Contenedor DI (`dependency-injector`, `punq`):** más indirección sin beneficio real aquí.
- **Arquitectura en capas clásica (MVC con servicios):** los servicios acabarían importando SDKs concretos.

## Consecuencias

- (+) Cambiar OpenAI ↔ Ollama es una variable de entorno; añadir otro proveedor = un adaptador nuevo.
- (+) Los casos de uso se testean con *fakes* en memoria, sin red (`backend/tests/unit`).
- (+) API y worker comparten el mismo `Container`, garantizando el mismo comportamiento.
- (−) Más ficheros y algo de *boilerplate* (DTOs de API ↔ entidades de dominio).
- (−) El dominio usa dataclasses propias; hay que mapearlas a payloads de Qdrant y a esquemas Pydantic.
