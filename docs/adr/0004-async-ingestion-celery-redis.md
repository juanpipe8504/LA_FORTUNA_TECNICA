# ADR 0004 – Ingesta asíncrona con Celery + Redis y seguimiento por `job_id`

- **Estado:** Aceptada
- **Fecha:** 2026-10-31

## Contexto

Procesar un manual (parseo, extracción de imágenes, descripción con LLM de visión, embeddings,
indexado) tarda de segundos a minutos. Hacerlo dentro de la petición HTTP bloquearía la API, provocaría
*timeouts* y no escalaría. El reto pide un endpoint de subida que devuelva un `job_id` y otro de estado.

## Decisión

- `POST /api/v1/documents` valida el fichero (magic bytes `%PDF`, tamaño ≤ `MAX_UPLOAD_MB`), lo guarda
  vía `BlobStoragePort`, crea un `IngestionJob` (`pending`) en `RedisJobRepository` y encola
  `ingestion.ingest_document(job_id)` mediante `CeleryTaskQueue` → **202 Accepted** con `job_id`.
- El **worker Celery** (`app/infrastructure/queue/tasks.py`, cola `ingestion`) ejecuta
  `ProcessDocumentUseCase`, que actualiza `status`, `progress` (0–100) y `message` en cada fase y marca
  `failed` con el error si algo falla.
- `GET /api/v1/jobs/{job_id}` lee el estado de Redis (TTL 7 días). El frontend hace *polling*.
- Fiabilidad de Celery (`app/infrastructure/queue/celery_app.py`): `task_acks_late`,
  `task_reject_on_worker_lost`, `worker_prefetch_multiplier=1` (tareas largas y pesadas en CPU),
  serialización JSON.
- Redis cumple doble rol (broker + repositorio de jobs) para minimizar piezas; con `appendonly yes`.
- El estado del job **no** depende del *result backend* de Celery: es un puerto propio
  (`JobRepositoryPort`), testeable con `InMemoryJobRepository`.

## Alternativas consideradas

- **`BackgroundTasks` de FastAPI:** se ejecuta en el proceso de la API; se pierde si el proceso muere y
  no escala de forma independiente.
- **RQ / Dramatiq / arq:** más simples, pero Celery es el estándar de facto, con routing, reintentos y
  monitorización (Flower) maduros.
- **RabbitMQ / SQS como broker:** más garantías de entrega; Redis basta aquí y se cambia por
  configuración de Celery sin tocar el dominio.
- **WebSockets / SSE para el progreso:** mejor UX pero más complejidad; el *polling* es suficiente.

## Consecuencias

- (+) La API responde en milisegundos; los workers escalan horizontalmente
  (`docker compose up --scale worker=N`).
- (+) Un worker caído no pierde la tarea (*acks late*); el reprocesado de un `document_id` sobrescribe
  sus puntos (ids deterministas), por lo que es idempotente.
- (−) Un componente más que operar (worker) y consistencia eventual entre subida y disponibilidad.
- (−) Redis como único almacén de jobs: para auditoría a largo plazo convendría una base relacional.
