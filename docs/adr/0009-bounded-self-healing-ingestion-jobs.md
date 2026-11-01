# ADR 0009 – Jobs de ingesta acotados y autorreparables

- **Estado:** Aceptada
- **Fecha:** 2026-10-31
- **Complementa:** [ADR 0004](0004-async-ingestion-celery-redis.md)

## Contexto

En una ejecución real un job se quedó ~3 h en `processing` al 55 % ("Describing images (1/2)"): el
equipo se suspendió, Ollama no pudo cargar el modelo de visión dos veces (`timed out waiting for
llama-server to start` → HTTP 500 tras 30–47 s) y la tarea Celery (`acks_late`, broker Redis) sólo se
re-entregó al vencer el *visibility timeout*. Nada marcaba el job como `failed` y la UI seguía
consultando un job que parecía vivo. Además, `--concurrency=2` cargaba los modelos de FastEmbed
(~600 MB) en cada proceso hijo dentro de una VM Docker de 8 GB.

## Decisión

1. **Límites de tiempo Celery** (`INGESTION_SOFT_TIME_LIMIT_SECONDS=1800`,
   `INGESTION_HARD_TIME_LIMIT_SECONDS=1900`, validado `hard > soft`) en la tarea y en la config.
   - El caso de uso sigue sin conocer Celery: `ProcessDocumentUseCase.execute(job_id,
     interrupt_errors=(...))` recibe tipos de excepción opacos que **nunca** se tragan (ni el
     `except` genérico del pipeline ni el de cada imagen) y se propagan intactos.
   - La tarea captura `SoftTimeLimitExceeded` y llama a `mark_timed_out(job_id)` →
     `failed` con "Tiempo máximo de procesamiento excedido…". La tarea termina con éxito (se hace
     *ack*: no hay re-entrega). `mark_failed` nunca pisa un job `completed`.
   - Se descartó un *deadline* cooperativo: no puede interrumpir una llamada HTTP colgada; la señal
     de Celery sí.
2. **`visibility_timeout` = hard limit + 300 s** (`broker_transport_options`). Con `acks_late`, Redis
   re-entrega cualquier tarea no confirmada tras ese tiempo; si fuera menor que la duración de una
   tarea sana, otro worker la ejecutaría en paralelo. Así sólo se re-entrega si el worker se perdió.
3. **Descripción de imágenes acotada** (paso opcional): presupuesto por documento
   (`DESCRIBE_IMAGES_BUDGET_SECONDS=600`), corte tras N fallos consecutivos
   (`DESCRIBE_IMAGES_MAX_CONSECUTIVE_FAILURES=2`) y corte inmediato con `CircuitOpenError`. Las
   imágenes restantes se indexan con el pie de figura del PDF; el job completa igualmente.
4. **Latido de progreso:** antes de cada imagen se actualiza el mensaje ("Describing image 2/5"),
   lo que refresca `updated_at` aunque la llamada al LLM sea larga.
5. **Watchdog de jobs huérfanos** (`JOB_STALE_AFTER_SECONDS=900`): `GetJobUseCase`
   (`app/application/jobs.py`) informa como `failed` ("El procesamiento se interrumpió…") un job
   `processing` cuyo `updated_at` supera el umbral. El veredicto se **deriva en lectura y no se
   persiste**: `GET /jobs/{id}` no tiene efectos secundarios ni compite en escritura con el worker,
   y si el worker sólo iba lento, su siguiente actualización lo devuelve a `processing`. El umbral
   debe superar la llamada más larga sin progreso (`LLM_TIMEOUT_SECONDS × RETRY_MAX_ATTEMPTS`).
6. **Semántica de re-entrega:** `completed` y `failed` son terminales y se omiten (un job fallido no
   resucita en silencio); `pending`/`processing` se (re)procesan desde cero (log
   `job_redelivered_restarting`), lo que es idempotente por `document_id`. Un job marcado como
   huérfano por el watchdog sigue `processing` en Redis, por lo que una re-entrega legítima lo termina.
7. **Worker con `--concurrency=1`**: cada hijo *prefork* carga sus propios modelos; se escala con
   réplicas (`--scale worker=N`) cuando hay memoria.

## Consecuencias

- (+) Ningún job queda `processing` indefinidamente: como mucho `soft limit` (fallo explícito) o
  `JOB_STALE_AFTER_SECONDS` sin progreso (fallo informado por el watchdog, p. ej. tras un *hard kill*,
  OOM o suspensión del equipo).
- (+) Un LLM de visión caído degrada la calidad (sin descripciones) pero no la disponibilidad.
- (−) Entre el veredicto del watchdog y una re-entrega (≤ `visibility_timeout`), la UI puede ver
  `failed` y después `processing`/`completed` si la tarea se reintenta; queda en logs.
- (−) Los límites de tiempo de Celery requieren el pool *prefork* (señales POSIX); con
  `--pool=solo` en Windows (desarrollo local) no se aplican y sólo queda el watchdog.
