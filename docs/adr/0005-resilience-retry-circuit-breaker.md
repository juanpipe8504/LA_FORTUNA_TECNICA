# ADR 0005 – Resiliencia: reintentos con backoff exponencial + circuit breaker como decoradores de puertos

- **Estado:** Aceptada
- **Fecha:** 2026-10-31

## Contexto

Los proveedores de LLM/embeddings fallan de forma transitoria (HTTP 429, 5xx, *timeouts*, cortes de
red) y a veces de forma prolongada (caída del proveedor, Ollama sin arrancar). Reintentar a ciegas
amplifica la carga y la latencia; no reintentar hace frágil el sistema.

## Decisión

- **Clasificación de errores en los adaptadores:** `OpenAILLM` / `OpenAIEmbeddings` traducen
  `RateLimitError`, `APITimeoutError`, `APIConnectionError` y 5xx a `TransientError`; los errores
  permanentes (401, 400...) se elevan como `ExternalServiceError` y **no** se reintentan. Igual en los
  adaptadores de Ollama (`httpx`).
- **Reintentos** (`app/infrastructure/resilience/retry.py`): `retry_with_backoff` / `RetryPolicy`
  con backoff exponencial y *jitter*: `delay = min(max_delay, base · 2^intento)` + jitter
  (`RETRY_MAX_ATTEMPTS=4`, `RETRY_BASE_DELAY_SECONDS=0.5`, `RETRY_MAX_DELAY_SECONDS=8`).
- **Circuit breaker** (`app/infrastructure/resilience/circuit_breaker.py`): estados
  `CLOSED → OPEN → HALF_OPEN`; se abre tras `CIRCUIT_FAILURE_THRESHOLD` fallos y deja pasar una
  llamada de prueba tras `CIRCUIT_RECOVERY_SECONDS`. Con el circuito abierto se lanza
  `CircuitOpenError` inmediatamente (*fail fast*), que la API traduce a **503**.
- **Composición como decoradores** (`app/infrastructure/resilience/resilient.py`):
  `ResilientLLM(inner, breaker, policy)` y `ResilientEmbeddings(...)` implementan el mismo puerto que
  envuelven. El caso de uso no sabe que existe resiliencia; el `Container` la aplica a cualquier
  proveedor (OpenAI u Ollama).
- El reloj del breaker es inyectable (`clock=`), lo que permite testear transiciones sin `sleep`.
- **Logging estructurado** (`app/core/logging.py`) de cada reintento y cambio de estado del circuito,
  con `request_id` / `job_id` en el contexto.

## Alternativas consideradas

- **Solo `tenacity` en cada llamada:** resuelve los reintentos pero no el circuit breaker y dispersa la
  política por el código.
- **`pybreaker`:** válido; la implementación propia es pequeña, sin dependencias y 100 % testeable.
- **Service mesh (Envoy / Istio):** fuera de alcance para un despliegue docker-compose.

## Consecuencias

- (+) Fallos transitorios absorbidos; caídas prolongadas detectadas rápido sin saturar al proveedor.
- (+) Política configurable por entorno; resiliencia ortogonal a la lógica de negocio.
- (−) Estado del breaker en memoria por proceso (no compartido entre réplicas); aceptable, cada
  réplica aprende por sí misma. Para estado compartido se podría mover a Redis.
