# ADR 0006 – Prompting defensivo contra alucinaciones y *prompt injection*

- **Estado:** Aceptada
- **Fecha:** 2026-10-31

## Contexto

En documentación técnica una respuesta inventada (un par de apriete, una tensión) es peor que no
responder. El reto pide explícitamente evitar alucinaciones y que el sistema diga "No tengo información
suficiente" cuando corresponda. Además, el contenido del PDF no es de confianza: puede contener texto
que intente dar instrucciones al modelo.

## Decisión

Defensa en varias capas (`app/application/prompts.py`, `app/application/query.py`):

1. **Guardia pre-LLM:** si la recuperación no devuelve chunks (o todos están por debajo de
   `RETRIEVAL_MIN_SCORE`) **no se llama al LLM**; se devuelve una respuesta fija de "no tengo
   información suficiente" con `has_sufficient_context=false`. Cero coste y cero alucinación.
2. **System prompt estricto (`SYSTEM_PROMPT`):** responder *solo* con los bloques de contexto
   numerados `[1]..[n]`; citar cada afirmación con `[n]`; no usar conocimiento externo; responder en el
   idioma del usuario; copiar cifras y unidades literalmente.
3. **Contexto como datos:** `build_context_block` delimita claramente cada chunk (con documento y
   página) y el prompt indica que cualquier instrucción dentro del contexto debe ignorarse (mitigación
   de *prompt injection* indirecta).
4. **Marcador de contexto insuficiente:** si el contexto no responde a la pregunta, el modelo debe
   responder con `[[NO_CONTEXT]]` (`INSUFFICIENT_CONTEXT_MARKER`) y un mensaje explícito. El backend
   detecta el marcador, lo elimina, pone `has_sufficient_context=false` y no adjunta imágenes.
5. **Generación determinista:** `LLM_TEMPERATURE=0.0` y `LLM_MAX_TOKENS` acotado.
6. **Citas verificables:** las fuentes devueltas son los chunks efectivamente citados (`[n]`), con
   documento, página y `bbox`, que la UI muestra como "Fuente: Manual_Motor.pdf, Página 4".
7. **Historial acotado** (`HISTORY_MAX_TURNS`) para que la conversación previa no desplace el contexto.

## Alternativas consideradas

- **Verificación posterior con un segundo LLM (*LLM-as-judge*) o NLI:** más robusto pero duplica coste
  y latencia; queda como mejora.
- **Umbral de score como único filtro:** los scores RRF no están calibrados; insuficiente por sí solo.
- **Salida estructurada (JSON con `answer` / `citations`):** más fácil de parsear, pero empeora el
  Markdown en modelos pequeños de Ollama; se optó por marcador + citas `[n]`.

## Consecuencias

- (+) Respuestas trazables; el sistema prefiere abstenerse a inventar.
- (−) Puede abstenerse de más cuando la información está repartida en chunks no recuperados
  (mitigado con `RETRIEVAL_TOP_K` y chunks con contexto de sección).
- (−) Los modelos locales pequeños siguen peor las instrucciones; la guardia pre-LLM y el marcador
  limitan el daño.
