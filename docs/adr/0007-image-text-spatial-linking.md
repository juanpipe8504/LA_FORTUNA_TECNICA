# ADR 0007 – Vinculación imagen ↔ texto por proximidad espacial y referencias "Figura N"

- **Estado:** Aceptada
- **Fecha:** 2026-10-31

## Contexto

El reto pide detectar qué imágenes están asociadas a los fragmentos recuperados y mostrarlas junto a la
respuesta. En un manual, el diagrama relevante suele estar en la misma página, cerca del párrafo que lo
explica, o se referencia explícitamente ("ver Figura 2").

## Decisión

- El parser extrae cada imagen con su `BoundingBox`, la persiste vía `BlobStoragePort`
  (`{images_dir}/{document_id}/{image_id}.png`, `image_id = {document_id}_p{page}_i{index}`) y detecta
  su pie de figura (bloque de texto cercano que empieza por "Figura N" / "Figure N" / "Fig. N").
- Opcionalmente (`DESCRIBE_IMAGES=true` y LLM con visión) se genera una descripción técnica de la
  imagen con `LLMPort.describe_image`, que enriquece el pie.
- En el chunking, cada chunk recibe `images: list[ImageRef]`: imágenes de la **misma página** cuya
  distancia mínima entre rectángulos (`BoundingBox.distance_to`) sea ≤ `IMAGE_LINK_MAX_DISTANCE`
  (150 pt), ordenadas por distancia. Si el texto del chunk menciona "Figura N" y existe esa figura,
  el vínculo se **fuerza** con distancia 0.
- Además, cada imagen genera su propio chunk `IMAGE` (pie + descripción + sección), recuperable por la
  búsqueda híbrida.
- En la consulta se agregan los `ImageRef` de los chunks recuperados con
  `relevance = score_chunk / (1 + distancia/100)`, se deduplican y se devuelven las
  `MAX_IMAGES_PER_ANSWER` mejores, con URL `/api/v1/documents/{document_id}/images/{image_id}`.
  Si la respuesta no tiene contexto suficiente no se muestran imágenes.

## Alternativas consideradas

- **Embeddings multimodales (CLIP / ColPali) indexando imágenes como vectores:** potente para
  búsquedas visuales, pero CLIP entiende mal los diagramas técnicos y añade otro espacio vectorial; la
  descripción textual vía LLM de visión se integra en la búsqueda híbrida existente.
- **Asociar todas las imágenes de la página:** ruidoso en páginas con varias figuras.
- **Enviar las imágenes al LLM en cada respuesta:** caro y lento; la visión se reserva para la ingesta.

## Consecuencias

- (+) Vínculos explicables (distancia en puntos, referencia explícita) y baratos de calcular.
- (+) Las imágenes también son recuperables por su contenido descrito.
- (−) Los diagramas **vectoriales** (dibujados con primitivas PDF, no raster, como la Figura 2 del
  manual de ejemplo) requieren rasterizar la región de dibujos para tratarlos como imagen.
- (−) Figuras en otra página distinta a su texto solo se vinculan si se referencian por número.
