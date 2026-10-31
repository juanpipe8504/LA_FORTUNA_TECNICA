#!/usr/bin/env bash
# End-to-end demo: upload a PDF, poll the ingestion job, ask a question.
# Requirements: bash, curl, python3 (only for JSON parsing; no jq needed).
#
#   bash scripts/demo.sh [PDF] ["question"]
#   API_URL=http://localhost:8000/api/v1 bash scripts/demo.sh
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000/api/v1}"
PDF="${1:-docs/sample/Manual_Motor.pdf}"
QUESTION="${2:-¿Cada cuántas horas se deben relubricar los rodamientos y qué cantidad de grasa se usa?}"
PY="$(command -v python3 || command -v python)"
TIMEOUT_S="${TIMEOUT_S:-300}"

json_get() {  # json_get '<json>' key
  "$PY" -c 'import json,sys; v=json.loads(sys.argv[1]).get(sys.argv[2]); print("" if v is None else v)' "$1" "$2"
}

echo "==> Health: $(curl -fsS "${API_URL%/api/v1}/health")"

echo "==> Uploading ${PDF}"
UPLOAD=$(curl -fsS -X POST -F "file=@${PDF};type=application/pdf" "${API_URL}/documents")
echo "    ${UPLOAD}"
JOB_ID=$(json_get "$UPLOAD" job_id)
DOCUMENT_ID=$(json_get "$UPLOAD" document_id)

echo "==> Polling job ${JOB_ID}"
START=$(date +%s)
while true; do
  JOB=$(curl -fsS "${API_URL}/jobs/${JOB_ID}")
  STATUS=$(json_get "$JOB" status)
  printf '    status=%-10s progress=%3s%%  %s\n' "$STATUS" "$(json_get "$JOB" progress)" "$(json_get "$JOB" message)"
  case "$STATUS" in
    completed) break ;;
    failed) echo "Ingestion failed: $(json_get "$JOB" error)" >&2; exit 1 ;;
  esac
  if (( $(date +%s) - START > TIMEOUT_S )); then echo "Timeout waiting for job" >&2; exit 1; fi
  sleep 2
done
echo "    pages=$(json_get "$JOB" page_count) chunks=$(json_get "$JOB" chunk_count) images=$(json_get "$JOB" image_count)"

echo "==> Question: ${QUESTION}"
BODY=$("$PY" -c 'import json,sys; print(json.dumps({"question": sys.argv[1], "history": [], "document_ids": [sys.argv[2]]}))' \
  "$QUESTION" "$DOCUMENT_ID")
ANSWER=$(curl -fsS -X POST -H "Content-Type: application/json" --data-binary "$BODY" "${API_URL}/chat")

"$PY" - "$ANSWER" <<'PYEOF'
import json, sys
a = json.loads(sys.argv[1])
print("\n--- Respuesta (has_sufficient_context=%s) ---\n" % a.get("has_sufficient_context"))
print(a.get("answer", ""))
print("\n--- Fuentes ---")
for s in a.get("sources", []):
    print(f"  Fuente: {s['document_name']}, Página {s['page']}  (score={s['score']:.3f})")
print("\n--- Imágenes ---")
for i in a.get("images", []):
    print(f"  {i['caption'] or i['image_id']}  (Página {i['page']})  -> {i['url']}")
PYEOF
