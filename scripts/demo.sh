#!/usr/bin/env bash
# End-to-end demo against a running stack (`docker compose up`).
#
# Submits a benchmark, polls until it finishes, prints the results, and
# demonstrates idempotent replay and the 409-before-completion contract along
# the way. Needs nothing but bash, curl and python3 (for pretty-printing).
#
#   ./scripts/demo.sh
#   BASE_URL=http://localhost:8000 API_KEY=local-development-key ./scripts/demo.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-local-development-key}"
MODEL="${MODEL:-fmnist-mlp-defender-0}"
ATTACKS="${ATTACKS:-[\"fgsm\", \"pgd\"]}"
EPSILON="${EPSILON:-0.1}"
ITERATIONS="${ITERATIONS:-20}"

auth=(-H "X-API-Key: ${API_KEY}")
json=(-H "Content-Type: application/json")

pretty() { python3 -m json.tool 2>/dev/null || cat; }
step()   { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

step "Waiting for ${BASE_URL}/healthz"
for _ in $(seq 1 60); do
    if curl -fsS "${BASE_URL}/healthz" >/dev/null 2>&1; then break; fi
    sleep 2
done
curl -fsS "${BASE_URL}/healthz" | pretty

step "Available models"
curl -fsS "${auth[@]}" "${BASE_URL}/v1/models" \
    | python3 -c 'import json,sys; [print(" ", m["name"], "(available)" if m["available"] else "(MISSING WEIGHTS)") for m in json.load(sys.stdin)["items"]]'

step "Available attacks"
curl -fsS "${auth[@]}" "${BASE_URL}/v1/attacks" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(" ", a["name"].ljust(10), a["description"]) for a in d["items"]]; print("\n  limits:", d["constraints"])'

# A fresh key per run, so re-running the demo submits a new job rather than
# replaying the previous one.
IDEMPOTENCY_KEY="$(python3 -c 'import uuid; print(uuid.uuid4())')"
BODY="{\"model_name\": \"${MODEL}\", \"attacks\": ${ATTACKS}, \"epsilon\": ${EPSILON}, \"max_iterations\": ${ITERATIONS}}"

step "Submitting a benchmark (expect 202 Accepted)"
response="$(curl -fsS -w '\n%{http_code}' -X POST "${BASE_URL}/v1/jobs" \
    "${auth[@]}" "${json[@]}" -H "Idempotency-Key: ${IDEMPOTENCY_KEY}" -d "${BODY}")"
status="$(tail -n1 <<<"${response}")"
payload="$(sed '$d' <<<"${response}")"
echo "HTTP ${status}"
echo "${payload}" | pretty
JOB_ID="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["job_id"])' "${payload}")"

step "Replaying the same Idempotency-Key (expect 200, same job id, no second job)"
replay="$(curl -fsS -w '\n%{http_code}' -X POST "${BASE_URL}/v1/jobs" \
    "${auth[@]}" "${json[@]}" -H "Idempotency-Key: ${IDEMPOTENCY_KEY}" -d "${BODY}")"
echo "HTTP $(tail -n1 <<<"${replay}")"
python3 -c '
import json, sys
original, replayed = sys.argv[1], json.loads(sys.argv[2])["job_id"]
print(f"  original: {original}\n  replayed: {replayed}\n  same job: {original == replayed}")
' "${JOB_ID}" "$(sed '$d' <<<"${replay}")"

step "Asking for results too early (expect 409 Conflict)"
curl -sS -o /tmp/demo-early.json -w '  HTTP %{http_code}\n' \
    "${auth[@]}" "${BASE_URL}/v1/jobs/${JOB_ID}/results" || true
pretty < /tmp/demo-early.json

step "Polling until the worker finishes"
for _ in $(seq 1 150); do
    state="$(curl -fsS "${auth[@]}" "${BASE_URL}/v1/jobs/${JOB_ID}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
    printf '  status: %s\n' "${state}"
    case "${state}" in
        succeeded|failed|cancelled) break ;;
    esac
    sleep 2
done

step "Job"
curl -fsS "${auth[@]}" "${BASE_URL}/v1/jobs/${JOB_ID}" | pretty

if [ "${state}" = "succeeded" ]; then
    step "Results"
    curl -fsS "${auth[@]}" "${BASE_URL}/v1/jobs/${JOB_ID}/results" | pretty
else
    step "Job did not succeed - the error is recorded on the job above"
    exit 1
fi

step "Job listing (newest first, cursor-paginated)"
curl -fsS "${auth[@]}" "${BASE_URL}/v1/jobs?limit=5" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(" ", j["job_id"], j["status"]) for j in d["items"]]; print("  next_cursor:", d["next_cursor"])'
