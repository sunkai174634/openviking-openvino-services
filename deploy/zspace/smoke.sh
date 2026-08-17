#!/usr/bin/env sh
# Post-deployment smoke: health + version + a real request + log API on both services.
set -eu
embedding_url=${EMBEDDING_URL:-http://127.0.0.1:33038}
intent_url=${INTENT_URL:-http://127.0.0.1:33039}

echo '== embedding health =='
curl -fsS "$embedding_url/health" | python3 -m json.tool | head -8

echo '== embedding real request + X-Request-ID echo =='
curl -fsS -D /tmp/smoke_h -H 'X-Request-ID: smoke-emb' -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-0.6b-openvino-int8","input":"OpenViking embedding smoke test"}' \
  "$embedding_url/v1/embeddings" \
  | python3 -c 'import json,sys; o=json.load(sys.stdin); print({"model": o["model"], "count": len(o["data"]), "dim": len(o["data"][0]["embedding"]), "input_truncated": o["meta"]["input_truncated"]})'
grep -i x-request-id /tmp/smoke_h || echo 'WARN: no X-Request-ID header'

echo '== embedding /v1/logs =='
curl -fsS "$embedding_url/v1/logs?limit=3" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("service:", d["service"], "version:", d["version"], "entries:", len(d["entries"]))'

echo '== intent health =='
curl -fsS "$intent_url/health" | python3 -m json.tool | head -8

echo '== intent real request =='
curl -fsS --max-time 120 -H 'Content-Type: application/json' \
  -d '{"model":"guoxuter/ov_intent_analysis_sft:v7_q8","input":"OpenViking /ready 503 那次怎么恢复的？"}' \
  "$intent_url/v1/intent" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print({"queries": len(d["plan"]["queries"]), "truncated": d["truncated"]})'

echo '== intent /v1/logs =='
curl -fsS "$intent_url/v1/logs?limit=3" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("service:", d["service"], "version:", d["version"], "entries:", len(d["entries"]))'

rm -f /tmp/smoke_h
echo 'SMOKE OK'
