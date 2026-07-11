#!/usr/bin/env bash
# Entrypoint for the single-container HF Space demo: uvicorn (:8000) and the
# Next standalone server (:3000) run in the background; nginx (:7860) is
# exec'd as PID 1 so container stop signals reach it directly and take the
# whole container down.
set -euo pipefail

uvicorn interpretability_backend.backend.main:app --host 127.0.0.1 --port 8000 &

(cd /app/frontend && PORT=3000 HOSTNAME=127.0.0.1 NODE_ENV=production exec node server.js) &

exec nginx -c /etc/nginx/nginx.conf -g 'daemon off;'
