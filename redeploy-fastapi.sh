#!/bin/bash
# Run this on the server (nexus1) inside
# /home/ubuntu/lms-evaluation-backend/fastapi-backend
# whenever FastAPI backend code changes need to go live.
#
# Post-cutover: FastAPI is now the sole backend, running on host port
# 5050 (Flask's lms-backend has been stopped and retired — see the
# docker-compose.yml comment for why the port mapping changed from
# 5051:5050 to 5050:5050). Flask's old redeploy.sh in the parent
# directory should no longer be used.
set -e

cd "$(dirname "$0")"

echo "==> Pulling latest code..."
git pull

echo "==> Rebuilding image (should be fast if the Docker cache is warm — dependency"
echo "    layers only re-run when requirements.txt or the Dockerfile change)..."
BUILD_START=$(date +%s)
docker compose build lms-backend-fastapi
BUILD_SECS=$(( $(date +%s) - BUILD_START ))

echo "==> Recreating container..."
docker compose up -d lms-backend-fastapi

echo "==> Verifying network attachment..."
docker network inspect lms-shared --format "Containers on lms-shared: {{range .Containers}}{{.Name}} {{end}}"

echo "==> Done. FastAPI is now live on port 5050 (sole backend — Flask has been retired)."
echo "==> Health check: curl http://localhost:5050/health"

echo "==> Build took ${BUILD_SECS}s."
if [ "$BUILD_SECS" -gt 120 ]; then
  echo "    ⚠ That's much slower than a cached build (normally under a minute)."
  echo "    Likely cause: requirements.txt/Dockerfile changed, or the Docker build"
  echo "    cache was evicted. Check disk space below — Docker auto-prunes its"
  echo "    cache when the disk gets full."
fi
echo "==> Docker disk usage (watch the 'RECLAIMABLE' build-cache row over time):"
docker system df
