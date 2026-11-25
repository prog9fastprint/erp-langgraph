#!/usr/bin/env bash
set -e

APP_DIR="${1:-/www/app/django/erp/staging/erp-langgraph}"
BRANCH="${2:-master}"

echo ">>> Deploying FastPrint ERP AI Orchestrator at ${APP_DIR} on branch ${BRANCH}..."

cd "${APP_DIR}"

if [ "${SKIP_GIT_SYNC:-0}" != "1" ]; then
  git fetch origin "${BRANCH}"
  git reset --hard "origin/${BRANCH}"
fi

if [ ! -f .env ]; then
  echo "WARNING: .env file not found in ${APP_DIR}. Creating from .env.example..."
  cp .env.example .env
  echo "Please configure ${APP_DIR}/.env before starting the service."
fi

# Determine Docker Compose command
if command -v docker &> /dev/null && (docker compose version &> /dev/null || command -v docker-compose &> /dev/null); then
  DOCKER_CMD="docker compose"
  if ! docker compose version &> /dev/null; then
    DOCKER_CMD="docker-compose"
  fi

  echo ">>> Rebuilding and restarting container using ${DOCKER_CMD}..."
  ${DOCKER_CMD} down --remove-orphans || true
  ${DOCKER_CMD} build --pull
  ${DOCKER_CMD} up -d
  echo ">>> Checking container status..."
  ${DOCKER_CMD} ps
  HEALTH_OK=0
  for i in $(seq 1 10); do
    if curl -fsS http://127.0.0.1:8002/health > /dev/null; then
      HEALTH_OK=1
      break
    fi
    echo "Waiting for service health check (${i}/10)..."
    sleep 2
  done
  if [ "${HEALTH_OK}" -ne 1 ]; then
    echo "ERROR: Service failed /health check after deployment."
    ${DOCKER_CMD} logs --tail=50
    exit 1
  fi
  echo ">>> Health check passed."
elif [ -f /etc/systemd/system/erp-langgraph.service ]; then
  echo ">>> Native systemd detected. Updating dependencies and restarting..."
  if [ -d .venv ]; then
    ./.venv/bin/pip install --no-cache-dir -r requirements.txt
  fi
  sudo systemctl restart erp-langgraph.service
  sudo systemctl status erp-langgraph.service --no-pager
else
  echo "ERROR: Neither Docker Compose nor erp-langgraph.service found."
  exit 1
fi

echo ">>> Deployment completed."
