#!/usr/bin/env bash
set -euxo pipefail

EC2_PATH="${EC2_PATH:-/home/ec2-user/transcription-automation}"
WORKER_IMAGE_TAG="${WORKER_IMAGE_TAG:-latest}"
WORKER_ECR_REPO="${WORKER_ECR_REPO:-transcription-worker}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
SLOT_CHOICE="${SLOT_CHOICE:-}"
TIMEZONE_INPUT="${TIMEZONE_INPUT:-}"
CLEAN_FIRST="${CLEAN_FIRST:-}"
TRIGGER_SOURCE="${TRIGGER_SOURCE:-}"
UPDATE_RUNTIME_ENV="${UPDATE_RUNTIME_ENV:-false}"

cd "$EC2_PATH"

echo "== Update code =="
git fetch --all
git reset --hard origin/main
git clean -fd || true

echo "== Ensure docker running =="
sudo systemctl enable docker || true
sudo systemctl start docker || true

echo "== Pre-clean docker state on server =="
docker compose down --volumes --rmi all --remove-orphans || true
docker system prune -a --volumes -f || true

echo "== Ensure .env exists =="
touch .env

upsert_env() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" .env; then
    sed -i "s|^${key}=.*|${key}=${value}|" .env
  else
    echo "${key}=${value}" >> .env
  fi
}

if [[ "$UPDATE_RUNTIME_ENV" == "true" ]]; then
  echo "== workflow_dispatch mode: updating runtime values in server .env =="
  upsert_env "WORKER_ECR_REPO" "${WORKER_ECR_REPO}"
  upsert_env "WORKER_IMAGE_TAG" "${WORKER_IMAGE_TAG}"
  upsert_env "SLOT_CHOICE" "${SLOT_CHOICE}"
  upsert_env "TIMEZONE_INPUT" "${TIMEZONE_INPUT}"
  upsert_env "CLEAN_FIRST" "${CLEAN_FIRST}"
  upsert_env "TRIGGER_SOURCE" "${TRIGGER_SOURCE}"
else
  echo "== push mode: keeping existing server .env unchanged =="
fi

echo "== Run detached pipeline =="
echo "STOP_ON_FAILURE=${STOP_ON_FAILURE}"
echo "UPDATE_RUNTIME_ENV=${UPDATE_RUNTIME_ENV}"
echo "SLOT_CHOICE=${SLOT_CHOICE}"
echo "TIMEZONE_INPUT=${TIMEZONE_INPUT}"
echo "CLEAN_FIRST=${CLEAN_FIRST}"
echo "TRIGGER_SOURCE=${TRIGGER_SOURCE}"

chmod +x run_detached.sh
STOP_ON_FAILURE="${STOP_ON_FAILURE}" \
SLOT_CHOICE="${SLOT_CHOICE}" \
TIMEZONE_INPUT="${TIMEZONE_INPUT}" \
CLEAN_FIRST="${CLEAN_FIRST}" \
TRIGGER_SOURCE="${TRIGGER_SOURCE}" \
UPDATE_RUNTIME_ENV="${UPDATE_RUNTIME_ENV}" \
EC2_PATH="${EC2_PATH}" \
./run_detached.sh

echo "== run_detached.sh started. Pipeline continues in background on EC2. =="