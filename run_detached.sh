#!/usr/bin/env bash
set -euo pipefail

EC2_PATH="${EC2_PATH:-/home/ec2-user/transcription-automation}"
cd "$EC2_PATH"
mkdir -p "$EC2_PATH/runs"

# pick compose command
if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
elif docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
else
  echo "ERROR: docker compose not found" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$EC2_PATH/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "$RUN_ID" > "$EC2_PATH/LAST_RUN_ID"

setsid nohup bash -lc "
  set +e
  cd '$EC2_PATH'

  echo '[INFO] ===== RUN START ====='
  echo '[INFO] RUN_ID=$RUN_ID'
  date

  echo '[INFO] Updating code...'
  git fetch --all
  git reset --hard origin/main
  git clean -fd || true

  echo '[INFO] Ensuring docker running...'
  sudo systemctl enable docker || true
  sudo systemctl start docker || true

  echo '[INFO] compose down...'
  $COMPOSE down || true

  echo '[INFO] compose up (build)...'
  $COMPOSE up -d --build

  # Track pipeline_rest logs (it runs after orchestrator)
  echo '[INFO] following orchestrator logs...'
  docker logs -f orchestrator || true

  echo '[INFO] orchestrator finished. following pipeline_rest logs...'
  docker logs -f pipeline_rest || true

  EC=\$(docker inspect -f '{{.State.ExitCode}}' pipeline_rest 2>/dev/null || echo 999)
  echo \"\$EC\" > '$RUN_DIR/exit_code.txt'
  date > '$RUN_DIR/DONE'

  echo '[INFO] compose down...'
  $COMPOSE down || true

  echo '[INFO] ===== RUN END ====='
  echo \"EXIT_CODE=\$EC\"
" > "$RUN_DIR/run.log" 2>&1 < /dev/null &

echo $! > "$RUN_DIR/pid.txt"
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$(cat "$RUN_DIR/pid.txt")"
echo "Log file: $RUN_DIR/run.log"