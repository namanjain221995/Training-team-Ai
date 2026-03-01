#!/usr/bin/env bash
set -euo pipefail

# -------- CONFIG --------
EC2_PATH="${EC2_PATH:-/home/ec2-user/transcription-automation}"
SERVICE_NAME="${SERVICE_NAME:-pipeline_rest}"   # only used if you still want to tail compose logs separately
RUN_MAIN_SCRIPT="${RUN_MAIN_SCRIPT:-main_ec2_pipeline.py}"

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

# start in background and survive SSH disconnect
setsid nohup bash -lc "
  set +e
  cd '$EC2_PATH'

  echo '[INFO] ===== RUN START ====='
  echo '[INFO] RUN_ID=$RUN_ID'
  date

  # Always pull latest (optional but recommended)
  if [ -d .git ]; then
    echo '[INFO] Updating code...'
    git fetch --all
    git reset --hard origin/main
    git clean -fd || true
  fi

  echo '[INFO] Ensuring docker is running...'
  sudo systemctl enable docker || true
  sudo systemctl start docker || true

  # Clean any previous compose stack (safe)
  echo '[INFO] compose down...'
  $COMPOSE down || true

  # Run main orchestrator (workers + logs + wait + compose + shutdown)
  if [ ! -f '$RUN_MAIN_SCRIPT' ]; then
    echo '[ERROR] $RUN_MAIN_SCRIPT not found in $EC2_PATH'
    echo '999' > '$RUN_DIR/exit_code.txt'
    date > '$RUN_DIR/DONE'
    exit 0
  fi

  echo '[INFO] Running main orchestrator: python3 $RUN_MAIN_SCRIPT'
  python3 '$RUN_MAIN_SCRIPT'
  EC=\$?

  echo \"\$EC\" > '$RUN_DIR/exit_code.txt'
  date > '$RUN_DIR/DONE'

  echo '[INFO] ===== RUN END (exit code) ====='
  echo \"\$EC\"

  # If main_ec2_pipeline.py triggers shutdown, code below may not execute.
  # But keep cleanup best-effort.
  echo '[INFO] compose down (best-effort)...'
  $COMPOSE down || true

  # Write a placeholder container_id.txt for compatibility with older monitor scripts
  echo 'main-ec2-pipeline' > '$RUN_DIR/container_id.txt' || true

" > "$RUN_DIR/run.log" 2>&1 < /dev/null &

echo $! > "$RUN_DIR/pid.txt"

echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$(cat "$RUN_DIR/pid.txt")"
echo "Log file: $RUN_DIR/run.log"