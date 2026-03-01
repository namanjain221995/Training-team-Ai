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

# service to track as the "main pipeline"
SERVICE_NAME="${SERVICE_NAME:-pipeline_rest}"

# start in background and survive SSH disconnect
setsid nohup bash -lc "
  set +e
  cd '$EC2_PATH'

  echo '[INFO] compose down...'
  $COMPOSE down || true

  echo '[INFO] compose up -d --build...'
  $COMPOSE up -d --build

  # wait for service container id
  CID=''
  for i in \$(seq 1 60); do
    CID=\$($COMPOSE ps -q '$SERVICE_NAME' 2>/dev/null | head -n 1 || true)
    if [ -n \"\$CID\" ]; then break; fi
    sleep 2
  done

  if [ -z \"\$CID\" ]; then
    echo '[ERROR] Could not find container for service: $SERVICE_NAME'
    echo '999' > '$RUN_DIR/exit_code.txt'
    date > '$RUN_DIR/DONE'
    $COMPOSE ps > '$RUN_DIR/compose_ps.txt' 2>&1 || true
    $COMPOSE logs > '$RUN_DIR/compose_logs.txt' 2>&1 || true
    $COMPOSE down || true
    echo '[INFO] Stopping EC2 now (no container found => fail fast)...'
    sudo shutdown -h now
    exit 0
  fi

  echo \"\$CID\" > '$RUN_DIR/container_id.txt'
  echo \"CONTAINER_ID=\$CID\"

  echo '[INFO] Following container logs...'
  docker logs -f \"\$CID\" 2>&1 | tee -a '$RUN_DIR/container_logs.txt'

  EC=\$(docker inspect -f '{{.State.ExitCode}}' \"\$CID\" 2>/dev/null || echo 999)
  echo \"\$EC\" > '$RUN_DIR/exit_code.txt'
  date > '$RUN_DIR/DONE'

  echo '[INFO] compose down...'
  $COMPOSE down || true

  echo '[INFO] Finished. Stopping EC2 now...'
  sudo shutdown -h now
" > "$RUN_DIR/run.log" 2>&1 < /dev/null &

echo $! > "$RUN_DIR/pid.txt"

echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$(cat "$RUN_DIR/pid.txt")"
echo "Log file: $RUN_DIR/run.log"