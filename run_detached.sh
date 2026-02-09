#!/usr/bin/env bash
set -euo pipefail

EC2_PATH="/home/ec2-user/transcription-automation"
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

  # start fresh
  $COMPOSE down || true

  # start detached
  $COMPOSE up -d --build
  CID=\$($COMPOSE ps -q test_runner | head -n 1)
  echo \"\$CID\" > '$RUN_DIR/container_id.txt'
  echo \"CONTAINER_ID=\$CID\"

  # follow logs until container exits
  docker logs -f \"\$CID\"
  EC=\$(docker inspect -f '{{.State.ExitCode}}' \"\$CID\" 2>/dev/null || echo 999)

  echo \"\$EC\" > '$RUN_DIR/exit_code.txt'
  date > '$RUN_DIR/DONE'

  # cleanup compose network/containers
  $COMPOSE down || true

  # stop the EC2 instance (OPTION A)
  sudo shutdown -h now
" > "$RUN_DIR/run.log" 2>&1 < /dev/null &

echo $! > "$RUN_DIR/pid.txt"

echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$(cat "$RUN_DIR/pid.txt")"
echo "Log file: $RUN_DIR/run.log"
