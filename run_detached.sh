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

# If 1 -> stop EC2 even if pipeline_rest fails
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

# Runtime trigger values
SLOT_CHOICE="${SLOT_CHOICE:-}"
TIMEZONE_INPUT="${TIMEZONE_INPUT:-America/New_York}"
CLEAN_FIRST="${CLEAN_FIRST:-false}"
TRIGGER_SOURCE="${TRIGGER_SOURCE:-manual}"

setsid nohup bash -lc "
  set +e
  cd '$EC2_PATH'

  upsert_env() {
    local key=\"\$1\"
    local value=\"\$2\"
    if grep -q \"^\${key}=\" .env; then
      sed -i \"s|^\${key}=.*|\${key}=\${value}|\" .env
    else
      echo \"\${key}=\${value}\" >> .env
    fi
  }

  echo '[INFO] ===== RUN START ====='
  echo '[INFO] RUN_ID=$RUN_ID'
  echo '[INFO] STOP_ON_FAILURE=$STOP_ON_FAILURE'
  echo '[INFO] SLOT_CHOICE=$SLOT_CHOICE'
  echo '[INFO] TIMEZONE_INPUT=$TIMEZONE_INPUT'
  echo '[INFO] CLEAN_FIRST=$CLEAN_FIRST'
  echo '[INFO] TRIGGER_SOURCE=$TRIGGER_SOURCE'
  date

  echo '[INFO] Fetch latest code from repo...'
  git fetch --all
  git pull origin main

  echo '[INFO] Ensuring docker running...'
  sudo systemctl enable docker || true
  sudo systemctl start docker || true

  echo '[INFO] Updating .env runtime values...'
  touch .env
  upsert_env 'SLOT_CHOICE' '$SLOT_CHOICE'
  upsert_env 'TIMEZONE_INPUT' '$TIMEZONE_INPUT'
  upsert_env 'CLEAN_FIRST' '$CLEAN_FIRST'
  upsert_env 'TRIGGER_SOURCE' '$TRIGGER_SOURCE'

  echo '[INFO] compose down before start...'
  $COMPOSE down --volumes --rmi all --remove-orphans || true

  echo '[INFO] docker cleanup before start...'
  docker system prune -a --volumes -f || true

  echo '[INFO] compose up (build)...'
  $COMPOSE up -d --build
  RC_UP=\$?
  echo \"[INFO] compose up exit code=\$RC_UP\"

  if [[ \"\$RC_UP\" != \"0\" ]]; then
    echo \"\$RC_UP\" > '$RUN_DIR/exit_code.txt'
    date > '$RUN_DIR/DONE'
    echo '[INFO] compose up failed. Exiting background runner.'
    exit \$RC_UP
  fi

  echo '[INFO] compose ps after startup...'
  $COMPOSE ps || true

  echo '[INFO] waiting 10 seconds for stabilization...'
  sleep 10

  echo '[INFO] compose ps after stabilization...'
  $COMPOSE ps || true

  echo '[INFO] following orchestrator logs...'
  docker logs -f orchestrator || true

  echo '[INFO] orchestrator finished. following pipeline_rest logs...'
  docker logs -f pipeline_rest || true

  EC=\$(docker inspect -f '{{.State.ExitCode}}' pipeline_rest 2>/dev/null || echo 999)
  echo \"\$EC\" > '$RUN_DIR/exit_code.txt'
  date > '$RUN_DIR/DONE'

  echo '[INFO] compose down (stops log_viewer too)...'
  $COMPOSE down || true

  echo '[INFO] ===== RUN END ====='
  echo \"EXIT_CODE=\$EC\"

  if [[ \"\$EC\" != \"0\" && \"$STOP_ON_FAILURE\" != \"1\" ]]; then
    echo '[INFO] pipeline_rest failed -> NOT stopping EC2 (STOP_ON_FAILURE=0)'
    exit 0
  fi

  echo '[INFO] Stopping EC2 instance (NOT terminate)...'
  TOKEN=\$(curl -sX PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' || true)
  imds() {
    local p=\"\$1\"
    if [[ -n \"\$TOKEN\" ]]; then
      curl -s -H \"X-aws-ec2-metadata-token: \$TOKEN\" \"http://169.254.169.254/latest/\$p\"
    else
      curl -s \"http://169.254.169.254/latest/\$p\"
    fi
  }

  REGION=\$(imds meta-data/placement/region)
  IID=\$(imds meta-data/instance-id)

  echo \"[INFO] REGION=\$REGION IID=\$IID\"
  aws ec2 stop-instances --region \"\$REGION\" --instance-ids \"\$IID\"
  echo '[INFO] Stop request sent.'
" > "$RUN_DIR/run.log" 2>&1 < /dev/null &

echo $! > "$RUN_DIR/pid.txt"
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$(cat "$RUN_DIR/pid.txt")"
echo "Log file: $RUN_DIR/run.log"