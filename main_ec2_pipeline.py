import os
import time
import base64
import random
from pathlib import Path
from typing import List, Dict, Optional

import boto3
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

load_dotenv()

# -----------------------------
# CONFIG (orchestrator container)
# -----------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

LAUNCH_TEMPLATE_ID = os.getenv("WORKER_LAUNCH_TEMPLATE_ID", "").strip()
LAUNCH_TEMPLATE_VERSION = os.getenv("WORKER_LAUNCH_TEMPLATE_VERSION", "$Default")

ROOT_2026_FOLDER_NAME = os.getenv("ROOT_2026_FOLDER_NAME", "2026")
SLOT_CHOICE = os.getenv("SLOT_CHOICE", "").strip()  # 1-based index
USE_SHARED_DRIVES = os.getenv("USE_SHARED_DRIVES", "0").strip().lower() in ("1", "true", "yes", "y")
USE_SHARED_DRIVES_STR = "1" if USE_SHARED_DRIVES else "0"

TOKEN_FILE = Path(os.getenv("TOKEN_FILE", "token.json"))
TOKEN_FILE_WRITE = Path(os.getenv("TOKEN_FILE_WRITE", "/app/state/token_refreshed.json"))

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
TRANSCRIPT_SUFFIX = "_transcripts.txt"

FOLDER_NAMES_TO_PROCESS = [
    "3. Introduction Video",
    "4. Mock Interview (First Call)",
    "5. Project Scenarios",
    "6. 30 Questions Related to Their Niche",
    "7. 50 Questions Related to the Resume",
    "8. Tools & Technology Videos",
    "9. System Design Video (with Draw.io)",
    "10. Persona Video",
    "11. Small Talk",
    "12. JD Video",
]

# Worker limits
MAX_LAUNCH = int(os.getenv("MAX_LAUNCH", "0"))  # 0 = launch all
LAUNCH_SLEEP_SECONDS = float(os.getenv("LAUNCH_SLEEP_SECONDS", "0.25"))

# Wait behavior
WAIT_POLL_SECONDS = int(os.getenv("WAIT_POLL_SECONDS", "15"))

# Unique RunId tag for this run
RUN_ID = os.getenv("RUN_ID", "") or time.strftime("%Y%m%d_%H%M%S")

# Drive flakiness (500 Internal Error) -> retries
GOOGLE_EXECUTE_RETRIES = int(os.getenv("GOOGLE_EXECUTE_RETRIES", "8"))
GOOGLE_PAGE_SIZE = int(os.getenv("GOOGLE_PAGE_SIZE", "500"))

# Worker SSM parameter names (used inside container)
SSM_ENV_PARAM = os.getenv("SSM_ENV_PARAM", "/transcription/worker/env")
SSM_CREDENTIALS_PARAM = os.getenv("SSM_CREDENTIALS_PARAM", "/transcription/worker/credentials_json")
SSM_TOKEN_PARAM = os.getenv("SSM_TOKEN_PARAM", "/transcription/worker/token_json")

# CloudWatch group for docker logs (awslogs driver)
WORKER_LOG_GROUP = os.getenv("WORKER_LOG_GROUP", "/transcription/workers")

# Worker image in ECR
WORKER_ECR_REPO = os.getenv("WORKER_ECR_REPO", "transcription-worker").strip()
WORKER_IMAGE_TAG = os.getenv("WORKER_IMAGE_TAG", "latest").strip()

# Worker whisper defaults (stable for CPU instances)
WORKER_WHISPER_MODEL = os.getenv("WORKER_WHISPER_MODEL", "Large").strip()
WORKER_DEVICE = os.getenv("WORKER_DEVICE", "cpu").strip()
WORKER_COMPUTE_TYPE = os.getenv("WORKER_COMPUTE_TYPE", "int8").strip()
WORKER_LANGUAGE = os.getenv("WORKER_LANGUAGE", "en").strip()
WORKER_USE_VAD = os.getenv("WORKER_USE_VAD", "true").strip().lower()
WORKER_CHUNK_SECONDS = os.getenv("WORKER_CHUNK_SECONDS", "540").strip()
WORKER_OVERLAP_SECONDS = os.getenv("WORKER_OVERLAP_SECONDS", "2").strip()

# Worker termination model:
# IMPORTANT: Set Launch Template "Shutdown behavior" = Terminate
USE_SHUTDOWN_TERMINATE = os.getenv("USE_SHUTDOWN_TERMINATE", "1").strip().lower() in ("1", "true", "yes", "y")

# -----------------------------
# AWS clients
# -----------------------------
ec2 = boto3.client("ec2", region_name=AWS_REGION)
sts = boto3.client("sts", region_name=AWS_REGION)
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "").strip() or sts.get_caller_identity()["Account"]

# -----------------------------
# Drive helpers
# -----------------------------
def _list_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True, "includeItemsFromAllDrives": True, "corpora": "allDrives"}
    return {}

def build_drive():
    if not TOKEN_FILE.exists():
        raise RuntimeError(f"token.json not found at {TOKEN_FILE}. Mount it into orchestrator container.")

    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        try:
            TOKEN_FILE_WRITE.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_FILE_WRITE.write_text(creds.to_json(), encoding="utf-8")
            print(f"[AUTH] Token refreshed; wrote updated token to {TOKEN_FILE_WRITE} (original token.json is read-only).")
        except Exception as e:
            print(f"[AUTH] Token refreshed but could not write refreshed token: {e}")

    return build("drive", "v3", credentials=creds)

def drive_list_children(drive, parent_id: str, mime_type: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed=false"]
    if mime_type:
        q_parts.append(f"mimeType='{mime_type}'")
    q = " and ".join(q_parts)

    token = None
    while True:
        req = drive.files().list(
            q=q,
            fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
            pageSize=GOOGLE_PAGE_SIZE,
            pageToken=token,
            **_list_kwargs(),
        )
        res = req.execute(num_retries=GOOGLE_EXECUTE_RETRIES)
        yield from res.get("files", [])
        token = res.get("nextPageToken")
        if not token:
            break

def drive_find_child(drive, parent_id: str, name: str, mime_type: Optional[str] = None):
    for f in drive_list_children(drive, parent_id, mime_type):
        if f["name"] == name:
            return f
    return None

def drive_search_folder_anywhere(drive, folder_name: str) -> List[dict]:
    q = f"name='{folder_name}' and mimeType='{FOLDER_MIME}' and trashed=false"
    out: List[dict] = []
    token = None
    while True:
        req = drive.files().list(
            q=q,
            fields="nextPageToken, files(id,name,parents,modifiedTime)",
            pageSize=GOOGLE_PAGE_SIZE,
            pageToken=token,
            **_list_kwargs(),
        )
        res = req.execute(num_retries=GOOGLE_EXECUTE_RETRIES)
        out.extend(res.get("files", []))
        token = res.get("nextPageToken")
        if not token:
            break
    return out

def is_video(name: str) -> bool:
    p = Path(name)
    if p.name.startswith("."):
        return False
    if p.suffix.lower() not in VIDEO_EXTS:
        return False
    if "__EYE_" in p.name:
        return False
    return True

def transcript_name(video_name: str) -> str:
    return f"{Path(video_name).stem}{TRANSCRIPT_SUFFIX}"

def pick_slot(drive, root_id: str) -> dict:
    slots = sorted(list(drive_list_children(drive, root_id, FOLDER_MIME)), key=lambda x: x["name"].lower())
    if not slots:
        raise RuntimeError("No slot folders found under 2026.")

    if SLOT_CHOICE.isdigit():
        idx = int(SLOT_CHOICE)
        if not (1 <= idx <= len(slots)):
            raise RuntimeError(f"SLOT_CHOICE out of range (1..{len(slots)})")
        return slots[idx - 1]

    return slots[0]

def find_tasks(drive) -> List[Dict]:
    roots = drive_search_folder_anywhere(drive, ROOT_2026_FOLDER_NAME)
    if not roots:
        raise RuntimeError(f"Drive folder '{ROOT_2026_FOLDER_NAME}' not found.")

    roots.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    root_id = roots[0]["id"]

    slot = pick_slot(drive, root_id)
    people = sorted(list(drive_list_children(drive, slot["id"], FOLDER_MIME)), key=lambda x: x["name"])

    tasks: List[Dict] = []
    for person in people:
        for folder_name in FOLDER_NAMES_TO_PROCESS:
            folder = drive_find_child(drive, person["id"], folder_name, FOLDER_MIME)
            if not folder:
                continue
            folder_id = folder["id"]

            children = list(drive_list_children(drive, folder_id, None))
            videos = [f for f in children if f["mimeType"] != FOLDER_MIME and is_video(f["name"])]

            for vid in videos:
                out = transcript_name(vid["name"])
                if drive_find_child(drive, folder_id, out, None):
                    continue

                tasks.append({
                    "video_file_id": vid["id"],
                    "target_folder_id": folder_id,
                    "video_name": vid["name"],
                    "slot": slot["name"],
                    "person": person["name"],
                    "folder": folder_name,
                })

    return tasks

def _sanitize_stream_name(s: str) -> str:
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.strip()
    if len(s) > 200:
        s = s[:200]
    return s or "video"

# -----------------------------
# WORKER USERDATA (Docker pull + docker run)
# -----------------------------
WORKER_USERDATA_TEMPLATE = r"""#!/bin/bash
set -euo pipefail

export VIDEO_FILE_ID="{VIDEO_FILE_ID}"
export TARGET_FOLDER_ID="{TARGET_FOLDER_ID}"
export VIDEO_NAME="{VIDEO_NAME}"

exec > >(tee -a /var/log/worker-userdata.log) 2>&1
echo "[BOOT] Worker starting..."
date

# Always shutdown on exit (prevents stuck workers)
{EXIT_TRAP}

# IMDSv2
TOKEN="$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)"
imds() {{
  local p="$1"
  if [[ -n "$TOKEN" ]]; then
    curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/$p"
  else
    curl -s "http://169.254.169.254/latest/$p"
  fi
}}

REGION="$(imds meta-data/placement/region)"
INSTANCE_ID="$(imds meta-data/instance-id)"
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"
echo "[BOOT] REGION=$REGION INSTANCE_ID=$INSTANCE_ID"

dnf update -y
dnf install -y docker awscli
systemctl enable docker
systemctl start docker

ACCOUNT_ID="{AWS_ACCOUNT_ID}"
REPO="{WORKER_ECR_REPO}"
TAG="{WORKER_IMAGE_TAG}"
IMAGE_URI="${{ACCOUNT_ID}}.dkr.ecr.${{REGION}}.amazonaws.com/${{REPO}}:${{TAG}}"
echo "[BOOT] Pulling image: $IMAGE_URI"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${{ACCOUNT_ID}}.dkr.ecr.${{REGION}}.amazonaws.com"

docker pull "$IMAGE_URI"

echo "[BOOT] Starting container..."
set +e
docker run --rm \
  --name transcription-worker \
  --log-driver awslogs \
  --log-opt awslogs-region="$REGION" \
  --log-opt awslogs-group="{WORKER_LOG_GROUP}" \
  --log-opt awslogs-stream="{LOG_STREAM}" \
  --log-opt awslogs-create-group="true" \
  -e AWS_REGION="$REGION" \
  -e VIDEO_FILE_ID="$VIDEO_FILE_ID" \
  -e TARGET_FOLDER_ID="$TARGET_FOLDER_ID" \
  -e VIDEO_NAME="$VIDEO_NAME" \
  -e USE_SHARED_DRIVES="{USE_SHARED_DRIVES}" \
  -e WHISPER_MODEL="{WHISPER_MODEL}" \
  -e DEVICE="{DEVICE}" \
  -e COMPUTE_TYPE="{COMPUTE_TYPE}" \
  -e LANGUAGE="{LANGUAGE}" \
  -e USE_VAD="{USE_VAD}" \
  -e CHUNK_SECONDS="{CHUNK_SECONDS}" \
  -e OVERLAP_SECONDS="{OVERLAP_SECONDS}" \
  -e SSM_ENV_PARAM="{SSM_ENV_PARAM}" \
  -e SSM_CREDENTIALS_PARAM="{SSM_CREDENTIALS_PARAM}" \
  -e SSM_TOKEN_PARAM="{SSM_TOKEN_PARAM}" \
  "$IMAGE_URI" \
  python -u /app/worker_entrypoint.py
RC=$?
echo "[BOOT] Container exit code=$RC"
set -e

echo "[BOOT] Done."
"""

def launch_one_worker(task: Dict) -> str:
    stream = _sanitize_stream_name(task["video_name"])
    log_stream = f"{RUN_ID}/{stream}"

    exit_trap = ""
    if USE_SHUTDOWN_TERMINATE:
        exit_trap = "trap 'echo \"[BOOT] EXIT trap -> shutdown\"; shutdown -h now' EXIT"

    ud = WORKER_USERDATA_TEMPLATE.format(
        VIDEO_FILE_ID=task["video_file_id"],
        TARGET_FOLDER_ID=task["target_folder_id"],
        VIDEO_NAME=task["video_name"].replace('"', "'"),
        AWS_ACCOUNT_ID=AWS_ACCOUNT_ID,
        WORKER_ECR_REPO=WORKER_ECR_REPO,
        WORKER_IMAGE_TAG=WORKER_IMAGE_TAG,
        WORKER_LOG_GROUP=WORKER_LOG_GROUP,
        LOG_STREAM=log_stream,
        EXIT_TRAP=exit_trap,
        USE_SHARED_DRIVES=USE_SHARED_DRIVES_STR,
        WHISPER_MODEL=WORKER_WHISPER_MODEL,
        DEVICE=WORKER_DEVICE,
        COMPUTE_TYPE=WORKER_COMPUTE_TYPE,
        LANGUAGE=WORKER_LANGUAGE,
        USE_VAD=WORKER_USE_VAD,
        CHUNK_SECONDS=WORKER_CHUNK_SECONDS,
        OVERLAP_SECONDS=WORKER_OVERLAP_SECONDS,
        SSM_ENV_PARAM=SSM_ENV_PARAM,
        SSM_CREDENTIALS_PARAM=SSM_CREDENTIALS_PARAM,
        SSM_TOKEN_PARAM=SSM_TOKEN_PARAM,
    )
    ud_b64 = base64.b64encode(ud.encode("utf-8")).decode("utf-8")

    for attempt in range(8):
        try:
            resp = ec2.run_instances(
                LaunchTemplate={"LaunchTemplateId": LAUNCH_TEMPLATE_ID, "Version": LAUNCH_TEMPLATE_VERSION},
                MinCount=1,
                MaxCount=1,
                UserData=ud_b64,
                TagSpecifications=[{
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": "transcription-worker"},
                        {"Key": "Purpose", "Value": "test2-docker"},
                        {"Key": "RunId", "Value": RUN_ID},
                        {"Key": "Slot", "Value": task["slot"][:256]},
                        {"Key": "Person", "Value": task["person"][:256]},
                        {"Key": "Folder", "Value": task["folder"][:256]},
                        {"Key": "VideoFileId", "Value": task["video_file_id"][:128]},
                    ]
                }],
            )
            return resp["Instances"][0]["InstanceId"]
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["RequestLimitExceeded", "Throttling", "Rate exceeded"]):
                sleep = (2 ** attempt) + random.random()
                print(f"[WARN] Throttled. Retry in {sleep:.1f}s ...")
                time.sleep(sleep)
                continue
            raise

def list_active_workers() -> List[str]:
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Purpose", "Values": ["test2-docker"]},
            {"Name": "tag:RunId", "Values": [RUN_ID]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "shutting-down"]},
        ]
    )
    ids: List[str] = []
    for r in resp.get("Reservations", []):
        for i in r.get("Instances", []):
            ids.append(i["InstanceId"])
    return ids

def main():
    if not LAUNCH_TEMPLATE_ID:
        raise RuntimeError("WORKER_LAUNCH_TEMPLATE_ID not set in .env (orchestrator container).")

    print(f"[RUN] RUN_ID={RUN_ID} SLOT_CHOICE={SLOT_CHOICE or '1(default)'}")
    drive = build_drive()

    tasks = find_tasks(drive)
    print(f"[INFO] Pending videos (missing transcripts): {len(tasks)}")

    if MAX_LAUNCH > 0:
        tasks = tasks[:MAX_LAUNCH]
        print(f"[INFO] MAX_LAUNCH applied -> launching {len(tasks)} workers")

    if not tasks:
        print("[DONE] No pending transcripts. Exiting ✅")
        return

    for idx, task in enumerate(tasks, start=1):
        iid = launch_one_worker(task)
        print(f"[LAUNCH] {idx}/{len(tasks)} -> {iid} | {task['person']} | {task['folder']} | {task['video_name']}")
        time.sleep(LAUNCH_SLEEP_SECONDS)

    print("[WAIT] Waiting for workers to finish...")
    while True:
        active = list_active_workers()
        print(f"[WAIT] active workers: {len(active)}")
        if not active:
            break
        time.sleep(WAIT_POLL_SECONDS)

    print("[DONE] All workers completed ✅ (orchestrator exiting)")

if __name__ == "__main__":
    main()