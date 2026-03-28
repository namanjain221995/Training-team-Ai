import os
import time
import random
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

import boto3
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.errors import HttpError

# ---------------------------
# Config
# ---------------------------
ROOT_2026_FOLDER_NAME = os.getenv("ROOT_2026_FOLDER_NAME", "2026").strip()
ROOT_2026_FOLDER_ID = os.getenv("ROOT_2026_FOLDER_ID", "").strip()
SHARED_DRIVE_NAME = os.getenv("SHARED_DRIVE_NAME", "").strip()

FOLDER_MIME = "application/vnd.google-apps.folder"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

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

TRANSCRIPT_SUFFIX = "_transcripts.txt"

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LAUNCH_TEMPLATE_ID = os.getenv("WORKER_LAUNCH_TEMPLATE_ID", "").strip()
LAUNCH_TEMPLATE_VERSION = os.getenv("WORKER_LAUNCH_TEMPLATE_VERSION", "$Latest")

LAUNCH_SLEEP_SECONDS = float(os.getenv("LAUNCH_SLEEP_SECONDS", "0.25"))
MAX_LAUNCH = int(os.getenv("MAX_LAUNCH", "1"))

SSM_ENV_PARAM = os.getenv("SSM_ENV_PARAM", "/transcription/worker/env")
SSM_SERVICE_ACCOUNT_PARAM = os.getenv("SSM_SERVICE_ACCOUNT_PARAM", "/transcription/worker/service_account_json")

AUTH_MODE = os.getenv("AUTH_MODE", "service_account").strip().lower()
USE_DELEGATION = os.getenv("USE_DELEGATION", "1").strip().lower() in ("1", "true", "yes", "y")
DELEGATED_USER_EMAIL = os.getenv("DELEGATED_USER_EMAIL", "").strip()

SCOPES = ["https://www.googleapis.com/auth/drive"]

WORKER_LOG_GROUP = os.getenv("WORKER_LOG_GROUP", "/transcription/workers")
USE_SHUTDOWN_TERMINATE = os.getenv("USE_SHUTDOWN_TERMINATE", "1").strip().lower() in ("1", "true", "yes", "y")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 6

# ---------------------------
# Helpers: SSM fetch
# ---------------------------
def ssm_get(ssm, name: str, with_decryption: bool = True) -> str:
    resp = ssm.get_parameter(Name=name, WithDecryption=with_decryption)
    return resp["Parameter"]["Value"]

# ---------------------------
# Retry helpers
# ---------------------------
def drive_execute(req, label: str = "Drive API call"):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return req.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                sleep_s = min(2 ** (attempt - 1), 20) + random.uniform(0, 1)
                print(f"[RETRY] {label} failed with HTTP {status}. Attempt {attempt}/{MAX_RETRIES}. Sleeping {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            raise
        except Exception as e:
            if attempt < MAX_RETRIES:
                sleep_s = min(2 ** (attempt - 1), 20) + random.uniform(0, 1)
                print(f"[RETRY] {label} failed ({e}). Attempt {attempt}/{MAX_RETRIES}. Sleeping {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            raise

# ---------------------------
# Drive helpers
# ---------------------------
def _escape_drive_q_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")

def _slot_sort_key(name: str):
    import re
    m = re.search(r"(\d+)", name or "")
    return (int(m.group(1)) if m else float("inf"), (name or "").lower())

def _list_kwargs_for_drive(drive_id: Optional[str] = None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
    }
    if drive_id:
        kwargs["corpora"] = "drive"
        kwargs["driveId"] = drive_id
    else:
        kwargs["corpora"] = "allDrives"
    return kwargs

def list_shared_drives(service) -> List[dict]:
    out = []
    page_token = None
    while True:
        res = drive_execute(
            service.drives().list(pageSize=100, pageToken=page_token),
            label="list shared drives",
        )
        out.extend(res.get("drives", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return out

def get_shared_drive_by_name(service, drive_name: str) -> dict:
    drives = list_shared_drives(service)
    matches = [d for d in drives if d.get("name") == drive_name]

    if not matches:
        available = sorted([d.get("name", "") for d in drives])
        raise RuntimeError(
            f"Shared Drive '{drive_name}' not found.\n"
            f"Visible Shared Drives: {available}"
        )

    if len(matches) > 1:
        print(f"[WARN] Multiple Shared Drives named '{drive_name}'. Using first match.")

    return matches[0]

def drive_get_file(service, file_id: str) -> dict:
    return drive_execute(
        service.files().get(
            fileId=file_id,
            fields="id,name,parents,modifiedTime,mimeType,driveId",
            supportsAllDrives=True,
        ),
        label=f"get file {file_id}",
    )

def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None, drive_id: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{_escape_drive_q_value(mime_type)}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = drive_execute(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,parents,driveId)",
                pageSize=1000,
                pageToken=page_token,
                **_list_kwargs_for_drive(drive_id),
            ),
            label=f"list children of parent {parent_id}",
        )
        for f in res.get("files", []):
            yield f
        page_token = res.get("nextPageToken")
        if not page_token:
            break

def drive_find_child(service, parent_id: str, name: str, mime_type: Optional[str] = None, drive_id: Optional[str] = None):
    safe_name = _escape_drive_q_value(name)
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe_name}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{_escape_drive_q_value(mime_type)}'")
    q = " and ".join(q_parts)

    res = drive_execute(
        service.files().list(
            q=q,
            fields="files(id,name,mimeType,modifiedTime,parents,driveId)",
            pageSize=50,
            **_list_kwargs_for_drive(drive_id),
        ),
        label=f"find child '{name}' in parent {parent_id}",
    )
    files = res.get("files", []) or []
    if not files:
        return None
    return sorted(files, key=lambda x: x.get("modifiedTime") or "", reverse=True)[0]

def drive_search_folder_anywhere_in_shared_drive(service, folder_name: str, drive_id: str) -> List[dict]:
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed = false"

    out = []
    page_token = None
    while True:
        res = drive_execute(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,parents,modifiedTime,driveId,mimeType)",
                pageSize=1000,
                pageToken=page_token,
                **_list_kwargs_for_drive(drive_id),
            ),
            label=f"search folder '{folder_name}' in shared drive {drive_id}",
        )
        out.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return out

def is_video_name(name: str) -> bool:
    p = Path(name)
    if p.name.startswith("."):
        return False
    if p.suffix.lower() not in VIDEO_EXTS:
        return False
    if "__EYE_" in p.name:
        return False
    return True

def transcript_output_name(video_name: str) -> str:
    return f"{Path(video_name).stem}{TRANSCRIPT_SUFFIX}"

# ---------------------------
# Drive auth (service account + delegation)
# ---------------------------
def build_drive_service(service_account_json_text: str):
    if AUTH_MODE != "service_account":
        raise RuntimeError(f"Unsupported AUTH_MODE='{AUTH_MODE}'. Use AUTH_MODE=service_account")

    creds = service_account.Credentials.from_service_account_info(
        eval_json(service_account_json_text),
        scopes=SCOPES,
    )

    if USE_DELEGATION:
        if not DELEGATED_USER_EMAIL:
            raise RuntimeError("USE_DELEGATION=true but DELEGATED_USER_EMAIL is empty.")
        creds = creds.with_subject(DELEGATED_USER_EMAIL)
        print(f"[AUTH] Service account with delegation -> {DELEGATED_USER_EMAIL}")
    else:
        print("[AUTH] Service account without delegation")

    return build("drive", "v3", credentials=creds)

def eval_json(text: str) -> Dict[str, Any]:
    import json
    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Invalid service account JSON from SSM: {e}")

# ---------------------------
# Worker UserData template (FULL)
# ---------------------------
WORKER_USERDATA_TEMPLATE = r"""#!/bin/bash
set -euo pipefail

exec > >(tee -a /var/log/worker-userdata.log) 2>&1
echo "[BOOT] Worker starting..."
date

IMDS_TOKEN="$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)"

imds() {{
  local path="$1"
  if [[ -n "${{IMDS_TOKEN}}" ]]; then
    curl -s -H "X-aws-ec2-metadata-token: ${{IMDS_TOKEN}}" "http://169.254.169.254/latest/${{path}}"
  else
    curl -s "http://169.254.169.254/latest/${{path}}"
  fi
}}

REGION="$(imds meta-data/placement/region)"
INSTANCE_ID="$(imds meta-data/instance-id)"
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"
echo "[BOOT] REGION=$REGION INSTANCE_ID=$INSTANCE_ID"

export VIDEO_FILE_ID="{VIDEO_FILE_ID}"
export TARGET_FOLDER_ID="{TARGET_FOLDER_ID}"
export VIDEO_NAME="{VIDEO_NAME}"

echo "[BOOT] Installing packages..."
dnf update -y
dnf install -y git python3-pip ffmpeg jq awscli amazon-cloudwatch-agent
python3 -m pip install --upgrade pip

echo "[BOOT] Starting CloudWatch Agent..."
cat >/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'JSON2'
{{
  "logs": {{
    "logs_collected": {{
      "files": {{
        "collect_list": [
          {{
            "file_path": "/var/log/worker-userdata.log",
            "log_group_name": "{WORKER_LOG_GROUP}",
            "log_stream_name": "{{instance_id}}/userdata"
          }},
          {{
            "file_path": "/var/log/cloud-init-output.log",
            "log_group_name": "{WORKER_LOG_GROUP}",
            "log_stream_name": "{{instance_id}}/cloud-init"
          }}
        ]
      }}
    }}
  }}
}}
JSON2

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

mkdir -p /app /app/cache
cd /app

echo "[BOOT] Fetching config from SSM..."
aws ssm get-parameter --region "$REGION" --name "{SSM_ENV_PARAM}" --with-decryption --query "Parameter.Value" --output text > /app/.env
aws ssm get-parameter --region "$REGION" --name "{SSM_SERVICE_ACCOUNT_PARAM}" --with-decryption --query "Parameter.Value" --output text > /app/service-account.json

set -a
source /app/.env || true
set +a

export AUTH_MODE=service_account
export SERVICE_ACCOUNT_FILE=/app/service-account.json
export HEADLESS_AUTH=1
export XDG_CACHE_HOME=/app/cache
export HF_HOME=/app/cache
export TRANSFORMERS_CACHE=/app/cache

echo "[BOOT] Cloning repo..."
rm -rf repo
git clone --depth 1 https://github.com/namanjain221995/Training-team-Ai.git repo
cd repo

echo "[BOOT] Installing requirements..."
python3 -m pip install -r requirements.txt

echo "[BOOT] Running test2.py..."
python3 test2.py || true

echo "[BOOT] Finished job for VIDEO_NAME=$VIDEO_NAME"

if [[ "{USE_SHUTDOWN_TERMINATE}" == "1" ]]; then
  echo "[BOOT] Shutting down (Launch Template should be set to Terminate on shutdown)..."
  sudo shutdown -h now
else
  echo "[BOOT] Terminating instance via EC2 API..."
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" || true
fi
"""

# ---------------------------
# Find pending video tasks
# ---------------------------
def find_pending_tasks(drive) -> List[Dict]:
    if not SHARED_DRIVE_NAME:
        raise RuntimeError("SHARED_DRIVE_NAME is required.")

    shared_drive = get_shared_drive_by_name(drive, SHARED_DRIVE_NAME)
    shared_drive_id = shared_drive["id"]
    print(f"[INFO] Using Shared Drive: {shared_drive['name']} ({shared_drive_id})")

    if ROOT_2026_FOLDER_ID:
        root = drive_get_file(drive, ROOT_2026_FOLDER_ID)
        if root.get("mimeType") != FOLDER_MIME:
            raise RuntimeError(f"ROOT_2026_FOLDER_ID={ROOT_2026_FOLDER_ID} is not a folder.")
        file_drive_id = root.get("driveId")
        if file_drive_id and file_drive_id != shared_drive_id:
            raise RuntimeError(
                f"ROOT_2026_FOLDER_ID belongs to driveId={file_drive_id}, "
                f"but SHARED_DRIVE_NAME resolved to driveId={shared_drive_id}."
            )
    else:
        roots = drive_search_folder_anywhere_in_shared_drive(drive, ROOT_2026_FOLDER_NAME, shared_drive_id)
        if not roots:
            raise RuntimeError(
                f"Could not find folder '{ROOT_2026_FOLDER_NAME}' inside Shared Drive '{SHARED_DRIVE_NAME}'."
            )
        roots.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
        root = roots[0]

    root_id = root["id"]

    slot_choice = os.getenv("SLOT_CHOICE", "1").strip()
    slot_folders = sorted(
        list(drive_list_children(drive, root_id, FOLDER_MIME, shared_drive_id)),
        key=lambda x: _slot_sort_key(x.get("name") or ""),
    )
    if not slot_folders:
        raise RuntimeError("No slot folders found under 2026.")

    if slot_choice.isdigit():
        idx = int(slot_choice)
        if not (1 <= idx <= len(slot_folders)):
            raise RuntimeError(f"SLOT_CHOICE out of range (1..{len(slot_folders)})")
        slot = slot_folders[idx - 1]
    else:
        slot = slot_folders[0]

    people = sorted(
        list(drive_list_children(drive, slot["id"], FOLDER_MIME, shared_drive_id)),
        key=lambda x: x["name"].lower(),
    )

    tasks = []
    for person in people:
        for folder_name in FOLDER_NAMES_TO_PROCESS:
            target = drive_find_child(drive, person["id"], folder_name, FOLDER_MIME, shared_drive_id)
            if not target:
                continue
            folder_id = target["id"]

            children = list(drive_list_children(drive, folder_id, None, shared_drive_id))
            videos = [f for f in children if f.get("mimeType") != FOLDER_MIME and is_video_name(f["name"])]

            for vid in videos:
                out_name = transcript_output_name(vid["name"])
                if drive_find_child(drive, folder_id, out_name, None, shared_drive_id):
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

# ---------------------------
# EC2 Launch
# ---------------------------
def launch_worker(ec2, task: Dict) -> str:
    userdata = WORKER_USERDATA_TEMPLATE.format(
        VIDEO_FILE_ID=task["video_file_id"],
        TARGET_FOLDER_ID=task["target_folder_id"],
        VIDEO_NAME=task["video_name"].replace('"', "'"),
        WORKER_LOG_GROUP=WORKER_LOG_GROUP,
        SSM_ENV_PARAM=SSM_ENV_PARAM,
        SSM_SERVICE_ACCOUNT_PARAM=SSM_SERVICE_ACCOUNT_PARAM,
        USE_SHUTDOWN_TERMINATE="1" if USE_SHUTDOWN_TERMINATE else "0",
    )
    userdata_b64 = base64.b64encode(userdata.encode("utf-8")).decode("utf-8")

    for attempt in range(8):
        try:
            resp = ec2.run_instances(
                LaunchTemplate={"LaunchTemplateId": LAUNCH_TEMPLATE_ID, "Version": LAUNCH_TEMPLATE_VERSION},
                MinCount=1,
                MaxCount=1,
                UserData=userdata_b64,
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": "transcription-worker"},
                            {"Key": "Purpose", "Value": "test2"},
                            {"Key": "Slot", "Value": task["slot"][:256]},
                            {"Key": "Person", "Value": task["person"][:256]},
                            {"Key": "Folder", "Value": task["folder"][:256]},
                            {"Key": "VideoFileId", "Value": task["video_file_id"][:128]},
                        ],
                    }
                ],
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

def main():
    if not LAUNCH_TEMPLATE_ID:
        raise RuntimeError("Missing WORKER_LAUNCH_TEMPLATE_ID env var.")
    if not SHARED_DRIVE_NAME:
        raise RuntimeError("Missing SHARED_DRIVE_NAME env var.")

    ssm = boto3.client("ssm", region_name=AWS_REGION)
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    service_account_json_text = ssm_get(ssm, SSM_SERVICE_ACCOUNT_PARAM, with_decryption=True)
    drive = build_drive_service(service_account_json_text)

    tasks = find_pending_tasks(drive)
    print(f"[INFO] Pending videos (missing transcripts): {len(tasks)}")

    if MAX_LAUNCH > 0:
        tasks = tasks[:MAX_LAUNCH]
        print(f"[INFO] MAX_LAUNCH applied -> launching {len(tasks)} workers")

    if not tasks:
        print("[DONE] Nothing to launch")
        return

    launched = 0
    for t in tasks:
        iid = launch_worker(ec2, t)
        launched += 1
        print(f"[LAUNCH] {launched}/{len(tasks)} -> {iid} | {t['person']} | {t['folder']} | {t['video_name']}")
        time.sleep(LAUNCH_SLEEP_SECONDS)

    print(f"[DONE] Launched {launched} worker instances")

if __name__ == "__main__":
    main()