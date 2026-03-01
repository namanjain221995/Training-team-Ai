import os
import time
import random
import base64
from pathlib import Path
from typing import Optional, List, Dict

import boto3
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# ---------------------------
# Config
# ---------------------------
ROOT_2026_FOLDER_NAME = os.getenv("ROOT_2026_FOLDER_NAME", "2026")
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
LAUNCH_TEMPLATE_ID = os.getenv("WORKER_LAUNCH_TEMPLATE_ID", "").strip()  # lt-...
LAUNCH_TEMPLATE_VERSION = os.getenv("WORKER_LAUNCH_TEMPLATE_VERSION", "$Latest")

# Throttle to avoid AWS API throttles
LAUNCH_SLEEP_SECONDS = float(os.getenv("LAUNCH_SLEEP_SECONDS", "0.25"))

# 0 = no limit (launch all)
MAX_LAUNCH = int(os.getenv("MAX_LAUNCH", "1"))

# Drive token/credentials fetched from SSM
SSM_ENV_PARAM = os.getenv("SSM_ENV_PARAM", "/transcription/worker/env")
SSM_CREDENTIALS_PARAM = os.getenv("SSM_CREDENTIALS_PARAM", "/transcription/worker/credentials_json")
SSM_TOKEN_PARAM = os.getenv("SSM_TOKEN_PARAM", "/transcription/worker/token_json")

USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES", "0").strip().lower() in ("1", "true", "yes", "y"))

SCOPES = ["https://www.googleapis.com/auth/drive"]

# CloudWatch worker logs
WORKER_LOG_GROUP = os.getenv("WORKER_LOG_GROUP", "/transcription/workers")

# ---------------------------
# Helpers: SSM fetch
# ---------------------------
def ssm_get(ssm, name: str, with_decryption: bool = True) -> str:
    resp = ssm.get_parameter(Name=name, WithDecryption=with_decryption)
    return resp["Parameter"]["Value"]

# ---------------------------
# Drive helpers
# ---------------------------
def _escape_drive_q_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")

def _list_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True, "includeItemsFromAllDrives": True, "corpora": "allDrives"}
    return {}

def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{_escape_drive_q_value(mime_type)}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = service.files().list(
            q=q,
            fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
            pageSize=1000,
            pageToken=page_token,
            **_list_kwargs(),
        ).execute()
        for f in res.get("files", []):
            yield f
        page_token = res.get("nextPageToken")
        if not page_token:
            break

def drive_find_child(service, parent_id: str, name: str, mime_type: Optional[str] = None):
    for f in drive_list_children(service, parent_id, mime_type):
        if f.get("name") == name:
            return f
    return None

def drive_search_folder_anywhere(service, folder_name: str) -> List[dict]:
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed=false"
    out = []
    page_token = None
    while True:
        res = service.files().list(
            q=q,
            fields="nextPageToken, files(id,name,parents,modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
            **_list_kwargs(),
        ).execute()
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
# Drive auth (non-interactive)
# ---------------------------
def build_drive_service(credentials_json_text: str, token_json_text: str):
    """
    Write creds/token to disk (Google libs work best with file paths).
    NOTE: In GitHub Actions we NEVER do interactive auth.
    """
    work = Path(".tmp_drive_auth")
    work.mkdir(exist_ok=True)
    cred_path = work / "credentials.json"
    token_path = work / "token.json"
    cred_path.write_text(credentials_json_text, encoding="utf-8")
    token_path.write_text(token_json_text, encoding="utf-8")

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
        else:
            raise RuntimeError("Drive token is missing/invalid. Update /transcription/worker/token_json in SSM.")

    return build("drive", "v3", credentials=creds)

# ---------------------------
# Worker UserData template (FULL + CloudWatch Logs)
# We override UserData per-instance so each worker knows which video to process.
# ---------------------------
WORKER_USERDATA_TEMPLATE = """#!/bin/bash
set -euo pipefail

# Log everything
exec > >(tee -a /var/log/worker-userdata.log) 2>&1
echo "[BOOT] Worker starting..."
date

# Injected task
export VIDEO_FILE_ID="{VIDEO_FILE_ID}"
export TARGET_FOLDER_ID="{TARGET_FOLDER_ID}"
export VIDEO_NAME="{VIDEO_NAME}"

# deps
dnf update -y
dnf install -y git python3-pip ffmpeg jq amazon-cloudwatch-agent
python3 -m pip install --upgrade pip

# CloudWatch log streaming
cat >/opt/aws/amazon-cloudwatch-agent/bin/config.json <<'EOF'
{{
  "logs": {{
    "logs_collected": {{
      "files": {{
        "collect_list": [
          {{
            "file_path": "/var/log/worker-userdata.log",
            "log_group_name": "{WORKER_LOG_GROUP}",
            "log_stream_name": "{{instance_id}}"
          }}
        ]
      }}
    }}
  }}
}}
EOF

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \\
  -a fetch-config -m ec2 \\
  -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json -s

mkdir -p /app /app/cache
cd /app

echo "[BOOT] Fetching config from SSM..."
aws ssm get-parameter --name "{SSM_ENV_PARAM}" --with-decryption --query "Parameter.Value" --output text > /app/.env
aws ssm get-parameter --name "{SSM_CREDENTIALS_PARAM}" --with-decryption --query "Parameter.Value" --output text > /app/credentials.json
aws ssm get-parameter --name "{SSM_TOKEN_PARAM}" --with-decryption --query "Parameter.Value" --output text > /app/token.json

set -a
source /app/.env || true
set +a

export CREDENTIALS_FILE=/app/credentials.json
export TOKEN_FILE=/app/token.json
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

echo "[BOOT] Terminating instance..."
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
"""

# ---------------------------
# Find pending video tasks
# ---------------------------
def find_pending_tasks(drive) -> List[Dict]:
    roots = drive_search_folder_anywhere(drive, ROOT_2026_FOLDER_NAME)
    if not roots:
        raise RuntimeError(f"Could not find Drive folder named '{ROOT_2026_FOLDER_NAME}'.")

    roots.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    root_id = roots[0]["id"]

    # Slot selection: use SLOT_CHOICE env var index (1-based). Default: first slot.
    slot_choice = os.getenv("SLOT_CHOICE", "4").strip()
    slot_folders = sorted(list(drive_list_children(drive, root_id, FOLDER_MIME)), key=lambda x: x["name"].lower())
    if not slot_folders:
        raise RuntimeError("No slot folders found under 2026.")

    if slot_choice.isdigit():
        idx = int(slot_choice)
        if not (1 <= idx <= len(slot_folders)):
            raise RuntimeError(f"SLOT_CHOICE out of range (1..{len(slot_folders)})")
        slot = slot_folders[idx - 1]
    else:
        slot = slot_folders[0]

    people = sorted(list(drive_list_children(drive, slot["id"], FOLDER_MIME)), key=lambda x: x["name"])

    tasks = []
    for person in people:
        for folder_name in FOLDER_NAMES_TO_PROCESS:
            target = drive_find_child(drive, person["id"], folder_name, FOLDER_MIME)
            if not target:
                continue
            folder_id = target["id"]

            children = list(drive_list_children(drive, folder_id, None))
            videos = [f for f in children if f.get("mimeType") != FOLDER_MIME and is_video_name(f["name"])]

            for vid in videos:
                out_name = transcript_output_name(vid["name"])
                if drive_find_child(drive, folder_id, out_name, None):
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
        SSM_CREDENTIALS_PARAM=SSM_CREDENTIALS_PARAM,
        SSM_TOKEN_PARAM=SSM_TOKEN_PARAM,
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
        raise RuntimeError("Missing WORKER_LAUNCH_TEMPLATE_ID env var (set it as GitHub Secret).")

    ssm = boto3.client("ssm", region_name=AWS_REGION)
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    # Read Drive auth from SSM
    credentials_json_text = ssm_get(ssm, SSM_CREDENTIALS_PARAM, with_decryption=True)
    token_json_text = ssm_get(ssm, SSM_TOKEN_PARAM, with_decryption=True)

    drive = build_drive_service(credentials_json_text, token_json_text)

    tasks = find_pending_tasks(drive)
    print(f"[INFO] Pending videos (missing transcripts): {len(tasks)}")

    if MAX_LAUNCH > 0:
        tasks = tasks[:MAX_LAUNCH]
        print(f"[INFO] MAX_LAUNCH applied -> launching {len(tasks)} workers")

    if not tasks:
        print("[DONE] Nothing to launch ✅")
        return

    launched = 0
    for t in tasks:
        iid = launch_worker(ec2, t)
        launched += 1
        print(f"[LAUNCH] {launched}/{len(tasks)} -> {iid} | {t['person']} | {t['folder']} | {t['video_name']}")
        time.sleep(LAUNCH_SLEEP_SECONDS)

    print(f"[DONE] Launched {launched} worker instances ✅")

if __name__ == "__main__":
    main()