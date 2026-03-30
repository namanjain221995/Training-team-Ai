import os
import time
import base64
import random
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

import boto3
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.errors import HttpError

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

LAUNCH_TEMPLATE_ID = os.getenv("WORKER_LAUNCH_TEMPLATE_ID", "").strip()
LAUNCH_TEMPLATE_VERSION = os.getenv("WORKER_LAUNCH_TEMPLATE_VERSION", "$Default")

ROOT_2026_FOLDER_NAME = os.getenv("ROOT_2026_FOLDER_NAME", "2026").strip()
ROOT_2026_FOLDER_ID = os.getenv("ROOT_2026_FOLDER_ID", "").strip()
SHARED_DRIVE_NAME = os.getenv("SHARED_DRIVE_NAME", "").strip()

SLOT_CHOICE = os.getenv("SLOT_CHOICE", "").strip()

AUTH_MODE = os.getenv("AUTH_MODE", "service_account").strip().lower()
USE_DELEGATION = os.getenv("USE_DELEGATION", "1").strip().lower() in ("1", "true", "yes", "y")
DELEGATED_USER_EMAIL = os.getenv("DELEGATED_USER_EMAIL", "").strip()

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

MAX_LAUNCH = int(os.getenv("MAX_LAUNCH", "0"))
MAX_CONCURRENT_WORKERS = int(os.getenv("MAX_CONCURRENT_WORKERS", "16"))
LAUNCH_SLEEP_SECONDS = float(os.getenv("LAUNCH_SLEEP_SECONDS", "0.10"))

WAIT_POLL_SECONDS = int(os.getenv("WAIT_POLL_SECONDS", "15"))

RUN_ID = os.getenv("RUN_ID", "") or time.strftime("%Y%m%d_%H%M%S")

GOOGLE_PAGE_SIZE = int(os.getenv("GOOGLE_PAGE_SIZE", "500"))
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = int(os.getenv("GOOGLE_EXECUTE_RETRIES", "8"))

SSM_ENV_PARAM = os.getenv("SSM_ENV_PARAM", "/transcription/worker/env")
SSM_SERVICE_ACCOUNT_PARAM = os.getenv("SSM_SERVICE_ACCOUNT_PARAM", "/transcription/worker/service_account_json")

WORKER_LOG_GROUP = os.getenv("WORKER_LOG_GROUP", "/transcription/workers")

WORKER_ECR_REPO = os.getenv("WORKER_ECR_REPO", "transcription-worker").strip()
WORKER_IMAGE_TAG = os.getenv("WORKER_IMAGE_TAG", "latest").strip()

WORKER_WHISPER_MODEL = os.getenv("WORKER_WHISPER_MODEL", "turbo").strip()
WORKER_DEVICE = os.getenv("WORKER_DEVICE", "cpu").strip()
WORKER_COMPUTE_TYPE = os.getenv("WORKER_COMPUTE_TYPE", "int8").strip()
WORKER_LANGUAGE = os.getenv("WORKER_LANGUAGE", "en").strip()
WORKER_USE_VAD = os.getenv("WORKER_USE_VAD", "true").strip().lower()
WORKER_CHUNK_SECONDS = os.getenv("WORKER_CHUNK_SECONDS", "540").strip()
WORKER_OVERLAP_SECONDS = os.getenv("WORKER_OVERLAP_SECONDS", "2").strip()

USE_SHUTDOWN_TERMINATE = os.getenv("USE_SHUTDOWN_TERMINATE", "1").strip().lower() in ("1", "true", "yes", "y")

ec2 = boto3.client("ec2", region_name=AWS_REGION)
sts = boto3.client("sts", region_name=AWS_REGION)
ssm = boto3.client("ssm", region_name=AWS_REGION)
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "").strip() or sts.get_caller_identity()["Account"]

_SLOT_PREFIX_RE = re.compile(r"^\s*(\d+)\.\s*")


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


def ssm_get(name: str, with_decryption: bool = True) -> str:
    resp = ssm.get_parameter(Name=name, WithDecryption=with_decryption)
    return resp["Parameter"]["Value"]


def parse_service_account_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Invalid service account JSON from SSM: {e}")


def build_drive(service_account_json_text: str):
    if AUTH_MODE != "service_account":
        raise RuntimeError(f"Unsupported AUTH_MODE='{AUTH_MODE}'. Use AUTH_MODE=service_account")

    creds = service_account.Credentials.from_service_account_info(
        parse_service_account_json(service_account_json_text),
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


def _escape_drive_q_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def extract_slot_prefix(name: str) -> Optional[int]:
    m = _SLOT_PREFIX_RE.match(name or "")
    return int(m.group(1)) if m else None


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


def _get_kwargs() -> Dict[str, Any]:
    return {"supportsAllDrives": True}


def list_shared_drives(drive) -> List[dict]:
    out: List[dict] = []
    token = None
    while True:
        res = drive_execute(
            drive.drives().list(pageSize=100, pageToken=token),
            label="list shared drives",
        )
        out.extend(res.get("drives", []))
        token = res.get("nextPageToken")
        if not token:
            break
    return out


def get_shared_drive_by_name(drive, drive_name: str) -> dict:
    drives = list_shared_drives(drive)
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


def drive_get_file(drive, file_id: str) -> dict:
    return drive_execute(
        drive.files().get(
            fileId=file_id,
            fields="id,name,parents,modifiedTime,mimeType,driveId",
            **_get_kwargs(),
        ),
        label=f"get file {file_id}",
    )


def drive_list_children(drive, parent_id: str, mime_type: Optional[str] = None, drive_id: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{_escape_drive_q_value(mime_type)}'")
    q = " and ".join(q_parts)

    token = None
    while True:
        res = drive_execute(
            drive.files().list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,parents,driveId)",
                pageSize=GOOGLE_PAGE_SIZE,
                pageToken=token,
                **_list_kwargs_for_drive(drive_id),
            ),
            label=f"list children of parent {parent_id}",
        )
        yield from res.get("files", [])
        token = res.get("nextPageToken")
        if not token:
            break


def drive_find_child(drive, parent_id: str, name: str, mime_type: Optional[str] = None, drive_id: Optional[str] = None):
    safe_name = _escape_drive_q_value(name)
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe_name}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{_escape_drive_q_value(mime_type)}'")
    q = " and ".join(q_parts)

    res = drive_execute(
        drive.files().list(
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


def drive_search_folder_anywhere_in_shared_drive(drive, folder_name: str, drive_id: str) -> List[dict]:
    safe_name = _escape_drive_q_value(folder_name)
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed = false"

    out: List[dict] = []
    token = None
    while True:
        res = drive_execute(
            drive.files().list(
                q=q,
                fields="nextPageToken, files(id,name,parents,modifiedTime,driveId,mimeType)",
                pageSize=GOOGLE_PAGE_SIZE,
                pageToken=token,
                **_list_kwargs_for_drive(drive_id),
            ),
            label=f"search folder '{folder_name}' in shared drive {drive_id}",
        )
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


def list_slot_folders(drive, root_id: str, drive_id: str) -> List[dict]:
    folders = list(drive_list_children(drive, root_id, FOLDER_MIME, drive_id))
    out = []

    for f in folders:
        slot_no = extract_slot_prefix(f.get("name", ""))
        if slot_no is not None:
            item = dict(f)
            item["_slot_no"] = slot_no
            out.append(item)

    return sorted(out, key=lambda x: (x["_slot_no"], x.get("name", "").lower()))


def pick_slot(drive, root_id: str, drive_id: str) -> dict:
    slots = list_slot_folders(drive, root_id, drive_id)
    if not slots:
        raise RuntimeError("No numbered slot folders found under 2026.")

    if SLOT_CHOICE.isdigit():
        wanted_slot_no = int(SLOT_CHOICE)
        for s in slots:
            if s["_slot_no"] == wanted_slot_no:
                print(f"[AUTO] Using SLOT_CHOICE={wanted_slot_no}: {s['name']}")
                return s

        available = ", ".join(str(s["_slot_no"]) for s in slots)
        raise RuntimeError(
            f"SLOT_CHOICE '{wanted_slot_no}' not found. Available folder numbers: {available}"
        )

    print("[INFO] SLOT_CHOICE not set. Defaulting to lowest numbered folder.")
    return slots[0]


def find_tasks(drive) -> List[Dict[str, str]]:
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

    slot = pick_slot(drive, root["id"], shared_drive_id)
    print(f"[SELECTED] Slot: {slot['name']}")

    people = sorted(
        list(drive_list_children(drive, slot["id"], FOLDER_MIME, shared_drive_id)),
        key=lambda x: x["name"].lower(),
    )

    tasks: List[Dict[str, str]] = []
    for person in people:
        for folder_name in FOLDER_NAMES_TO_PROCESS:
            folder = drive_find_child(drive, person["id"], folder_name, FOLDER_MIME, shared_drive_id)
            if not folder:
                continue
            folder_id = folder["id"]

            children = list(drive_list_children(drive, folder_id, None, shared_drive_id))
            videos = [f for f in children if f["mimeType"] != FOLDER_MIME and is_video(f["name"])]

            for vid in videos:
                out = transcript_name(vid["name"])
                if drive_find_child(drive, folder_id, out, None, shared_drive_id):
                    continue

                tasks.append(
                    {
                        "video_file_id": vid["id"],
                        "target_folder_id": folder_id,
                        "video_name": vid["name"],
                        "slot": slot["name"],
                        "person": person["name"],
                        "folder": folder_name,
                    }
                )

    return tasks


def _sanitize_stream_name(s: str) -> str:
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.strip()
    if len(s) > 200:
        s = s[:200]
    return s or "video"


WORKER_USERDATA_TEMPLATE = r"""#!/bin/bash
set -euo pipefail

export VIDEO_FILE_ID="{VIDEO_FILE_ID}"
export TARGET_FOLDER_ID="{TARGET_FOLDER_ID}"
export VIDEO_NAME="{VIDEO_NAME}"

exec > >(tee -a /var/log/worker-userdata.log) 2>&1
echo "[BOOT] Worker starting..."
date

{EXIT_TRAP}

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
  -e AUTH_MODE="service_account" \
  -e USE_DELEGATION="{USE_DELEGATION}" \
  -e DELEGATED_USER_EMAIL="{DELEGATED_USER_EMAIL}" \
  -e SHARED_DRIVE_NAME="{SHARED_DRIVE_NAME}" \
  -e ROOT_2026_FOLDER_NAME="{ROOT_2026_FOLDER_NAME}" \
  -e ROOT_2026_FOLDER_ID="{ROOT_2026_FOLDER_ID}" \
  -e WHISPER_MODEL="{WHISPER_MODEL}" \
  -e DEVICE="{DEVICE}" \
  -e COMPUTE_TYPE="{COMPUTE_TYPE}" \
  -e LANGUAGE="{LANGUAGE}" \
  -e USE_VAD="{USE_VAD}" \
  -e CHUNK_SECONDS="{CHUNK_SECONDS}" \
  -e OVERLAP_SECONDS="{OVERLAP_SECONDS}" \
  -e SSM_ENV_PARAM="{SSM_ENV_PARAM}" \
  -e SSM_SERVICE_ACCOUNT_PARAM="{SSM_SERVICE_ACCOUNT_PARAM}" \
  "$IMAGE_URI" \
  python -u /app/worker_entrypoint.py
RC=$?
echo "[BOOT] Container exit code=$RC"
set -e

echo "[BOOT] Done."
"""


def launch_one_worker(task: Dict[str, str]) -> str:
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
        USE_DELEGATION="1" if USE_DELEGATION else "0",
        DELEGATED_USER_EMAIL=DELEGATED_USER_EMAIL,
        SHARED_DRIVE_NAME=SHARED_DRIVE_NAME,
        ROOT_2026_FOLDER_NAME=ROOT_2026_FOLDER_NAME,
        ROOT_2026_FOLDER_ID=ROOT_2026_FOLDER_ID,
        WHISPER_MODEL=WORKER_WHISPER_MODEL,
        DEVICE=WORKER_DEVICE,
        COMPUTE_TYPE=WORKER_COMPUTE_TYPE,
        LANGUAGE=WORKER_LANGUAGE,
        USE_VAD=WORKER_USE_VAD,
        CHUNK_SECONDS=WORKER_CHUNK_SECONDS,
        OVERLAP_SECONDS=WORKER_OVERLAP_SECONDS,
        SSM_ENV_PARAM=SSM_ENV_PARAM,
        SSM_SERVICE_ACCOUNT_PARAM=SSM_SERVICE_ACCOUNT_PARAM,
    )
    ud_b64 = base64.b64encode(ud.encode("utf-8")).decode("utf-8")

    for attempt in range(8):
        try:
            resp = ec2.run_instances(
                LaunchTemplate={"LaunchTemplateId": LAUNCH_TEMPLATE_ID, "Version": LAUNCH_TEMPLATE_VERSION},
                MinCount=1,
                MaxCount=1,
                UserData=ud_b64,
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": "transcription-worker"},
                            {"Key": "Purpose", "Value": "test2-docker"},
                            {"Key": "RunId", "Value": RUN_ID},
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
                sleep_s = (2 ** attempt) + random.random()
                print(f"[WARN] Throttled. Retry in {sleep_s:.1f}s ...")
                time.sleep(sleep_s)
                continue
            raise


def describe_run_workers() -> List[Dict[str, str]]:
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Purpose", "Values": ["test2-docker"]},
            {"Name": "tag:RunId", "Values": [RUN_ID]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "shutting-down", "stopped"]},
        ]
    )

    workers: List[Dict[str, str]] = []
    for r in resp.get("Reservations", []):
        for i in r.get("Instances", []):
            workers.append(
                {
                    "instance_id": i["InstanceId"],
                    "state": i["State"]["Name"],
                }
            )
    return workers


def list_active_workers() -> List[str]:
    workers = describe_run_workers()
    active_states = {"pending", "running", "stopping", "shutting-down"}
    return [w["instance_id"] for w in workers if w["state"] in active_states]


def main():
    if not LAUNCH_TEMPLATE_ID:
        raise RuntimeError("WORKER_LAUNCH_TEMPLATE_ID not set in .env (orchestrator container).")
    if not SHARED_DRIVE_NAME:
        raise RuntimeError("SHARED_DRIVE_NAME not set in .env.")
    if AUTH_MODE != "service_account":
        raise RuntimeError("This orchestrator now supports AUTH_MODE=service_account only.")
    if MAX_CONCURRENT_WORKERS < 1:
        raise RuntimeError("MAX_CONCURRENT_WORKERS must be >= 1")

    print(f"[RUN] RUN_ID={RUN_ID} SLOT_CHOICE={SLOT_CHOICE or 'not set'}")
    print(f"[RUN] MAX_CONCURRENT_WORKERS={MAX_CONCURRENT_WORKERS}")

    service_account_json_text = ssm_get(SSM_SERVICE_ACCOUNT_PARAM, with_decryption=True)
    drive = build_drive(service_account_json_text)

    tasks = find_tasks(drive)
    print(f"[INFO] Pending videos (missing transcripts): {len(tasks)}")

    if MAX_LAUNCH > 0:
        tasks = tasks[:MAX_LAUNCH]
        print(f"[INFO] MAX_LAUNCH applied -> total tasks limited to {len(tasks)}")

    if not tasks:
        print("[DONE] No pending transcripts. Exiting ✅")
        return

    launched = 0
    total = len(tasks)
    next_task_idx = 0

    while True:
        active = list_active_workers()
        active_count = len(active)

        while next_task_idx < total and active_count < MAX_CONCURRENT_WORKERS:
            task = tasks[next_task_idx]
            iid = launch_one_worker(task)
            launched += 1
            next_task_idx += 1
            active_count += 1

            print(
                f"[LAUNCH] {launched}/{total} -> {iid} | "
                f"{task['person']} | {task['folder']} | {task['video_name']}"
            )
            time.sleep(LAUNCH_SLEEP_SECONDS)

        active = list_active_workers()
        active_count = len(active)

        print(
            f"[WAIT] launched={launched}/{total} | "
            f"next_task_idx={next_task_idx} | active_workers={active_count}"
        )

        if next_task_idx >= total and active_count == 0:
            break

        time.sleep(WAIT_POLL_SECONDS)

    print("[DONE] All workers completed ✅ (orchestrator exiting)")


if __name__ == "__main__":
    main()