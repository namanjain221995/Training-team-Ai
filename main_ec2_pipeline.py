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
# CONFIG (from .env on MAIN EC2 / orchestrator container)
# -----------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

LAUNCH_TEMPLATE_ID = os.getenv("WORKER_LAUNCH_TEMPLATE_ID", "").strip()
LAUNCH_TEMPLATE_VERSION = os.getenv("WORKER_LAUNCH_TEMPLATE_VERSION", "$Default")

ROOT_2026_FOLDER_NAME = os.getenv("ROOT_2026_FOLDER_NAME", "2026")
SLOT_CHOICE = os.getenv("SLOT_CHOICE", "").strip()  # 1-based index
USE_SHARED_DRIVES = os.getenv("USE_SHARED_DRIVES", "0").strip().lower() in ("1", "true", "yes", "y")

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
MAX_LAUNCH = int(os.getenv("MAX_LAUNCH", "1"))  # 0 = launch all
LAUNCH_SLEEP_SECONDS = float(os.getenv("LAUNCH_SLEEP_SECONDS", "0.25"))

# Wait behavior
WAIT_POLL_SECONDS = int(os.getenv("WAIT_POLL_SECONDS", "15"))

# Unique RunId tag for this run
RUN_ID = os.getenv("RUN_ID", "") or time.strftime("%Y%m%d_%H%M%S")

# -----------------------------
# AWS client
# -----------------------------
ec2 = boto3.client("ec2", region_name=AWS_REGION)

# -----------------------------
# Drive helpers
# -----------------------------
def _list_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True, "includeItemsFromAllDrives": True, "corpora": "allDrives"}
    return {}

def build_drive():
    """
    Non-interactive Drive auth on main EC2:
    token.json must exist (mounted read-only is OK).
    If refresh happens, we write refreshed token to TOKEN_FILE_WRITE (writable).
    """
    if not TOKEN_FILE.exists():
        raise RuntimeError(f"token.json not found at {TOKEN_FILE}. Mount it into orchestrator container.")

    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

        # token.json is mounted read-only; write refreshed token elsewhere
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
        res = drive.files().list(
            q=q,
            fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
            pageSize=1000,
            pageToken=token,
            **_list_kwargs(),
        ).execute()
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
        res = drive.files().list(
            q=q,
            fields="nextPageToken, files(id,name,parents,modifiedTime)",
            pageSize=1000,
            pageToken=token,
            **_list_kwargs(),
        ).execute()
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

# -----------------------------
# Launch workers (per video)
# Worker Launch Template contains the real logic (docker build/run test2 + terminate).
# We override only to inject VIDEO_FILE_ID / TARGET_FOLDER_ID / VIDEO_NAME.
# -----------------------------
USERDATA_INJECT_ONLY = """#!/bin/bash
set -euo pipefail
export VIDEO_FILE_ID="{VIDEO_FILE_ID}"
export TARGET_FOLDER_ID="{TARGET_FOLDER_ID}"
export VIDEO_NAME="{VIDEO_NAME}"
"""

def launch_one_worker(task: Dict) -> str:
    ud = USERDATA_INJECT_ONLY.format(
        VIDEO_FILE_ID=task["video_file_id"],
        TARGET_FOLDER_ID=task["target_folder_id"],
        VIDEO_NAME=task["video_name"].replace('"', "'"),
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
                        {"Key": "Purpose", "Value": "test2"},
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
            {"Name": "tag:Purpose", "Values": ["test2"]},
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