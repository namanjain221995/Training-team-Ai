import os
import random
import re
import socket
import ssl
import time
from pathlib import Path
from typing import Optional, List, Dict, Iterable

import httplib2
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


# =========================
# CONFIG
# =========================
SCOPES = ["https://www.googleapis.com/auth/drive"]

CREDENTIALS_FILE = Path(os.environ.get("CREDENTIALS_FILE", "credentials.json"))
TOKEN_FILE = Path(os.environ.get("TOKEN_FILE", "token.json"))

ROOT_2026_FOLDER_NAME = os.environ.get("ROOT_2026_FOLDER_NAME", "2026")
USE_SHARED_DRIVES = os.environ.get("USE_SHARED_DRIVES", "false").strip().lower() in ("1", "true", "yes")

# 0 => ALL slots
# N => only selected slot
SLOT_CHOICE = int(os.environ.get("SLOT_CHOICE", "5"))

OUTPUT_ROOT_FOLDER_NAME = os.environ.get("OUTPUT_ROOT_FOLDER_NAME", "Candidate Result")
OUTPUT_ROOT_FOLDER_ID = os.environ.get("OUTPUT_ROOT_FOLDER_ID", "").strip() or None

# Optional: set DRY_RUN=true to only print what would be deleted
DRY_RUN = os.environ.get("DRY_RUN", "false").strip().lower() in ("1", "true", "yes")

FOLDER_MIME = "application/vnd.google-apps.folder"
GDOC_MIME = "application/vnd.google-apps.document"
GSHEET_MIME = "application/vnd.google-apps.spreadsheet"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

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

SKIP_PERSON_FOLDERS = {"1. Format"}

# test2
LLM_OUTPUT_PATTERN = re.compile(r"^LLM_OUTPUT__.*\.txt$", re.IGNORECASE)

# test6
EYE_OUTPUT_PATTERN = re.compile(
    r".*__EYE_(annotated_h264\.mp4|summary\.json|result\.json|metrics\.csv)$",
    re.IGNORECASE,
)


# =========================
# Robust Drive execute retry
# =========================
def drive_execute_with_retry(request, *, max_retries: int = 8, base_sleep: float = 1.0):
    last_exc = None

    for attempt in range(max_retries):
        try:
            return request.execute()
        except (TimeoutError, socket.timeout, ssl.SSLError, ConnectionError, httplib2.HttpLib2Error) as e:
            last_exc = e
        except HttpError as e:
            last_exc = e
            status = getattr(e.resp, "status", None)
            if status not in (429, 500, 502, 503, 504):
                raise

        sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(min(sleep_s, 30))

    if isinstance(last_exc, TimeoutError):
        raise last_exc

    raise TimeoutError(f"Drive API request failed after retries. Last error: {last_exc}")


# =========================
# Google Drive Auth
# =========================
def get_drive_service():
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(f"{CREDENTIALS_FILE} not found next to this script.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def _list_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True, "includeItemsFromAllDrives": True}
    return {}


def _delete_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}


def _get_kwargs():
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}


# =========================
# Drive helpers
# =========================
def drive_search_folder_anywhere(service, folder_name: str) -> List[dict]:
    safe_name = folder_name.replace("'", "\\'")
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed=false"

    out: List[dict] = []
    page_token = None

    while True:
        req = service.files().list(
            q=q,
            fields="nextPageToken, files(id,name,parents,modifiedTime,mimeType)",
            pageSize=100,
            pageToken=page_token,
            **_list_kwargs(),
        )
        res = drive_execute_with_retry(req)
        out.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break

    return out


def pick_best_named_folder(candidates: List[dict]) -> dict:
    if not candidates:
        raise RuntimeError("No folder candidates available.")
    return sorted(candidates, key=lambda c: (c.get("modifiedTime") or ""), reverse=True)[0]


def drive_get_file(service, file_id: str) -> dict:
    req = service.files().get(
        fileId=file_id,
        fields="id,name,mimeType,parents,modifiedTime",
        **_get_kwargs(),
    )
    return drive_execute_with_retry(req)


def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None) -> List[dict]:
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    out: List[dict] = []
    page_token = None

    while True:
        req = service.files().list(
            q=q,
            fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
            pageSize=500,
            pageToken=page_token,
            **_list_kwargs(),
        )
        res = drive_execute_with_retry(req)
        out.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break

    return out


def build_name_map(children: Iterable[dict]) -> Dict[str, dict]:
    """
    For duplicate names, keep the latest modified item.
    """
    by_name: Dict[str, dict] = {}
    for item in children:
        name = item.get("name") or ""
        prev = by_name.get(name)
        if not prev or (item.get("modifiedTime") or "") > (prev.get("modifiedTime") or ""):
            by_name[name] = item
    return by_name


def drive_delete_file(service, file_id: str, file_name: str) -> bool:
    if DRY_RUN:
        print(f"  [DRY RUN] Would delete: {file_name}")
        return True

    try:
        req = service.files().delete(fileId=file_id, **_delete_kwargs())
        drive_execute_with_retry(req)
        print(f"  ✓ Deleted: {file_name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to delete {file_name}: {e}")
        return False


# =========================
# Output location helpers
# =========================
def resolve_output_root_folder(service) -> Optional[dict]:
    if OUTPUT_ROOT_FOLDER_ID:
        try:
            root = drive_get_file(service, OUTPUT_ROOT_FOLDER_ID)
            if root.get("mimeType") != FOLDER_MIME:
                print("⚠ OUTPUT_ROOT_FOLDER_ID exists but is not a folder. Ignoring.")
                return None
            return root
        except Exception as e:
            print(f"⚠ Failed to fetch OUTPUT_ROOT_FOLDER_ID: {e}")
            return None

    cands = drive_search_folder_anywhere(service, OUTPUT_ROOT_FOLDER_NAME)
    if not cands:
        return None
    return pick_best_named_folder(cands)


def resolve_output_slot_folder_map(service, output_root_id: str) -> Dict[str, dict]:
    slot_children = drive_list_children(service, output_root_id, FOLDER_MIME)
    return build_name_map(slot_children)


# =========================
# Slot helpers
# =========================
def _slot_num_from_name(name: str) -> Optional[int]:
    if not name:
        return None
    m = re.search(r"\bslot\s*(\d+)\b", name, flags=re.I) or re.search(r"(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    slots = drive_list_children(service, slots_parent_id, FOLDER_MIME)

    def _sort_key(x):
        nm = x.get("name") or ""
        sn = _slot_num_from_name(nm)
        return (0, sn) if sn is not None else (1, nm.lower())

    return sorted(slots, key=_sort_key)


def resolve_target_slots(service, slots_parent_id: str) -> List[dict]:
    slots = list_slot_folders(service, slots_parent_id)
    if not slots:
        return []

    if SLOT_CHOICE == 0:
        print("Mode: SLOT_CHOICE=0 -> delete outputs for ALL slots")
        return slots

    for s in slots:
        if _slot_num_from_name(s.get("name", "")) == SLOT_CHOICE:
            print(f"Mode: SLOT_CHOICE={SLOT_CHOICE} -> delete outputs only for '{s['name']}'")
            return [s]

    raise RuntimeError(
        f"SLOT_CHOICE={SLOT_CHOICE} did not match any slot folder under '{ROOT_2026_FOLDER_NAME}'"
    )


def iter_people(service, slot_id: str) -> List[dict]:
    people = drive_list_children(service, slot_id, FOLDER_MIME)
    people = [
        p for p in people
        if (p.get("name") or "").strip() not in SKIP_PERSON_FOLDERS
    ]
    return sorted(people, key=lambda x: (x.get("name") or "").lower())


# =========================
# Fast person-folder pass
# =========================
def delete_test2_and_test6_outputs(service, target_slots: List[dict]) -> int:
    print("\n" + "=" * 80)
    print("DELETING TEST2 + TEST6 OUTPUTS IN SINGLE PASS")
    print("TEST2 -> LLM_OUTPUT__*.txt")
    print("TEST6 -> __EYE_* files")
    print("=" * 80)

    deleted_count = 0

    needed_folder_names = set(FOLDER_NAMES_TO_PROCESS)

    for slot in target_slots:
        people = iter_people(service, slot["id"])

        for person in people:
            # List person child folders ONCE
            person_children = drive_list_children(service, person["id"], FOLDER_MIME)
            person_folder_map = build_name_map(person_children)

            for folder_name in needed_folder_names:
                target_folder = person_folder_map.get(folder_name)
                if not target_folder:
                    continue

                # List files in deliverable folder ONCE
                folder_files = drive_list_children(service, target_folder["id"], None)

                files_to_delete = []
                for f in folder_files:
                    if f.get("mimeType") == FOLDER_MIME:
                        continue
                    name = f.get("name") or ""
                    if LLM_OUTPUT_PATTERN.match(name) or EYE_OUTPUT_PATTERN.match(name):
                        files_to_delete.append(f)

                if not files_to_delete:
                    continue

                print(f"\n[{slot['name']}] {person['name']} > {folder_name}")
                for f in files_to_delete:
                    if drive_delete_file(service, f["id"], f["name"]):
                        deleted_count += 1

    return deleted_count


# =========================
# TEST3
# =========================
def delete_test3_outputs(service, target_slots: List[dict]) -> int:
    print("\n" + "=" * 80)
    print("DELETING TEST3 OUTPUTS ('Deliverables Analysis' Google Docs)")
    print("=" * 80)

    deleted_count = 0

    for slot in target_slots:
        people = iter_people(service, slot["id"])

        for person in people:
            # List person children ONCE
            person_children = drive_list_children(service, person["id"], None)
            person_item_map = build_name_map(person_children)

            doc = person_item_map.get("Deliverables Analysis")
            if doc and doc.get("mimeType") == GDOC_MIME:
                print(f"[{slot['name']}] {person['name']}")
                if drive_delete_file(service, doc["id"], doc["name"]):
                    deleted_count += 1

    return deleted_count


# =========================
# TEST4 + TEST5
# =========================
def delete_test4_and_test5_outputs(service, target_slots: List[dict], output_root: Optional[dict]) -> int:
    print("\n" + "=" * 80)
    print("DELETING TEST4 + TEST5 OUTPUTS")
    print("TEST4 -> 'All Deliverables Analysis' Google Docs")
    print("TEST5 -> 'Deliverables Analysis Sheet.xlsx'")
    print("=" * 80)

    if not output_root:
        print(f"⚠ Output root '{OUTPUT_ROOT_FOLDER_NAME}' not found. Skipping TEST4/TEST5 deletions.")
        return 0

    deleted_count = 0

    # Build output slot map ONCE
    output_slot_map = resolve_output_slot_folder_map(service, output_root["id"])

    for slot in target_slots:
        out_slot = output_slot_map.get(slot["name"])
        if not out_slot:
            continue

        out_children = drive_list_children(service, out_slot["id"], None)
        out_map = build_name_map(out_children)

        # TEST4
        doc = out_map.get("All Deliverables Analysis")
        if doc and doc.get("mimeType") == GDOC_MIME:
            print(f"[{slot['name']}] (Candidate Result) -> All Deliverables Analysis")
            if drive_delete_file(service, doc["id"], doc["name"]):
                deleted_count += 1

        # TEST5 - support both true XLSX file and Google Sheet fallback
        xlsx = out_map.get("Deliverables Analysis Sheet.xlsx")
        if xlsx and xlsx.get("mimeType") in (XLSX_MIME, GSHEET_MIME):
            print(f"[{slot['name']}] (Candidate Result) -> Deliverables Analysis Sheet.xlsx")
            if drive_delete_file(service, xlsx["id"], xlsx["name"]):
                deleted_count += 1

    return deleted_count


# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("AUTO DELETE SCRIPT - OPTIMIZED")
    print("=" * 80)
    print(f"ROOT_2026_FOLDER_NAME   = {ROOT_2026_FOLDER_NAME}")
    print(f"OUTPUT_ROOT_FOLDER_NAME = {OUTPUT_ROOT_FOLDER_NAME}")
    print(f"USE_SHARED_DRIVES       = {USE_SHARED_DRIVES}")
    print(f"SLOT_CHOICE             = {SLOT_CHOICE}")
    print(f"DRY_RUN                 = {DRY_RUN}")
    print("Delete scope            = test2 + test3 + test4 + test5 + test6")
    print("Transcript deletion     = DISABLED (test1 not touched)")
    print("=" * 80)

    service = get_drive_service()

    # Resolve 2026 root ONCE
    candidates = drive_search_folder_anywhere(service, ROOT_2026_FOLDER_NAME)
    if not candidates:
        raise RuntimeError(f"Could not find folder '{ROOT_2026_FOLDER_NAME}' in Drive.")

    base_2026 = pick_best_named_folder(candidates)
    slots_parent_id = base_2026["id"]

    print(
        f"Using 2026 folder: {base_2026['name']} "
        f"(id={base_2026['id']}, modified={base_2026.get('modifiedTime')})"
    )

    target_slots = resolve_target_slots(service, slots_parent_id)
    if not target_slots:
        raise RuntimeError("No target slots found.")

    # Resolve output root ONCE
    output_root = resolve_output_root_folder(service)
    if output_root:
        print(
            f"Using output root: {output_root['name']} "
            f"(id={output_root['id']}, modified={output_root.get('modifiedTime')})"
        )
    else:
        print(f"⚠ Output root '{OUTPUT_ROOT_FOLDER_NAME}' not found.")

    total_deleted = 0

    # Combined fast pass for test2 + test6
    total_deleted += delete_test2_and_test6_outputs(service, target_slots)

    # test3
    total_deleted += delete_test3_outputs(service, target_slots)

    # combined fast pass for test4 + test5
    total_deleted += delete_test4_and_test5_outputs(service, target_slots, output_root)

    print("\n" + "=" * 80)
    if DRY_RUN:
        print(f"✓ DRY RUN COMPLETE - {total_deleted} file(s) would be deleted")
    else:
        print(f"✓ DELETION COMPLETE - {total_deleted} file(s) deleted")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()