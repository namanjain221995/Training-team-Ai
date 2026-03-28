import random
import re
import socket
import ssl
import time
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Set

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
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")

ROOT_2026_FOLDER_NAME = "2026"
USE_SHARED_DRIVES = False

# Output root folder for ONLY test4 + test5
OUTPUT_ROOT_FOLDER_NAME = "Candidate Result"
# If you want to hardcode the ID for safety (recommended), set it here:
OUTPUT_ROOT_FOLDER_ID = None  # e.g. "1AbCdefGhIJK..."

FOLDER_MIME = "application/vnd.google-apps.folder"
GDOC_MIME = "application/vnd.google-apps.document"
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

# File names and patterns created by each script
SCRIPT_OUTPUTS = {
    "test1": {
        "description": "Transcript .txt files",
        "pattern": r"^[^/]*\.txt$",
        "mime_types": ["text/plain"],
        "location": "in 2026/<Slot>/<Person>/<FolderName>/",
    },
    "test2": {
        "description": "LLM_OUTPUT__*.txt files",
        "pattern": r"^LLM_OUTPUT__.*\.txt$",
        "mime_types": ["text/plain"],
        "location": "in 2026/<Slot>/<Person>/<FolderName>/",
    },
    "test3": {
        "description": "'Deliverables Analysis' Google Docs",
        "filename": "Deliverables Analysis",
        "mime_types": [GDOC_MIME],
        "location": "in 2026/<Slot>/<Person>/",
    },
    "test4": {
        "description": "'All Deliverables Analysis' Google Docs",
        "filename": "All Deliverables Analysis",
        "mime_types": [GDOC_MIME],
        "location": "in Candidate Result/<Slot>/",
    },
    "test5": {
        "description": "'Deliverables Analysis Sheet.xlsx' Excel files",
        "filename": "Deliverables Analysis Sheet.xlsx",
        "mime_types": [XLSX_MIME],
        "location": "in Candidate Result/<Slot>/",
    },
    "test6": {
        "description": "Eye/Face tracking outputs (__EYE_*)",
        "pattern": r".*__EYE_(annotated_h264\.mp4|summary\.json|result\.json|metrics\.csv)$",
        "mime_types": ["video/mp4", "application/json", "text/csv"],
        "location": "in 2026/<Slot>/<Person>/<FolderName>/",
    },
}


# =========================
# Robust Drive execute retry
# =========================
def drive_execute_with_retry(request, *, max_retries: int = 8, base_sleep: float = 1.0):
    """
    Robust retry for Drive API calls.
    Retries on:
      - transient network errors/timeouts
      - HTTP 429 (rate limit)
      - HTTP 5xx (transient server errors)
    """
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
                raise FileNotFoundError("credentials.json not found next to this script.")
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


# =========================
# Drive helpers
# =========================
def drive_search_folder_anywhere(service, folder_name: str) -> List[dict]:
    """Search for folder by name anywhere in Drive (paginated)."""
    safe_name = folder_name.replace("'", "\\'")
    q = f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed=false"

    out: List[dict] = []
    page_token = None
    while True:
        req = service.files().list(
            q=q,
            fields="nextPageToken, files(id,name,parents,modifiedTime)",
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
    """Pick the most recent folder if multiple found."""
    return sorted(candidates, key=lambda c: (c.get("modifiedTime") or ""), reverse=True)[0]


def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None):
    """List all children of a folder."""
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

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

        for f in res.get("files", []):
            yield f

        page_token = res.get("nextPageToken")
        if not page_token:
            break


def drive_find_child(service, parent_id: str, name: str, mime_type: Optional[str] = None) -> Optional[dict]:
    """Find a specific child by name."""
    safe_name = name.replace("'", "\\'")
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe_name}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    req = service.files().list(q=q, fields="files(id,name,mimeType,modifiedTime)", pageSize=50, **_list_kwargs())
    res = drive_execute_with_retry(req)
    files = res.get("files", []) or []
    if not files:
        return None
    files.sort(key=lambda x: (x.get("modifiedTime") or ""), reverse=True)
    return files[0]


def drive_delete_file(service, file_id: str, file_name: str) -> bool:
    """Delete a file by ID."""
    try:
        req = service.files().delete(fileId=file_id, **_delete_kwargs())
        drive_execute_with_retry(req)
        print(f"  ✓ Deleted: {file_name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to delete {file_name}: {e}")
        return False


# =========================
# Output location helpers (ONLY for test4/test5)
# =========================
def drive_get_folder_by_id(service, folder_id: str) -> dict:
    req = service.files().get(
        fileId=folder_id,
        fields="id,name,mimeType,parents,modifiedTime",
        **_list_kwargs(),
    )
    return drive_execute_with_retry(req)


def resolve_output_root_folder(service) -> Optional[dict]:
    """
    Resolve the OUTPUT root folder (Candidate Result).
    Returns folder dict or None if not found.
    """
    if OUTPUT_ROOT_FOLDER_ID:
        try:
            root = drive_get_folder_by_id(service, OUTPUT_ROOT_FOLDER_ID)
            if root.get("mimeType") != FOLDER_MIME:
                print("⚠ OUTPUT_ROOT_FOLDER_ID is not a folder. Ignoring.")
                return None
            return root
        except Exception as e:
            print(f"⚠ Failed to fetch OUTPUT_ROOT_FOLDER_ID: {e}")
            return None

    cands = drive_search_folder_anywhere(service, OUTPUT_ROOT_FOLDER_NAME)
    if not cands:
        return None
    return pick_best_named_folder(cands)


def resolve_output_slot_folder_existing(service, slot_name: str) -> Optional[dict]:
    """
    Returns Candidate Result/<SlotName> folder if it exists.
    IMPORTANT: does NOT create folders.
    """
    out_root = resolve_output_root_folder(service)
    if not out_root:
        return None
    return drive_find_child(service, out_root["id"], slot_name, FOLDER_MIME)


# =========================
# Slot / Candidate selection helpers
# =========================
def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    return sorted(
        list(drive_list_children(service, slots_parent_id, FOLDER_MIME)),
        key=lambda x: (x.get("name") or "").lower(),
    )


def list_people(service, slot_id: str) -> List[dict]:
    return sorted(
        [
            p
            for p in drive_list_children(service, slot_id, FOLDER_MIME)
            if (p.get("name") or "").strip() not in SKIP_PERSON_FOLDERS
        ],
        key=lambda x: (x.get("name") or "").lower(),
    )


def choose_slot(service, slots_parent_id: str) -> Optional[dict]:
    """
    Choose one slot or ALL.
    Returns:
      - slot dict if one slot selected
      - None if ALL slots selected
    """
    slots = list_slot_folders(service, slots_parent_id)
    if not slots:
        print("No slot folders found under '2026'.")
        return None

    print("\n" + "=" * 80)
    print("SELECT SLOT")
    print("=" * 80)
    for i, s in enumerate(slots, start=1):
        print(f"  {i:2}. {s['name']}")
    print("  ALL - All slots")
    print("  EXIT - Exit\n")

    while True:
        choice = input("Choose slot number / ALL / EXIT: ").strip().lower()
        if choice == "exit":
            print("✓ Exiting without changes.")
            raise SystemExit(0)
        if choice == "all":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(slots):
                return slots[idx - 1]
        print("Invalid choice. Try again.")


def choose_candidate_mode() -> str:
    """
    Candidate mode only applies when one slot is selected.
    Returns:
      - one
      - multiple
      - all
      - exit
    """
    print("\n" + "=" * 80)
    print("SELECT CANDIDATE SCOPE")
    print("=" * 80)
    print("  1. One candidate")
    print("  2. Multiple candidates")
    print("  3. All candidates in this slot")
    print("  4. Exit\n")

    valid = {"1": "one", "2": "multiple", "3": "all", "4": "exit"}
    while True:
        choice = input("Choose option (1/2/3/4): ").strip()
        if choice in valid:
            return valid[choice]
        print("Invalid choice. Try again.")


def choose_candidates_for_slot(service, slot: dict) -> Optional[List[dict]]:
    """
    For one selected slot, let user choose:
      - one candidate
      - multiple candidates
      - all candidates
    Returns:
      - list[dict] of selected candidates
      - None means all candidates in that slot
    """
    candidates = list_people(service, slot["id"])
    if not candidates:
        print(f"No candidate folders found inside slot '{slot['name']}'.")
        return []

    mode = choose_candidate_mode()
    if mode == "exit":
        print("✓ Exiting without changes.")
        raise SystemExit(0)

    if mode == "all":
        return None

    print("\n" + "=" * 80)
    print(f"SELECT CANDIDATE(S) IN SLOT: {slot['name']}")
    print("=" * 80)
    for i, c in enumerate(candidates, start=1):
        print(f"  {i:2}. {c['name']}")
    print("")

    if mode == "one":
        while True:
            choice = input("Choose one candidate number: ").strip()
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(candidates):
                    return [candidates[idx - 1]]
            print("Invalid choice. Try again.")

    # mode == "multiple"
    while True:
        raw = input("Enter candidate numbers separated by comma (example: 1,3,5): ").strip()
        if not raw:
            print("Please enter at least one candidate number.")
            continue

        parts = [x.strip() for x in raw.split(",") if x.strip()]
        if not parts:
            print("Invalid input. Try again.")
            continue

        bad = [p for p in parts if not p.isdigit()]
        if bad:
            print(f"Invalid entries: {', '.join(bad)}")
            continue

        idxs = sorted(set(int(p) for p in parts))
        if any(idx < 1 or idx > len(candidates) for idx in idxs):
            print("One or more numbers are out of range.")
            continue

        return [candidates[idx - 1] for idx in idxs]


def iter_target_slots(service, slots_parent_id: str, only_slot: Optional[dict]) -> Iterable[dict]:
    if only_slot:
        yield only_slot
    else:
        for s in list_slot_folders(service, slots_parent_id):
            yield s


def iter_target_people(service, slot: dict, selected_people: Optional[List[dict]]) -> Iterable[dict]:
    """
    selected_people:
      - None => all people in slot
      - []   => none
      - list => selected people only
    """
    if selected_people is None:
        for p in list_people(service, slot["id"]):
            yield p
    else:
        for p in selected_people:
            yield p


# =========================
# Delete scope context
# =========================
class DeleteScope:
    def __init__(self, slot: Optional[dict], selected_people: Optional[List[dict]]):
        self.slot = slot
        self.selected_people = selected_people

    @property
    def is_all_slots(self) -> bool:
        return self.slot is None

    @property
    def is_slot_level_all_candidates(self) -> bool:
        return self.slot is not None and self.selected_people is None

    @property
    def is_candidate_level(self) -> bool:
        return self.slot is not None and self.selected_people is not None

    def describe(self) -> str:
        if self.is_all_slots:
            return "ALL SLOTS / ALL CANDIDATES"
        if self.selected_people is None:
            return f"SLOT = {self.slot['name']} / ALL CANDIDATES"
        names = ", ".join(p["name"] for p in self.selected_people) if self.selected_people else "(none)"
        return f"SLOT = {self.slot['name']} / CANDIDATE(S) = {names}"


# =========================
# Deletion functions
# =========================
def delete_test1_outputs(service, slots_parent_id: str, scope: DeleteScope) -> int:
    """Delete transcript .txt files created by test.py."""
    print("\n" + "=" * 80)
    print("DELETING TEST1 OUTPUTS (Transcript .txt files)")
    print("=" * 80)

    deleted_count = 0
    pattern = re.compile(r"^[^/]*\.txt$", re.IGNORECASE)

    for slot in iter_target_slots(service, slots_parent_id, scope.slot):
        selected_people = scope.selected_people if scope.slot and slot["id"] == scope.slot["id"] else None
        if scope.is_all_slots:
            selected_people = None

        for person in iter_target_people(service, slot, selected_people):
            for folder_name in FOLDER_NAMES_TO_PROCESS:
                target = drive_find_child(service, person["id"], folder_name, FOLDER_MIME)
                if not target:
                    continue

                files = list(drive_list_children(service, target["id"], None))
                txt_files = [
                    f
                    for f in files
                    if f.get("mimeType") != FOLDER_MIME
                    and pattern.match(f.get("name") or "")
                    and not (f.get("name") or "").startswith("LLM_OUTPUT__")
                ]

                if txt_files:
                    print(f"\n[{slot['name']}] {person['name']} > {folder_name}")
                    for f in txt_files:
                        if drive_delete_file(service, f["id"], f["name"]):
                            deleted_count += 1

    return deleted_count


def delete_test2_outputs(service, slots_parent_id: str, scope: DeleteScope) -> int:
    """Delete LLM_OUTPUT__*.txt files created by test2.py."""
    print("\n" + "=" * 80)
    print("DELETING TEST2 OUTPUTS (LLM_OUTPUT__*.txt files)")
    print("=" * 80)

    deleted_count = 0
    pattern = re.compile(r"^LLM_OUTPUT__.*\.txt$", re.IGNORECASE)

    for slot in iter_target_slots(service, slots_parent_id, scope.slot):
        selected_people = scope.selected_people if scope.slot and slot["id"] == scope.slot["id"] else None
        if scope.is_all_slots:
            selected_people = None

        for person in iter_target_people(service, slot, selected_people):
            for folder_name in FOLDER_NAMES_TO_PROCESS:
                target = drive_find_child(service, person["id"], folder_name, FOLDER_MIME)
                if not target:
                    continue

                files = list(drive_list_children(service, target["id"], None))
                llm_files = [
                    f
                    for f in files
                    if f.get("mimeType") != FOLDER_MIME and pattern.match(f.get("name") or "")
                ]

                if llm_files:
                    print(f"\n[{slot['name']}] {person['name']} > {folder_name}")
                    for f in llm_files:
                        if drive_delete_file(service, f["id"], f["name"]):
                            deleted_count += 1

    return deleted_count


def delete_test3_outputs(service, slots_parent_id: str, scope: DeleteScope) -> int:
    """Delete 'Deliverables Analysis' Google Docs created by test3.py."""
    print("\n" + "=" * 80)
    print("DELETING TEST3 OUTPUTS ('Deliverables Analysis' Google Docs)")
    print("=" * 80)

    deleted_count = 0

    for slot in iter_target_slots(service, slots_parent_id, scope.slot):
        selected_people = scope.selected_people if scope.slot and slot["id"] == scope.slot["id"] else None
        if scope.is_all_slots:
            selected_people = None

        for person in iter_target_people(service, slot, selected_people):
            doc = drive_find_child(service, person["id"], "Deliverables Analysis", GDOC_MIME)
            if doc:
                print(f"[{slot['name']}] {person['name']}")
                if drive_delete_file(service, doc["id"], "Deliverables Analysis"):
                    deleted_count += 1

    return deleted_count


def delete_test4_outputs(service, slots_parent_id: str, scope: DeleteScope) -> int:
    """
    Delete 'All Deliverables Analysis' Google Docs created by test4.py.

    Stored in:
      Candidate Result/<SlotName>/All Deliverables Analysis
    """
    print("\n" + "=" * 80)
    print("DELETING TEST4 OUTPUTS ('All Deliverables Analysis' Google Docs with tabs)")
    print("=" * 80)

    if scope.is_candidate_level:
        print("⚠ TEST4 is a SLOT-LEVEL file in Candidate Result/<Slot>/, not candidate-level.")
        print("⚠ You selected specific candidate(s), so TEST4 deletion is skipped.")
        return 0

    deleted_count = 0

    out_root = resolve_output_root_folder(service)
    if not out_root:
        print(f"⚠ Output root '{OUTPUT_ROOT_FOLDER_NAME}' not found. Skipping TEST4 deletions.")
        return 0

    for slot in iter_target_slots(service, slots_parent_id, scope.slot):
        out_slot = resolve_output_slot_folder_existing(service, slot["name"])
        if not out_slot:
            continue

        doc = drive_find_child(service, out_slot["id"], "All Deliverables Analysis", GDOC_MIME)
        if doc:
            print(f"[{slot['name']}] (Candidate Result)")
            if drive_delete_file(service, doc["id"], "All Deliverables Analysis"):
                deleted_count += 1

    return deleted_count


def delete_test5_outputs(service, slots_parent_id: str, scope: DeleteScope) -> int:
    """
    Delete 'Deliverables Analysis Sheet.xlsx' Excel files created by test5.py.

    Stored in:
      Candidate Result/<SlotName>/Deliverables Analysis Sheet.xlsx
    """
    print("\n" + "=" * 80)
    print("DELETING TEST5 OUTPUTS ('Deliverables Analysis Sheet.xlsx' Excel files)")
    print("=" * 80)

    if scope.is_candidate_level:
        print("⚠ TEST5 is a SLOT-LEVEL file in Candidate Result/<Slot>/, not candidate-level.")
        print("⚠ You selected specific candidate(s), so TEST5 deletion is skipped.")
        return 0

    deleted_count = 0

    out_root = resolve_output_root_folder(service)
    if not out_root:
        print(f"⚠ Output root '{OUTPUT_ROOT_FOLDER_NAME}' not found. Skipping TEST5 deletions.")
        return 0

    for slot in iter_target_slots(service, slots_parent_id, scope.slot):
        out_slot = resolve_output_slot_folder_existing(service, slot["name"])
        if not out_slot:
            continue

        xlsx_file = drive_find_child(service, out_slot["id"], "Deliverables Analysis Sheet.xlsx", XLSX_MIME)
        if xlsx_file:
            print(f"[{slot['name']}] (Candidate Result)")
            if drive_delete_file(service, xlsx_file["id"], "Deliverables Analysis Sheet.xlsx"):
                deleted_count += 1

    return deleted_count


def delete_test6_outputs(service, slots_parent_id: str, scope: DeleteScope) -> int:
    """
    Delete __EYE_* outputs created by test6.py.
    """
    print("\n" + "=" * 80)
    print("DELETING TEST6 OUTPUTS (__EYE_* eye/face tracking files)")
    print("=" * 80)

    deleted_count = 0
    pattern = re.compile(
        r".*__EYE_(annotated_h264\.mp4|summary\.json|result\.json|metrics\.csv)$",
        re.IGNORECASE,
    )

    for slot in iter_target_slots(service, slots_parent_id, scope.slot):
        selected_people = scope.selected_people if scope.slot and slot["id"] == scope.slot["id"] else None
        if scope.is_all_slots:
            selected_people = None

        for person in iter_target_people(service, slot, selected_people):
            for folder_name in FOLDER_NAMES_TO_PROCESS:
                target = drive_find_child(service, person["id"], folder_name, FOLDER_MIME)
                if not target:
                    continue

                files = list(drive_list_children(service, target["id"], None))
                eye_files = [
                    f
                    for f in files
                    if f.get("mimeType") != FOLDER_MIME and pattern.match(f.get("name") or "")
                ]

                if eye_files:
                    print(f"\n[{slot['name']}] {person['name']} > {folder_name}")
                    for f in eye_files:
                        if drive_delete_file(service, f["id"], f["name"]):
                            deleted_count += 1

    return deleted_count


# =========================
# Menu
# =========================
def show_delete_menu():
    print("\n" + "=" * 80)
    print("DELETE SCRIPT OUTPUTS FROM GOOGLE DRIVE")
    print("=" * 80)
    print("\nChoose which script outputs to delete:\n")

    for key, info in SCRIPT_OUTPUTS.items():
        print(f"  {key.upper():6} - {info['description']:50} {info['location']}")

    print(f"  {'ALL':6} - Delete outputs from ALL scripts above")
    print(f"  {'EXIT':6} - Exit without deleting anything\n")


def get_delete_choice() -> str:
    valid_choices = list(SCRIPT_OUTPUTS.keys()) + ["all", "exit"]
    while True:
        choice = input("Enter your choice (test1/test2/test3/test4/test5/test6/all/exit): ").strip().lower()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")


# =========================
# Main
# =========================
def main():
    service = get_drive_service()

    # Find 2026 root
    candidates = drive_search_folder_anywhere(service, ROOT_2026_FOLDER_NAME)
    if not candidates:
        print("Could not find folder '2026' in Drive.")
        return

    base_2026 = pick_best_named_folder(candidates)
    slots_parent_id = base_2026["id"]

    print(
        f"\nUsing 2026 folder: {base_2026['name']} "
        f"(id={base_2026['id']}, modified={base_2026.get('modifiedTime')})"
    )

    # Step 1: choose slot
    selected_slot = choose_slot(service, slots_parent_id)

    # Step 2: if one slot chosen, optionally choose one/multiple/all candidates
    if selected_slot is None:
        scope = DeleteScope(slot=None, selected_people=None)
    else:
        selected_people = choose_candidates_for_slot(service, selected_slot)
        scope = DeleteScope(slot=selected_slot, selected_people=selected_people)

    print("\n" + "=" * 80)
    print("CURRENT TARGET")
    print("=" * 80)
    print(scope.describe())

    # Step 3: choose what to delete
    show_delete_menu()
    choice = get_delete_choice()

    if choice == "exit":
        print("\n✓ Exiting without making any changes.")
        return

    # Confirm
    print("\n" + "=" * 80)
    print("CONFIRM DELETE")
    print("=" * 80)
    print(f"Target: {scope.describe()}")
    print(f"Delete mode: {choice.upper()}")
    if choice == "all":
        print("This will delete outputs from ALL test groups that apply to the selected scope.")
        print("NOTE: TEST4 and TEST5 are slot-level files inside Candidate Result/<Slot>/.")

    confirm = input("Type 'YES' to confirm: ").strip()
    if confirm != "YES":
        print("Cancelled. No files were deleted.")
        return

    total_deleted = 0

    if choice == "all":
        total_deleted += delete_test1_outputs(service, slots_parent_id, scope)
        time.sleep(0.5)
        total_deleted += delete_test2_outputs(service, slots_parent_id, scope)
        time.sleep(0.5)
        total_deleted += delete_test3_outputs(service, slots_parent_id, scope)
        time.sleep(0.5)
        total_deleted += delete_test4_outputs(service, slots_parent_id, scope)
        time.sleep(0.5)
        total_deleted += delete_test5_outputs(service, slots_parent_id, scope)
        time.sleep(0.5)
        total_deleted += delete_test6_outputs(service, slots_parent_id, scope)
    elif choice == "test1":
        total_deleted = delete_test1_outputs(service, slots_parent_id, scope)
    elif choice == "test2":
        total_deleted = delete_test2_outputs(service, slots_parent_id, scope)
    elif choice == "test3":
        total_deleted = delete_test3_outputs(service, slots_parent_id, scope)
    elif choice == "test4":
        total_deleted = delete_test4_outputs(service, slots_parent_id, scope)
    elif choice == "test5":
        total_deleted = delete_test5_outputs(service, slots_parent_id, scope)
    elif choice == "test6":
        total_deleted = delete_test6_outputs(service, slots_parent_id, scope)

    print("\n" + "=" * 80)
    print(f"✓ DELETION COMPLETE - {total_deleted} file(s) deleted")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()