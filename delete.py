import re
import time
from pathlib import Path
from typing import Optional, List

from googleapiclient.discovery import build
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
        "location": "NOW STORED in Candidate Result/<Slot>/ (NOT in 2026/<Slot>/)",
    },
    "test5": {
        "description": "'Deliverables Analysis Sheet.xlsx' Excel files",
        "filename": "Deliverables Analysis Sheet.xlsx",
        "mime_types": [XLSX_MIME],
        "location": "NOW STORED in Candidate Result/<Slot>/ (NOT in 2026/<Slot>/)",
    },
    "test6": {
        "description": "Eye/Face tracking outputs (__EYE_*)",
        "pattern": r".*__EYE_(annotated_h264\.mp4|summary\.json|result\.json|metrics\.csv)$",
        "mime_types": ["video/mp4", "application/json", "text/csv"],
        "location": "in 2026/<Slot>/<Person>/<FolderName>/ (same folder as input video)",
    },
}


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
        res = (
            service.files()
            .list(
                q=q,
                fields="nextPageToken, files(id,name,parents,modifiedTime)",
                pageSize=100,
                pageToken=page_token,
                **_list_kwargs(),
            )
            .execute()
        )

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
        res = (
            service.files()
            .list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
                pageSize=1000,
                pageToken=page_token,
                **_list_kwargs(),
            )
            .execute()
        )

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

    res = (
        service.files()
        .list(q=q, fields="files(id,name,mimeType,modifiedTime)", pageSize=50, **_list_kwargs())
        .execute()
    )
    files = res.get("files", []) or []
    if not files:
        return None
    # pick most recent if duplicates
    files.sort(key=lambda x: (x.get("modifiedTime") or ""), reverse=True)
    return files[0]


def drive_delete_file(service, file_id: str, file_name: str) -> bool:
    """Delete a file by ID."""
    try:
        service.files().delete(fileId=file_id, **_delete_kwargs()).execute()
        print(f"  ✓ Deleted: {file_name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to delete {file_name}: {e}")
        return False


# =========================
# Output location helpers (ONLY for test4/test5)
# =========================
def drive_get_folder_by_id(service, folder_id: str) -> dict:
    return (
        service.files()
        .get(
            fileId=folder_id,
            fields="id,name,mimeType,parents,modifiedTime",
            **_list_kwargs(),
        )
        .execute()
    )


def resolve_output_root_folder(service) -> Optional[dict]:
    """
    Resolve the OUTPUT root folder (Candidate Result).
    Returns folder dict or None if not found.
    """
    if OUTPUT_ROOT_FOLDER_ID:
        try:
            root = drive_get_folder_by_id(service, OUTPUT_ROOT_FOLDER_ID)
            if root.get("mimeType") != FOLDER_MIME:
                print(" ⚠ OUTPUT_ROOT_FOLDER_ID is not a folder. Ignoring.")
                return None
            return root
        except Exception as e:
            print(f" ⚠ Failed to fetch OUTPUT_ROOT_FOLDER_ID: {e}")
            return None

    cands = drive_search_folder_anywhere(service, OUTPUT_ROOT_FOLDER_NAME)
    if not cands:
        return None
    return pick_best_named_folder(cands)


def resolve_output_slot_folder_existing(service, slot_name: str) -> Optional[dict]:
    """
    Returns Candidate Result/<SlotName> folder if it exists.
    IMPORTANT: this function does NOT create folders (safe for deletion script).
    """
    out_root = resolve_output_root_folder(service)
    if not out_root:
        return None
    slot_folder = drive_find_child(service, out_root["id"], slot_name, FOLDER_MIME)
    return slot_folder


# =========================
# Slot selection helpers
# =========================
def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    """List slot folders under 2026 root."""
    return sorted(
        list(drive_list_children(service, slots_parent_id, FOLDER_MIME)),
        key=lambda x: (x.get("name") or "").lower(),
    )


def choose_slot(service, slots_parent_id: str) -> Optional[dict]:
    """
    Let user pick a slot folder.
    Returns:
      - dict for selected slot
      - None if user chooses ALL
    """
    slots = list_slot_folders(service, slots_parent_id)
    if not slots:
        print(" No slot folders found under '2026'.")
        return None

    print("\n" + "=" * 80)
    print("SELECT SLOT FOLDER")
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
        print(" Invalid choice. Try again.")


def iter_target_slots(service, slots_parent_id: str, only_slot: Optional[dict]):
    """Yield selected slot only, or all slots."""
    if only_slot:
        yield only_slot
    else:
        for s in list_slot_folders(service, slots_parent_id):
            yield s


# =========================
# Deletion functions for each script
# =========================
def delete_test1_outputs(service, slots_parent_id: str, only_slot: Optional[dict]) -> int:
    """Delete transcript .txt files created by test.py (within selected slot)."""
    print("\n" + "=" * 80)
    print("DELETING TEST1 OUTPUTS (Transcript .txt files)")
    print("=" * 80)

    deleted_count = 0
    pattern = re.compile(r"^[^/]*\.txt$", re.IGNORECASE)

    for slot in iter_target_slots(service, slots_parent_id, only_slot):
        people = sorted(
            list(drive_list_children(service, slot["id"], FOLDER_MIME)),
            key=lambda x: (x.get("name") or "").lower(),
        )

        for person in people:
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


def delete_test2_outputs(service, slots_parent_id: str, only_slot: Optional[dict]) -> int:
    """Delete LLM_OUTPUT__*.txt files created by test2.py (within selected slot)."""
    print("\n" + "=" * 80)
    print("DELETING TEST2 OUTPUTS (LLM_OUTPUT__*.txt files)")
    print("=" * 80)

    deleted_count = 0
    pattern = re.compile(r"^LLM_OUTPUT__.*\.txt$", re.IGNORECASE)

    for slot in iter_target_slots(service, slots_parent_id, only_slot):
        people = sorted(
            list(drive_list_children(service, slot["id"], FOLDER_MIME)),
            key=lambda x: (x.get("name") or "").lower(),
        )

        for person in people:
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


def delete_test3_outputs(service, slots_parent_id: str, only_slot: Optional[dict]) -> int:
    """Delete 'Deliverables Analysis' Google Docs created by test3.py (within selected slot)."""
    print("\n" + "=" * 80)
    print("DELETING TEST3 OUTPUTS ('Deliverables Analysis' Google Docs)")
    print("=" * 80)

    deleted_count = 0

    for slot in iter_target_slots(service, slots_parent_id, only_slot):
        people = sorted(
            [
                p
                for p in drive_list_children(service, slot["id"], FOLDER_MIME)
                if (p.get("name") or "").strip() not in SKIP_PERSON_FOLDERS
            ],
            key=lambda x: (x.get("name") or "").lower(),
        )

        for person in people:
            doc = drive_find_child(service, person["id"], "Deliverables Analysis", GDOC_MIME)
            if doc:
                print(f"[{slot['name']}] {person['name']}")
                if drive_delete_file(service, doc["id"], "Deliverables Analysis"):
                    deleted_count += 1

    return deleted_count


def delete_test4_outputs(service, slots_parent_id: str, only_slot: Optional[dict]) -> int:
    """
    Delete 'All Deliverables Analysis' Google Docs created by test4.py.

    Stored in:
      Candidate Result/<SlotName>/All Deliverables Analysis
    """
    print("\n" + "=" * 80)
    print("DELETING TEST4 OUTPUTS ('All Deliverables Analysis' Google Docs with tabs)")
    print("=" * 80)

    deleted_count = 0

    out_root = resolve_output_root_folder(service)
    if not out_root:
        print(f" ⚠ Output root '{OUTPUT_ROOT_FOLDER_NAME}' not found. Skipping TEST4 deletions.")
        return 0

    for slot in iter_target_slots(service, slots_parent_id, only_slot):
        out_slot = resolve_output_slot_folder_existing(service, slot["name"])
        if not out_slot:
            continue

        doc = drive_find_child(service, out_slot["id"], "All Deliverables Analysis", GDOC_MIME)
        if doc:
            print(f"[{slot['name']}] (Candidate Result)")
            if drive_delete_file(service, doc["id"], "All Deliverables Analysis"):
                deleted_count += 1

    return deleted_count


def delete_test5_outputs(service, slots_parent_id: str, only_slot: Optional[dict]) -> int:
    """
    Delete 'Deliverables Analysis Sheet.xlsx' Excel files created by test5.py.

    Stored in:
      Candidate Result/<SlotName>/Deliverables Analysis Sheet.xlsx
    """
    print("\n" + "=" * 80)
    print("DELETING TEST5 OUTPUTS ('Deliverables Analysis Sheet.xlsx' Excel files)")
    print("=" * 80)

    deleted_count = 0

    out_root = resolve_output_root_folder(service)
    if not out_root:
        print(f" ⚠ Output root '{OUTPUT_ROOT_FOLDER_NAME}' not found. Skipping TEST5 deletions.")
        return 0

    for slot in iter_target_slots(service, slots_parent_id, only_slot):
        out_slot = resolve_output_slot_folder_existing(service, slot["name"])
        if not out_slot:
            continue

        xlsx_file = drive_find_child(service, out_slot["id"], "Deliverables Analysis Sheet.xlsx", XLSX_MIME)
        if xlsx_file:
            print(f"[{slot['name']}] (Candidate Result)")
            if drive_delete_file(service, xlsx_file["id"], "Deliverables Analysis Sheet.xlsx"):
                deleted_count += 1

    return deleted_count


def delete_test6_outputs(service, slots_parent_id: str, only_slot: Optional[dict]) -> int:
    """
    Delete __EYE_* outputs created by test6.py inside:
      2026/<Slot>/<Person>/<FolderName>/
    """
    print("\n" + "=" * 80)
    print("DELETING TEST6 OUTPUTS (__EYE_* eye/face tracking files)")
    print("=" * 80)

    deleted_count = 0
    pattern = re.compile(
        r".*__EYE_(annotated_h264\.mp4|summary\.json|result\.json|metrics\.csv)$",
        re.IGNORECASE,
    )

    for slot in iter_target_slots(service, slots_parent_id, only_slot):
        people = sorted(
            [
                p for p in drive_list_children(service, slot["id"], FOLDER_MIME)
                if (p.get("name") or "").strip() not in SKIP_PERSON_FOLDERS
            ],
            key=lambda x: (x.get("name") or "").lower(),
        )

        for person in people:
            for folder_name in FOLDER_NAMES_TO_PROCESS:
                target = drive_find_child(service, person["id"], folder_name, FOLDER_MIME)
                if not target:
                    continue

                files = list(drive_list_children(service, target["id"], None))
                eye_files = [
                    f for f in files
                    if f.get("mimeType") != FOLDER_MIME
                    and pattern.match(f.get("name") or "")
                ]

                if eye_files:
                    print(f"\n[{slot['name']}] {person['name']} > {folder_name}")
                    for f in eye_files:
                        if drive_delete_file(service, f["id"], f["name"]):
                            deleted_count += 1

    return deleted_count


# =========================
# Menu & Main
# =========================
def show_menu():
    """Display the deletion menu."""
    print("\n" + "=" * 80)
    print("DELETE SCRIPT OUTPUTS FROM GOOGLE DRIVE")
    print("=" * 80)
    print("\nChoose which script outputs to delete:\n")

    for key, info in SCRIPT_OUTPUTS.items():
        print(f"  {key.upper():6} - {info['description']:50} {info['location']}")

    print(f"  {'ALL':6} - Delete outputs from ALL scripts above")
    print(f"  {'EXIT':6} - Exit without deleting anything\n")


def get_user_choice() -> str:
    """Get valid user input."""
    valid_choices = list(SCRIPT_OUTPUTS.keys()) + ["all", "exit"]
    while True:
        choice = input("Enter your choice (test1/test2/test3/test4/test5/test6/all/exit): ").strip().lower()
        if choice in valid_choices:
            return choice
        print(f" Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def main():
    service = get_drive_service()

    # Find 2026
    candidates = drive_search_folder_anywhere(service, ROOT_2026_FOLDER_NAME)
    if not candidates:
        print(" Could not find folder '2026' in Drive.")
        return

    base_2026 = pick_best_named_folder(candidates)
    slots_parent_id = base_2026["id"]

    print(f"\nUsing 2026 folder: {base_2026['name']} (id={base_2026['id']}, modified={base_2026.get('modifiedTime')})")

    show_menu()
    choice = get_user_choice()

    if choice == "exit":
        print("\n✓ Exiting without making any changes.")
        return

    # Ask which slot folder to target
    selected_slot = choose_slot(service, slots_parent_id)

    total_deleted = 0

    if choice == "all":
        print("\n  YOU ARE ABOUT TO DELETE ALL SCRIPT OUTPUTS (within selected slot/all slots)!")
        print("  NOTE: TEST4 & TEST5 outputs are deleted from Candidate Result/<Slot>/ (not from 2026).")
        confirm = input("Type 'YES' to confirm: ").strip()
        if confirm != "YES":
            print(" Cancelled. No files were deleted.")
            return

        total_deleted += delete_test1_outputs(service, slots_parent_id, selected_slot)
        time.sleep(0.5)
        total_deleted += delete_test2_outputs(service, slots_parent_id, selected_slot)
        time.sleep(0.5)
        total_deleted += delete_test3_outputs(service, slots_parent_id, selected_slot)
        time.sleep(0.5)
        total_deleted += delete_test4_outputs(service, slots_parent_id, selected_slot)
        time.sleep(0.5)
        total_deleted += delete_test5_outputs(service, slots_parent_id, selected_slot)
        time.sleep(0.5)
        total_deleted += delete_test6_outputs(service, slots_parent_id, selected_slot)

    else:
        if choice == "test1":
            total_deleted = delete_test1_outputs(service, slots_parent_id, selected_slot)
        elif choice == "test2":
            total_deleted = delete_test2_outputs(service, slots_parent_id, selected_slot)
        elif choice == "test3":
            total_deleted = delete_test3_outputs(service, slots_parent_id, selected_slot)
        elif choice == "test4":
            total_deleted = delete_test4_outputs(service, slots_parent_id, selected_slot)
        elif choice == "test5":
            total_deleted = delete_test5_outputs(service, slots_parent_id, selected_slot)
        elif choice == "test6":
            total_deleted = delete_test6_outputs(service, slots_parent_id, selected_slot)

    print("\n" + "=" * 80)
    print(f"✓ DELETION COMPLETE - {total_deleted} file(s) deleted")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
