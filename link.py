from __future__ import annotations

import re
from typing import Iterable

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ===== USER SETTINGS =====
SERVICE_ACCOUNT_FILE = "service-account.json"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1hb3mRsR7O8zVFZxPAf9Q0ogGlehyYjHjZW20xm7vStM/edit?usp=sharing"
WORKSHEET_NAME = "Slot8"

# Share the destination folder with the service account and paste that folder ID here.
# Example: https://drive.google.com/drive/folders/<THIS_PART_IS_THE_ID>
DESTINATION_PARENT_FOLDER_ID = "1GRwwb95GK2lv6XUM1gUn3X_j64v6dTaO"

# Optional: create one slot folder first, then candidates inside it.
CREATE_SLOT_FOLDER = True
SLOT_FOLDER_NAME = "8. March 8th Slot (March 30 to April 10)"

FOLDERS_TO_CREATE = [
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
# =========================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive",
]


def get_clients():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    return gc, drive


def clean_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def get_candidate_names(gc) -> list[str]:
    sh = gc.open_by_url(SPREADSHEET_URL)
    ws = sh.worksheet(WORKSHEET_NAME)
    names = ws.col_values(1)
    cleaned = []
    seen = set()
    for raw in names:
        name = clean_name(raw)
        if not name:
            continue
        # skip header-like rows if present
        if name.lower() in {"candidate name", "name", "candidates"}:
            continue
        if name not in seen:
            seen.add(name)
            cleaned.append(name)
    return cleaned


def find_folder(drive, name: str, parent_id: str | None = None) -> str | None:
    safe_name = name.replace("'", "\\'")
    query_parts = [
        "mimeType='application/vnd.google-apps.folder'",
        f"name='{safe_name}'",
        "trashed=false",
    ]
    if parent_id:
        query_parts.append(f"'{parent_id}' in parents")
    resp = drive.files().list(
        q=" and ".join(query_parts),
        spaces="drive",
        fields="files(id,name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def create_folder(drive, name: str, parent_id: str) -> str:
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    file = drive.files().create(
        body=metadata,
        fields="id,name",
        supportsAllDrives=True,
    ).execute()
    return file["id"]


def get_or_create_folder(drive, name: str, parent_id: str) -> tuple[str, bool]:
    existing = find_folder(drive, name, parent_id)
    if existing:
        return existing, False
    return create_folder(drive, name, parent_id), True


def main() -> None:
    if DESTINATION_PARENT_FOLDER_ID == "PASTE_PARENT_FOLDER_ID_HERE":
        raise ValueError("Please set DESTINATION_PARENT_FOLDER_ID before running the script.")

    gc, drive = get_clients()
    candidate_names = get_candidate_names(gc)

    if not candidate_names:
        raise RuntimeError("No candidate names found in column A of sheet 'slot8'.")

    parent_id = DESTINATION_PARENT_FOLDER_ID
    slot_created = False

    if CREATE_SLOT_FOLDER:
        parent_id, slot_created = get_or_create_folder(drive, SLOT_FOLDER_NAME, DESTINATION_PARENT_FOLDER_ID)

    print(f"Found {len(candidate_names)} candidate names in worksheet '{WORKSHEET_NAME}'.")
    if CREATE_SLOT_FOLDER:
        print(f"Slot folder: {SLOT_FOLDER_NAME} -> {'created' if slot_created else 'already exists'}")

    created_candidates = 0
    existing_candidates = 0
    created_subfolders = 0
    existing_subfolders = 0

    for candidate in candidate_names:
        candidate_folder_id, was_created = get_or_create_folder(drive, candidate, parent_id)
        if was_created:
            created_candidates += 1
            print(f"\nCandidate folder created: {candidate}")
        else:
            existing_candidates += 1
            print(f"\nCandidate folder exists: {candidate}")

        for subfolder in FOLDERS_TO_CREATE:
            _, child_created = get_or_create_folder(drive, subfolder, candidate_folder_id)
            if child_created:
                created_subfolders += 1
                print(f"  Created -> {subfolder}")
            else:
                existing_subfolders += 1
                print(f"  Already exists -> {subfolder}")

    print("\n" + "=" * 90)
    print("GOOGLE DRIVE FOLDER CREATION COMPLETE")
    print("=" * 90)
    print(f"Candidate folders created: {created_candidates}")
    print(f"Candidate folders already existing: {existing_candidates}")
    print(f"Subfolders created: {created_subfolders}")
    print(f"Subfolders already existing: {existing_subfolders}")


if __name__ == "__main__":
    main()
