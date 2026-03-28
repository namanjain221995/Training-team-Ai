#!/usr/bin/env python3
"""
================================================================================
Smart Copy Script: Skip Duplicates & Resume
================================================================================

This IMPROVED version:
1. Checks if folders already exist in destination
2. Skips existing files (doesn't overwrite)
3. Only copies NEW files
4. Safe to run multiple times
5. Perfect for resuming incomplete copies

Usage:
1. Run the script
2. Paste your Google Drive folder link
3. Script copies only NEW files/folders
4. No duplicates created!

Author: TechSara Solutions
Date: 2026-03-21
================================================================================
"""

import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SERVICE_ACCOUNT_FILE = Path(os.getenv("SERVICE_ACCOUNT_FILE") or "service-account.json")
SHARED_DRIVE_NAME = (os.getenv("SHARED_DRIVE_NAME") or "2026_Shared_Drive").strip()
SCOPES = ["https://www.googleapis.com/auth/drive"]

FOLDER_MIME = "application/vnd.google-apps.folder"

# ==============================================================================
# LOGGING
# ==============================================================================

def log_info(msg: str):
    print(f"[INFO] {msg}")

def log_success(msg: str):
    print(f"[✓] {msg}")

def log_skip(msg: str):
    print(f"[⊘] {msg}")

def log_warn(msg: str):
    print(f"[WARN] {msg}")

def log_error(msg: str):
    print(f"[ERROR] {msg}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def extract_folder_id_from_link(drive_link: str) -> Optional[str]:
    """Extract folder ID from Google Drive link."""
    match = re.search(r'/folders/([a-zA-Z0-9-_]+)', drive_link)
    if match:
        return match.group(1)
    
    match = re.search(r'id=([a-zA-Z0-9-_]+)', drive_link)
    if match:
        return match.group(1)
    
    return None

def get_drive_service():
    """Authenticate with Google Drive."""
    print("\n" + "=" * 80)
    print("AUTHENTICATION")
    print("=" * 80)

    if not SERVICE_ACCOUNT_FILE.exists():
        raise FileNotFoundError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")

    creds = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=SCOPES,
    )

    print(f"[INFO] Service Account: {creds.service_account_email}")
    service = build("drive", "v3", credentials=creds)
    print("[✓] Authentication successful\n")
    return service

def find_shared_drive(service, drive_name: str) -> Optional[Dict]:
    """Find a Shared Drive by name."""
    try:
        result = service.drives().list(
            q=f"name = '{drive_name.replace(chr(39), chr(92) + chr(39))}'",
            pageSize=1,
        ).execute()

        drives = result.get("drives", [])
        return drives[0] if drives else None
    except Exception as e:
        log_error(f"Error finding Shared Drive: {e}")
        return None

def check_folder_access(service, folder_id: str) -> Optional[Dict]:
    """Check if service account can access a folder and get its metadata."""
    try:
        result = service.files().get(
            fileId=folder_id,
            fields="id, name, mimeType, owners",
            supportsAllDrives=True,
        ).execute()
        
        return result
    except HttpError as e:
        if e.resp.status == 404:
            log_error(f"Folder not found")
        elif e.resp.status == 403:
            log_error(f"Permission denied accessing folder")
        else:
            log_error(f"Error accessing folder: {e}")
        return None

def folder_exists_in_destination(service, parent_id: str, folder_name: str) -> Optional[str]:
    """
    Check if a folder with the same name already exists in destination.
    Returns folder ID if exists, None if doesn't exist.
    """
    safe_name = folder_name.replace("\\", "\\\\").replace("'", "\\'")
    query = f"'{parent_id}' in parents and name = '{safe_name}' and mimeType = '{FOLDER_MIME}' and trashed = false"

    try:
        result = service.files().list(
            q=query,
            spaces="drive",
            fields="files(id)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        files = result.get("files", [])
        return files[0]["id"] if files else None
    except:
        return None

def file_exists_in_destination(service, parent_id: str, file_name: str) -> bool:
    """Check if a file with the same name already exists in destination."""
    safe_name = file_name.replace("\\", "\\\\").replace("'", "\\'")
    query = f"'{parent_id}' in parents and name = '{safe_name}' and trashed = false"

    try:
        result = service.files().list(
            q=query,
            spaces="drive",
            fields="files(id)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        files = result.get("files", [])
        return len(files) > 0
    except:
        return False

def list_folder_contents(service, folder_id: str) -> List[Dict]:
    """List all files and folders in a given parent folder."""
    items = []
    page_token = None

    try:
        while True:
            result = service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageSize=100,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()

            items.extend(result.get("files", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        return items
    except Exception as e:
        log_error(f"Error listing folder contents: {e}")
        return []

def copy_folder_recursive(
    service,
    source_folder_id: str,
    dest_folder_id: str,
    folder_name: str,
    level: int = 0,
    stats: Dict = None,
) -> Dict:
    """Recursively copy folder structure and files, SKIPPING EXISTING ONES."""
    if stats is None:
        stats = {
            "folders_created": 0,
            "files_copied": 0,
            "folders_skipped": 0,
            "files_skipped": 0,
            "errors": 0,
        }

    indent = "  " * level

    # List contents
    contents = list_folder_contents(service, source_folder_id)

    if not contents:
        print(f"{indent}📁 {folder_name} (empty)")
        return stats

    # Separate folders and files
    folders = [f for f in contents if f["mimeType"] == FOLDER_MIME]
    files = [f for f in contents if f["mimeType"] != FOLDER_MIME]

    print(f"{indent}📁 {folder_name} ({len(folders)} folders, {len(files)} files)")

    # Copy folders recursively
    for folder in folders:
        folder_name = folder["name"]
        folder_id = folder["id"]

        # CHECK IF FOLDER ALREADY EXISTS
        existing_folder_id = folder_exists_in_destination(service, dest_folder_id, folder_name)
        
        if existing_folder_id:
            # Folder exists, skip creation but continue recursively
            log_skip(f"Folder exists: {folder_name}")
            stats["folders_skipped"] += 1
            dest_sub_folder_id = existing_folder_id
        else:
            # Folder doesn't exist, create it
            try:
                file_metadata = {
                    "name": folder_name,
                    "mimeType": FOLDER_MIME,
                    "parents": [dest_folder_id],
                }

                result = service.files().create(
                    body=file_metadata,
                    fields="id",
                    supportsAllDrives=True,
                ).execute()

                dest_sub_folder_id = result.get("id")
                stats["folders_created"] += 1
                log_success(f"Created folder: {folder_name}")
            except Exception as e:
                log_warn(f"Error creating folder {folder_name}: {e}")
                stats["errors"] += 1
                continue

        # Recursively copy contents
        copy_folder_recursive(
            service,
            folder_id,
            dest_sub_folder_id,
            folder_name,
            level + 1,
            stats,
        )

    # Copy files
    for file in files:
        file_name = file["name"]
        file_id = file["id"]

        # CHECK IF FILE ALREADY EXISTS
        if file_exists_in_destination(service, dest_folder_id, file_name):
            log_skip(f"File exists: {file_name}")
            stats["files_skipped"] += 1
            continue

        # File doesn't exist, copy it
        try:
            file_metadata = {
                "name": file_name,
                "parents": [dest_folder_id],
            }

            service.files().copy(
                fileId=file_id,
                body=file_metadata,
                supportsAllDrives=True,
                fields="id",
            ).execute()

            stats["files_copied"] += 1
            log_success(f"Copied file: {file_name}")
            time.sleep(0.2)  # Rate limiting
        except Exception as e:
            log_warn(f"Error copying file {file_name}: {e}")
            stats["errors"] += 1

    return stats

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main workflow."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Smart Copy: Skip Duplicates & Resume".center(78) + "║")
    print("║" + "  Safe to run multiple times!".center(78) + "║")
    print("║" + "  TechSara Solutions".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝\n")

    # STEP 1: Get folder link
    print("=" * 80)
    print("STEP 1: Provide Google Drive Folder Link")
    print("=" * 80)
    print("\nPaste your Google Drive folder link:\n")

    folder_link = input("Paste link here: ").strip()

    if not folder_link:
        log_error("No link provided")
        return

    # Extract folder ID
    folder_id = extract_folder_id_from_link(folder_link)
    if not folder_id:
        log_error("Could not extract folder ID from link")
        print("\nMake sure you pasted a valid Google Drive folder link")
        return

    log_success(f"Extracted folder ID: {folder_id}")

    # Authenticate
    service = get_drive_service()

    # STEP 2: Check access to source folder
    print("=" * 80)
    print("STEP 2: Checking Access to Source Folder")
    print("=" * 80)

    source_folder = check_folder_access(service, folder_id)
    if not source_folder:
        log_error("Cannot access the folder from the link")
        return

    source_folder_name = source_folder["name"]
    log_success(f"Can access folder: {source_folder_name}")

    # STEP 3: Find Shared Drive
    print("\n" + "=" * 80)
    print("STEP 3: Finding Shared Drive")
    print("=" * 80)

    shared_drive = find_shared_drive(service, SHARED_DRIVE_NAME)
    if not shared_drive:
        log_error(f"Could not find Shared Drive: {SHARED_DRIVE_NAME}")
        return

    shared_drive_id = shared_drive["id"]
    log_success(f"Found Shared Drive: {SHARED_DRIVE_NAME}")

    # STEP 4: Check if destination folder exists
    print("\n" + "=" * 80)
    print("STEP 4: Checking Destination Folder")
    print("=" * 80)

    dest_folder_id = folder_exists_in_destination(service, shared_drive_id, source_folder_name)
    
    if dest_folder_id:
        log_skip(f"Destination folder already exists: {source_folder_name}")
        log_info("Will skip existing files and only copy NEW files")
    else:
        # Create destination folder
        try:
            file_metadata = {
                "name": source_folder_name,
                "mimeType": FOLDER_MIME,
                "parents": [shared_drive_id],
            }

            result = service.files().create(
                body=file_metadata,
                fields="id",
                supportsAllDrives=True,
            ).execute()

            dest_folder_id = result.get("id")
            log_success(f"Created destination folder: {source_folder_name}")
        except Exception as e:
            log_error(f"Error creating destination folder: {e}")
            return

    # STEP 5: Smart copy
    print("\n" + "=" * 80)
    print("STEP 5: Smart Copy (Skipping Existing Files)")
    print("=" * 80)
    print(f"\nCopying from: {source_folder_name}")
    print(f"Destination: {SHARED_DRIVE_NAME}/{source_folder_name}")
    print("\nThis will skip any files/folders that already exist...\n")

    stats = copy_folder_recursive(
        service,
        folder_id,
        dest_folder_id,
        source_folder_name,
        level=0,
    )

    # STEP 6: Summary
    print("\n" + "=" * 80)
    print("COPY COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"  ✓ Folders created: {stats['folders_created']}")
    print(f"  ⊘ Folders skipped (already exist): {stats['folders_skipped']}")
    print(f"  ✓ Files copied: {stats['files_copied']}")
    print(f"  ⊘ Files skipped (already exist): {stats['files_skipped']}")
    print(f"  ⚠ Errors: {stats['errors']}")
    print(f"  📁 Total new items: {stats['folders_created'] + stats['files_copied']}")

    if stats['files_copied'] == 0 and stats['folders_created'] == 0:
        log_info("No new files to copy. Everything is already up to date!")
    
    if stats['errors'] > 0:
        log_warn(f"Completed with {stats['errors']} error(s)")

    print("\n" + "=" * 80)
    print("Safe to Run Again!")
    print("=" * 80)
    print("\nYou can run this script again anytime:")
    print("  • New files will be copied")
    print("  • Existing files will be skipped")
    print("  • No duplicates will be created")
    print("\nNext Steps:")
    print(f"  1. Run transcription: python transcription_fixed_shared_drive.py")
    print()

if __name__ == "__main__":
    main()