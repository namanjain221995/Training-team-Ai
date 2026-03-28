#!/usr/bin/env python3
"""
Diagnostic script to debug Shared Drive folder access issues.
Run this to see exactly what the service account can see.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

load_dotenv()

SERVICE_ACCOUNT_FILE = Path(os.getenv("SERVICE_ACCOUNT_FILE") or "service-account.json")
SHARED_DRIVE_NAME = (os.getenv("SHARED_DRIVE_NAME") or "2026_Shared_Drive").strip()

print("=" * 80)
print("DIAGNOSTIC: Shared Drive Folder Access")
print("=" * 80)

# Authenticate
creds = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=["https://www.googleapis.com/auth/drive"]
)
service = build("drive", "v3", credentials=creds)
print(f"\n[✓] Authenticated as: {creds.service_account_email}")

# Step 1: List Shared Drives
print("\n" + "=" * 80)
print("STEP 1: List Shared Drives")
print("=" * 80)

drives_result = service.drives().list(pageSize=10).execute()
drives = drives_result.get('drives', [])

if not drives:
    print("[ERROR] No Shared Drives found!")
    print("[HINT] Is the service account a member of any Shared Drive?")
    exit(1)

print(f"\n[FOUND] {len(drives)} Shared Drive(s):")
for drive in drives:
    print(f"  - {drive['name']}")
    print(f"    ID: {drive['id']}")

# Step 2: Find the target Shared Drive
print("\n" + "=" * 80)
print(f"STEP 2: Find Shared Drive '{SHARED_DRIVE_NAME}'")
print("=" * 80)

target_drive = None
for drive in drives:
    if drive['name'] == SHARED_DRIVE_NAME:
        target_drive = drive
        break

if not target_drive:
    print(f"[ERROR] Could not find Shared Drive named '{SHARED_DRIVE_NAME}'")
    print(f"\nAvailable Shared Drives:")
    for drive in drives:
        print(f"  - {drive['name']}")
    exit(1)

drive_id = target_drive['id']
print(f"[✓] Found: {target_drive['name']}")
print(f"    ID: {drive_id}")

# Step 3: List contents of Shared Drive
print("\n" + "=" * 80)
print("STEP 3: List Contents of Shared Drive")
print("=" * 80)

# Method 1: Query with corpora=allDrives
print("\n[METHOD 1] Using corpora=allDrives (what the main script uses)...")
try:
    files_result = service.files().list(
        corpora="allDrives",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        q=f"'{drive_id}' in parents and trashed=false",
        fields="files(id,name,mimeType,size)",
        pageSize=50
    ).execute()
    
    files = files_result.get('files', [])
    print(f"[✓] Found {len(files)} item(s) in Shared Drive:")
    for f in files:
        mime = "📁 Folder" if f['mimeType'] == "application/vnd.google-apps.folder" else "📄 File"
        print(f"  {mime} {f['name']} (ID: {f['id'][:20]}...)")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    files = []

# Method 2: Query with just the drive ID (fallback)
if not files:
    print("\n[METHOD 2] Fallback: Using direct parent query...")
    try:
        files_result = service.files().list(
            supportsAllDrives=True,
            q=f"'{drive_id}' in parents and trashed=false",
            fields="files(id,name,mimeType,size)",
            pageSize=50
        ).execute()
        
        files = files_result.get('files', [])
        print(f"[✓] Found {len(files)} item(s):")
        for f in files:
            mime = "📁 Folder" if f['mimeType'] == "application/vnd.google-apps.folder" else "📄 File"
            print(f"  {mime} {f['name']} (ID: {f['id'][:20]}...)")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        files = []

# Step 4: Find 2026 folder
print("\n" + "=" * 80)
print("STEP 4: Search for '2026' Folder")
print("=" * 80)

folder_2026 = None
for f in files:
    if f['name'] == '2026' and f['mimeType'] == 'application/vnd.google-apps.folder':
        folder_2026 = f
        break

if folder_2026:
    print(f"[✓] Found '2026' folder!")
    print(f"    ID: {folder_2026['id']}")
    
    # Step 5: List contents of 2026 folder
    print("\n" + "=" * 80)
    print("STEP 5: List Contents of '2026' Folder")
    print("=" * 80)
    
    try:
        slot_result = service.files().list(
            supportsAllDrives=True,
            q=f"'{folder_2026['id']}' in parents and trashed=false",
            fields="files(id,name,mimeType)",
            pageSize=50
        ).execute()
        
        slots = slot_result.get('files', [])
        print(f"\n[✓] Found {len(slots)} item(s) in 2026 folder:")
        for slot in slots:
            mime = "📁 Folder" if slot['mimeType'] == "application/vnd.google-apps.folder" else "📄 File"
            print(f"  {mime} {slot['name']}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
else:
    print("[ERROR] Could not find '2026' folder in Shared Drive!")
    print(f"\nItems found at Shared Drive root:")
    if files:
        for f in files:
            mime = "📁 Folder" if f['mimeType'] == "application/vnd.google-apps.folder" else "📄 File"
            print(f"  {mime} {f['name']}")
    else:
        print("  (no items listed)")
    
    print("\n[POSSIBLE CAUSES]:")
    print("  1. '2026' folder was moved or deleted")
    print("  2. Service account permissions are limited")
    print("  3. Folder structure changed")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)