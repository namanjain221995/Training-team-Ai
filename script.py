import re
from pathlib import Path

ENV_PATH = Path(".env")

# ===== HARD-CODE SLOT HERE =====
SLOT_CHOICE = "1"   # change this to 1/2/3..
# ===============================

def update_env_var(path: Path, key: str, value: str) -> None:
    if not path.exists():
        path.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    content = path.read_text(encoding="utf-8")

    # Match lines like: SLOT_CHOICE=..., export SLOT_CHOICE=..., SLOT_CHOICE = ...
    pattern = re.compile(rf"^(?:export\s+)?{re.escape(key)}\s*=\s*.*$", re.MULTILINE)

    if pattern.search(content):
        content = pattern.sub(f"{key}={value}", content)
    else:
        if not content.endswith("\n"):
            content += "\n"
        content += f"{key}={value}\n"

    path.write_text(content, encoding="utf-8")

def main():
    update_env_var(ENV_PATH, "SLOT_CHOICE", SLOT_CHOICE)
    print(f" Updated .env -> SLOT_CHOICE={SLOT_CHOICE}")

if __name__ == "__main__":
    main()
