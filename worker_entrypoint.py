import os
import subprocess
import boto3


def must(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def ssm_get(region: str, name: str) -> str:
    ssm = boto3.client("ssm", region_name=region)
    return ssm.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]


def main():
    region = must("AWS_REGION")
    video_file_id = must("VIDEO_FILE_ID")
    target_folder_id = must("TARGET_FOLDER_ID")
    video_name = os.getenv("VIDEO_NAME", "").strip()

    env_param = os.getenv("SSM_ENV_PARAM", "/transcription/worker/env").strip()
    service_account_param = os.getenv(
        "SSM_SERVICE_ACCOUNT_PARAM",
        "/transcription/worker/service_account_json",
    ).strip()

    os.makedirs("/app/state", exist_ok=True)

    # Write worker .env from SSM
    with open("/app/.env", "w", encoding="utf-8") as f:
        f.write(ssm_get(region, env_param))

    # Write service account JSON from SSM
    with open("/app/service-account.json", "w", encoding="utf-8") as f:
        f.write(ssm_get(region, service_account_param))

    # Force service-account mode for downstream script
    os.environ["AUTH_MODE"] = "service_account"
    os.environ["SERVICE_ACCOUNT_FILE"] = "/app/service-account.json"
    os.environ["HEADLESS_AUTH"] = "1"

    # Keep task context
    os.environ["VIDEO_FILE_ID"] = video_file_id
    os.environ["TARGET_FOLDER_ID"] = target_folder_id
    if video_name:
        os.environ["VIDEO_NAME"] = video_name

    print(f"[WORKER] Starting test2.py for {video_name or video_file_id}")
    subprocess.run(["python", "-u", "test2.py"], check=True)


if __name__ == "__main__":
    main()