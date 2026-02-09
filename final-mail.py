import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
from email.utils import format_datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")


def must_get(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def send_email(subject: str, body: str) -> None:
    email_to = must_get("EMAIL_TO")
    email_from = must_get("EMAIL_FROM")

    host = must_get("SMTP_HOST")
    port = int((os.getenv("SMTP_PORT") or "587").strip())
    user = must_get("SMTP_USER")
    password = must_get("SMTP_PASS")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to

    # Proper RFC Email Date Header (IST)
    msg["Date"] = format_datetime(datetime.now(IST))

    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(user, password)
        smtp.send_message(msg)


def main():
    now_ist = datetime.now(IST)
    time_str = now_ist.strftime("%d-%m-%Y %I:%M %p")  # Example: 07-02-2026 02:42 AM

    slot = os.getenv("SLOT_CHOICE", "Not Provided")

    # Clean Professional Subject
    subject = f"Pipeline Execution Completed Successfully ✅ ({time_str} IST)"

    # Beautiful Informational Email Body
    body = f"""
Hello,

This is an automated notification to confirm that the pipeline execution has completed successfully.

===========================
        Information
===========================

Status       : SUCCESS ✅
Completed At : {time_str} (IST)
Slot Choice  : {slot}

===========================

No further action is required.

Regards,  
Pipeline Automation System
"""

    send_email(subject, body)
    print("[MAIL] Success email sent to", os.getenv("EMAIL_TO"))


if __name__ == "__main__":
    main()
