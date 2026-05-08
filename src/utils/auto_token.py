"""
Fully Automated Zerodha Token Generation

No browser needed. Uses:
  - Zerodha user credentials
  - TOTP secret for 2FA
  - Kite Connect API

Run manually or via cron:
  python src/utils/auto_token.py

Secrets needed in GCP Secret Manager:
  KITE_API_KEY
  KITE_API_SECRET
  ZERODHA_USER_ID
  ZERODHA_PASSWORD
  ZERODHA_TOTP_SECRET
"""

import re
import time
import pyotp
import requests
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from google.cloud import secretmanager

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID = "hedge-fund-494103"
VM_NAME    = "data-collector"
VM_ZONE    = "asia-south1-c"

# GCloud path for VM restart
GCLOUD_PATH = r"C:\Users\abhim\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"


# ─────────────────────────────────────
# Secret Manager helpers
# ─────────────────────────────────────
def get_secret(secret_id: str) -> str:
    client   = secretmanager.SecretManagerServiceClient()
    name     = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8").strip()


def store_secret(secret_id: str, value: str):
    client  = secretmanager.SecretManagerServiceClient()
    parent  = f"projects/{PROJECT_ID}/secrets/{secret_id}"
    client.add_secret_version(
        request={
            "parent":  parent,
            "payload": {"data": value.encode("utf-8")}
        }
    )
    print(f"Secret {secret_id} updated.")


# ─────────────────────────────────────
# Automated login
# ─────────────────────────────────────
def generate_totp(totp_secret: str) -> str:
    """Generate current TOTP code from secret."""
    totp = pyotp.TOTP(totp_secret)
    return totp.now()


def automated_login(api_key: str, api_secret: str,
                    user_id: str, password: str,
                    totp_secret: str) -> str:
    """
    Fully automated Zerodha login.
    Returns access_token.
    """
    session = requests.Session()

    # Step 1: Get login page (establish session)
    login_url = f"https://kite.zerodha.com/api/login"
    print("Step 1: Logging in...")

    r = session.post(login_url, data={
        "user_id":  user_id,
        "password": password,
    })

    if r.status_code != 200:
        raise Exception(f"Login failed: {r.status_code} {r.text}")

    data = r.json()
    if data.get("status") != "success":
        raise Exception(f"Login error: {data}")

    request_id = data["data"]["request_id"]
    print(f"  Login OK, request_id: {request_id}")

    # Step 2: Submit TOTP
    print("Step 2: Submitting TOTP...")
    totp_code = generate_totp(totp_secret)
    print(f"  TOTP: {totp_code}")

    r = session.post("https://kite.zerodha.com/api/twofa", data={
        "user_id":    user_id,
        "request_id": request_id,
        "twofa_value": totp_code,
        "twofa_type": "totp",
        "skip_session": "",
    })

    if r.status_code != 200:
        raise Exception(f"TOTP failed: {r.status_code} {r.text}")

    data = r.json()
    if data.get("status") != "success":
        raise Exception(f"TOTP error: {data}")

    print("  TOTP OK")

    # Step 3: Get request token via Kite Connect login URL
    print("Step 3: Getting request token...")
    kite      = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    r = session.get(login_url, allow_redirects=False)

    # Follow redirects manually to capture request_token
    location = r.headers.get("Location", "")
    max_redirects = 5
    while "request_token" not in location and max_redirects > 0:
        if not location:
            # Try to get it from response
            r = session.get(login_url, allow_redirects=True)
            location = r.url
            break
        r = session.get(location, allow_redirects=False)
        location = r.headers.get("Location", r.url)
        max_redirects -= 1

    # Extract request_token from URL
    match = re.search(r"request_token=([^&]+)", location)
    if not match:
        # Try the final URL
        match = re.search(r"request_token=([^&]+)", r.url)
    if not match:
        raise Exception(
            f"Could not find request_token in URL: {location}"
        )

    request_token = match.group(1)
    print(f"  request_token: {request_token[:10]}...")

    # Step 4: Generate session
    print("Step 4: Generating session...")
    data         = kite.generate_session(
        request_token, api_secret=api_secret
    )
    access_token = data["access_token"]
    print(f"  access_token: {access_token[:10]}...")

    return access_token


# ─────────────────────────────────────
# Restart VM collector
# ─────────────────────────────────────
def restart_collector():
    """Restart collector on VM."""
    print("\nRestarting collector on VM...")
    try:
        result = subprocess.run(
            [
                GCLOUD_PATH,
                "compute", "ssh", VM_NAME,
                f"--zone={VM_ZONE}",
                f"--project={PROJECT_ID}",
                "--command=sudo systemctl reset-failed collector && "
                          "sudo systemctl restart collector && "
                          "sleep 3 && "
                          "sudo systemctl status collector --no-pager | head -5"
            ],
            capture_output = True,
            text           = True,
            timeout        = 30,
            encoding       = "utf-8",
            errors         = "replace",
        )
        if result.returncode == 0:
            print("Collector restarted successfully")
            print(result.stdout[:200])
        else:
            print(f"Restart failed: {result.stderr[:100]}")
            print("Restart manually:")
            print(f"  gcloud compute ssh {VM_NAME} --zone={VM_ZONE}")
            print(f"  sudo systemctl reset-failed collector")
            print(f"  sudo systemctl restart collector")
    except Exception as e:
        print(f"Restart error: {e}")


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────
def run_auto_token():
    """
    Fully automated token refresh.
    No user interaction needed.
    """
    print("=" * 55)
    print("AUTOMATED TOKEN REFRESH")
    print("=" * 55)

    # Load secrets
    print("\nLoading credentials from Secret Manager...")
    api_key      = get_secret("KITE_API_KEY")
    api_secret   = get_secret("KITE_API_SECRET")
    user_id      = get_secret("ZERODHA_USER_ID")
    password     = get_secret("ZERODHA_PASSWORD")
    totp_secret  = get_secret("ZERODHA_TOTP_SECRET")
    print("Credentials loaded.")

    # Generate token
    access_token = automated_login(
        api_key, api_secret,
        user_id, password, totp_secret
    )

    # Save token
    print("\nSaving token to Secret Manager...")
    store_secret("KITE_ACCESS_TOKEN", access_token)

    # Save to .env
    env_path = ENV_PATH
    lines    = []
    if env_path.exists():
        with open(env_path) as f:
            lines = f.readlines()
    with open(env_path, "w") as f:
        found = False
        for line in lines:
            if line.startswith("KITE_ACCESS_TOKEN="):
                f.write(f"KITE_ACCESS_TOKEN={access_token}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"KITE_ACCESS_TOKEN={access_token}\n")

    print("Token saved.")

    # Restart collector
    restart_collector()

    print("\n" + "=" * 55)
    print("DONE — Token refreshed automatically")
    print("=" * 55)

    return access_token


if __name__ == "__main__":
    run_auto_token()