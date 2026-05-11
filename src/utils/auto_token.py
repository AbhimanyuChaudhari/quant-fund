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

Compatible with:
  - Windows (laptop) — uses gcloud.cmd, SSH to restart collector
  - Linux VM        — restarts collector directly via systemctl
"""

import os
import re
import sys
import time
import platform
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

# ─── Environment detection ────────────────────────────────────────────────────

def is_running_on_vm() -> bool:
    """
    Detect if we're running on the GCP VM or on a local laptop.
    Checks hostname and platform.
    """
    hostname = os.uname().nodename if hasattr(os, "uname") else ""
    is_linux = platform.system() == "Linux"
    is_vm    = is_linux and (
        "data-collector" in hostname or
        "google" in hostname.lower() or
        os.path.exists("/etc/google_cloud")   # GCP marker file
    )
    return is_vm


def get_gcloud_path() -> str:
    """Get correct gcloud path for current OS."""
    if platform.system() == "Windows":
        # Common Windows install locations
        candidates = [
            r"C:\Users\abhim\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            r"C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return "gcloud"   # fallback — hope it's in PATH
    else:
        # Linux/Mac — gcloud should be in PATH
        return "gcloud"


# ─── Secret Manager helpers ───────────────────────────────────────────────────

def get_secret(secret_id: str) -> str:
    client   = secretmanager.SecretManagerServiceClient()
    name     = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8").strip()


def store_secret(secret_id: str, value: str):
    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{PROJECT_ID}/secrets/{secret_id}"
    client.add_secret_version(
        request={
            "parent":  parent,
            "payload": {"data": value.encode("utf-8")}
        }
    )
    print(f"Secret {secret_id} updated.")


# ─── Automated login ──────────────────────────────────────────────────────────

def generate_totp(totp_secret: str) -> str:
    """Generate current TOTP code from secret."""
    totp = pyotp.TOTP(totp_secret)
    return totp.now()


def automated_login(
    api_key:     str,
    api_secret:  str,
    user_id:     str,
    password:    str,
    totp_secret: str,
) -> str:
    """
    Fully automated Zerodha login.
    Returns access_token.
    """
    session = requests.Session()

    # Step 1: Login
    print("Step 1: Logging in...")
    r = session.post("https://kite.zerodha.com/api/login", data={
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
        "user_id":      user_id,
        "request_id":   request_id,
        "twofa_value":  totp_code,
        "twofa_type":   "totp",
        "skip_session": "",
    })

    if r.status_code != 200:
        raise Exception(f"TOTP failed: {r.status_code} {r.text}")

    data = r.json()
    if data.get("status") != "success":
        raise Exception(f"TOTP error: {data}")

    print("  TOTP OK")

    # Step 3: Get request token
    print("Step 3: Getting request token...")
    kite      = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    r = session.get(login_url, allow_redirects=False)

    location     = r.headers.get("Location", "")
    max_redirects = 5
    while "request_token" not in location and max_redirects > 0:
        if not location:
            r        = session.get(login_url, allow_redirects=True)
            location = r.url
            break
        r        = session.get(location, allow_redirects=False)
        location = r.headers.get("Location", r.url)
        max_redirects -= 1

    match = re.search(r"request_token=([^&]+)", location)
    if not match:
        match = re.search(r"request_token=([^&]+)", r.url)
    if not match:
        raise Exception(f"Could not find request_token in URL: {location}")

    request_token = match.group(1)
    print(f"  request_token: {request_token[:10]}...")

    # Step 4: Generate session
    print("Step 4: Generating session...")
    data         = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    print(f"  access_token: {access_token[:10]}...")

    return access_token


# ─── Collector restart ────────────────────────────────────────────────────────

def restart_collector():
    """
    Restart collector service.
    On VM: directly via systemctl (no SSH needed).
    On laptop: SSH into VM via gcloud.
    """
    print("\nRestarting collector on VM...")

    if is_running_on_vm():
        # ── On VM — restart directly ──
        try:
            subprocess.run(
                ["sudo", "systemctl", "reset-failed", "collector"],
                check=True, capture_output=True
            )
            subprocess.run(
                ["sudo", "systemctl", "restart", "collector"],
                check=True, capture_output=True
            )
            time.sleep(3)
            result = subprocess.run(
                ["sudo", "systemctl", "status", "collector",
                 "--no-pager", "--lines=5"],
                capture_output=True, text=True
            )
            print("Collector restarted successfully")
            # Show just the active/inactive line
            for line in result.stdout.split("\n"):
                if "Active:" in line or "active" in line.lower():
                    print(f"  {line.strip()}")
                    break
        except Exception as e:
            print(f"Restart error: {e}")
            print("Run manually: sudo systemctl restart collector")

    else:
        # ── On laptop — SSH into VM ──
        gcloud = get_gcloud_path()
        try:
            result = subprocess.run(
                [
                    gcloud,
                    "compute", "ssh", VM_NAME,
                    f"--zone={VM_ZONE}",
                    f"--project={PROJECT_ID}",
                    "--command=sudo systemctl reset-failed collector && "
                              "sudo systemctl restart collector && "
                              "sleep 3 && "
                              "sudo systemctl status collector "
                              "--no-pager | head -5"
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
            print("Restart manually:")
            print(f"  gcloud compute ssh {VM_NAME} --zone={VM_ZONE}")
            print(f"  sudo systemctl restart collector")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_auto_token():
    """
    Fully automated token refresh.
    No user interaction needed.
    Works on both laptop and VM.
    """
    on_vm = is_running_on_vm()

    print("=" * 55)
    print("AUTOMATED TOKEN REFRESH")
    print(f"Running on: {'VM (Linux)' if on_vm else 'Laptop (local)'}")
    print("=" * 55)

    # Load secrets
    print("\nLoading credentials from Secret Manager...")
    api_key     = get_secret("KITE_API_KEY")
    api_secret  = get_secret("KITE_API_SECRET")
    user_id     = get_secret("ZERODHA_USER_ID")
    password    = get_secret("ZERODHA_PASSWORD")
    totp_secret = get_secret("ZERODHA_TOTP_SECRET")
    print("Credentials loaded.")

    # Generate token
    access_token = automated_login(
        api_key, api_secret,
        user_id, password, totp_secret,
    )

    # Save to Secret Manager
    print("\nSaving token to Secret Manager...")
    store_secret("KITE_ACCESS_TOKEN", access_token)

    # Save to .env (useful for local development)
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