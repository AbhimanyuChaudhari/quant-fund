import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from google.cloud import secretmanager

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID = os.getenv("GCP_PROJECT_ID")

# VM config
VM_NAME = "data-collector"
VM_ZONE = "asia-south1-c"


# ─────────────────────────────────────
# Secret Manager helpers
# ─────────────────────────────────────
def get_secret(secret_id: str) -> str:
    """Fetch latest version of a secret from GCP Secret Manager."""
    client   = secretmanager.SecretManagerServiceClient()
    name     = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8").strip()


def store_secret(secret_id: str, value: str):
    """Store a new version of a secret. Creates secret if it doesn't exist."""
    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{PROJECT_ID}"
    name   = f"{parent}/secrets/{secret_id}"

    # Create secret if it doesn't exist
    try:
        client.get_secret(request={"name": name})
    except Exception:
        client.create_secret(request={
            "parent":    parent,
            "secret_id": secret_id,
            "secret": {
                "replication": {"automatic": {}}
            }
        })
        print(f"Created secret {secret_id}")

    # Add new version
    client.add_secret_version(
        request={
            "parent":  name,
            "payload": {"data": value.encode("utf-8")}
        }
    )
    print(f"Secret {secret_id} updated.")


# ─────────────────────────────────────
# VM restart helper
# ─────────────────────────────────────
def restart_collector():
    """
    Restart the collector service on the VM via gcloud SSH.
    Called automatically after token refresh.
    """
    print("\nRestarting collector on VM...")
    try:
        result = subprocess.run(
            [
                r"C:\Users\abhim\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
                    "compute", "ssh", VM_NAME,
                f"--zone={VM_ZONE}",
                f"--project={PROJECT_ID}",
                "--command=sudo systemctl restart collector && "
                "sudo systemctl status collector --no-pager | head -5"
            ],
            capture_output = True,
            text           = True,
            timeout        = 30,
            encoding       = 'utf-8',
            errors         = 'replace',
        )
        if result.returncode == 0:
            print("Collector restarted successfully.")
            print(result.stdout)
        else:
            print(f"Auto-restart failed. Restart manually:")
            print(f"  gcloud compute ssh {VM_NAME} --zone={VM_ZONE}")
            print(f"  sudo systemctl restart collector")
            if result.stderr:
                print(f"  Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("SSH timeout. Restart manually:")
        print(f"  gcloud compute ssh {VM_NAME} --zone={VM_ZONE}")
        print(f"  sudo systemctl restart collector")
    except FileNotFoundError:
        print("gcloud not found. Restart manually:")
        print(f"  gcloud compute ssh {VM_NAME} --zone={VM_ZONE}")
        print(f"  sudo systemctl restart collector")


# ─────────────────────────────────────
# Kite client
# ─────────────────────────────────────
def get_kite_client() -> KiteConnect:
    """
    Returns authenticated KiteConnect client.
    Fetches credentials from Secret Manager.
    Falls back to .env for local development.
    """
    try:
        api_key      = get_secret("KITE_API_KEY")
        access_token = get_secret("KITE_ACCESS_TOKEN")
        print("Credentials loaded from Secret Manager.")
    except Exception:
        api_key      = os.getenv("KITE_API_KEY")
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        print("Credentials loaded from .env")

    if not api_key:
        raise ValueError("KITE_API_KEY not found.")
    if not access_token:
        raise ValueError("KITE_ACCESS_TOKEN not found — run auth.py first.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# ─────────────────────────────────────
# Token generation
# ─────────────────────────────────────
def generate_token():
    """
    Run this every night before market opens.
    Saves token to Secret Manager AND restarts collector on VM.
    
    Complete daily routine in ONE command:
        python src/utils/auth.py
    """
    try:
        api_key    = get_secret("KITE_API_KEY")
        api_secret = get_secret("KITE_API_SECRET")
    except Exception:
        api_key    = os.getenv("KITE_API_KEY")
        api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("API key or secret not found.")

    kite = KiteConnect(api_key=api_key)

    print("\n" + "="*55)
    print("DAILY TOKEN REFRESH")
    print("="*55)
    print("\nStep 1: Open this URL in browser and login:")
    print(kite.login_url())
    print("\nAfter login, copy the request_token from the URL.")
    print("URL looks like: https://127.0.0.1/?request_token=XXXXX\n")

    request_token = input("Paste request_token here: ").strip()
    data          = kite.generate_session(request_token,
                                          api_secret=api_secret)
    access_token  = data["access_token"]

    # Step 2: Save to Secret Manager
    print("\nStep 2: Saving token to Secret Manager...")
    store_secret("KITE_ACCESS_TOKEN", access_token)

    # Step 3: Save to .env as backup
    lines = []
    if ENV_PATH.exists():
        with open(ENV_PATH, "r") as f:
            lines = f.readlines()

    with open(ENV_PATH, "w") as f:
        found = False
        for line in lines:
            if line.startswith("KITE_ACCESS_TOKEN="):
                f.write(f"KITE_ACCESS_TOKEN={access_token}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"KITE_ACCESS_TOKEN={access_token}\n")

    print("Token saved to Secret Manager and .env")

    # Step 4: Restart collector on VM automatically
    print("\nStep 3: Restarting collector on VM...")
    restart_collector()

    print("\n" + "="*55)
    print("DONE — Token refreshed, collector restarted")
    print("Market opens at 9:15am IST (3:45am EST)")
    print("="*55 + "\n")

    return access_token


if __name__ == "__main__":
    generate_token()