import os
from pathlib import Path
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from google.cloud import secretmanager

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID = os.getenv("GCP_PROJECT_ID")


# ─────────────────────────────────────
# Secret Manager helpers
# ─────────────────────────────────────
def get_secret(secret_id: str) -> str:
    """Fetch latest version of a secret from GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name   = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8").strip()


def store_secret(secret_id: str, value: str):
    """Store a new version of a secret in GCP Secret Manager."""
    client  = secretmanager.SecretManagerServiceClient()
    parent  = f"projects/{PROJECT_ID}/secrets/{secret_id}"
    payload = value.encode("utf-8")
    client.add_secret_version(
        request={
            "parent": parent,
            "payload": {"data": payload}
        }
    )
    print(f"Secret {secret_id} updated in Secret Manager.")


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
        # Fallback to .env for local dev
        api_key      = os.getenv("KITE_API_KEY")
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        print("Credentials loaded from .env")

    if not api_key:
        raise ValueError("KITE_API_KEY not found.")
    if not access_token:
        raise ValueError("KITE_ACCESS_TOKEN not found — run generate_token first.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# ─────────────────────────────────────
# Token generation
# ─────────────────────────────────────
def generate_token():
    """
    Run this manually each morning to get a fresh access token.
    Saves token to both .env and Secret Manager.
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

    print("\n" + "="*50)
    print("Open this URL in your browser and login:")
    print("="*50)
    print(kite.login_url())
    print("="*50)
    print("\nAfter login copy the request_token from the URL.")
    print("URL looks like: https://127.0.0.1/?request_token=XXXXX\n")

    request_token = input("Paste request_token here: ").strip()
    data          = kite.generate_session(request_token, api_secret=api_secret)
    access_token  = data["access_token"]

    # Save to Secret Manager
    store_secret("KITE_ACCESS_TOKEN", access_token)

    # Also save to .env as backup
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

    print("\n" + "="*50)
    print("Token saved to Secret Manager and .env")
    print("="*50 + "\n")

    return access_token


if __name__ == "__main__":
    generate_token()