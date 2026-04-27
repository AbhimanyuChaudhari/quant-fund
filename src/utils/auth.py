import os
from pathlib import Path
from dotenv import load_dotenv
from kiteconnect import KiteConnect

# Always load from project root
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def get_kite_client() -> KiteConnect:
    """
    Returns an authenticated KiteConnect client.
    Reads API key and access token from .env
    """
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key:
        raise ValueError("KITE_API_KEY missing from .env")
    if not access_token:
        raise ValueError("KITE_ACCESS_TOKEN missing — run generate_token.py first")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def generate_token():
    """
    Run this manually each morning to get a fresh access token.
    Step 1: Opens login URL
    Step 2: You paste back the request token
    Step 3: Saves access token to .env automatically
    """
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("KITE_API_KEY or KITE_API_SECRET missing from .env")

    kite = KiteConnect(api_key=api_key)

    print("\n" + "="*50)
    print("STEP 1: Open this URL in your browser and login:")
    print("="*50)
    print(kite.login_url())
    print("="*50)
    print("\nAfter login, Zerodha redirects you to a URL.")
    print("Copy the 'request_token' value from that URL.")
    print("It looks like: https://yourapp.com/?request_token=XXXXXX&action=login")
    print()

    request_token = input("Paste request_token here: ").strip()

    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]

    # Save back to .env
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
    print("Access token saved to .env successfully")
    print("="*50 + "\n")

    return access_token


if __name__ == "__main__":
    generate_token()