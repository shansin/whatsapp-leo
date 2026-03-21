"""One-time X (Twitter) session setup — run this before starting the MCP server.

Reads x.com cookies directly from your Firefox or Chrome/Chromium profile.
The browser must be logged in to x.com before running this.

Usage:
    uv run python x-mcp/x-mcp-server/setup.py              # auto-detect browser
    uv run python x-mcp/x-mcp-server/setup.py --firefox
    uv run python x-mcp/x-mcp-server/setup.py --chrome
"""

import glob
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile

from dotenv import load_dotenv

load_dotenv(override=True)

REQUIRED_COOKIES = {"auth_token", "ct0"}
X_HOSTS = {".x.com", "x.com", ".twitter.com", "twitter.com"}


# ── Firefox ───────────────────────────────────────────────────────────────────

def find_firefox_cookie_db() -> str | None:
    base = os.path.expanduser("~/.mozilla/firefox")
    if not os.path.exists(base):
        return None
    for pattern in ["*.default-release", "*.default", "*"]:
        for profile_dir in glob.glob(os.path.join(base, pattern)):
            db = os.path.join(profile_dir, "cookies.sqlite")
            if os.path.exists(db):
                return db
    return None


def read_firefox_cookies(db_path: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shutil.copy2(db_path, tmp_path)
        conn = sqlite3.connect(tmp_path)
        placeholders = ",".join("?" * len(X_HOSTS))
        rows = conn.execute(
            f"SELECT name, value FROM moz_cookies WHERE host IN ({placeholders})",
            list(X_HOSTS),
        ).fetchall()
        conn.close()
        return {name: value for name, value in rows}
    finally:
        os.unlink(tmp_path)


# ── Chrome / Chromium ─────────────────────────────────────────────────────────

CHROME_PATHS = [
    "~/.config/google-chrome/Default/Cookies",
    "~/.config/chromium/Default/Cookies",
    "~/.config/google-chrome-beta/Default/Cookies",
    "~/.config/google-chrome-unstable/Default/Cookies",
    "~/.config/BraveSoftware/Brave-Browser/Default/Cookies",
]


def find_chrome_cookie_db() -> str | None:
    for path in CHROME_PATHS:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            return expanded
    return None


def get_chrome_encryption_key() -> bytes:
    """Return the AES key used by Chrome to encrypt cookies on Linux.

    Tries GNOME Keyring first; falls back to the hardcoded Linux default.
    """
    try:
        import secretstorage
        conn = secretstorage.dbus_init()
        collection = secretstorage.get_default_collection(conn)
        for item in collection.get_all_items():
            if item.get_label() == "Chrome Safe Storage":
                return item.get_secret()
    except Exception:
        pass
    return b"peanuts"


def decrypt_chrome_value(encrypted: bytes, key: bytes) -> str:
    """Decrypt a Chrome cookie value (v10/v11 AES-128-CBC)."""
    if not encrypted:
        return ""
    if not encrypted.startswith((b"v10", b"v11")):
        # Unencrypted (old-style) value
        return encrypted.decode("utf-8", errors="replace")

    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    payload = encrypted[3:]  # strip 'v10'/'v11' prefix
    derived_key = hashlib.pbkdf2_hmac("sha1", key, b"saltysalt", 1, dklen=16)
    iv = b" " * 16

    cipher = Cipher(algorithms.AES(derived_key), modes.CBC(iv))
    decrypted = cipher.decryptor().update(payload)

    # Remove PKCS7 padding
    pad = decrypted[-1]
    return decrypted[:-pad].decode("utf-8", errors="replace")


def read_chrome_cookies(db_path: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shutil.copy2(db_path, tmp_path)
        conn = sqlite3.connect(tmp_path)

        # Column name differs between Chrome versions
        try:
            rows = conn.execute(
                "SELECT name, encrypted_value FROM cookies WHERE host_key IN ({})".format(
                    ",".join("?" * len(X_HOSTS))
                ),
                list(X_HOSTS),
            ).fetchall()
        except sqlite3.OperationalError:
            # Older Chrome schema uses 'host' not 'host_key'
            rows = conn.execute(
                "SELECT name, encrypted_value FROM cookies WHERE host IN ({})".format(
                    ",".join("?" * len(X_HOSTS))
                ),
                list(X_HOSTS),
            ).fetchall()

        conn.close()

        enc_key = get_chrome_encryption_key()
        return {name: decrypt_chrome_value(enc_value, enc_key) for name, enc_value in rows}
    finally:
        os.unlink(tmp_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    flag = sys.argv[1] if len(sys.argv) > 1 else None
    cookie_path = os.getenv("X_COOKIE_PATH", "/tmp/x_cookies.json")

    cookies = None

    if flag in (None, "--firefox"):
        db = find_firefox_cookie_db()
        if db:
            print(f"Reading Firefox cookies from: {db}")
            cookies = read_firefox_cookies(db)
        elif flag == "--firefox":
            print("Firefox profile not found.")
            sys.exit(1)

    if cookies is None and flag in (None, "--chrome"):
        db = find_chrome_cookie_db()
        if db:
            print(f"Reading Chrome cookies from: {db}")
            cookies = read_chrome_cookies(db)
        elif flag == "--chrome":
            print("Chrome/Chromium profile not found.")
            sys.exit(1)

    if cookies is None:
        print("No Firefox or Chrome profile found.")
        sys.exit(1)

    missing = REQUIRED_COOKIES - cookies.keys()
    if missing:
        print(f"Missing required cookies: {', '.join(missing)}")
        print("Make sure you are logged in to x.com in your browser, then re-run.")
        sys.exit(1)

    if os.path.exists(cookie_path):
        overwrite = input(f"Cookie file already exists at {cookie_path}. Overwrite? [y/N] ").strip().lower()
        if overwrite != "y":
            print("Aborted.")
            return

    with open(cookie_path, "w") as f:
        json.dump(cookies, f)

    print(f"Done. {len(cookies)} cookies saved to {cookie_path}")


if __name__ == "__main__":
    main()
