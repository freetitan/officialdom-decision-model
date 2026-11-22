#!/usr/bin/env python3
"""Deploy the officialdom-demo module Worker to Cloudflare (durable copy).

Cloudflare's module Workers REQUIRE the multipart order: metadata part FIRST,
script part second, with metadata.main_module matching the script part's
filename. cf_api.py --script-file sends script first, which makes Cloudflare
compile the module as a classic script and fail on `export`. This script
uses the correct order. Success = response.result.has_modules == true.

Usage: deploy_worker.py <worker.js path>  (account/script fixed below)
Auth: same surrogate mechanism as cf_api.py (custom.cloudflare credential).
"""
import json
import os
import sys
import urllib.request
import uuid

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request

ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")  # set in env, never hardcode
SCRIPT_NAME = "officialdom-demo"
BASE = "https://api.cloudflare.com/client/v4"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def main(argv):
    if len(argv) < 2:
        print("usage: deploy_worker.py <worker.js path>", file=sys.stderr)
        return 2
    with open(argv[1], "rb") as f:
        script_bytes = f.read()
    metadata = {
        "main_module": "worker.js",
        "compatibility_date": "2025-01-01",
        "bindings": [{"type": "ai", "name": "AI"}],
    }
    boundary = uuid.uuid4().hex
    parts = []
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n'
        f"Content-Type: application/json\r\n\r\n".encode()
        + json.dumps(metadata).encode()
        + b"\r\n"
    )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="worker.js"; filename="worker.js"\r\n'
        f"Content-Type: application/javascript+module\r\n\r\n".encode()
        + script_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    url = f"{BASE}/accounts/{ACCOUNT_ID}/workers/scripts/{SCRIPT_NAME}"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="PUT",
    )
    add_surrogate_to_request(req, "custom.cloudflare", allowed_hosts=["api.cloudflare.com"])
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=120) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        data = e.read()
        print(f"HTTP {e.code}", file=sys.stderr)
        print(data[:4000].decode("utf-8", "replace"), file=sys.stderr)
        return 1
    result = json.loads(data)
    print(json.dumps(result, ensure_ascii=False, indent=1)[:2000])
    ok = result.get("success") and (result.get("result") or {}).get("has_modules") is True
    print("deploy success:", ok, "has_modules:", (result.get("result") or {}).get("has_modules"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
