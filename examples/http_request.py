"""Submit your own UTF-8 passage to an already running localhost service."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("document", type=Path)
parser.add_argument("--url", default="http://127.0.0.1:8765")
args = parser.parse_args()
body = json.dumps({"text": args.document.read_text(encoding="utf-8")}).encode()
request = Request(args.url.rstrip("/") + "/detect", data=body, headers={"Content-Type": "application/json"})
with urlopen(request, timeout=300) as response:
    print(response.read().decode())
