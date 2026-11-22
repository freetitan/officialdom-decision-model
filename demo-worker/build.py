import os
#!/usr/bin/env python3
"""Generate worker.js from worker.template.js + corpus.json, with a size check.

The Worker script limit on the free plan is 1 MiB; keep gzip size <= 900 KB.
Usage: python3 build.py   (then deploy with /tmp/deploy_worker.py)
"""
import gzip
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
MARKER = "const RAG_CORPUS = __RAG_CORPUS_JSON__;"

with open(f"{BASE}/worker.template.js", encoding="utf-8") as f:
    tpl = f.read()
with open(f"{BASE}/corpus.json", encoding="utf-8") as f:
    corpus_text = f.read()

assert MARKER in tpl, "code placeholder missing from template"
out = tpl.replace(MARKER, "const RAG_CORPUS = " + corpus_text + ";", 1)
assert MARKER not in out, "placeholder left unreplaced"

with open(f"{BASE}/worker.js", "w", encoding="utf-8") as f:
    f.write(out)

raw = len(out.encode("utf-8"))
gz = len(gzip.compress(out.encode("utf-8"), compresslevel=9))
print(f"worker.js bytes: {raw}, gzip: {gz}")
if gz > 900_000:
    print("WARNING: gzip size near/over the 1 MiB Worker script limit")
    sys.exit(2)
print("OK")
