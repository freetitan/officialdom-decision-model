#!/usr/bin/env python3
"""Build the RAG corpus for the demo Worker from the training data.

Sources (both are Siqi's own training materials, nothing else):
  A. DPO round-1 pairs: the CHOSEN response of every pair (fact-checked).
  B. SFT v6 rule fragments: only the 【法规要求】【综合判断】【同类案例实际处理】
     sections of non-CBDB items. Novel/folk passages are NEVER included.

Excluded: all CBDB trivia items, duplicates, fragments shorter than 30 chars.
Historical chunks (Qing/Ming/Song/ancient) get "h": 1: per data plan v2.2,
history is precedent/background only and must never serve as the basis for a
modern decision, so retrieval skips them unless the query itself is historical.
Output: corpus.json, a compact list of {id, t, s, x} (+h when historical).
"""
import json
import os
import re
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "officialdom-model", "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.json")

CBDB_PAT = re.compile(r"进士|及第|登科|CBDB|状元|榜眼|探花|殿试|会试|乡试")
HIST_PAT = re.compile(r"清代|清朝|古代|明代|宋代|元代|大清律|封建|旧时|古时")
SECTION_PAT = re.compile(r"【([^】]{1,12})】")
KEEP_SECTIONS = {"法规要求", "综合判断", "同类案例实际处理"}
DROP_SECTIONS = {"小说里的通常做法", "民间说法", "清代规矩", "清代的规矩"}


def norm(s):
    return re.sub(r"\s+", "", s)


def split_sections(output):
    parts = SECTION_PAT.split(output)
    sections = []
    for i in range(1, len(parts) - 1, 2):
        sections.append((parts[i].strip(), parts[i + 1].strip()))
    return sections


def is_hist(*texts):
    return bool(HIST_PAT.search(" ".join(texts)))


def main():
    chunks = []
    with open(f"{DATA}/dpo-round1-pairs.jsonl", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    seen = set()
    for i, p in enumerate(pairs, 1):
        text = norm(p["chosen"])
        if len(text) < 20 or text in seen:
            continue
        seen.add(text)
        c = {
            "id": f"d{i}",
            "t": p["prompt"][:24],
            "s": "DPO核验回答",
            "x": p["chosen"].strip()[:400],
        }
        if is_hist(p["prompt"], p["chosen"]):
            c["h"] = 1
        chunks.append(c)
    n_dpo = len(chunks)
    with open(f"{DATA}/sft-train-v6.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    n_cbdb = 0
    frag_seen = set()
    frags = []
    for r in recs:
        out = r["output"]
        instr = r["instruction"]
        if CBDB_PAT.search(out) or CBDB_PAT.search(instr):
            n_cbdb += 1
            continue
        for header, body in split_sections(out):
            if header in DROP_SECTIONS or header not in KEEP_SECTIONS:
                continue
            body = body.strip()
            if len(body) < 30:
                continue
            key = norm(body)[:60]
            if key in frag_seen:
                continue
            frag_seen.add(key)
            frags.append((instr, header, body))

    def rank(item):
        _instr, _header, body = item
        cites = bool(re.search(r"条例|办法|规定|准则|党章|公务员法|监察法|预算法|问责", body))
        return (1 if cites else 0, len(body))

    frags.sort(key=rank, reverse=True)
    for j, (instr, header, body) in enumerate(frags[:250], 1):
        c = {
            "id": f"s{j}",
            "t": instr[:24],
            "s": "法规片段",
            "x": body[:300],
        }
        if is_hist(instr, body):
            c["h"] = 1
        chunks.append(c)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, separators=(",", ":"))
    n_hist = sum(1 for c in chunks if c.get("h"))
    print(f"DPO chunks: {n_dpo}")
    print(f"CBDB items excluded: {n_cbdb}")
    print(f"SFT fragments kept: {len(chunks) - n_dpo} (from {len(frags)} unique)")
    print(f"historical chunks flagged: {n_hist}")
    print(f"total chunks: {len(chunks)}")
    print(f"corpus.json bytes: {os.path.getsize(OUT)}")


if __name__ == "__main__":
    sys.exit(main())
