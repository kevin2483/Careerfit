"""
블라인드 풀링 후보 생성 — 시스템 N개의 예측을 합쳐 출처를 지우고 섞는다.

diff_extractions.py 는 두 파일만 받는다. 2차 교정(2026-08-15)의 교훈대로 비교할 모든 시스템을 한 번에 넣어야
어느 시스템 단독 태그도 후보에서 빠지지 않는다.

    python -m src.label.make_pool \\
        --pred regex=data/processed/2026-07-30/regex_extract.jsonl \\
        --pred haiku=data/processed/2026-07-30/llm_extract_claude-haiku-4-5-20251001.jsonl \\
        --pred sonnet=data/processed/2026-07-30/llm_extract_claude-sonnet-5.jsonl \\
        --pred opus=data/processed/2026-07-30/llm_extract_claude-opus-5.jsonl \\
        --ids-file data/labeled/gold_ids_round2.txt \\
        --pool-out data/labeled/pool_candidates_round2.jsonl \\
        --key-out  data/labeled/pool_key_round2.jsonl

pool_candidates: {job_id, tag, verdict=null} — 검수자는 이것만 본다. 문서 안 순서는 seed 로 섞인다.
pool_key:        {job_id, tag, sources=[...]} — 채점 때만 연다. 검수 중에 열면 블라인드가 깨진다.
요청한 id 에 예측이 없는 시스템이 하나라도 있으면 중단한다 (빠진 시스템의 단독 태그가 후보에서 사라지므로).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.common.jsonl import read_jsonl, write_jsonl


def build_pool(preds: dict[str, dict[str, set[str]]], ids: list[str], seed: int) -> tuple[list[dict], list[dict]]:
    for name, p in preds.items():
        missing = [j for j in ids if j not in p]
        if missing:
            raise SystemExit(f"[중단] {name} 에 예측 없는 공고 {len(missing)}건: {missing[:5]}")
    rng = random.Random(seed)
    cands, keys = [], []
    for jid in ids:
        tags = sorted({t for p in preds.values() for t in p[jid]})   # set 순회 금지 — seed 가 무의미해진다
        rng.shuffle(tags)
        for tag in tags:
            cands.append({"job_id": jid, "tag": tag, "verdict": None})
            keys.append({"job_id": jid, "tag": tag, "sources": [n for n, p in preds.items() if tag in p[jid]]})
    return cands, keys


def load_pred(path: str | Path, tags_key: str = "tags") -> dict[str, set[str]]:
    return {str(r["job_id"]): set(r.get(tags_key) or []) for r in read_jsonl(path)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", required=True, help="이름=경로. 여러 번")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--pool-out", required=True)
    ap.add_argument("--key-out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    preds = {}
    for item in a.pred:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"[중단] --pred 형식은 이름=경로: {item!r}")
        preds[name] = load_pred(path)
    ids = [ln.strip() for ln in Path(a.ids_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    for out in (a.pool_out, a.key_out):
        if Path(out).exists():
            raise SystemExit(f"[중단] 이미 있음: {out}. 판정 중인 풀을 덮어쓰지 않는다")
    cands, keys = build_pool(preds, ids, a.seed)
    write_jsonl(a.pool_out, cands)
    write_jsonl(a.key_out, keys)
    by_src = {n: sum(1 for k in keys if k["sources"] == [n]) for n in preds}
    print(f"후보 {len(cands)}건 / 공고 {len(ids)}건 / 시스템 {list(preds)}  단독 후보: {by_src}")
    print(f"-> {a.pool_out}  (key: {a.key_out} — 판정 끝날 때까지 열지 말 것)")


if __name__ == "__main__":
    main()
