"""
CareerFit — 두 추출 산출물 비교 / 블라인드 풀링 후보 생성

위치: scripts/diff_extractions.py

용도 두 가지 (같은 도구를 2단계와 4단계에서 재사용):
  2단계  조사 패치 전/후 regex 출력을 비교 → 전체 1,190건에서 무엇이 몇 개 늘었나
  4단계  regex 출력 vs LLM 출력을 합집합으로 풀링 → 사람이 블라인드로 재검수

사용
----
    # 조사 패치 전후 비교
    python scripts/diff_extractions.py \
        --a data/processed/2026-08-12/regex_before.jsonl \
        --b data/processed/2026-08-12/regex_after.jsonl \
        --label-a regex_b --label-b regex_patched

    # 풀링 후보 생성 (정답셋 38건만)
    python scripts/diff_extractions.py \
        --a data/processed/2026-08-12/regex_after.jsonl \
        --b data/processed/2026-08-12/llm_extract_claude-haiku-4-5-20251001.jsonl \
        --label-a regex --label-b llm \
        --ids-file data/labeled/gold_ids.txt \
        --pool-out data/labeled/pool_candidates.jsonl \
        --key-out  data/labeled/pool_key.jsonl

pool_candidates.jsonl 은 출처가 지워지고 순서가 섞여 있다. 검수자는 이것만 본다.
pool_key.jsonl 에 (job_id, tag) -> 어느 쪽이 냈는지가 들어 있고, 채점 때 조인한다.
검수 중에 key 파일을 열면 블라인드가 깨진다. 열지 말 것.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load(path: str, id_key: str, tags_key: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:  # splitlines() 쓰지 말 것 (lab_log 기록됨)
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            jid = str(rec[id_key])
            out[jid] = set(rec.get(tags_key) or [])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--id-key", default="job_id")
    ap.add_argument("--tags-key", default="tags")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--pool-out", default=None)
    ap.add_argument("--key-out", default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    A = load(a.a, a.id_key, a.tags_key)
    B = load(a.b, a.id_key, a.tags_key)

    keep = None
    if a.ids_file:
        keep = {ln.strip() for ln in Path(a.ids_file).read_text(encoding="utf-8").split("\n") if ln.strip()}

    ids = sorted((set(A) & set(B)) if keep is None else (set(A) & set(B) & keep))
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))

    print(f"공통 문서 {len(ids)}건 / {a.label_a}만 {len(only_a)}건 / {a.label_b}만 {len(only_b)}건")
    if only_a or only_b:
        print("  ! 문서 집합이 다르다. 같은 raw·같은 date로 돌렸는지 먼저 확인할 것.")

    n_a = sum(len(A[i]) for i in ids)
    n_b = sum(len(B[i]) for i in ids)
    added: Counter = Counter()
    removed: Counter = Counter()
    n_diff_docs = 0

    for i in ids:
        d_add, d_rem = B[i] - A[i], A[i] - B[i]
        if d_add or d_rem:
            n_diff_docs += 1
        added.update(d_add)
        removed.update(d_rem)

    print(f"\n총 태그수  {a.label_a}={n_a}  {a.label_b}={n_b}  "
          f"({n_b - n_a:+d}, {(n_b - n_a) / max(n_a, 1) * 100:+.1f}%)")
    print(f"문서당 평균 {n_a / max(len(ids), 1):.2f} -> {n_b / max(len(ids), 1):.2f}")
    print(f"태그셋이 달라진 문서 {n_diff_docs}/{len(ids)}건 ({n_diff_docs / max(len(ids), 1) * 100:.1f}%)")

    print(f"\n{a.label_b}에서 늘어난 태그 top{a.top}")
    for tag, c in added.most_common(a.top):
        print(f"  +{c:5d}  {tag}")
    if removed:
        print(f"\n{a.label_b}에서 사라진 태그 top{a.top}  (패치 비교라면 여기가 비어 있어야 정상)")
        for tag, c in removed.most_common(a.top):
            print(f"  -{c:5d}  {tag}")

    if a.pool_out:
        if not a.key_out:
            raise SystemExit("--pool-out 을 쓰면 --key-out 도 필요하다")
        rng = random.Random(a.seed)
        cands, keys = [], []
        for i in ids:
            for tag in (A[i] | B[i]):
                src = []
                if tag in A[i]:
                    src.append(a.label_a)
                if tag in B[i]:
                    src.append(a.label_b)
                cands.append({"job_id": i, "tag": tag, "verdict": None})
                keys.append({"job_id": i, "tag": tag, "sources": src})
        rng.shuffle(cands)

        Path(a.pool_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(a.pool_out).open("w", encoding="utf-8") as f:
            for c in cands:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        with Path(a.key_out).open("w", encoding="utf-8") as f:
            for k in keys:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")

        both = sum(1 for k in keys if len(k["sources"]) == 2)
        print(f"\n풀링 후보 {len(cands)}건 -> {a.pool_out}")
        print(f"  양쪽 일치 {both} / {a.label_a}만 {sum(1 for k in keys if k['sources'] == [a.label_a])}"
              f" / {a.label_b}만 {sum(1 for k in keys if k['sources'] == [a.label_b])}")
        print(f"  정답키 -> {a.key_out} (검수 끝나기 전에 열지 말 것)")


if __name__ == "__main__":
    main()
