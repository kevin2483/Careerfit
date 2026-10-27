"""
정답셋 공개판 생성 — 비공개본(label_draft.jsonl)에서 공고 본문(body_raw)만 뺀다.

공고 원문은 저작권 문제로 레포에 넣지 않는다. 본문 외 필드(gold_tags, regex_prefilled,
evidence_sections, review_flags, pooled 등)는 그대로 둔다. 이 파일만으로 LLM·regex 예측 파일을
채점할 수 있다 (evaluate_extraction.py --pred-file).

비공개본을 고친 뒤에는 반드시 다시 돌려 공개판을 맞춘다.

    python -m scripts.make_public_labels
"""
from __future__ import annotations

import argparse

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.paths import LABELS_PRIVATE, LABELS_PUBLIC

STRIP = ("body_raw",)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(LABELS_PRIVATE))
    ap.add_argument("--out", default=str(LABELS_PUBLIC))
    a = ap.parse_args(argv)
    rows = read_jsonl(a.src)
    pub = [{k: v for k, v in r.items() if k not in STRIP} for r in rows]
    write_jsonl(a.out, pub)
    n_rev = sum(1 for r in pub if r.get("reviewed"))
    print(f"공개판 {len(pub)}건 (reviewed {n_rev}) -> {a.out}  (제거 필드: {', '.join(STRIP)})")


if __name__ == "__main__":
    main()
