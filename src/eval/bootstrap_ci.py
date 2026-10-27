"""
부트스트랩 신뢰구간 — "정답셋 38건에서 F1 0.012 차이는 의미가 있는가"를 숫자로 답한다.

방법
  공고 단위로 38건을 복원추출해 F1 을 다시 계산하는 것을 B 회 반복한다 (percentile 95% CI).
  방식 간 차이는 같은 리샘플에서 두 방식을 함께 계산한 '짝지은(paired)' 차이의 분포로 본다.
  차이의 CI 가 0 을 포함하면 "이 정답셋 크기로는 우위를 확정할 수 없다"는 뜻이다.

입력은 전부 커밋된 파일: 공개 정답셋 + <processed>/<date>/{regex_extract,llm_extract_*}.jsonl

    python -m src.eval.bootstrap_ci --date 2026-07-30
산출: <processed>/bootstrap_ci.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.common.jsonl import read_jsonl
from src.common.paths import DICT_PATH, PROCESSED_DIR, labels_path, resolve_date
from src.common.skill_dict import field_map

PRED_FILES = {
    "regex": "regex_extract.jsonl",
    "haiku": "llm_extract_claude-haiku-4-5-20251001.jsonl",
    "sonnet": "llm_extract_claude-sonnet-5.jsonl",
    "opus": "llm_extract_claude-opus-5.jsonl",
}


def doc_counts(gold: set, pred: set, fm: dict) -> tuple[int, int, int]:
    g = {t for t in gold if fm.get(t) != 3}
    p = {t for t in pred if fm.get(t) != 3}
    return len(g & p), len(p - g), len(g - p)


def f1_from(counts: list[tuple[int, int, int]], idx: list[int]) -> float:
    tp = sum(counts[i][0] for i in idx)
    fp = sum(counts[i][1] for i in idx)
    fn = sum(counts[i][2] for i in idx)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run(labels: Path, pred_dir: Path, fm: dict, b: int, seed: int, methods: dict[str, str]):
    rows = [r for r in read_jsonl(labels) if r.get("reviewed")]
    ids = [str(r["job_id"]) for r in rows]
    gold = {str(r["job_id"]): set(r.get("gold_tags") or []) for r in rows}

    counts: dict[str, list[tuple[int, int, int]]] = {}
    for name, fname in methods.items():
        p = pred_dir / fname
        if not p.exists():
            print(f"[건너뜀] {name}: {p} 없음")
            continue
        pred = {str(r["job_id"]): set(r.get("tags") or []) for r in read_jsonl(p)}
        missing = [j for j in ids if j not in pred]
        if missing:
            raise SystemExit(f"[중단] {name} 예측 없는 정답 {len(missing)}건: {missing[:5]}")
        counts[name] = [doc_counts(gold[j], pred[j], fm) for j in ids]

    n = len(ids)
    rng = random.Random(seed)
    samples = [[rng.randrange(n) for _ in range(n)] for _ in range(b)]
    point = {m: f1_from(c, list(range(n))) for m, c in counts.items()}
    boot = {m: [f1_from(c, s) for s in samples] for m, c in counts.items()}

    out = []
    for m in counts:
        out.append({"방식": m, "지표": "F1", "점추정": point[m],
                    "CI_하한": percentile(boot[m], 0.025), "CI_상한": percentile(boot[m], 0.975)})
    names = list(counts)
    for i, a in enumerate(names):
        for bname in names[i + 1:]:
            diffs = [x - y for x, y in zip(boot[a], boot[bname])]
            out.append({"방식": f"{a} - {bname}", "지표": "F1 차이", "점추정": point[a] - point[bname],
                        "CI_하한": percentile(diffs, 0.025), "CI_상한": percentile(diffs, 0.975),
                        "P(차이>0)": sum(d > 0 for d in diffs) / len(diffs)})
    return n, out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None)
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--dict", dest="dict_path", default=str(DICT_PATH))
    ap.add_argument("--b", type=int, default=10_000, help="리샘플 횟수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    pdir = Path(a.processed_dir)
    date = resolve_date(a.date, processed_dir=pdir)
    out = Path(a.out) if a.out else pdir / "bootstrap_ci.csv"
    n, rows = run(labels_path(a.labels), pdir / date, field_map(a.dict_path), a.b, a.seed, PRED_FILES)

    print("=" * 70)
    print(f"부트스트랩 95% CI — 정답셋 {n}건 · 공고 단위 복원추출 B={a.b} · seed={a.seed}")
    print("=" * 70)
    for r in rows:
        extra = f"   P(차이>0)={r['P(차이>0)']:.3f}" if "P(차이>0)" in r else ""
        print(f"  {r['방식']:<18}{r['지표']:<8}{r['점추정']:>+8.3f}   [{r['CI_하한']:+.3f}, {r['CI_상한']:+.3f}]{extra}")
    print("\n※ 차이의 CI 가 0 을 포함하면 이 정답셋 크기로는 우위를 확정할 수 없다.")

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["방식", "지표", "점추정", "CI_하한", "CI_상한", "P(차이>0)", "n", "B", "seed"]
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                        "n": n, "B": a.b, "seed": a.seed})
    print(f"저장 -> {out}")


if __name__ == "__main__":
    main()
