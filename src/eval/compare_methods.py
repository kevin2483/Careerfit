"""
CareerFit — 추출 방식별 정확도·비용·지연 비교표 (계획서 5번)

위치: src/eval/compare_methods.py

무엇을 하나
-----------
evaluation_* 폴더의 eval_summary.csv 와 llm_extract_*_summary.json 을 모아
하나의 표로 만든다. 정확도-비용 파레토 판정까지 붙인다.

    python -m src.eval.compare_methods

산출
----
    data/processed/method_comparison.csv   전체 + 필드별
    화면 출력                              리포트에 그대로 붙일 표
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# (표시이름, eval 디렉토리, LLM summary 경로 또는 None)
METHODS = [
    ("regex",  "data/processed/evaluation_regex_pooled", None),
    ("haiku",  "data/processed/evaluation_llm_pooled",
     "data/processed/2026-07-30/llm_extract_claude-haiku-4-5-20251001_summary.json"),
    ("sonnet", "data/processed/evaluation_llm_sonnet",
     "data/processed/2026-07-30/llm_extract_claude-sonnet-5_summary.json"),
    ("opus",   "data/processed/evaluation_llm_opus",
     "data/processed/2026-07-30/llm_extract_claude-opus-5_summary.json"),
]


def read_eval(path: Path) -> dict[str, dict]:
    """eval_summary.csv -> {구분_항목: {p, r, f1, tp, fp, fn}}"""
    out: dict[str, dict] = {}
    with (path / "eval_summary.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            key = row["항목"] if row["구분"] != "전체" else "전체"
            out[key] = {
                "p": float(row["precision"]), "r": float(row["recall"]), "f1": float(row["F1"]),
                "tp": int(row["TP"]), "fp": int(row["FP"]), "fn": int(row["FN"]),
            }
    return out


def read_cost(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {"cost_per_doc_usd": 0.0, "latency_ms_p50": 0, "avg_tags": None,
                "n_hallucinated_tags": 0, "n_parse_fail": 0, "n_empty": 0}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/method_comparison.csv")
    ap.add_argument("--n-docs", type=int, default=1132, help="전체 코퍼스 건수 (비용 환산용)")
    a = ap.parse_args()

    rows = []
    for name, evdir, costpath in METHODS:
        p = Path(evdir)
        if not (p / "eval_summary.csv").exists():
            print(f"[건너뜀] {name}: {p}/eval_summary.csv 없음")
            continue
        ev, cost = read_eval(p), read_cost(costpath)
        rows.append({"방식": name, "ev": ev, "cost": cost})

    if not rows:
        raise SystemExit("[중단] 비교할 결과가 없다")

    # ── 전체 ──
    print("=" * 78)
    print("추출 방식별 비교 — 정답셋 38건 (풀링 재검수 후)")
    print("=" * 78)
    print(f"{'방식':<8}{'F1':>7}{'P':>7}{'R':>7}{'TP':>6}{'FP':>5}{'FN':>5}"
          f"{'$/건':>10}{'p50ms':>8}{'전체비용':>10}")
    for r in rows:
        e, c = r["ev"]["전체"], r["cost"]
        total = c["cost_per_doc_usd"] * a.n_docs
        print(f"{r['방식']:<8}{e['f1']:>7.3f}{e['p']:>7.3f}{e['r']:>7.3f}"
              f"{e['tp']:>6}{e['fp']:>5}{e['fn']:>5}"
              f"{c['cost_per_doc_usd']:>10.4f}{c['latency_ms_p50'] or 0:>8}"
              f"{'$' + format(total, '.2f'):>10}")

    # ── 필드별 ──
    print(f"\n{'─' * 78}\n필드 난이도별 F1")
    labels = [k for k in rows[0]["ev"] if k != "전체"]
    print(f"{'방식':<8}" + "".join(f"{lb:>16}" for lb in labels))
    for r in rows:
        line = f"{r['방식']:<8}"
        for lb in labels:
            v = r["ev"].get(lb)
            line += f"{(v['f1'] if v else 0):>16.3f}"
        print(line)

    # ── 파레토 ──
    # 더 싸고(<=) 더 정확한(>=) 방식이 존재하면 지배당한 것으로 본다
    print(f"\n{'─' * 78}\n정확도-비용 파레토")
    for r in rows:
        f1, cpd = r["ev"]["전체"]["f1"], r["cost"]["cost_per_doc_usd"]
        dominators = [
            o["방식"] for o in rows
            if o is not r
            and o["ev"]["전체"]["f1"] >= f1
            and o["cost"]["cost_per_doc_usd"] <= cpd
            and (o["ev"]["전체"]["f1"] > f1 or o["cost"]["cost_per_doc_usd"] < cpd)
        ]
        mark = "파레토 최적" if not dominators else f"지배당함 ← {', '.join(dominators)}"
        print(f"  {r['방식']:<8} F1 {f1:.3f} · ${cpd:.4f}/건   {mark}")

    # ── 출력 품질 ──
    print(f"\n{'─' * 78}\n출력 품질 (LLM만)")
    print(f"{'방식':<8}{'avg_tags':>10}{'환각':>7}{'파싱실패':>10}{'빈출력':>8}")
    for r in rows:
        c = r["cost"]
        if c.get("avg_tags") is None:
            continue
        print(f"{r['방식']:<8}{c['avg_tags']:>10.2f}{c.get('n_hallucinated_tags', 0):>7}"
              f"{c.get('n_parse_fail', 0):>10}{c.get('n_empty', 0):>8}")

    # ── CSV ──
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["방식", "구분", "precision", "recall", "F1", "TP", "FP", "FN",
                    "비용_건당USD", "지연_p50ms", f"전체{a.n_docs}건_예상비용USD"])
        for r in rows:
            c = r["cost"]
            for key, e in r["ev"].items():
                w.writerow([
                    r["방식"], key, e["p"], e["r"], e["f1"], e["tp"], e["fp"], e["fn"],
                    round(c["cost_per_doc_usd"], 6) if key == "전체" else "",
                    c["latency_ms_p50"] if key == "전체" else "",
                    round(c["cost_per_doc_usd"] * a.n_docs, 2) if key == "전체" else "",
                ])
    print(f"\n저장 -> {out}")


if __name__ == "__main__":
    main()
