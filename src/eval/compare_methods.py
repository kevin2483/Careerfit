"""
추출 방식별 정확도·비용·지연 비교표

evaluation_*/eval_summary.csv 와 <processed>/<date>/llm_extract_*_summary.json 을 모아 한 표로 만든다.
bootstrap_ci.csv 가 있으면 F1 95% 신뢰구간을 붙인다. 정확도-비용 파레토 판정까지 붙인다.

    python -m src.eval.compare_methods --date 2026-07-30 --n-docs 892

산출: <processed>/method_comparison.csv  (전체 + 필드별 + 직무별)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.common.paths import PROCESSED_DIR, resolve_date

# (표시이름, eval 디렉토리명, 모델 id 또는 None)  — 디렉토리는 <processed>/ 아래, 요약은 <processed>/<date>/ 아래
METHODS = [
    ("regex",  "evaluation_regex_pooled", None),
    ("haiku",  "evaluation_llm_haiku",    "claude-haiku-4-5-20251001"),
    ("sonnet", "evaluation_llm_sonnet",   "claude-sonnet-5"),
    ("opus",   "evaluation_llm_opus",     "claude-opus-5"),
]


def read_eval(path: Path) -> dict[str, dict]:
    """eval_summary.csv -> {항목: {p, r, f1, tp, fp, fn}}"""
    out: dict[str, dict] = {}
    with (path / "eval_summary.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            key = row["항목"] if row["구분"] != "전체" else "전체"
            out[key] = {
                "p": float(row["precision"]), "r": float(row["recall"]), "f1": float(row["F1"]),
                "tp": int(row["TP"]), "fp": int(row["FP"]), "fn": int(row["FN"]),
            }
    return out


def read_cost(path: Path | None) -> dict:
    if not path or not path.exists():
        return {"cost_per_doc_usd": 0.0, "latency_ms_p50": 0, "avg_tags": None,
                "n_hallucinated_tags": 0, "n_parse_fail": 0, "n_empty": 0, "pricing_as_of": ""}
    return json.loads(path.read_text(encoding="utf-8"))


def read_ci(path: Path) -> dict[str, dict]:
    """bootstrap_ci.csv -> {방식: {lo, hi}} (있을 때만)"""
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["지표"] == "F1":
                out[row["방식"]] = {"lo": float(row["CI_하한"]), "hi": float(row["CI_상한"])}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None)
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--n-docs", type=int, default=892, help="전체 코퍼스 고유 공고 수 (비용 환산용)")
    ap.add_argument("--out", default=None, help="기본: <processed>/method_comparison.csv")
    a = ap.parse_args(argv)

    pdir = Path(a.processed_dir)
    date = resolve_date(a.date, processed_dir=pdir)
    out = Path(a.out) if a.out else pdir / "method_comparison.csv"
    ci = read_ci(pdir / "bootstrap_ci.csv")

    rows = []
    for name, evdir, model in METHODS:
        p = pdir / evdir
        if not (p / "eval_summary.csv").exists():
            print(f"[건너뜀] {name}: {p}/eval_summary.csv 없음")
            continue
        costpath = pdir / date / f"llm_extract_{model}_summary.json" if model else None
        rows.append({"방식": name, "ev": read_eval(p), "cost": read_cost(costpath), "ci": ci.get(name)})
    if not rows:
        raise SystemExit("[중단] 비교할 결과가 없다")

    print("=" * 96)
    print(f"추출 방식별 비교 — 블라인드 풀링 재검수 정답셋 · 전체 {a.n_docs}건 환산")
    print("=" * 96)
    print(f"{'방식':<8}{'F1':>7}{'95% CI':>16}{'P':>7}{'R':>7}{'TP':>6}{'FP':>5}{'FN':>5}"
          f"{'$/건':>10}{'p50ms':>8}{'전체비용':>10}")
    for r in rows:
        e, c = r["ev"]["전체"], r["cost"]
        total = c["cost_per_doc_usd"] * a.n_docs
        ci_s = f"[{r['ci']['lo']:.3f}, {r['ci']['hi']:.3f}]" if r["ci"] else "-"
        print(f"{r['방식']:<8}{e['f1']:>7.3f}{ci_s:>16}{e['p']:>7.3f}{e['r']:>7.3f}"
              f"{e['tp']:>6}{e['fp']:>5}{e['fn']:>5}"
              f"{c['cost_per_doc_usd']:>10.4f}{c['latency_ms_p50'] or 0:>8}"
              f"{'$' + format(total, '.2f'):>10}")

    print(f"\n{'─' * 96}\n필드 난이도별 F1")
    labels = [k for k in rows[0]["ev"] if k != "전체"]
    print(f"{'방식':<8}" + "".join(f"{lb:>16}" for lb in labels))
    for r in rows:
        print(f"{r['방식']:<8}" + "".join(f"{(r['ev'].get(lb) or {'f1': 0})['f1']:>16.3f}" for lb in labels))

    print(f"\n{'─' * 96}\n정확도-비용 파레토  (더 싸고(<=) 더 정확한(>=) 방식이 있으면 지배당함)")
    for r in rows:
        f1, cpd = r["ev"]["전체"]["f1"], r["cost"]["cost_per_doc_usd"]
        dominators = [
            o["방식"] for o in rows
            if o is not r and o["ev"]["전체"]["f1"] >= f1 and o["cost"]["cost_per_doc_usd"] <= cpd
            and (o["ev"]["전체"]["f1"] > f1 or o["cost"]["cost_per_doc_usd"] < cpd)
        ]
        mark = "파레토 최적" if not dominators else f"지배당함 ← {', '.join(dominators)}"
        print(f"  {r['방식']:<8} F1 {f1:.3f} · ${cpd:.4f}/건   {mark}")

    print(f"\n{'─' * 96}\n출력 품질 (LLM만)")
    print(f"{'방식':<8}{'avg_tags':>10}{'환각':>7}{'파싱실패':>10}{'빈출력':>8}{'단가기준일':>14}")
    for r in rows:
        c = r["cost"]
        if c.get("avg_tags") is None:
            continue
        print(f"{r['방식']:<8}{c['avg_tags']:>10.2f}{c.get('n_hallucinated_tags', 0):>7}"
              f"{c.get('n_parse_fail', 0):>10}{c.get('n_empty', 0):>8}{c.get('pricing_as_of', ''):>14}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["방식", "구분", "precision", "recall", "F1", "F1_CI95_하한", "F1_CI95_상한", "TP", "FP", "FN",
                    "비용_건당USD", "지연_p50ms", f"전체{a.n_docs}건_예상비용USD"])
        for r in rows:
            c = r["cost"]
            for key, e in r["ev"].items():
                is_total = key == "전체"
                w.writerow([
                    r["방식"], key, e["p"], e["r"], e["f1"],
                    r["ci"]["lo"] if (is_total and r["ci"]) else "",
                    r["ci"]["hi"] if (is_total and r["ci"]) else "",
                    e["tp"], e["fp"], e["fn"],
                    round(c["cost_per_doc_usd"], 6) if is_total else "",
                    c["latency_ms_p50"] if is_total else "",
                    round(c["cost_per_doc_usd"] * a.n_docs, 2) if is_total else "",
                ])
    print(f"\n저장 -> {out}")


if __name__ == "__main__":
    main()
