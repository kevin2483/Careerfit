"""
프롬프트 변형 평가 — 기준(haiku base)과 변형들을 같은 채점기·같은 부트스트랩으로 비교한다.

    python -m src.eval.prompt_variants --date 2026-07-30
    → data/processed/prompt_variants.csv

행: 변형별 F1·CI·①②필드 F1·환각·파싱실패, 그리고 '변형 − base' 짝지은 차이와 P(차이>0).
판정은 notes/prompt_experiment_2026-10-29.md 에 실행 전 고정한 기준(CI 가 0 을 넘어야 '개선 확정')을 따른다.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.common.jsonl import read_jsonl
from src.common.paths import DICT_PATH, PROCESSED_DIR, labels_path, resolve_date
from src.common.skill_dict import field_map, implies_map
from src.eval.bootstrap_ci import run as bootstrap
from src.eval.evaluate_extraction import evaluate, prf
from src.extract.llm_skill_extract import pred_stem

MODEL = "claude-haiku-4-5-20251001"
VARIANT_ORDER = ["base", "alias", "criteria", "schema"]
COLUMNS = ["변형", "지표", "F1", "CI_하한", "CI_상한", "P(차이>0)", "precision", "recall", "TP", "FP", "FN",
           "F1_①명시", "F1_②서술형", "환각태그", "파싱실패", "건당비용USD", "n", "B", "seed"]


def field_scores(labels: Path, pred: Path, fm: dict, implies: dict) -> dict:
    rows = [r for r in read_jsonl(labels) if r.get("reviewed")]
    pm = {str(r["job_id"]): r.get("tags") or [] for r in read_jsonl(pred)}
    for r in rows:
        r["_pred"] = pm.get(str(r["job_id"]), [])
    total, by_field, _by_job, _err = evaluate(rows, fm, implies=implies)
    pr, rc, f1 = prf(**total)
    f = {k: prf(**v)[2] for k, v in by_field.items()}
    return {"precision": pr, "recall": rc, "F1": f1, **total, "F1_①명시": f.get(1, ""), "F1_②서술형": f.get(2, "")}


def run(labels: Path, pred_dir: Path, fm: dict, implies: dict, model: str, b: int, seed: int) -> list[dict]:
    files = {v: pred_dir / f"{pred_stem(model, v)}.jsonl" for v in VARIANT_ORDER}
    present = {v: p for v, p in files.items() if p.exists()}
    if "base" not in present or len(present) < 2:
        raise SystemExit(f"[중단] base 와 변형 파일이 최소 하나씩 있어야 한다. 있음: {sorted(present)}")
    n, boot = bootstrap(labels, pred_dir, fm, b, seed, {v: p.name for v, p in present.items()}, implies=implies)
    point = {r["방식"]: r for r in boot if r["지표"] == "F1"}
    diffs = {r["방식"]: r for r in boot if r["지표"] == "F1 차이"}

    out = []
    for v, p in present.items():
        fs = field_scores(labels, p, fm, implies)
        summ = json.loads(p.with_name(p.stem + "_summary.json").read_text(encoding="utf-8")) if p.with_name(p.stem + "_summary.json").exists() else {}
        out.append({"변형": v, "지표": "F1", "F1": fs["F1"], "CI_하한": point[v]["CI_하한"], "CI_상한": point[v]["CI_상한"], "P(차이>0)": "",
                    "precision": fs["precision"], "recall": fs["recall"], "TP": fs["tp"], "FP": fs["fp"], "FN": fs["fn"],
                    "F1_①명시": fs["F1_①명시"], "F1_②서술형": fs["F1_②서술형"],
                    "환각태그": summ.get("n_hallucinated_tags", ""), "파싱실패": summ.get("n_parse_fail", ""),
                    "건당비용USD": summ.get("cost_per_doc_usd", ""), "n": n, "B": b, "seed": seed})
    for v in present:
        if v == "base":
            continue
        key = f"base - {v}" if f"base - {v}" in diffs else f"{v} - base"
        d = diffs[key]
        sign = -1 if key.startswith("base") else 1   # '변형 − base' 방향으로 통일
        out.append({"변형": f"{v} - base", "지표": "F1 차이", "F1": sign * d["점추정"],
                    "CI_하한": sign * (d["CI_상한"] if sign < 0 else d["CI_하한"]),
                    "CI_상한": sign * (d["CI_하한"] if sign < 0 else d["CI_상한"]),
                    "P(차이>0)": (1 - d["P(차이>0)"]) if sign < 0 else d["P(차이>0)"],
                    "precision": "", "recall": "", "TP": "", "FP": "", "FN": "", "F1_①명시": "", "F1_②서술형": "",
                    "환각태그": "", "파싱실패": "", "건당비용USD": "", "n": n, "B": b, "seed": seed})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None)
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--dict", dest="dict_path", default=str(DICT_PATH))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--b", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    pdir = Path(a.processed_dir)
    date = resolve_date(a.date, processed_dir=pdir, prefer="processed")
    out = Path(a.out) if a.out else pdir / "prompt_variants.csv"
    rows = run(labels_path(a.labels), pdir / date, field_map(a.dict_path), implies_map(a.dict_path), a.model, a.b, a.seed)

    print("프롬프트 변형 vs 기준 (haiku) — 정답셋 기준, 짝지은 부트스트랩")
    for r in rows:
        if r["지표"] == "F1":
            print(f"  {r['변형']:<10} F1 {r['F1']:.3f} [{r['CI_하한']:.3f}, {r['CI_상한']:.3f}]  P {r['precision']:.3f} R {r['recall']:.3f}  "
                  f"① {r['F1_①명시']:.3f} ② {r['F1_②서술형']:.3f}  환각 {r['환각태그']} 파싱실패 {r['파싱실패']}")
    for r in rows:
        if r["지표"] == "F1 차이":
            settled = "개선 확정" if r["CI_하한"] > 0 else ("악화 확정" if r["CI_상한"] < 0 else "확정 불가")
            print(f"  {r['변형']:<16} {r['F1']:+.4f} [{r['CI_하한']:+.4f}, {r['CI_상한']:+.4f}]  P(>0)={r['P(차이>0)']:.3f}  → {settled}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"저장 -> {out}")


if __name__ == "__main__":
    main()
