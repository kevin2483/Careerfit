"""
추출 성능 평가 (계획서 4번) — 정답셋 vs regex 베이스라인

무엇을 하나:
  사람이 확정한 정답(gold_tags, reviewed=true)과 regex 결과(regex_prefilled)를
  대조해 precision / recall / F1 을 계산한다.

왜 이게 본체인가:
  "LLM 추출을 정성적으로 믿지 않고 직접 구축한 정답셋으로 정량 평가한다"는
  계획서의 핵심 차별점이 여기서 숫자로 나온다.
  지금은 regex 만 평가하고, LLM 추출을 붙이면 같은 정답셋으로 동일 비교한다.

평가 축:
  ① 전체
  ② 필드 난이도별 (field 1=명시 기술 / 2=서술형 역량) ← 개선효과 측정 대상
  ③ 직무별 (ds/de/mle/da)

주의:
  field 3(암묵조건)은 사전 설계상 평가에서 제외한다(변별력 없음·regex 매칭 불가).

사용:
  python -m src.eval.evaluate_extraction
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml


def load_field_map(dict_path: Path) -> dict[str, int]:
    """canonical -> field(1/2/3) 매핑."""
    raw = yaml.safe_load(dict_path.read_text(encoding="utf-8"))
    m: dict[str, int] = {}
    for _cat, entries in raw.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            m[e["canonical"]] = e.get("field")
    return m


def rerun_regex(rows: list[dict], dict_path: Path) -> None:
    """
    라벨 파일에 저장된 regex_prefilled 는 '그 당시 사전' 기준이라 낡았다.
    사전을 개정(v1)하고도 이 값을 그대로 쓰면 개선이 평가에 반영되지 않는다.
    그래서 지금 사전으로 body_raw 에 regex 를 다시 돌려 pred 를 새로 만든다.
    """
    from src.extract.job_skill_extract_report import load_dict, compile_patterns
    patterns = compile_patterns(load_dict(dict_path))
    for r in rows:
        body = r.get("body_raw") or ""
        r["_pred"] = sorted({c for c, _cat, _f, pat in patterns if pat.search(body)})


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """precision / recall / F1. 분모 0 방어."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def evaluate(rows: list[dict], field_map: dict[str, int], exclude_field: int = 3):
    """
    반환: (전체, 필드별, 직무별, 오류사례)
    TP: 정답에도 있고 regex도 뽑음
    FP: regex는 뽑았는데 정답에 없음  (오탐)
    FN: 정답에는 있는데 regex가 놓침  (누락)
    """
    def keep(tag: str) -> bool:
        # 사전에 없는 태그(사람이 추가한 새 역량)는 field 미상 -> 평가에 포함하되
        # field 3(암묵조건)만 제외한다.
        return field_map.get(tag) != exclude_field

    total = {"tp": 0, "fp": 0, "fn": 0}
    by_field: dict[object, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    by_job: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    errors = {"fp": defaultdict(int), "fn": defaultdict(int)}

    for r in rows:
        gold = {t for t in r.get("gold_tags", []) if keep(t)}
        pred = {t for t in r.get("_pred", r.get("regex_prefilled", [])) if keep(t)}
        job = r.get("직무", "?")

        tp_set, fp_set, fn_set = gold & pred, pred - gold, gold - pred

        for name, s in (("tp", tp_set), ("fp", fp_set), ("fn", fn_set)):
            total[name] += len(s)
            by_job[job][name] += len(s)
            for tag in s:
                fld = field_map.get(tag, "사전밖")
                by_field[fld][name] += 1

        for tag in fp_set:
            errors["fp"][tag] += 1
        for tag in fn_set:
            errors["fn"][tag] += 1

    return total, by_field, by_job, errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/labeled/label_draft.jsonl")
    p.add_argument("--dict-path", default="data/raw/dict/skill_tags_v0.yaml")
    p.add_argument("--out-dir", default="data/processed/evaluation")
    p.add_argument("--use-stored", action="store_true",
                   help="regex 재실행 없이 라벨 파일에 저장된 값으로 평가(과거 사전 기준 비교용)")
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.infile).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("reviewed")]
    if not rows:
        raise SystemExit("[중단] 검수 완료(reviewed=true) 항목이 없습니다.")

    field_map = load_field_map(Path(args.dict_path))
    if not args.use_stored:
        rerun_regex(rows, Path(args.dict_path))
        print(f"[정보] 사전 {Path(args.dict_path).name} 로 regex 재실행 후 평가합니다.")
    else:
        print("[정보] 라벨 파일에 저장된 regex_prefilled(과거 사전 기준)로 평가합니다.")
    total, by_field, by_job, errors = evaluate(rows, field_map)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 화면 출력 ──
    print("=" * 62)
    print(f"추출 성능 평가 — 정답셋 {len(rows)}건 (regex 베이스라인)")
    print("=" * 62)

    pr, rc, f1 = prf(total["tp"], total["fp"], total["fn"])
    print(f"\n[전체]  precision {pr}  recall {rc}  F1 {f1}")
    print(f"        TP {total['tp']} / FP(오탐) {total['fp']} / FN(누락) {total['fn']}")

    print(f"\n[필드 난이도별]  ※ field3(암묵조건) 제외")
    label = {1: "①명시 기술", 2: "②서술형 역량", "사전밖": "사전에 없던 역량"}
    field_rows = []
    for fld in [1, 2, "사전밖"]:
        d = by_field.get(fld)
        if not d:
            continue
        pr_, rc_, f1_ = prf(d["tp"], d["fp"], d["fn"])
        print(f"  {label[fld]:<16} P {pr_:<6} R {rc_:<6} F1 {f1_:<6} (TP {d['tp']} FP {d['fp']} FN {d['fn']})")
        field_rows.append([label[fld], pr_, rc_, f1_, d["tp"], d["fp"], d["fn"]])

    print(f"\n[직무별]")
    job_rows = []
    for job in ["ds", "de", "mle", "da"]:
        d = by_job.get(job)
        if not d:
            continue
        pr_, rc_, f1_ = prf(d["tp"], d["fp"], d["fn"])
        print(f"  {job:<5} P {pr_:<6} R {rc_:<6} F1 {f1_:<6} (TP {d['tp']} FP {d['fp']} FN {d['fn']})")
        job_rows.append([job, pr_, rc_, f1_, d["tp"], d["fp"], d["fn"]])

    print(f"\n[오류 유형 — 자주 틀린 태그]")
    print("  FP(regex가 잘못 뽑음):", ", ".join(f"{t}({c})" for t, c in sorted(errors["fp"].items(), key=lambda x: -x[1])[:8]) or "(없음)")
    print("  FN(regex가 놓침):    ", ", ".join(f"{t}({c})" for t, c in sorted(errors["fn"].items(), key=lambda x: -x[1])[:8]) or "(없음)")

    # ── 파일 저장 ──
    with (out_dir / "eval_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["구분", "항목", "precision", "recall", "F1", "TP", "FP", "FN"])
        w.writerow(["전체", "-", pr, rc, f1, total["tp"], total["fp"], total["fn"]])
        for row in field_rows:
            w.writerow(["필드", *row])
        for row in job_rows:
            w.writerow(["직무", *row])

    with (out_dir / "error_analysis.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["유형", "태그", "건수"])
        for t, c in sorted(errors["fp"].items(), key=lambda x: -x[1]):
            w.writerow(["FP_오탐", t, c])
        for t, c in sorted(errors["fn"].items(), key=lambda x: -x[1]):
            w.writerow(["FN_누락", t, c])

    print(f"\n저장 -> {out_dir}/eval_summary.csv, error_analysis.csv")


if __name__ == "__main__":
    main()