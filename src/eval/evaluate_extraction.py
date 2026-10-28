"""
추출 성능 평가 — 사람이 확정한 정답(gold_tags, reviewed=true) vs 예측

왜 이게 본체인가:
  "추출 결과를 정성적으로 믿지 않고 자체 정답셋으로 정량 평가한다"는 원칙이 여기서 숫자가 된다.
  regex 와 LLM 을 같은 정답셋·같은 산식으로 채점한다.

평가 축:
  ① 전체
  ② 필드 난이도별 (field 1=명시 기술 / 2=서술형 역량 / 사전밖)
  ③ 직무별 (ds/de/mle/da) — 정답셋 38건이라 직무별 수치는 참고용. 결론에 쓰지 않는다.

주의:
  field 3(암묵조건)은 설계상 평가에서 제외한다(변별력 없음·규칙 매칭 불가).
  사전에 없는 정답 태그는 '사전밖'으로 분류해 FN 으로 센다 (recall 상한을 드러내기 위해 뺀다).

예측 소스 (셋 중 하나):
  --pred-file F     미리 계산된 예측 jsonl (LLM 또는 regex_predict.py 출력)  ← 공개 데이터만으로 재현 가능
  (기본)            정답셋의 body_raw 에 현재 사전으로 regex 를 다시 돌림      ← 본문 포함 비공개본 필요
  --use-stored      라벨 파일의 regex_prefilled(라벨링 당시 사전) 그대로      ← 과거 비교용

사용:
  python -m src.eval.evaluate_extraction --pred-file data/processed/2026-07-30/llm_extract_claude-opus-5.jsonl \
      --out-dir data/processed/evaluation_llm_opus
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src.common.jsonl import read_jsonl
from src.common.paths import DICT_PATH, JOBS, labels_path
from src.common.skill_dict import close_tags, extract_tags, field_map, implies_map, load_patterns
from src.eval.pred_meta import report_failed_rows

FIELD_LABEL = {1: "①명시 기술", 2: "②서술형 역량", "사전밖": "사전에 없던 역량"}


def load_gold(infile: str | Path) -> list[dict]:
    rows = [r for r in read_jsonl(infile) if r.get("reviewed")]
    if not rows:
        raise SystemExit(f"[중단] 검수 완료(reviewed=true) 항목이 없습니다: {infile}")
    return rows


def rerun_regex(rows: list[dict], dict_path: str | Path) -> None:
    """라벨 파일의 regex_prefilled 는 '그 당시 사전' 기준이라 낡았다. 지금 사전으로 다시 돌린다."""
    missing = [r.get("job_id") for r in rows if not r.get("body_raw")]
    if missing:
        raise SystemExit(
            f"[중단] body_raw 없는 정답 {len(missing)}건 (예: {missing[:3]}). "
            "공개 정답셋에는 본문이 없다. --pred-file 로 예측 파일을 지정할 것 "
            "(regex 는 data/processed/<date>/regex_extract.jsonl)."
        )
    patterns = load_patterns(dict_path)
    for r in rows:
        r["_pred"] = sorted(extract_tags(r["body_raw"], patterns))


def attach_pred_file(rows: list[dict], pred_file: str | Path, tag_key: str = "tags") -> None:
    recs = read_jsonl(pred_file)
    report_failed_rows(Path(pred_file).name, recs, [str(r["job_id"]) for r in rows])
    pred_map = {str(rec["job_id"]): list(rec.get(tag_key) or []) for rec in recs}
    missing = [r["job_id"] for r in rows if str(r["job_id"]) not in pred_map]
    if missing:
        raise SystemExit(f"[중단] 예측 없는 정답 {len(missing)}건: {missing[:5]}")
    for r in rows:
        r["_pred"] = sorted(pred_map[str(r["job_id"])])


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """precision / recall / F1. 분모 0 방어."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def evaluate(rows: list[dict], fmap: dict[str, int | None], exclude_field: int = 3,
             implies: dict[str, list[str]] | None = None):
    """
    반환: (전체, 필드별, 직무별, 오류사례)
    TP: 정답에도 있고 예측도 함 / FP: 예측했지만 정답 아님(오탐) / FN: 정답인데 놓침(누락)
    implies 가 있으면 정답과 예측을 같은 함의 규칙으로 닫은 뒤 비교한다 (v2: PySpark→Spark).
    """
    def keep(tag: str) -> bool:
        return fmap.get(tag) != exclude_field

    total = {"tp": 0, "fp": 0, "fn": 0}
    by_field: dict[object, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    by_job: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    errors = {"fp": defaultdict(int), "fn": defaultdict(int)}

    for r in rows:
        gold = {t for t in close_tags(r.get("gold_tags", []), implies) if keep(t)}
        pred = {t for t in close_tags(r.get("_pred", r.get("regex_prefilled", [])), implies) if keep(t)}
        job = r.get("직무", "?")
        for name, s in (("tp", gold & pred), ("fp", pred - gold), ("fn", gold - pred)):
            total[name] += len(s)
            by_job[job][name] += len(s)
            for tag in s:
                by_field[fmap.get(tag, "사전밖")][name] += 1
                if name != "tp":
                    errors[name][tag] += 1
    return total, by_field, by_job, errors


def write_outputs(out_dir: Path, total, by_field, by_job, errors) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pr, rc, f1 = prf(**total)
    with (out_dir / "eval_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["구분", "항목", "precision", "recall", "F1", "TP", "FP", "FN"])
        w.writerow(["전체", "-", pr, rc, f1, total["tp"], total["fp"], total["fn"]])
        for fld in [1, 2, "사전밖"]:
            d = by_field.get(fld)
            if d:
                w.writerow(["필드", FIELD_LABEL[fld], *prf(**d), d["tp"], d["fp"], d["fn"]])
        for job in JOBS:
            d = by_job.get(job)
            if d:
                w.writerow(["직무", job, *prf(**d), d["tp"], d["fp"], d["fn"]])
    with (out_dir / "error_analysis.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["유형", "태그", "건수"])
        for t, c in sorted(errors["fp"].items(), key=lambda x: (-x[1], x[0])):
            w.writerow(["FP_오탐", t, c])
        for t, c in sorted(errors["fn"].items(), key=lambda x: (-x[1], x[0])):
            w.writerow(["FN_누락", t, c])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", "--infile", dest="infile", default=None, help="정답셋. 기본: 비공개본이 있으면 그것, 없으면 공개본")
    p.add_argument("--dict", "--dict-path", dest="dict_path", default=str(DICT_PATH))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pred-file", default=None, help="예측 jsonl. 지정하면 regex 재실행 안 함")
    p.add_argument("--tag-key", default="tags")
    p.add_argument("--use-stored", action="store_true", help="라벨 파일의 regex_prefilled(과거 사전)로 평가")
    p.add_argument("--label", default=None, help="화면 표시용 예측기 이름")
    args = p.parse_args(argv)

    infile = labels_path(args.infile)
    rows = load_gold(infile)
    fmap = field_map(args.dict_path)

    if args.pred_file:
        attach_pred_file(rows, args.pred_file, args.tag_key)
        label = args.label or Path(args.pred_file).stem
        print(f"[정보] 예측파일로 평가: {args.pred_file}")
    elif args.use_stored:
        label = args.label or "regex_prefilled(저장값)"
        print("[정보] 라벨 파일에 저장된 regex_prefilled(라벨링 당시 사전)로 평가")
    else:
        rerun_regex(rows, args.dict_path)
        label = args.label or f"regex({Path(args.dict_path).name})"
        print(f"[정보] 사전 {Path(args.dict_path).name} 로 regex 재실행 후 평가")

    total, by_field, by_job, errors = evaluate(rows, fmap, implies=implies_map(args.dict_path))
    out_dir = Path(args.out_dir)
    write_outputs(out_dir, total, by_field, by_job, errors)

    pr, rc, f1 = prf(**total)
    print("=" * 62)
    print(f"추출 성능 평가 — 정답셋 {len(rows)}건 ({infile.name}) · 예측기: {label}")
    print("=" * 62)
    print(f"\n[전체]  precision {pr}  recall {rc}  F1 {f1}")
    print(f"        TP {total['tp']} / FP(오탐) {total['fp']} / FN(누락) {total['fn']}")
    print("\n[필드 난이도별]  ※ field3(암묵조건) 제외")
    for fld in [1, 2, "사전밖"]:
        d = by_field.get(fld)
        if d:
            pr_, rc_, f1_ = prf(**d)
            print(f"  {FIELD_LABEL[fld]:<16} P {pr_:<6} R {rc_:<6} F1 {f1_:<6} (TP {d['tp']} FP {d['fp']} FN {d['fn']})")
    print("\n[직무별]  ※ 표본이 작아 참고용")
    for job in JOBS:
        d = by_job.get(job)
        if d:
            pr_, rc_, f1_ = prf(**d)
            print(f"  {job:<5} P {pr_:<6} R {rc_:<6} F1 {f1_:<6} (TP {d['tp']} FP {d['fp']} FN {d['fn']})")
    top = lambda d: ", ".join(f"{t}({c})" for t, c in sorted(d.items(), key=lambda x: (-x[1], x[0]))[:8]) or "(없음)"  # noqa: E731
    print("\n[오류 유형 — 자주 틀린 태그]")
    print("  FP(예측이 잘못 뽑음):", top(errors["fp"]))
    print("  FN(예측이 놓침):    ", top(errors["fn"]))
    print(f"\n저장 -> {out_dir}/eval_summary.csv, error_analysis.csv")


if __name__ == "__main__":
    main()
