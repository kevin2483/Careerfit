"""
예측 실패 행 민감도 — 실패 행(출력 잘림·파싱 실패·API 오류)이 있는 문서를 뺐을 때 네 방식의 F1.

채점기는 실패 행을 빈 예측으로 두고 점수를 낸다(정답셋 38건을 그대로 쓰기 위해). 그 선택이 결론을 바꾸는지
여기서 확인한다. 어느 한 방식이라도 실패한 문서는 네 방식 모두에서 빼서 같은 문서 집합으로 비교한다.

    python -m src.eval.failed_rows --date 2026-07-30
    → data/processed/failed_rows_sensitivity.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.common.jsonl import read_jsonl
from src.common.paths import DICT_PATH, PROCESSED_DIR, labels_path, resolve_date
from src.common.skill_dict import close_tags, field_map, implies_map
from src.eval.bootstrap_ci import PRED_FILES
from src.eval.evaluate_extraction import prf
from src.eval.pred_meta import failed_ids

COLUMNS = ["방식", "구분", "n", "precision", "recall", "F1", "TP", "FP", "FN", "제외문서"]


def score(gold: dict[str, set], pred: dict[str, set], ids: list[str], fm: dict) -> dict:
    tp = fp = fn = 0
    for j in ids:
        g = {t for t in gold[j] if fm.get(t) != 3}
        p = {t for t in pred[j] if fm.get(t) != 3}
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    pr, rc, f1 = prf(tp, fp, fn)
    return {"n": len(ids), "precision": pr, "recall": rc, "F1": f1, "TP": tp, "FP": fp, "FN": fn}


def run(labels: Path, pred_dir: Path, fm: dict, implies: dict, methods: dict[str, str]) -> list[dict]:
    rows = [r for r in read_jsonl(labels) if r.get("reviewed")]
    ids = [str(r["job_id"]) for r in rows]
    gold = {str(r["job_id"]): close_tags(r.get("gold_tags") or [], implies) for r in rows}
    preds: dict[str, dict[str, set]] = {}
    failed: dict[str, dict[str, str]] = {}
    for name, fname in methods.items():
        p = pred_dir / fname
        if not p.exists():
            raise SystemExit(f"[중단] {name} 예측 파일 없음: {p}")
        recs = read_jsonl(p)
        failed[name] = failed_ids(recs, ids)
        preds[name] = {str(r["job_id"]): close_tags(r.get("tags") or [], implies) for r in recs}
        missing = [j for j in ids if j not in preds[name]]
        if missing:
            raise SystemExit(f"[중단] {name} 예측 없는 정답 {len(missing)}건: {missing[:5]}")
    drop = sorted({j for f in failed.values() for j in f})
    kept = [j for j in ids if j not in drop]
    out = []
    for name in methods:
        out.append({"방식": name, "구분": "전체", **score(gold, preds[name], ids, fm),
                    "제외문서": "; ".join(f"{j}({w})" for j, w in sorted(failed[name].items()))})
    for name in methods:
        out.append({"방식": name, "구분": "실패문서제외", **score(gold, preds[name], kept, fm),
                    "제외문서": ", ".join(drop)})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None)
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--dict", dest="dict_path", default=str(DICT_PATH))
    ap.add_argument("--out", default=None, help="기본: <processed>/failed_rows_sensitivity.csv")
    a = ap.parse_args(argv)
    pdir = Path(a.processed_dir)
    date = resolve_date(a.date, processed_dir=pdir, prefer="processed")
    out = Path(a.out) if a.out else pdir / "failed_rows_sensitivity.csv"
    rows = run(labels_path(a.labels), pdir / date, field_map(a.dict_path), implies_map(a.dict_path), PRED_FILES)

    print("예측 실패 행 민감도 (실패 문서를 네 방식 모두에서 제외)")
    for r in rows:
        print(f"  {r['방식']:<8}{r['구분']:<10} n={r['n']:<3} F1 {r['F1']:.3f}  (TP {r['TP']} FP {r['FP']} FN {r['FN']})"
              + (f"  실패: {r['제외문서']}" if r["구분"] == "전체" and r["제외문서"] else ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"저장 -> {out}")


if __name__ == "__main__":
    main()
