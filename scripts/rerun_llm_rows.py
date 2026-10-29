"""
예측 파일에서 특정 job_id 행만 지우고 그 공고만 다시 호출한다 (나머지 행은 그대로).

용도: 잘린 응답·파싱 실패 등 호출 설정 문제로 생긴 행을 모델 성능과 섞지 않기 위해 그 행만 다시 받는다.
      (2026-10-28 리뷰: sonnet 361662 가 max_tokens=512 에서 잘려 빈 예측으로 채점됨)

    export ANTHROPIC_API_KEY=...
    python -m scripts.rerun_llm_rows --model claude-sonnet-5 --job-ids 361662 --date 2026-07-30

절차: 백업(.bak_YYYYMMDD) → 해당 행 제거 → resume 모드로 그 id 만 호출(append) → 요약 재생성.
이후 `make reproduce` 로 평가 표를 다시 만든다. API 키가 없으면 아무것도 바꾸지 않고 중단한다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
from pathlib import Path

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.paths import DICT_PATH, PROCESSED_DIR, RAW_DIR, resolve_date
from src.extract.llm_skill_extract import _slug, run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--job-ids", required=True, help="쉼표 구분")
    ap.add_argument("--date", default=None)
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--dict", dest="dict_path", default=str(DICT_PATH))
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    a = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("[중단] ANTHROPIC_API_KEY 가 환경에 없다. 아무것도 바꾸지 않았다.")
    date = resolve_date(a.date, raw_dir=a.raw_dir, processed_dir=a.processed_dir, prefer="processed")
    ids = {s.strip() for s in a.job_ids.split(",") if s.strip()}
    pred_path = Path(a.processed_dir) / date / f"llm_extract_{_slug(a.model)}.jsonl"
    if not pred_path.exists():
        raise SystemExit(f"[중단] 예측 파일 없음: {pred_path}")

    rows = read_jsonl(pred_path)
    present = {str(r["job_id"]) for r in rows}
    missing = sorted(ids - present)
    if missing:
        print(f"[주의] 파일에 없는 job_id {missing} — 새로 추가된다")
    backup = pred_path.with_suffix(f".jsonl.bak_{dt.date.today():%Y%m%d}")
    shutil.copy2(pred_path, backup)
    kept = [r for r in rows if str(r["job_id"]) not in ids]
    write_jsonl(pred_path, kept)
    print(f"[준비] {pred_path.name}: {len(rows)}행 → {len(kept)}행 (제거 {len(rows) - len(kept)}), 백업 {backup.name}")

    summary = run(raw_dir=a.raw_dir, dict_path=a.dict_path, out_dir=a.processed_dir, date=date,
                  model=a.model, ids=ids, workers=1, resume=True)
    after = {str(r["job_id"]): r for r in read_jsonl(pred_path)}
    for j in sorted(ids):
        r = after.get(j)
        if r is None:
            print(f"  {j}: 호출되지 않음 (원문 배치 {date} 에 그 공고가 없는지 확인)")
            continue
        m = r["_meta"]
        print(f"  {j}: tags={len(r['tags'])} stop={m.get('stop_reason')} parse_ok={m.get('parse_ok')} "
              f"out_tokens={m.get('output_tokens')} cost=${m.get('cost_usd', 0):.4f}")
    print(f"[완료] n_processed={summary['n_processed']} n_parse_fail={summary['n_parse_fail']}. 다음: make reproduce")


if __name__ == "__main__":
    main()
