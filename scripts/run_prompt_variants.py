"""
프롬프트 변형 실험 실행기 — notes/prompt_experiment_2026-10-29.md 에 사전 등록한 변형을 각 한 번씩 돌린다.

    export ANTHROPIC_API_KEY=...
    python -m scripts.run_prompt_variants --date 2026-07-30            # alias · criteria · schema
    python -m scripts.run_prompt_variants --date 2026-07-30 --variants schema

- 대상은 정답셋 38건(gold_ids.txt)만. 기준(base) 파일은 건드리지 않는다.
- 이미 결과 파일이 있는 변형은 건너뛴다(사전 등록 원칙: 한 변형은 한 번만). 다시 돌리려면 파일을 직접 지워야 한다.
- 키가 없으면 아무것도 하지 않고 중단한다.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.common.paths import DICT_PATH, LABELED_DIR, PROCESSED_DIR, RAW_DIR, resolve_date
from src.extract.llm_skill_extract import VARIANTS, pred_stem, run

MODEL = "claude-haiku-4-5-20251001"
EXPERIMENT_VARIANTS = ["alias", "criteria", "schema"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--variants", default=",".join(EXPERIMENT_VARIANTS), help="쉼표 구분")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--dict", dest="dict_path", default=str(DICT_PATH))
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--ids-file", default=str(LABELED_DIR / "gold_ids.txt"))
    a = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("[중단] ANTHROPIC_API_KEY 가 환경에 없다. 아무것도 하지 않았다.")
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS or v == "base"]
    if bad:
        raise SystemExit(f"[중단] 실험 변형이 아님: {bad}. 가능: {EXPERIMENT_VARIANTS}")
    date = resolve_date(a.date, raw_dir=a.raw_dir, processed_dir=a.processed_dir)
    ids = {ln.strip() for ln in Path(a.ids_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
    print(f"[실험] model={a.model} date={date} 정답셋 {len(ids)}건 변형 {variants}")

    for v in variants:
        pred_path = Path(a.processed_dir) / date / f"{pred_stem(a.model, v)}.jsonl"
        if pred_path.exists():
            print(f"[건너뜀] {v}: {pred_path.name} 이미 있음 (한 변형은 한 번만 돌린다)")
            continue
        print(f"\n=== 변형 {v}: {VARIANTS[v]}")
        s = run(raw_dir=a.raw_dir, dict_path=a.dict_path, out_dir=a.processed_dir, date=date,
                model=a.model, ids=ids, workers=4, resume=False, variant=v)
        print(f"  → n={s['n_processed']} 오류={s['n_error']} 파싱실패={s['n_parse_fail']} 환각={s['n_hallucinated_tags']} "
              f"비용=${s['total_cost_usd']:.3f}")
    print("\n[완료] 다음: python -m src.eval.prompt_variants --date", date)


if __name__ == "__main__":
    main()
