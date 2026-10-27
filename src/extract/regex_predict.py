"""
정규식 추출기를 정답셋 공고에 돌려 LLM 추출기와 같은 형식의 예측 파일을 남긴다.

왜 필요한가:
  공고 본문(body_raw)은 저작권 문제로 레포에 넣지 않는다. 그런데 평가·캐스케이드 시뮬레이션이
  본문에 정규식을 다시 돌리는 구조면 레포만 받은 사람은 재현할 수 없다.
  LLM 예측은 이미 파일로 커밋돼 있으니, 정규식 예측도 같은 형식으로 커밋해 두면
  공개 정답셋(label_draft.public.jsonl) + 예측 파일 4개만으로 모든 평가가 재현된다.

출력 형식 (llm_skill_extract.py 와 동일 키):
  {"job_id", "role", "tags", "_meta": {"model": "regex", "dict", "n_chars"}}

사용:
  python -m src.extract.regex_predict --date 2026-07-30
  → data/processed/2026-07-30/regex_extract.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.paths import DICT_PATH, PROCESSED_DIR, labels_path, resolve_date
from src.common.skill_dict import close_tags, extract_tags, implies_map, load_patterns


def predict(rows: list[dict], dict_path: str | Path) -> list[dict]:
    patterns = load_patterns(dict_path)
    implies = implies_map(dict_path)
    out = []
    for r in rows:
        body = r.get("body_raw")
        if body is None:
            raise SystemExit(
                f"[중단] job_id={r.get('job_id')} 에 body_raw 가 없다. 본문이 있는 정답셋(비공개본)이 필요하다."
            )
        out.append({
            "job_id": str(r["job_id"]),
            "role": r.get("직무"),
            "tags": sorted(close_tags(extract_tags(body, patterns), implies)),
            "_meta": {"model": "regex", "dict": Path(dict_path).name, "n_chars": len(body)},
        })
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=None, help="본문 포함 정답셋. 기본: data/labeled/label_draft.jsonl")
    ap.add_argument("--dict", dest="dict_path", default=str(DICT_PATH))
    ap.add_argument("--out-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--date", default=None)
    a = ap.parse_args(argv)

    date = resolve_date(a.date)
    rows = read_jsonl(labels_path(a.labels))
    preds = predict(rows, a.dict_path)
    out = Path(a.out_dir) / date / "regex_extract.jsonl"
    write_jsonl(out, preds)
    print(f"regex 예측 {len(preds)}건 -> {out}  (dict={Path(a.dict_path).name})")


if __name__ == "__main__":
    main()
