"""
프로젝트 표준 경로와 날짜 해석.

규칙 (CLAUDE.md ④): 날짜·파일명을 코드에 하드코딩하지 않는다.
  - 날짜는 항상 인자로 받되, 생략하면 디렉토리에서 '가장 최신 배치'를 찾아 쓴다.
  - 어떤 날짜가 쓰였는지는 항상 화면에 찍는다.
"""
from __future__ import annotations

import re
from pathlib import Path

JOBS = ["ds", "de", "mle", "da"]

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DICT_PATH = RAW_DIR / "dict" / "skill_tags_v2.yaml"   # v1 + implies(PySpark→Spark). canonical·alias 는 v1 과 동일
LABELED_DIR = DATA_DIR / "labeled"
LABELS_PRIVATE = LABELED_DIR / "label_draft.jsonl"          # body_raw 포함. 레포 미포함
LABELS_PUBLIC = LABELED_DIR / "label_draft.public.jsonl"    # body_raw 제거. 레포 포함
PROCESSED_DIR = DATA_DIR / "processed"
EXTRACT_REPORT_DIR = PROCESSED_DIR / "job_skill_extract_report"
MARKET_STATS_DIR = PROCESSED_DIR / "market_stats"

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def labels_path(explicit: str | Path | None = None) -> Path:
    """정답셋 경로. 비공개본(본문 포함)이 있으면 그것, 없으면 공개본."""
    if explicit:
        return Path(explicit)
    return LABELS_PRIVATE if LABELS_PRIVATE.exists() else LABELS_PUBLIC


def raw_dates(raw_dir: str | Path = RAW_DIR) -> list[str]:
    """data/raw/{job}/wanted_YYYY-MM-DD.jsonl 에서 날짜 목록(오름차순)."""
    found: set[str] = set()
    for job in JOBS:
        for p in Path(raw_dir).glob(f"{job}/wanted_*.jsonl"):
            m = _DATE_RE.search(p.name)
            if m:
                found.add(m.group(1))
    return sorted(found)


def processed_dates(processed_dir: str | Path = PROCESSED_DIR) -> list[str]:
    """data/processed/YYYY-MM-DD/ 디렉토리 이름에서 날짜 목록(오름차순)."""
    return sorted(
        p.name for p in Path(processed_dir).iterdir()
        if p.is_dir() and _DATE_RE.fullmatch(p.name)
    ) if Path(processed_dir).exists() else []


def resolve_date(explicit: str | None, *, raw_dir: str | Path = RAW_DIR,
                 processed_dir: str | Path = PROCESSED_DIR, prefer: str = "raw") -> str:
    """--date 가 주어지면 그대로. 아니면 prefer="raw" 는 raw → processed, prefer="processed" 는 그 반대 순으로 최신 배치.
    추출은 raw 를, 평가(예측 파일만 읽음)는 processed 를 우선해야 새 크롤 직후에 예측 없는 날짜를 고르지 않는다."""
    if explicit:
        return explicit
    order = (raw_dates(raw_dir), processed_dates(processed_dir))
    if prefer == "processed":
        order = order[::-1]
    for cands in order:
        if cands:
            print(f"[날짜] --date 미지정 → 최신 배치 {cands[-1]} 사용")
            return cands[-1]
    raise SystemExit(f"[중단] 날짜를 찾을 수 없음. --date 를 지정할 것 (raw={raw_dir}, processed={processed_dir})")
