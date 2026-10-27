"""
JSONL 읽기/쓰기 — 프로젝트 전체가 이 한 곳만 쓴다.

왜 따로 두나:
  `read_text().splitlines()` 는 \\n 뿐 아니라 \\u2028 같은 유니코드 줄바꿈에서도 줄을 끊는다.
  공고 본문에 그 문자가 들어 있으면 한 줄짜리 레코드가 두 동강 난다 (2026-07-25 사고, lab_log).
  파일 객체를 직접 순회하면 \\n 에서만 끊긴다. 같은 실수를 네 곳에서 반복했기 때문에 함수로 고정한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl(path: str | Path, *, strict: bool = True) -> Iterator[tuple[int, dict | None]]:
    """(줄번호, 레코드) 를 낸다. strict=False 면 파싱 실패 줄을 (lineno, None) 으로 낸다."""
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise
                yield lineno, None


def read_jsonl(path: str | Path) -> list[dict]:
    return [rec for _lineno, rec in iter_jsonl(path, strict=True) if rec is not None]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
