"""JSONL 읽기 — 유니코드 줄바꿈(\\u2028)이 본문에 있어도 레코드가 절단되지 않아야 한다 (2026-07-25 사고)."""
import json

import pytest

from src.common.jsonl import iter_jsonl, read_jsonl, write_jsonl


def test_u2028_inside_body_is_one_record(tmp_path):
    p = tmp_path / "a.jsonl"
    rec = {"job_id": "1", "body_raw": "첫 줄 둘째 줄"}
    p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    # 잘못된 방식은 두 동강 낸다는 것을 함께 보여 둔다
    assert len(p.read_text(encoding="utf-8").splitlines()) == 2
    assert read_jsonl(p) == [rec]


def test_blank_lines_skipped_and_roundtrip(tmp_path):
    p = tmp_path / "b.jsonl"
    rows = [{"a": 1}, {"b": "한글"}]
    write_jsonl(p, rows)
    p.write_text(p.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert read_jsonl(p) == rows


def test_strict_vs_lenient(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"ok": 1}\n{broken\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(p)
    assert [r for _, r in iter_jsonl(p, strict=False)] == [{"ok": 1}, None]
