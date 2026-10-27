"""블라인드 풀링 --apply — 전원 n 공고도 정답이 갱신되고, 수기 태그는 canonical 로 정규화된다."""
import argparse

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.paths import DICT_PATH
from src.label.pool_cli import apply


def test_apply_updates_all_rejected_doc_and_normalizes_manual_tag(tmp_path, capsys):
    pool = tmp_path / "pool.jsonl"
    labels = tmp_path / "labels.jsonl"
    write_jsonl(pool, [
        {"job_id": "A", "tag": "Python", "verdict": True},
        {"job_id": "A", "tag": "fastapi", "verdict": True, "source": "manual"},
        {"job_id": "B", "tag": "LLM", "verdict": False},          # 전원 n → 정답은 빈 집합이어야 한다
        {"job_id": "B", "tag": "파인튜닝", "verdict": False},
        {"job_id": "C", "tag": "SQL", "verdict": None},            # 미판정 → --force 없이는 중단
    ])
    write_jsonl(labels, [
        {"job_id": "A", "reviewed": True, "gold_tags": ["Python", "SQL"]},
        {"job_id": "B", "reviewed": True, "gold_tags": ["LLM", "파인튜닝"]},
        {"job_id": "D", "reviewed": True, "gold_tags": ["R"]},     # 풀링 대상 아님 → 그대로
    ])
    args = argparse.Namespace(pool=str(pool), labels=str(labels), force=False, dict_path=str(DICT_PATH))
    try:
        apply(args)
        raised = False
    except SystemExit as e:
        raised = "미판정" in str(e)
    assert raised

    args.force = True
    apply(args)
    rows = {r["job_id"]: r for r in read_jsonl(labels)}
    assert rows["A"]["gold_tags"] == ["FastAPI", "Python"]          # fastapi → FastAPI, SQL 제거
    assert rows["A"]["gold_tags_prepool"] == ["Python", "SQL"]
    assert rows["B"]["gold_tags"] == [] and rows["B"]["pooled"] is True
    assert rows["D"]["gold_tags"] == ["R"] and "pooled" not in rows["D"]
    assert "정규화: 'fastapi' -> 'FastAPI'" in capsys.readouterr().out
    assert (tmp_path / "labels.jsonl.bak_prepool").exists()
