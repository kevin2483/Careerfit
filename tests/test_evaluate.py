"""평가 산식 — TP/FP/FN, field 3 제외, 사전 밖 태그는 FN 으로 포함."""
import pytest

from src.eval.cascade_sim import score
from src.eval.evaluate_extraction import evaluate, prf, rerun_regex

FM = {"Python": 1, "SQL": 1, "대시보드": 2, "협업커뮤니케이션": 3}


def test_prf_basic_and_zero_division():
    assert prf(2, 1, 1) == (0.667, 0.667, 0.667)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_evaluate_counts_and_field_split():
    rows = [
        {"직무": "ds", "gold_tags": ["Python", "대시보드", "협업커뮤니케이션", "새역량"],
         "_pred": ["Python", "SQL", "협업커뮤니케이션"]},
    ]
    total, by_field, by_job, errors = evaluate(rows, FM)
    assert total == {"tp": 1, "fp": 1, "fn": 2}          # 협업커뮤니케이션(field 3)은 양쪽에서 빠진다
    assert by_field[1] == {"tp": 1, "fp": 1, "fn": 0}
    assert by_field[2] == {"tp": 0, "fp": 0, "fn": 1}
    assert by_field["사전밖"] == {"tp": 0, "fp": 0, "fn": 1}
    assert by_job["ds"]["fn"] == 2
    assert errors["fp"] == {"SQL": 1} and errors["fn"] == {"대시보드": 1, "새역량": 1}


def test_cascade_score_matches_evaluate():
    rows = [{"직무": "ds", "gold_tags": ["Python", "대시보드", "협업커뮤니케이션"], "_pred": ["Python", "SQL"]},
            {"직무": "de", "gold_tags": ["SQL"], "_pred": ["SQL", "협업커뮤니케이션"]}]
    total, *_ = evaluate(rows, FM)
    s = score([(set(r["gold_tags"]), set(r["_pred"])) for r in rows], FM)
    assert (s["TP"], s["FP"], s["FN"]) == (total["tp"], total["fp"], total["fn"])
    assert prf(**total)[2] == round(s["F1"], 3)


def test_rerun_regex_refuses_rows_without_body():
    with pytest.raises(SystemExit, match="body_raw 없는 정답"):
        rerun_regex([{"job_id": "1", "gold_tags": []}], "data/raw/dict/skill_tags_v1.yaml")
