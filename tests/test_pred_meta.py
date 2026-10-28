"""실패 예측 행 판정과 민감도 표 — 실패 행은 빈 예측으로 채점되되 반드시 알리고, 제외 수치를 따로 낸다."""
import json
from pathlib import Path

from src.eval.failed_rows import run as sensitivity
from src.eval.pred_meta import failed_ids, pred_failure, report_failed_rows


def test_pred_failure_reasons():
    assert pred_failure({"job_id": "1", "tags": []}) is None                                  # regex 행
    assert pred_failure({"_meta": {"parse_ok": True, "stop_reason": "end_turn"}}) is None
    assert pred_failure({"_meta": {"stop_reason": "max_tokens", "parse_ok": False}}) == "max_tokens(출력 잘림)"
    assert pred_failure({"_meta": {"parse_ok": False}}) == "parse_fail"
    assert pred_failure({"_meta": {"error": "empty_text"}}) == "error=empty_text"
    assert pred_failure({"_meta": {"stop_reason": "end_turn"}}) is None                      # 옛 파서 행(parse_ok 없음)


def test_report_filters_to_requested_ids(capsys):
    recs = [{"job_id": "1", "_meta": {"parse_ok": False}}, {"job_id": "2", "_meta": {"parse_ok": False}}]
    assert failed_ids(recs, ["1"]) == {"1": "parse_fail"}
    assert report_failed_rows("x", recs, ["2"]) == {"2": "parse_fail"}
    assert "예측 실패 행 1건" in capsys.readouterr().out


def test_sensitivity_drops_failed_doc_from_every_method(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"job_id": j, "reviewed": True, "gold_tags": ["Python", "SQL"]}) + "\n"
                              for j in ("1", "2")), encoding="utf-8")
    pdir = tmp_path / "p"
    pdir.mkdir()
    (pdir / "a.jsonl").write_text(json.dumps({"job_id": "1", "tags": ["Python", "SQL"]}) + "\n"
                                  + json.dumps({"job_id": "2", "tags": [], "_meta": {"stop_reason": "max_tokens"}}) + "\n",
                                  encoding="utf-8")
    (pdir / "b.jsonl").write_text("".join(json.dumps({"job_id": j, "tags": ["Python"]}) + "\n" for j in ("1", "2")),
                                  encoding="utf-8")
    rows = sensitivity(labels, Path(pdir), {}, {}, {"a": "a.jsonl", "b": "b.jsonl"})
    by = {(r["방식"], r["구분"]): r for r in rows}
    assert by[("a", "전체")]["F1"] == 0.667 and "2(max_tokens" in by[("a", "전체")]["제외문서"]
    assert by[("a", "실패문서제외")] | {"n": 1, "F1": 1.0, "제외문서": "2"} == by[("a", "실패문서제외")]
    assert by[("b", "실패문서제외")]["n"] == 1 and by[("b", "실패문서제외")]["F1"] == 0.667   # b 도 같은 문서를 뺀다
