"""LLM 추출기의 API 없이 검증 가능한 부분 — 출력 파싱, 요약 집계(옛 파서 행 포함), 라벨 공간이 사전과 같은지."""
import json

from src.common.paths import DICT_PATH
from src.common.skill_dict import canonical_names
from src.extract.llm_skill_extract import MIN_CANONICAL, load_canonical_tags, parse_tags, summarize


def test_parse_tags_fence_empty_and_failure():
    canon = {"Python", "SQL"}
    assert parse_tags('["Python", "SQL", "Excel"]', canon) == (["Python", "SQL"], ["Excel"], True)
    assert parse_tags("```json\n[]\n```", canon) == ([], [], True)        # 빈 배열은 실패가 아니다
    assert parse_tags('{"a": 1}', canon) == ([], [], False)
    assert parse_tags('["Python", "SQ', canon)[2] is False                  # max_tokens 로 잘린 출력


def test_label_space_equals_dictionary():
    assert load_canonical_tags(DICT_PATH) == canonical_names(DICT_PATH)
    assert len(load_canonical_tags(DICT_PATH)) >= MIN_CANONICAL


def test_summarize_reclassifies_legacy_rows(tmp_path):
    """parse_ok 키가 없는 옛 행: raw_output 이 'None' 이면 정상, 펜스 친 [] 는 빈출력, 깨진 JSON 은 파싱실패."""
    rows = [
        {"job_id": "1", "tags": ["Python"], "_meta": {"error": None, "cost_usd": 0.001, "latency_ms": 10, "raw_output": "None"}},
        {"job_id": "2", "tags": [], "_meta": {"error": None, "cost_usd": 0.001, "latency_ms": 20, "raw_output": "```json\n[]\n```"}},
        {"job_id": "3", "tags": [], "_meta": {"error": None, "cost_usd": 0.001, "latency_ms": 30, "raw_output": "[\"Py", "stop_reason": "max_tokens"}},
        {"job_id": "4", "tags": [], "_meta": {"error": "empty_text"}},
    ]
    p = tmp_path / "pred.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    s = summarize(p, "m", "2099-01-01")
    assert (s["n_processed"], s["n_error"], s["n_parse_fail"], s["n_empty"]) == (4, 1, 1, 1)
    assert s["total_cost_usd"] == 0.003
