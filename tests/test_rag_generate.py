"""생성·주장 판정·생성 평가 — 가짜 응답과 합성 데이터만 쓴다 (실제 질문·API 호출 없음)."""
import builtins
import json
from types import SimpleNamespace

import pytest

from src.common.jsonl import write_jsonl
from src.eval import rag_gen_eval
from src.rag import claim_judge, generate

QUERIES = [{"qid": "q1", "type": "A", "question": "가", "filter": {}},
           {"qid": "q2", "type": "B", "question": "나", "filter": {}},
           {"qid": "q3", "type": "D", "question": "다", "filter": {}}]
CHUNKS = {f"{i}:a:0": {"chunk_id": f"{i}:a:0", "text": f"본문{i}", "title": "t", "company": f"회사{i}", "section": "a"}
          for i in range(1, 8)}
RUN = [{"qid": q, "rank": r, "chunk_id": f"{r}:a:0", "score": 1.0} for q in ("q1", "q2", "q3") for r in range(1, 8)]


class FakeClient:
    """messages.create 를 흉내 낸다. replies: 호출 순서대로의 (stop_reason, 본문 텍스트)."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        stop, text = self.replies.pop(0)
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def _ans(answerable, *claims):
    return ("end_turn", json.dumps({"answerable": answerable, "claims": [{"text": t, "source": s} for t, s in claims]}))


def _feed(monkeypatch, answers):
    it = iter(answers)

    def fake(_prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr(builtins, "input", fake)


# ── 생성 ──────────────────────────────────────────────────────────────
def test_passed_chunks_follow_judged_depth():
    passed = generate.passed_chunks(QUERIES, RUN)
    assert passed["q1"] == [f"{r}:a:0" for r in range(1, 6)]
    assert passed["q3"] == [f"{r}:a:0" for r in range(1, 4)]      # D 유형은 상위 3


def test_user_message_numbers_chunks_in_rank_order():
    msg = generate.user_message("가", [CHUNKS["2:a:0"], CHUNKS["1:a:0"]])
    assert msg.startswith("질문: 가") and msg.index("[1] t · 회사2 · a") < msg.index("[2] t · 회사1 · a")


def test_generate_maps_source_number_to_chunk_id_and_calls_once_per_question():
    client = FakeClient([_ans(True, ("회사2는 X를 요구한다.", 2)), _ans(True, ("ㄱ", 1), ("ㄴ", 5)), _ans(False)])
    rows = generate.generate(client, QUERIES, generate.passed_chunks(QUERIES, RUN), CHUNKS)
    assert len(client.calls) == 3
    assert rows[0]["claims"] == [{"claim_id": "q1:1", "text": "회사2는 X를 요구한다.", "chunk_id": "2:a:0"}]
    assert [c["claim_id"] for c in rows[1]["claims"]] == ["q2:1", "q2:2"]
    assert rows[2]["answerable"] is False and rows[2]["claims"] == [] and rows[2]["error"] is None
    kw = client.calls[0]
    assert kw["model"] == "claude-sonnet-5-5" and kw["system"] == generate.SYSTEM_PROMPT
    assert kw["output_config"]["effort"] == "medium" and "temperature" not in kw


@pytest.mark.parametrize("reply, error", [
    (_ans(True, ("ㄱ", 1), ("ㄴ", 2), ("ㄷ", 3)), "too_many_claims"),
    (_ans(True, ("ㄱ", 6)), "source_out_of_range"),
    (_ans(True), "answerable_without_claims"),
    (_ans(False, ("ㄱ", 1)), "claims_when_unanswerable"),
    (("end_turn", "{깨진"), "json_parse"),
    (("max_tokens", "{}"), "stop_reason=max_tokens"),
    (("refusal", ""), "stop_reason=refusal"),
])
def test_bad_responses_become_failed_rows_without_retry(reply, error):
    client = FakeClient([reply])
    rows = generate.generate(client, QUERIES[:1], generate.passed_chunks(QUERIES, RUN), CHUNKS)
    assert rows[0]["error"] == error and rows[0]["claims"] == [] and rows[0]["answerable"] is None
    assert len(client.calls) == 1


def test_main_refuses_to_overwrite_existing_answers(tmp_path):
    write_jsonl(tmp_path / "a.jsonl", [])
    with pytest.raises(SystemExit, match="이미 있다"):
        generate.main(["--out", str(tmp_path / "a.jsonl")])


def test_llm_judge_waits_for_human_judgments(tmp_path):
    write_jsonl(tmp_path / "a.jsonl", [{"qid": "q1", "claims": [{"claim_id": "q1:1", "text": "ㄱ", "chunk_id": "1:a:0"}]}])
    with pytest.raises(SystemExit, match="사람 판정이 1건 남음"):
        generate.main(["--llm-judge", "--out", str(tmp_path / "a.jsonl"),
                       "--judgments", str(tmp_path / "none.jsonl"), "--llm-out", str(tmp_path / "l.jsonl")])


def test_llm_judge_rows():
    answers = [{"qid": "q1", "claims": [{"claim_id": "q1:1", "text": "ㄱ", "chunk_id": "1:a:0"},
                                        {"claim_id": "q1:2", "text": "ㄴ", "chunk_id": "2:a:0"}]}]
    client = FakeClient([("end_turn", '{"supported": true}'), ("max_tokens", "")])
    rows = generate.llm_judge(client, answers, CHUNKS)
    assert [(r["claim_id"], r["supported"], r["error"]) for r in rows] == [
        ("q1:1", True, None), ("q1:2", None, "stop_reason=max_tokens")]
    assert client.calls[0]["model"] == "claude-opus-5-5" and "주장: ㄱ" in client.calls[0]["messages"][0]["content"]


# ── 주장 판정 ─────────────────────────────────────────────────────────
def test_claim_session_has_no_default_and_saves_each(monkeypatch, tmp_path, capsys):
    todo = [{"claim_id": "q1:1", "qid": "q1", "text": "주장ㄱ", "chunk_id": "1:a:0"},
            {"claim_id": "q1:2", "qid": "q1", "text": "주장ㄴ", "chunk_id": "2:a:0"}]
    out, judged = tmp_path / "j.jsonl", {}
    _feed(monkeypatch, ["", "y", "q"])
    assert claim_judge.session(todo, {"q1": "가"}, CHUNKS, judged, out) == 1
    assert claim_judge.load_claim_judgments(out) == {"q1:1": True}
    shown = capsys.readouterr().out
    assert "주장: 주장ㄱ" in shown and "본문1" in shown and "claude" not in shown


def test_rejudge_ids_are_ten_percent_rounded_up_and_stable():
    ids = [f"q{i}:1" for i in range(41)]
    assert len(claim_judge.rejudge_ids(ids)) == 5
    assert claim_judge.rejudge_ids(ids) == claim_judge.rejudge_ids(list(reversed(ids)))


# ── 생성 평가 ─────────────────────────────────────────────────────────
def _answers():
    def row(qid, answerable, claims, error=None):
        return {"qid": qid, "model": "m", "retriever": "hybrid", "passed": ["1:a:0", "2:a:0"],
                "answerable": answerable, "error": error, "ms": 100.0, "input_tokens": 10, "output_tokens": 5,
                "claims": [{"claim_id": f"{qid}:{i}", "text": "t", "chunk_id": c} for i, c in enumerate(claims, 1)]}
    return [row("q1", True, ["1:a:0", "2:a:0"]), row("q2", True, ["2:a:0"]), row("q3", False, [])]


SEARCH = {("q1", "1:a:0"): True, ("q1", "2:a:0"): False, ("q2", "1:a:0"): False, ("q2", "2:a:0"): False,
          ("q3", "1:a:0"): False, ("q3", "2:a:0"): False}
HUMAN = {"q1:1": True, "q1:2": False, "q2:1": True}


def test_gen_eval_values_by_hand():
    llm = [{"claim_id": "q1:1", "model": "j", "supported": True, "error": None},
           {"claim_id": "q1:2", "model": "j", "supported": True, "error": None},
           {"claim_id": "q2:1", "model": "j", "supported": None, "error": "json_parse"}]
    rows = rag_gen_eval.evaluate(QUERIES, _answers(), SEARCH, HUMAN, {"q1:1": True}, llm, b=200)
    assert all(list(r) == ["항목", "값", "근거"] and r["근거"] for r in rows)
    by = {r["항목"]: r["값"] for r in rows}
    assert by["충실도 (주 지표)"].startswith("0.667")          # y 2 / 주장 3
    assert by["관련 문단 있음 · 답함"] == "1" and by["관련 문단 없음 · 답함"] == "1"   # q2 는 근거 없이 답함
    assert by["관련 문단 없음 · 거절"] == "1" and by["올바른 거절률"] == "0.500"
    assert by["인용 적합도 (보조)"] == "0.333"                  # q1:1 만 검색 판정 y 문단을 인용
    assert by["자기 일치율 (재판정)"] == "1.000"
    assert by["LLM 판정 일치율 (보조)"] == "0.500"              # 유효 2건 중 q1:1 만 일치
    assert by["LLM 판정 2×2 (사람 y/n × LLM y/n)"] == "yy 1 · yn 0 · ny 1 · nn 0"


def test_gen_eval_stops_on_unjudged_claim():
    with pytest.raises(SystemExit, match="미판정 주장 1건"):
        rag_gen_eval.evaluate(QUERIES, _answers(), SEARCH, {"q1:1": True, "q1:2": False}, b=10)


def test_gen_eval_counts_failed_rows_and_flags_suspicious_score():
    answers = _answers()
    answers[1] = dict(answers[1], error="json_parse", answerable=None, claims=[])
    by = {r["항목"]: r for r in rag_gen_eval.evaluate(QUERIES, answers, SEARCH, {"q1:1": True, "q1:2": True}, b=50)}
    assert by["실패 행"]["값"] == "1" and "q2(json_parse)" in by["실패 행"]["근거"]
    assert "0.98 이상" in by["충실도 (주 지표)"]["근거"]
    assert by["LLM 판정 일치율 (보조)"]["값"] == "미실시"
