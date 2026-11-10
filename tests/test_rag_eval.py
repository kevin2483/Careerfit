"""판정 도구·검색 평가 — 합성 실행 기록과 합성 판정만 쓴다 (실제 질문·판정 없음)."""
import builtins
import math

import pytest

from src.common.jsonl import read_jsonl, write_jsonl
from src.eval import rag_eval
from src.rag import judge
from src.rag.run import METHODS

QUERIES = [{"qid": "q1", "type": "A", "question": "가", "filter": {}},
           {"qid": "q2", "type": "B", "question": "나", "filter": {}},
           {"qid": "q3", "type": "C", "question": "다", "filter": {}},   # 풀 안 정답 0개
           {"qid": "q4", "type": "D", "question": "라", "filter": {}}]


def _run(order: dict[str, list[str]], score=1.0):
    return [{"qid": q, "rank": i, "chunk_id": c, "score": score - i * 0.01}
            for q, ids in order.items() for i, c in enumerate(ids, 1)]


def _runs():
    good = {"q1": ["1:a:0", "2:a:0", "3:a:0"], "q2": ["5:a:0", "6:a:0"], "q3": ["8:a:0"], "q4": ["9:a:0"]}
    bad = {"q1": ["3:a:0", "4:a:0", "1:a:0"], "q2": ["6:a:0", "7:a:0"], "q3": ["8:a:0"], "q4": ["9:a:0"]}
    return {m: _run(good if i % 2 == 0 else bad) for i, m in enumerate(METHODS)}


RELEVANT = {("q1", "1:a:0"), ("q1", "2:a:0"), ("q2", "5:a:0")}


def _judgments(runs):
    return {(q, c): (q, c) in RELEVANT for q, ids in judge.build_pool(runs).items() for c in ids}


def _feed(monkeypatch, answers):
    it = iter(answers)

    def fake(_prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr(builtins, "input", fake)


# ── 풀링 ──────────────────────────────────────────────────────────────
def test_load_runs_refuses_until_all_six_methods_exist(tmp_path):
    write_jsonl(tmp_path / "bm25.jsonl", [])
    with pytest.raises(SystemExit, match="실행 기록 없는 방식"):
        judge.load_runs(tmp_path)


def test_pool_is_union_without_method_info_and_order_is_stable():
    pool = judge.build_pool(_runs())
    assert sorted(pool["q1"]) == ["1:a:0", "2:a:0", "3:a:0", "4:a:0"]
    assert pool == judge.build_pool(_runs())                     # 재개해도 같은 순서
    assert all(isinstance(c, str) for ids in pool.values() for c in ids)   # 방식·순위·점수가 따라오지 않는다


def test_pool_ignores_ranks_beyond_top_k():
    runs = {"bm25": [{"qid": "q1", "rank": r, "chunk_id": f"{r}:a:0", "score": 0.0} for r in range(1, 13)]}
    assert len(judge.build_pool(runs)["q1"]) == 10


# ── y/n 강제 ──────────────────────────────────────────────────────────
def test_ask_yn_has_no_default(monkeypatch, capsys):
    _feed(monkeypatch, ["", "  ", "yes", "Y"])
    assert judge.ask_yn("? ") == "y"
    assert capsys.readouterr().out.count("엔터만으로는") == 3


def test_ask_yn_treats_eof_as_quit_not_as_answer(monkeypatch):
    _feed(monkeypatch, [])
    assert judge.ask_yn("? ") == "q"


def test_session_saves_each_judgment_and_stops_on_q(monkeypatch, tmp_path):
    queries = {q["qid"]: q for q in QUERIES}
    chunks = {c: {"chunk_id": c, "text": "본문", "title": "t", "company": "c", "section": "a"}
              for c in ["1:a:0", "2:a:0", "3:a:0"]}
    todo = [("q1", "1:a:0"), ("q1", "2:a:0"), ("q1", "3:a:0")]
    out, judged = tmp_path / "j.jsonl", {}
    _feed(monkeypatch, ["y", "", "n", "q"])
    assert judge.session(todo, queries, chunks, judged, out) == 2
    assert judge.load_judgments(out) == {("q1", "1:a:0"): True, ("q1", "2:a:0"): False}


def test_screen_never_shows_method_names(monkeypatch, capsys, tmp_path):
    queries = {q["qid"]: q for q in QUERIES}
    chunks = {"1:a:0": {"chunk_id": "1:a:0", "text": "본문", "title": "t", "company": "c", "section": "a"}}
    _feed(monkeypatch, ["y"])
    judge.session([("q1", "1:a:0")], queries, chunks, {}, tmp_path / "j.jsonl")
    shown = capsys.readouterr().out
    assert not any(m in shown for m in METHODS)


def test_rejudge_sample_is_ten_percent_and_deterministic():
    pairs = [("q1", f"{i}:a:0") for i in range(95)]
    sample = judge.rejudge_sample(pairs)
    assert len(sample) == 10 and set(sample) <= set(pairs)
    assert sample == judge.rejudge_sample(list(reversed(pairs)))


def test_rejudge_refuses_while_first_pass_incomplete(tmp_path):
    for m, rows in _runs().items():
        write_jsonl(tmp_path / f"{m}.jsonl", rows)
    write_jsonl(tmp_path / "q.jsonl", QUERIES)
    with pytest.raises(SystemExit, match="1차 판정이"):
        judge.main(["--runs-dir", str(tmp_path), "--queries", str(tmp_path / "q.jsonl"),
                    "--out", str(tmp_path / "j.jsonl"), "--rejudge"])


def test_bad_judgment_value_is_rejected(tmp_path):
    write_jsonl(tmp_path / "j.jsonl", [{"qid": "q1", "chunk_id": "1:a:0", "relevant": "y"}])
    with pytest.raises(SystemExit, match="true/false"):
        judge.load_judgments(tmp_path / "j.jsonl")


# ── 지표 ──────────────────────────────────────────────────────────────
def test_query_metrics_by_hand():
    m = rag_eval.query_metrics(["x:a:0", "1:a:0", "1:b:0"], {"1:a:0", "1:b:0", "2:a:0"})
    assert m["Recall@10"] == pytest.approx(2 / 3)
    assert m["MRR@10"] == pytest.approx(1 / 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert m["nDCG@10"] == pytest.approx(dcg / idcg)
    assert m["공고 Recall@10"] == pytest.approx(1 / 2)   # 공고 1 은 찾았고 2 는 못 찾음


def test_ndcg_ideal_is_capped_at_ten_when_more_relevant_than_k():
    rel = {f"{i}:a:0" for i in range(15)}
    m = rag_eval.query_metrics([f"{i}:a:0" for i in range(10)], rel)
    assert m["nDCG@10"] == pytest.approx(1.0)       # 상위 10 이 전부 정답이면 1
    assert m["Recall@10"] == pytest.approx(10 / 15)  # Recall 은 상한이 1 미만 — nDCG 를 같이 보는 이유


def test_unjudged_chunk_stops_evaluation():
    runs = _runs()
    j = _judgments(runs)
    j.pop(("q1", "4:a:0"))
    with pytest.raises(SystemExit, match="미판정"):
        rag_eval.evaluate(QUERIES, runs, j, b=50)


def test_evaluate_rows_have_three_columns_and_expected_values():
    runs = _runs()
    rows = rag_eval.evaluate(QUERIES, runs, _judgments(runs), b=200)
    assert all(list(r) == ["항목", "값", "근거"] and r["근거"] for r in rows)
    by = {r["항목"]: r["값"] for r in rows}
    assert by["평가 질문 수 (A~C, 풀 안 정답 ≥1)"] == "2"
    assert by["정답 0개라 제외한 A~C 질문"] == "1"
    assert by["D 유형 중 정답이 발견된 질문"] == "0 / 1"
    assert by["bm25 · Recall@10"].startswith("1.000")          # good: q1 2/2, q2 1/1
    assert by["vector · Recall@10"].startswith("0.250")        # bad:  q1 1/2, q2 0/1
    assert by["자기 일치율 (재판정)"] == "미실시"
    assert "hybrid_rerank · D 유형 상위 1 점수" not in by       # 순위 기반 점수는 보고하지 않는다


def test_only_primary_pairs_get_a_verdict():
    runs = _runs()
    by = {r["항목"]: r["값"] for r in rag_eval.evaluate(QUERIES, runs, _judgments(runs), b=200)}
    primary = by["hybrid − vector · Recall@10"]
    assert primary.endswith("확정") or primary.endswith("확정 불가")
    assert by["bm25 − sec3 · Recall@10"].endswith("참고 (판정 안 함)")
    assert sum(k.endswith("· Recall@10") and "−" in k and not v.endswith("참고 (판정 안 함)")
               for k, v in by.items()) == len(rag_eval.PRIMARY_PAIRS)


def test_identical_methods_are_not_confirmed():
    runs = _runs()   # bm25 와 hybrid 는 같은 순위 → 차이 0, CI [0, 0]
    by = {r["항목"]: r["값"] for r in rag_eval.evaluate(QUERIES, runs, _judgments(runs), b=200)}
    assert by["hybrid − bm25 · Recall@10"].endswith("확정 불가")


def test_self_agreement_uses_rejudgments_but_scores_use_first_pass():
    runs = _runs()
    j = _judgments(runs)
    rej = {("q1", "1:a:0"): True, ("q1", "2:a:0"): False}
    by = {r["항목"]: r["값"] for r in rag_eval.evaluate(QUERIES, runs, j, rej, b=50)}
    assert by["자기 일치율 (재판정)"] == "0.500"
    assert by["bm25 · Recall@10"].startswith("1.000")


def test_cli_output_is_byte_reproducible(tmp_path):
    runs = _runs()
    for m, rows in runs.items():
        write_jsonl(tmp_path / f"{m}.jsonl", rows)
    write_jsonl(tmp_path / "q.jsonl", QUERIES)
    judge.save_judgments(tmp_path / "j.jsonl", _judgments(runs))
    outs = []
    for name in ("a.csv", "b.csv"):
        rag_eval.main(["--queries", str(tmp_path / "q.jsonl"), "--runs-dir", str(tmp_path),
                       "--judgments", str(tmp_path / "j.jsonl"), "--rejudgments", str(tmp_path / "none.jsonl"),
                       "--b", "100", "--out", str(tmp_path / name)])
        outs.append((tmp_path / name).read_bytes())
    assert outs[0] == outs[1]
    assert read_jsonl(tmp_path / "j.jsonl")[0].keys() == {"qid", "chunk_id", "relevant"}
