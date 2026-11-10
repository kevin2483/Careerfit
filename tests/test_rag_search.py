"""필터·hybrid·실행 기록 — 합성 데이터만 쓴다 (실제 질문·임베딩 모델 없이 돈다)."""
import pytest

from src.common.jsonl import read_jsonl, write_jsonl
from src.rag import run as rag_run
from src.rag.bm25 import BM25
from src.rag.filter import allowed_mask, validate
from src.rag.hybrid import HybridIndex, rrf


def _chunk(cid, text, annual, jobs):
    return {"chunk_id": cid, "text": text, "annual_from": annual, "all_jobs": jobs}


CHUNKS = [
    _chunk("1:requirements:0", "Airflow 배치 파이프라인 운영", 0, ["de"]),
    _chunk("2:requirements:0", "Airflow 와 Spark 경험", 3, ["de", "mle"]),
    _chunk("3:requirements:0", "Airflow 워크플로 관리", None, ["ds"]),
    _chunk("4:benefits:0", "재택 근무 가능", 0, ["da"]),
]


# ── 필터 ──────────────────────────────────────────────────────────────
def test_empty_filter_means_no_mask():
    assert allowed_mask(CHUNKS, {}) is None
    assert allowed_mask(CHUNKS, None) is None


def test_annual_max_zero_excludes_experienced_and_unstated():
    # 부록 A2: None(미기재)은 "신입 가능"이 확인되지 않았으므로 제외
    assert allowed_mask(CHUNKS, {"annual_max": 0}) == [True, False, False, True]


def test_jobs_filter_matches_any_merged_job():
    assert allowed_mask(CHUNKS, {"jobs": ["mle"]}) == [False, True, False, False]
    assert allowed_mask(CHUNKS, {"annual_max": 0, "jobs": ["de", "ds"]}) == [True, False, False, False]


@pytest.mark.parametrize("bad", [{"anual_max": 0}, {"annual_max": "0"}, {"annual_max": True},
                                 {"jobs": []}, {"jobs": ["pm"]}, {"jobs": "de"}])
def test_bad_filter_stops_instead_of_running_unfiltered(bad):
    with pytest.raises(ValueError):
        validate(bad)


def test_bm25_filter_cuts_before_top_k_and_keeps_corpus_stats():
    bm = BM25([c["text"] for c in CHUNKS])
    plain = dict(bm.search("Airflow", 10))
    masked = bm.search("Airflow", 1, allowed=allowed_mask(CHUNKS, {"annual_max": 3, "jobs": ["mle"]}))
    assert [i for i, _ in masked] == [1]
    assert masked[0][1] == plain[1]   # 점수는 필터와 무관 (IDF·평균 길이는 색인 전체 기준)


def test_vector_filter_cuts_before_top_k(monkeypatch):
    np = pytest.importorskip("numpy")   # numpy 는 requirements-rag.txt 에만 있다 (CI 는 건너뜀)
    from src.rag import vector
    mat = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(vector, "embed", lambda texts, model, kind="passage": np.array([[1.0, 0.0]], dtype=np.float32))
    ix = vector.VectorIndex(mat)
    assert [i for i, _ in ix.search("q", 2)] == [0, 1]
    assert [i for i, _ in ix.search("q", 2, allowed=[False, False, True, True])] == [2, 3]
    assert [i for i, _ in ix.search("q", 5, allowed=[False, True, False, False])] == [1]   # k 미만이면 있는 만큼만


# ── hybrid ────────────────────────────────────────────────────────────
def test_rrf_uses_ranks_not_scores():
    fused = rrf([[10, 20, 30], [20, 40]], k=60)
    assert [d for d, _ in fused] == [20, 10, 40, 30]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_tie_breaks_by_doc_index():
    assert [d for d, _ in rrf([[5], [3]], k=60)] == [3, 5]


class _Fake:
    def __init__(self, order):
        self.order, self.calls = order, []

    def search(self, query, k=10, allowed=None):
        self.calls.append((k, allowed))
        return [(i, 0.0) for i in self.order if allowed is None or allowed[i]][:k]


def test_hybrid_asks_each_index_for_depth_and_passes_filter():
    a, b = _Fake([0, 1, 2, 3]), _Fake([3, 2, 1, 0])
    mask = [True, False, True, True]
    out = HybridIndex(a, b, depth=50).search("q", 2, allowed=mask)
    assert a.calls == [(50, mask)] and b.calls == [(50, mask)]
    assert len(out) == 2 and all(mask[i] for i, _ in out)


# ── 실행 기록 ─────────────────────────────────────────────────────────
def _queries(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    write_jsonl(p, rows)
    return p


def test_load_queries_refuses_blank_questions(tmp_path):
    p = _queries(tmp_path, [{"qid": "q01", "type": "A", "question": "", "filter": {}}])
    with pytest.raises(SystemExit, match="비어 있음"):
        rag_run.load_queries(p)


def test_load_queries_refuses_duplicates_bad_type_and_bad_filter(tmp_path):
    p = _queries(tmp_path, [{"qid": "q01", "type": "A", "question": "x", "filter": {}},
                            {"qid": "q01", "type": "Z", "question": "y", "filter": {"jobs": ["pm"]}}])
    with pytest.raises(SystemExit) as e:
        rag_run.load_queries(p)
    msg = str(e.value)
    assert "중복" in msg and "유형" in msg and "jobs" in msg


def test_run_method_writes_rank_rows_with_filter_applied():
    queries = [{"qid": "q01", "type": "A", "question": "Airflow", "filter": {}},
               {"qid": "q02", "type": "A", "question": "Airflow", "filter": {"annual_max": 0}}]
    rows = rag_run.run_method(BM25([c["text"] for c in CHUNKS]), CHUNKS, queries)
    assert set(rows[0]) == {"qid", "rank", "chunk_id", "score"}
    assert [r["rank"] for r in rows if r["qid"] == "q01"] == [1, 2, 3]
    assert [r["chunk_id"] for r in rows if r["qid"] == "q02"] == ["1:requirements:0"]


def test_main_refuses_to_overwrite_existing_run(tmp_path):
    (tmp_path / "bm25.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="이미 실행 기록"):
        rag_run.main(["--method", "bm25", "--out-dir", str(tmp_path)])


def test_committed_query_template_is_not_runnable_until_filled():
    """질문 틀이 비어 있는 동안에는 실행 기록을 만들 수 없어야 한다 (사람이 채우기 전 실수 방지)."""
    if not rag_run.QUERIES_PATH.exists():
        pytest.skip("질문 파일 없음")
    rows = read_jsonl(rag_run.QUERIES_PATH)
    if all(str(r.get("question") or "").strip() for r in rows):
        rag_run.load_queries()   # 다 채워졌으면 형식 검사를 통과해야 한다
    else:
        with pytest.raises(SystemExit):
            rag_run.load_queries()


# ── 변형·지연 ─────────────────────────────────────────────────────────
def test_masked_index_combines_base_mask_with_query_filter():
    from src.rag.filter import MaskedIndex
    inner = _Fake([0, 1, 2, 3])
    ix = MaskedIndex(inner, [True, True, False, True])
    assert [i for i, _ in ix.search("q", 10)] == [0, 1, 3]
    assert [i for i, _ in ix.search("q", 10, allowed=[False, True, True, True])] == [1, 3]


def test_ctx_prefix_text_puts_title_company_section_before_body():
    pytest.importorskip("numpy")
    from src.rag.vector import cache_paths, ctx_prefix_text
    c = {"title": "데이터 엔지니어", "company": "가나다", "section": "preferred_points", "text": "본문"}
    assert ctx_prefix_text(c) == "데이터 엔지니어 · 가나다 · 우대사항\n본문"
    assert cache_paths("2026-01-01", "a/b")[0] != cache_paths("2026-01-01", "a/b", "ctx_prefix")[0]   # 캐시가 섞이지 않는다


def test_run_method_records_one_timing_per_query():
    queries = [{"qid": "q01", "type": "A", "question": "Airflow", "filter": {}},
               {"qid": "q02", "type": "A", "question": "없는단어", "filter": {}}]
    timings = []
    rag_run.run_method(BM25([c["text"] for c in CHUNKS]), CHUNKS, queries, timings=timings)
    assert [t["qid"] for t in timings] == ["q01", "q02"] and all(t["ms"] >= 0 for t in timings)
