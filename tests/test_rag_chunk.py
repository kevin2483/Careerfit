from src.rag.chunk import make_chunks, split_long


def test_short_section_is_one_chunk():
    assert split_long("  Python 3년 이상  ") == ["Python 3년 이상"]
    assert split_long("   ") == []


def test_long_section_splits_on_paragraphs_under_limit():
    paras = [f"문단{i} " + "가" * 50 for i in range(10)]
    pieces = split_long("\n\n".join(paras), max_chars=130)
    assert all(len(p) <= 130 for p in pieces)
    assert "".join(pieces).replace("\n", "") == "".join(paras)  # 내용 손실 없음


def test_tiny_tail_is_merged_into_previous():
    text = "가" * 100 + "\n\n" + "끝"
    pieces = split_long(text, max_chars=100)
    assert pieces == ["가" * 100 + "\n끝"]


def test_chunks_dedup_by_job_and_carry_metadata():
    postings = {"7": {"title": "t", "company": "c", "all_jobs": ["ds", "mle"],
                      "sections": {"requirements": "SQL", "benefits": ""}}}
    chunks = make_chunks(postings)
    assert [c["chunk_id"] for c in chunks] == ["7:requirements:0"]
    assert chunks[0]["all_jobs"] == ["ds", "mle"]
