from src.rag.bm25 import BM25, tokenize


def test_tokenize_handles_particles_and_tech_names():
    assert tokenize("Spark를 활용한") == ["spark", "활용", "용한"]
    assert tokenize("C++ 및 Node.js") == ["c++", "node.js"]
    # 조사가 달라도 어간 2-gram 은 공유된다
    assert set(tokenize("파이프라인을")) & set(tokenize("파이프라인의")) >= {"파이", "이프", "프라", "라인"}


def test_bm25_ranks_matching_doc_first_and_ignores_unknown_terms():
    docs = ["Python 과 SQL 경험", "쿠버네티스 운영 경험", "복지: 재택근무 가능"]
    idx = BM25(docs)
    assert idx.search("쿠버네티스를 다뤄 본 분", k=1)[0][0] == 1
    assert idx.search("zzz 없는단어") == []


def test_rare_term_outweighs_common_term():
    docs = ["경험 경험 경험", "경험 airflow", "경험"]
    top = BM25(docs).search("airflow 경험", k=3)
    assert top[0][0] == 1
