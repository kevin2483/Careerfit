"""데모 로더 — 커밋된 CSV 의 컬럼이 바뀌면 데모가 깨지므로 CI 에서 같이 검사한다. 계산은 하지 않고 읽기만 확인."""
from demo import loaders as L
from src.analyze import gap_diagnosis as gd


def test_market_loaders_have_expected_columns():
    for job in L.JOB_NAMES:
        rank = L.rank_by_job(job, 20)
        assert 1 <= len(rank) <= 20 and {"canonical", "출현율%", "순위"} <= set(rank[0])
        assert [r["순위"] for r in rank] == sorted(r["순위"] for r in rank)
        assert {"신입가능%", "경력요구%"} <= set(L.junior_vs_senior(job)[0])
        assert {"tag_a", "tag_b", "동시출현수"} <= set(L.cooccurrence(job)[0])
    pop = L.population()
    assert set(pop) == set(L.JOB_NAMES) and pop["da"]["직무불일치"] > 0 and L.da_note("da")


def test_eval_loaders_match_readme_shape():
    tot = L.totals()
    assert [r["방식"] for r in tot] == ["regex", "haiku", "sonnet", "opus"]
    assert all(isinstance(r["전체비용"], (int, float)) for r in tot)
    assert {r["방식"] for r in L.field_f1()} == set(L.METHOD_NAMES.values())
    assert len(L.paired_diffs()) == 6
    curves = L.routing_curves()
    assert {c["정책"] for c in curves} == {"length", "f2count", "random"}
    assert all(c["F1_sd"] > 0 for c in curves if c["정책"] == "random")


def test_gap_diagnosis_runs_without_freshness_check():
    """데모는 git clone 직후(mtime 무의미) 돌아가야 한다. 산식은 같은 run() 이다."""
    sample = open("demo/sample_resume.txt", encoding="utf-8").read()
    r = gd.run(resume_text=sample, target_job="de", check_fresh=False)
    assert "Python" in r["보유역량"] and "SQL" in r["보유역량"]
    assert 1 <= len(r["이번분기_학습로드맵(top5)"]) <= 5
    assert all(x["canonical"] not in r["보유역량"] for x in r["이번분기_학습로드맵(top5)"])
