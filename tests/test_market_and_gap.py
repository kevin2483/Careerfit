"""시장 통계 판단 기준(3건 미만 제외, 출현율 분모=통과)과 갭 진단 산식(출현율 × 미보유 × 신입접근성, top5)."""
import csv

from src.analyze import gap_diagnosis as gd
from src.analyze.market_stats import (compute_cooccurrence, compute_excluded, compute_junior_vs_senior,
                                      compute_rank_by_job, implied_pairs)


def test_lowfreq_excluded_and_pct_uses_pass_count():
    freq = [{"canonical": "Python", "ds": "10", "de": "2", "mle": "0", "da": "3"},
            {"canonical": "R", "ds": "1", "de": "0", "mle": "0", "da": "0"}]
    excluded, rows = compute_excluded(freq)
    assert excluded == {("Python", "de"), ("R", "ds")}
    pop = {"ds": {"통과": 20, "입력공고수": 20, "직무불일치": 0}, "de": {"통과": 5, "입력공고수": 5, "직무불일치": 0},
           "mle": {"통과": 0, "입력공고수": 0, "직무불일치": 0}, "da": {"통과": 6, "입력공고수": 12, "직무불일치": 6}}
    rank = compute_rank_by_job(freq, pop, excluded, da_note="da 비고")
    by = {(r["canonical"], r["직무"]): r for r in rank}
    assert ("Python", "de") not in by and ("R", "ds") not in by
    assert by[("Python", "ds")]["출현율%"] == 50.0
    assert by[("Python", "da")]["출현율%"] == 50.0 and by[("Python", "da")]["비고"] == "da 비고"  # 분모 6, 12 아님


def test_cooccurrence_skips_implied_pairs():
    ext = [{"job_id": "1", "직무": "de", "tags": ["PySpark", "Spark", "Python"]}]
    rows = compute_cooccurrence(ext, set(), skip_pairs=implied_pairs({"PySpark": ["Spark"]}))
    assert {(r["tag_a"], r["tag_b"]) for r in rows} == {("PySpark", "Python"), ("Python", "Spark")}
    assert len(compute_cooccurrence(ext, set())) == 3


def test_junior_vs_senior_buckets():
    ext = [{"job_id": "1", "직무": "ds", "tags": ["Python"]}, {"job_id": "2", "직무": "ds", "tags": ["Python"]},
           {"job_id": "3", "직무": "ds", "tags": ["Python"]}]
    rows = compute_junior_vs_senior(ext, {"1": 0, "2": 3, "3": None}, set())
    assert rows == [{"canonical": "Python", "직무": "ds", "신입가능": 1, "경력요구": 1, "미기재": 1,
                     "신입가능%": 33.3, "경력요구%": 33.3}]


def _csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_gap_diagnosis_formula_and_split(tmp_path):
    rank = tmp_path / "rank.csv"
    jvs = tmp_path / "jvs.csv"
    tags = ["Python", "SQL", "Airflow", "Kafka", "Spark", "Docker", "dbt", "협업커뮤니케이션"]
    _csv(rank, ["canonical", "직무", "출현수", "모집단수", "출현율%", "비고", "순위"],
         [[t, "de", 10, 100, 80 - 10 * i, "", i + 1] for i, t in enumerate(tags)])
    _csv(jvs, ["canonical", "직무", "신입가능", "경력요구", "미기재", "신입가능%", "경력요구%"],
         [[t, "de", 5, 5, 0, 50.0, 50.0] for t in tags])
    rank.touch()  # analyst 산출물이 extractor 산출물보다 최신이어야 한다

    r = gd.run(resume_text="Python과 SQL을 씁니다", target_job="de",
               rank_by_job_path=rank, junior_senior_path=jvs, top_n=5)
    assert r["보유역량"] == ["Python", "SQL"]
    top = r["이번분기_학습로드맵(top5)"]
    assert [x["canonical"] for x in top] == ["Airflow", "Kafka", "Spark", "Docker", "dbt"]
    assert top[0]["우선순위점수"] == round(60 * 1.0 * 0.5, 4)
    assert r["다음분기_후순위"] == []
    assert [x["canonical"] for x in r["제외_field3_암묵조건"]] == ["협업커뮤니케이션"]
