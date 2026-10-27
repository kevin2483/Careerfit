"""부트스트랩 CI — 산식과 결정성."""
from src.eval.bootstrap_ci import doc_counts, f1_from, percentile


def test_percentile_edges():
    xs = [0.1, 0.2, 0.3, 0.4]
    assert percentile(xs, 0.0) == 0.1 and percentile(xs, 1.0) == 0.4
    assert abs(percentile(xs, 0.5) - 0.25) < 1e-12


def test_f1_from_counts():
    counts = [(1, 0, 0), (1, 1, 1)]                      # 전체 TP2 FP1 FN1 → F1 = 4/(4+2)
    assert abs(f1_from(counts, [0, 1]) - 2 / 3) < 1e-12
    assert f1_from(counts, [0]) == 1.0
    assert doc_counts({"Python", "협업커뮤니케이션"}, {"Python", "SQL"}, {"협업커뮤니케이션": 3}) == (1, 1, 0)


def test_run_is_seeded(tmp_path):
    """같은 입력·시드면 같은 CI. 공개 데이터로 돌리는 CI 가 바이트 단위 diff 를 하므로 중요하다."""
    import json
    from pathlib import Path
    from src.eval.bootstrap_ci import run
    labels = tmp_path / "labels.jsonl"
    preds = tmp_path / "p"
    preds.mkdir()
    labels.write_text("".join(json.dumps({"job_id": str(i), "reviewed": True, "gold_tags": ["Python", "SQL"]}) + "\n"
                              for i in range(10)), encoding="utf-8")
    (preds / "a.jsonl").write_text("".join(json.dumps({"job_id": str(i), "tags": ["Python"] + (["SQL"] if i % 2 else [])}) + "\n"
                                           for i in range(10)), encoding="utf-8")
    out1 = run(labels, Path(preds), {}, b=200, seed=1, methods={"a": "a.jsonl"})
    out2 = run(labels, Path(preds), {}, b=200, seed=1, methods={"a": "a.jsonl"})
    assert out1 == out2
    n, rows = out1
    assert n == 10 and rows[0]["CI_하한"] <= rows[0]["점추정"] <= rows[0]["CI_상한"]
