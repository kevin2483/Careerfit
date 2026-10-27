"""추출 리포트 — 합성 이상입력에서 불변식(입력 = 통과 + 모든 스킵 사유)이 유지되고, 결측 파일이 후속 단계를 깨지 않는다."""
import csv
import json

from src.analyze.market_stats import load_population
from src.common.paths import DICT_PATH
from src.extract.job_skill_extract_report import VALIDATION_COLUMNS, run

DATE = "2099-01-01"


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rec(job_id, title, body, sections=None):
    return json.dumps({"job_id": job_id, "title": title, "body_raw": body,
                       "sections": sections or {"main_tasks": body, "requirements": body, "preferred_points": "x"}},
                      ensure_ascii=False)


def test_invariant_with_synthetic_bad_inputs(tmp_path):
    raw = tmp_path / "raw"
    body = "[자격요건]\nPython과 SQL을 활용한 데이터 파이프라인 설계 및 운영 경험. " * 3
    _write(raw / "ds" / f"wanted_{DATE}.jsonl", [
        _rec("1", "데이터 사이언티스트", body),            # 정상
        json.dumps({"job_id": "2", "title": "x"}),         # 필드누락 (body 없음)
        _rec("3", "짧은 공고", "너무 짧다"),                # 본문결측
        "{broken json",                                    # 파싱실패
    ])
    _write(raw / "da" / f"wanted_{DATE}.jsonl", [
        _rec("10", "데이터 분석가", body),                 # 통과
        _rec("11", "화장품 마케팅 팀장", body),            # 직무불일치
    ])
    # de, mle 파일은 일부러 만들지 않는다 (결측 파일 경로)
    out = tmp_path / "out"
    r = run(raw_dir=raw, dict_path=DICT_PATH, out_dir=out, date=DATE)

    assert r["총입력"] == 6 and r["통과"] == 2 and r["직무불일치"] == 1
    with open(out / "validation_summary.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["직무", *VALIDATION_COLUMNS]
    for row in rows[1:]:
        assert len(row) == len(VALIDATION_COLUMNS) + 1, row         # 결측 직무도 열 수가 같다
        입력, *나머지 = map(int, row[1:])
        assert 입력 == sum(나머지), row                                # 불변식
    assert rows[-1] == ["전체합계", "6", "2", "1", "1", "1", "1"]

    # 후속 단계가 결측 직무 행을 읽을 수 있다 (예전엔 int(None) 으로 죽었다)
    pop = load_population(out / "validation_summary.csv")
    assert pop["de"]["통과"] == 0 and pop["ds"]["통과"] == 1

    tags = [json.loads(l) for l in (out / "extracted_tags.jsonl").read_text(encoding="utf-8").splitlines() if l]
    assert {t["job_id"] for t in tags} == {"1", "10"}
    assert "Python" in tags[0]["tags"] and "SQL" in tags[0]["tags"]
    assert "requirements" in tags[0]["matched_fields"]


def test_run_is_deterministic(tmp_path):
    raw = tmp_path / "raw"
    body = "[자격요건]\nAirflow, Spark 경험. Kafka 우대. " * 5
    _write(raw / "de" / f"wanted_{DATE}.jsonl", [_rec(str(i), "데이터 엔지니어", body) for i in range(5)])
    outs = []
    for k in range(2):
        out = tmp_path / f"out{k}"
        run(raw_dir=raw, dict_path=DICT_PATH, out_dir=out, date=DATE)
        outs.append({f: (out / f).read_bytes() for f in ("validation_summary.csv", "tag_frequency_by_job.csv",
                                                          "extracted_tags.jsonl", "field_difficulty_summary.csv")})
    assert outs[0] == outs[1]
