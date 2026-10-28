"""
채용 역량 추출 리포트 (job-skill-extract-report) — 정규식 베이스라인

입력: data/raw/{ds,de,mle,da}/wanted_{date}.jsonl + 사전(기본 src/common/paths.DICT_PATH)
출력: {out_dir}/
  validation_summary.csv       직무별 입력·통과·스킵 사유 (불변식: 입력 = 통과 + 모든 스킵 사유)
  extracted_tags.jsonl         공고별 태그 (직무순 → job_id 오름차순)
  tag_frequency_by_job.csv     canonical × 직무 출현수. "전체" 열은 직무 합산이라 교차직무 중복 공고가 여러 번 세어진다
  field_difficulty_summary.csv field 1(명시)/2(서술)만. field 3(암묵조건)은 집계 제외
결정적: 같은 입력 = 같은 출력. LLM 추출은 llm_skill_extract.py.

사용:
  python -m src.extract.job_skill_extract_report --date 2026-07-30
  python -m src.extract.job_skill_extract_report              # --date 생략 시 data/raw 의 최신 배치
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.common.jsonl import iter_jsonl
from src.common.paths import DICT_PATH, EXTRACT_REPORT_DIR, JOBS, RAW_DIR, resolve_date
from src.common.skill_dict import close_tags, compile_patterns, implies_from_tags, load_dict

OUT_DIR = EXTRACT_REPORT_DIR
SECTION_KEYS = ["main_tasks", "requirements", "preferred_points"]
# da(경영·비즈니스 직군) 유입 공고를 제목 키워드로 데이터 직무 여부 판정.
# category_tags 가 실데이터에서 전부 null 이라(2026-07-25 확인) 제목 키워드로 대체.
DATA_TITLE_KEYWORDS = ["데이터", "data", "분석가", "analyst", "analytics"]

VALIDATION_COLUMNS = ["입력공고수", "통과", "스킵_파싱실패", "스킵_필드누락", "스킵_본문결측", "직무불일치"]


def load_jsonl(path: Path):
    """(줄번호, 레코드|None). 파싱 실패 줄은 None — 호출자가 스킵 사유로 집계한다."""
    yield from iter_jsonl(path, strict=False)


def is_data_job(title: str) -> bool:
    t = (title or "").lower()
    return any(kw.lower() in t for kw in DATA_TITLE_KEYWORDS)


def run(raw_dir: str | Path = RAW_DIR, dict_path: str | Path = DICT_PATH,
        out_dir: str | Path = OUT_DIR, date: str | None = None) -> dict:
    """
    regex 베이스라인으로 채용공고에서 역량 태그를 추출하고 4개 표를 저장한다.
    date 를 생략하면 raw_dir 의 최신 배치를 쓴다 (어느 날짜를 썼는지 화면과 반환값에 남긴다).
    반환: 처리 요약 dict (호출한 쪽이 대화에서 바로 쓸 수 있도록).
    """
    raw_dir = Path(raw_dir)
    dict_path = Path(dict_path)
    out_dir = Path(out_dir)
    date = resolve_date(date, raw_dir=raw_dir)

    tags = load_dict(dict_path)
    patterns = compile_patterns(tags)
    tag_meta = {c: (cat, f) for c, cat, f, _ in patterns}
    implies = implies_from_tags(tags)
    excluded = sorted(t["canonical"] for t in tags if t["field"] == 3)
    if excluded:
        print(f"[로그] ④ 집계 제외 태그(field=3, 암묵조건) {len(excluded)}개: {excluded}")

    out_dir.mkdir(parents=True, exist_ok=True)

    fail_log: list[dict] = []
    mismatch_by_job: dict[str, list[str]] = defaultdict(list)
    validation_rows: dict[str, list[int]] = {}
    extracted_rows: list[dict] = []
    freq: dict[str, Counter] = defaultdict(Counter)  # canonical -> {job: count}

    for job in JOBS:
        path = raw_dir / job / f"wanted_{date}.jsonl"
        counts = Counter()
        if not path.exists():
            # 열 수를 다른 행과 똑같이 맞춘다. 짧은 행을 넣으면 합계 zip 이 조용히 잘리고
            # market_stats 가 int(None) 으로 죽는다 (2026-10-27 발견).
            validation_rows[job] = [0] * len(VALIDATION_COLUMNS)
            fail_log.append({"file": str(path), "reason": "raw_file_missing"})
            continue

        for lineno, rec in load_jsonl(path):
            counts["입력"] += 1
            if rec is None:
                counts["스킵_파싱실패"] += 1
                fail_log.append({"file": str(path), "line": lineno, "reason": "json_decode_error"})
                continue

            job_id = rec.get("job_id")
            body_raw = rec.get("body_raw") or ""
            if not job_id or not body_raw:
                counts["스킵_필드누락"] += 1
                fail_log.append({"file": str(path), "line": lineno, "job_id": job_id, "reason": "missing_required_field"})
                continue
            if len(body_raw) < 100:
                counts["스킵_본문결측"] += 1
                fail_log.append({"file": str(path), "line": lineno, "job_id": job_id, "reason": "body_too_short"})
                continue

            sections = rec.get("sections") or {}
            for k in SECTION_KEYS:
                if not (sections.get(k) or "").strip():
                    fail_log.append({"file": str(path), "line": lineno, "job_id": job_id, "reason": f"section_missing:{k}"})

            if job == "da" and not is_data_job(rec.get("title", "")):
                mismatch_by_job[job].append(job_id)
                counts["직무불일치"] += 1
                continue

            counts["통과"] += 1

            matched_canonicals: set[str] = set()
            matched_fields: set[str] = set()
            for canonical, _category, _field, pat in patterns:
                if pat.search(body_raw):
                    matched_canonicals.add(canonical)
                    for sk in SECTION_KEYS:
                        if pat.search(sections.get(sk) or ""):
                            matched_fields.add(sk)
            matched_canonicals = close_tags(matched_canonicals, implies)  # v2: PySpark → Spark

            for c in matched_canonicals:
                freq[c][job] += 1

            extracted_rows.append({
                "job_id": job_id,
                "직무": job,
                "title": rec.get("title", ""),
                "tags": sorted(matched_canonicals),
                "tag_count": len(matched_canonicals),
                "matched_fields": sorted(matched_fields),
            })
            if not matched_canonicals:
                fail_log.append({"file": str(path), "line": lineno, "job_id": job_id, "reason": "zero_tags_matched"})

        validation_rows[job] = [
            counts["입력"], counts["통과"], counts["스킵_파싱실패"],
            counts["스킵_필드누락"], counts["스킵_본문결측"], counts["직무불일치"],
        ]
        # 불변식: 입력공고수 = 통과 + 스킵_파싱실패 + 스킵_필드누락 + 스킵_본문결측 + 직무불일치
        입력, *나머지 = validation_rows[job]
        if 입력 != sum(나머지):
            raise SystemExit(f"[전체중단] 불변식 위반 (job={job}): 입력공고수={입력} != 합계={sum(나머지)}")

    # ① validation_summary.csv
    with open(out_dir / "validation_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["직무", *VALIDATION_COLUMNS])
        totals = [0] * len(VALIDATION_COLUMNS)
        for job in JOBS:
            row = validation_rows[job]
            assert len(row) == len(VALIDATION_COLUMNS), (job, row)
            w.writerow([job, *row])
            totals = [a + b for a, b in zip(totals, row)]
        w.writerow(["전체합계", *totals])

    # ② extracted_tags.jsonl : 직무순(ds->de->mle->da) -> job_id 오름차순
    extracted_rows.sort(key=lambda r: (JOBS.index(r["직무"]), r["job_id"]))
    with open(out_dir / "extracted_tags.jsonl", "w", encoding="utf-8") as f:
        for r in extracted_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ③ tag_frequency_by_job.csv : 전체 빈도 내림차순, 동점이면 canonical 알파벳순
    freq_rows = []
    for canonical, per_job in freq.items():
        category, field = tag_meta[canonical]
        total = sum(per_job.values())
        freq_rows.append([canonical, category, field, *[per_job.get(j, 0) for j in JOBS], total])
    freq_rows.sort(key=lambda r: (-r[-1], r[0]))
    with open(out_dir / "tag_frequency_by_job.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["canonical", "category", "field", *JOBS, "전체"])
        w.writerows(freq_rows)

    # ④ field_difficulty_summary.csv : field 1(명시)/2(서술)만. field 3(암묵조건)은 집계 제외.
    field_stats = {1: {"types": set(), "total": 0}, 2: {"types": set(), "total": 0}}
    for canonical, per_job in freq.items():
        _category, field = tag_meta[canonical]
        if field in field_stats:
            field_stats[field]["types"].add(canonical)
            field_stats[field]["total"] += sum(per_job.values())

    n_postings = len(extracted_rows)
    with open(out_dir / "field_difficulty_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["field", "태그종류수", "총매칭수", "공고당평균태그수"])
        for field, label in [(1, "①명시"), (2, "②서술")]:
            st = field_stats[field]
            avg = round(st["total"] / n_postings, 2) if n_postings else 0.0
            w.writerow([label, len(st["types"]), st["total"], avg])

    # 공식 4개 출력 외 진단용 로그
    with open(out_dir / "_fail_log.jsonl", "w", encoding="utf-8") as f:
        for r in fail_log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for job, ids in mismatch_by_job.items():
        with open(out_dir / f"_job_mismatch_{job}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(ids))

    total_mismatch = sum(len(v) for v in mismatch_by_job.values())
    print(f"완료 -> {out_dir}  (date={date}, dict={dict_path.name})")
    print(f"직무불일치(별도분리, 통과 미포함): {total_mismatch}건")
    print(f"진단 로그: {len(fail_log)}건 -> _fail_log.jsonl")

    total_input = sum(validation_rows[j][0] for j in JOBS)
    # field 3(암묵조건)은 변별력이 없어 집계에서 제외한다(CSV ④와 동일 기준).
    top_tags = [
        {"canonical": c, "전체": sum(pj.values())}
        for c, pj in sorted(freq.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
        if tag_meta[c][1] != 3
    ][:10]
    return {
        "output_dir": str(out_dir),
        "date": date,
        "dict": str(dict_path),
        "총입력": total_input,
        "통과": len(extracted_rows),
        "직무불일치": total_mismatch,
        "진단로그건수": len(fail_log),
        "직무별통과": {j: validation_rows[j][1] for j in JOBS},
        "상위태그_top10": top_tags,
        "출력파일": [
            "validation_summary.csv",
            "extracted_tags.jsonl",
            "tag_frequency_by_job.csv",
            "field_difficulty_summary.csv",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--dict", dest="dict_path", default=str(DICT_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--date", default=None, help="wanted_{date}.jsonl 의 date. 생략 시 raw-dir 의 최신 배치")
    a = ap.parse_args(argv)
    r = run(raw_dir=a.raw_dir, dict_path=a.dict_path, out_dir=a.out_dir, date=a.date)
    print(json.dumps({k: v for k, v in r.items() if k != "출력파일"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
