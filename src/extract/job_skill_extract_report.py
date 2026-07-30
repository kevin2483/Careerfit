"""
채용 역량 추출 리포트 (job-skill-extract-report)
data/raw/{ds,de,mle,da}/*.jsonl + data/raw/dict/skill_tags_v0.yaml
-> data/processed/job_skill_extract_report/{validation_summary.csv, extracted_tags.jsonl,
   tag_frequency_by_job.csv, field_difficulty_summary.csv}
regex 베이스라인만 사용 (결정적: 같은 입력 = 같은 출력). LLM 추출·정확도 비교는 4회차로 이관.
사용:
  python -m src.extract.job_skill_extract_report
"""
from __future__ import annotations
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import yaml

JOBS = ["ds", "de", "mle", "da"]
RAW_DIR = Path("data/raw")
DICT_PATH = RAW_DIR / "dict" / "skill_tags_v0.yaml"
OUT_DIR = Path("data/processed/job_skill_extract_report")
SECTION_KEYS = ["main_tasks", "requirements", "preferred_points"]
# STEP 0-5: da(경영·비즈니스 직군) 유입 공고를 제목 키워드로 데이터 직무 여부 판정.
# category_tags가 실데이터에서 전부 null이라(2026-07-25 확인) 제목 키워드로 대체.
DATA_TITLE_KEYWORDS = ["데이터", "data", "분석가", "analyst", "analytics"]


def load_dict(path: Path) -> list[dict]:
    # STEP 0-4: 사전 없음/파싱 실패는 공용 자원 문제 -> 개별 스킵이 아니라 전체 중단.
    if not path.exists():
        raise SystemExit(f"[전체중단] 태그 사전 없음: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SystemExit(f"[전체중단] 태그 사전 파싱 실패: {e}")
    tags: list[dict] = []
    seen: set[str] = set()
    for category, entries in raw.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            canonical = e["canonical"]
            if canonical in seen:
                print(f"[경고] canonical 중복 정의: {canonical}")
            seen.add(canonical)
            tags.append({
                "canonical": canonical,
                "aliases": e.get("aliases") or [],
                "field": e.get("field"),
                "category": category,
            })
    excluded = sorted(t["canonical"] for t in tags if t["field"] == 3)
    if excluded:
        print(f"[로그] ④ 집계 제외 태그(field=3, 암묵조건) {len(excluded)}개: {excluded}")
    return tags


def compile_patterns(tags: list[dict]):
    patterns = []
    for t in tags:
        alts = sorted({a for a in [*t["aliases"], t["canonical"]] if a}, key=len, reverse=True)
        alts = [re.escape(a) for a in alts]
        if not alts:
            continue
        pat = re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)
        patterns.append((t["canonical"], t["category"], t["field"], pat))
    return patterns


def load_jsonl(path: Path):
    # STEP 0-1: 파일 객체 직접 순회 (splitlines()는 유니코드 줄바꿈에서 레코드 절단 위험)
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                yield lineno, None


def is_data_job(title: str) -> bool:
    t = (title or "").lower()
    return any(kw.lower() in t for kw in DATA_TITLE_KEYWORDS)


def run(raw_dir: Path = RAW_DIR, dict_path: Path = DICT_PATH,
        out_dir: Path = OUT_DIR) -> dict:
    """
    regex 베이스라인으로 채용공고에서 역량 태그를 추출하고 4개 표를 저장한다.
    입력 경로를 인자로 받아 MCP 도구가 감쌀 수 있게 했다(기본값은 프로젝트 표준 경로).
    반환: 처리 요약 dict (호출한 쪽=Claude가 대화에서 바로 쓸 수 있도록).
    """
    raw_dir = Path(raw_dir)
    dict_path = Path(dict_path)
    out_dir = Path(out_dir)

    tags = load_dict(dict_path)
    patterns = compile_patterns(tags)
    tag_meta = {c: (cat, f) for c, cat, f, _ in patterns}

    out_dir.mkdir(parents=True, exist_ok=True)

    fail_log: list[dict] = []
    mismatch_by_job: dict[str, list[str]] = defaultdict(list)
    validation_rows: dict[str, list[int]] = {}
    extracted_rows: list[dict] = []
    freq: dict[str, Counter] = defaultdict(Counter)  # canonical -> {job: count}

    for job in JOBS:
        path = raw_dir / job / "wanted_2026-07-25.jsonl"
        counts = Counter()
        if not path.exists():
            validation_rows[job] = [0, 0, 0, 0, 0]
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
        w.writerow(["직무", "입력공고수", "통과", "스킵_파싱실패", "스킵_필드누락", "스킵_본문결측", "직무불일치"])
        totals = [0, 0, 0, 0, 0, 0]
        for job in JOBS:
            row = validation_rows[job]
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
        freq_rows.append([canonical, category, field, per_job.get("ds", 0), per_job.get("de", 0),
                           per_job.get("mle", 0), per_job.get("da", 0), total])
    freq_rows.sort(key=lambda r: (-r[-1], r[0]))
    with open(out_dir / "tag_frequency_by_job.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["canonical", "category", "field", "ds", "de", "mle", "da", "전체"])
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

    # 공식 4개 출력 외 진단용 로그 (재현 검증·6번 분석 재료)
    with open(out_dir / "_fail_log.jsonl", "w", encoding="utf-8") as f:
        for r in fail_log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for job, ids in mismatch_by_job.items():
        with open(out_dir / f"_job_mismatch_{job}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(ids))

    total_mismatch = sum(len(v) for v in mismatch_by_job.values())
    print(f"완료 -> {out_dir}")
    print(f"직무불일치(별도분리, 통과 미포함): {total_mismatch}건")
    print(f"진단 로그: {len(fail_log)}건 -> _fail_log.jsonl")

    # --- MCP/호출자용 요약 반환 ---
    # 파일 저장은 그대로 하되, 호출한 쪽이 대화에서 바로 쓸 수 있게 핵심 수치를 돌려준다.
    total_input = sum(validation_rows[j][0] for j in JOBS)
    total_pass = len(extracted_rows)
    # field 3(암묵조건)은 변별력이 없어 집계에서 제외한다(CSV ④와 동일 기준).
    # 이 필터가 없으면 '협업커뮤니케이션'이 거의 모든 공고에 나와 상위를 차지해 오해를 부른다.
    top_tags = [
        {"canonical": c, "전체": sum(pj.values())}
        for c, pj in sorted(freq.items(), key=lambda kv: -sum(kv[1].values()))
        if tag_meta[c][1] != 3
    ][:10]
    return {
        "output_dir": str(out_dir),
        "총입력": total_input,
        "통과": total_pass,
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


if __name__ == "__main__":
    run()