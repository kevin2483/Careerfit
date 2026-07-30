"""
라벨링 템플릿 생성기 (4번 정답셋 구축의 출발점)

무엇을 하나:
  07-30 수집분에서 직무별 15건씩(da는 데이터 직무만) 총 60건을 seed 고정으로 뽑아,
  '검수 방식' 라벨링 템플릿(JSONL)을 만든다.

왜 검수 방식인가:
  백지에서 태그를 다는 것보다 regex 결과를 미리 채우고 맞나/틀리나 확인하는 게 빠르다.
  단, regex가 '놓친' 태그를 사람도 놓치면 정답셋이 regex 쪽으로 편향된다.
  그래서 body_raw 전문을 함께 실어, 라벨러가 regex 누락을 직접 잡도록 강제한다.

라벨러(사람)가 할 일 — 각 레코드에서:
  1. regex_prefilled 의 각 태그를 보고, 틀린 건 gold_tags 에서 지운다.
  2. body_raw 를 읽고, regex 가 놓친 태그를 gold_tags 에 추가한다.
  3. 사전에 없는 새 역량은 new_tag_candidates 에 자유 문자열로 적는다(v1 후보).
  4. 각 gold 태그가 어느 섹션에서 나왔는지 evidence_sections 에 표기.
  5. reviewed 를 true 로 바꾼다.

사용:
  python -m src.label.make_label_template
  python -m src.label.make_label_template --date 2026-07-30 --per-job 15 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# 추출 코드의 함수를 재활용 (중복 구현 금지)
from src.extract.job_skill_extract_report import (
    load_dict, compile_patterns, load_jsonl, is_data_job,
    SECTION_KEYS, RAW_DIR, DICT_PATH, JOBS,
)


def collect_candidates(raw_dir: Path, date: str, job: str) -> list[dict]:
    """한 직무의 유효 공고만 모은다 (검증 규칙 축약판 + da 오염 필터)."""
    path = raw_dir / job / f"wanted_{date}.jsonl"
    out: list[dict] = []
    if not path.exists():
        return out
    for _lineno, rec in load_jsonl(path):
        if rec is None:
            continue
        job_id = rec.get("job_id")
        body = rec.get("body_raw") or ""
        if not job_id or len(body) < 100:
            continue
        if job == "da" and not is_data_job(rec.get("title", "")):
            continue  # da 오염(영업·인사) 제외 — 라벨링 헛수고 방지
        out.append(rec)
    return out


def regex_prefill(rec: dict, patterns) -> tuple[list[str], dict]:
    """regex 가 찾은 canonical 태그와, 태그별 발견 섹션을 만든다."""
    body = rec.get("body_raw") or ""
    sections = rec.get("sections") or {}
    hits: list[str] = []
    ev: dict[str, list[str]] = {}
    for canonical, _cat, _field, pat in patterns:
        if pat.search(body):
            hits.append(canonical)
            found_in = [sk for sk in SECTION_KEYS if pat.search(sections.get(sk) or "")]
            ev[canonical] = found_in or ["(본문 기타)"]
    return sorted(hits), ev


def build_unique_pool(raw_dir: Path, date: str) -> dict[str, dict]:
    """
    4개 직무를 합쳐 고유 job_id 풀을 만든다.
    - 귀속(직무): 처음 등장한 직무 (JOBS 순서 ds->de->mle->da). 결정적.
    - all_jobs: 그 공고가 걸린 '모든' 직무 (6번/10번 분석용, 정보 보존).
    da 는 데이터 직무만 포함 (오염 제외).
    """
    pool: dict[str, dict] = {}
    for job in JOBS:  # 순서 고정 -> 귀속이 결정적
        for rec in collect_candidates(raw_dir, date, job):
            jid = str(rec.get("job_id"))
            if jid not in pool:
                pool[jid] = {"rec": rec, "직무": job, "all_jobs": [job]}
            else:
                pool[jid]["all_jobs"].append(job)  # 겹침 이력 보존
    return pool


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2026-07-30")
    p.add_argument("--per-job", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-dir", default=str(RAW_DIR))
    p.add_argument("--dict-path", default=str(DICT_PATH))
    p.add_argument("--out", default="data/labeled/label_template.jsonl")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    patterns = compile_patterns(load_dict(Path(args.dict_path)))
    rng = random.Random(args.seed)

    # 1) 고유 공고 풀 (중복 제거 + 겹침 이력 보존)
    pool = build_unique_pool(raw_dir, args.date)

    # 2) 귀속 직무별로 그룹핑
    by_job: dict[str, list[dict]] = {j: [] for j in JOBS}
    for entry in pool.values():
        by_job[entry["직무"]].append(entry)

    # 3) 직무별 per-job 건씩 표본 추출 (seed 고정)
    template_rows: list[dict] = []
    picked_summary: dict[str, int] = {}
    for job in JOBS:
        entries = by_job[job]
        rng.shuffle(entries)
        chosen = entries[: args.per_job]
        picked_summary[job] = len(chosen)
        for entry in chosen:
            rec = entry["rec"]
            prefilled, ev = regex_prefill(rec, patterns)
            template_rows.append({
                "job_id": rec.get("job_id"),
                "직무": job,                       # 귀속 직무 (라벨링 기준)
                "all_jobs": entry["all_jobs"],     # 원래 걸린 모든 직무 (분석 보존)
                "title": rec.get("title", ""),
                "body_raw": rec.get("body_raw", ""),
                "regex_prefilled": prefilled,
                "gold_tags": list(prefilled),
                "evidence_sections": ev,
                "new_tag_candidates": [],
                "reviewed": False,
            })

    template_rows.sort(key=lambda r: (JOBS.index(r["직무"]), str(r["job_id"])))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in template_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_multi = sum(1 for r in template_rows if len(r["all_jobs"]) > 1)
    print(f"라벨링 템플릿 생성: {len(template_rows)}건 -> {out_path}")
    print(f"직무별(귀속 기준): {picked_summary}")
    print(f"고유 공고 풀: {len(pool)}건 (중복 제거 완료)")
    print(f"뽑힌 60건 중 복수직무 공고: {n_multi}건 (all_jobs로 이력 보존)")
    print(f"seed={args.seed} (같은 seed면 같은 표본)")
    print("다음: 이 파일을 열어 각 레코드의 gold_tags 를 검수하고 reviewed=true 로 바꾼다.")


if __name__ == "__main__":
    main()
