"""
시장 통계 (market-stats)
data/processed/job_skill_extract_report/{tag_frequency_by_job.csv, extracted_tags.jsonl, validation_summary.csv}
+ data/raw/{ds,de,mle,da}/wanted_{date}.jsonl (annual_from 조인용)
-> data/processed/market_stats/{rank_by_job.csv, cooccurrence.csv, junior_vs_senior.csv, excluded_lowfreq.csv}

analyst 담당자가 이 모듈의 run()만 호출한다. 판단 기준(CLAUDE.md/판단기준)을 코드에
고정해서 매번 같은 입력이면 같은 출력이 나오게 한다 (재현성).

  판단 기준 ①: (canonical, 직무) 조합의 출현수가 3건 미만이면 그 직무의 순위·비교·동시출현·
              신입경력 통계에서 빼고 excluded_lowfreq.csv로 분리한다.
  판단 기준 ②: 직무 비교는 절대건수가 아니라 출현율(%)로 한다. 직무별 모집단 크기가
              다르기 때문이다 (특히 da는 원티드 분류 오염으로 모집단이 입력공고수와 다르다).

    python -m src.analyze.market_stats --date 2026-07-30
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from src.common.jsonl import iter_jsonl, read_jsonl
from src.common.paths import EXTRACT_REPORT_DIR, JOBS, MARKET_STATS_DIR, RAW_DIR, resolve_date

LOWFREQ_THRESHOLD = 3  # 판단 기준 ①

TAG_FREQ_PATH = EXTRACT_REPORT_DIR / "tag_frequency_by_job.csv"
EXTRACTED_TAGS_PATH = EXTRACT_REPORT_DIR / "extracted_tags.jsonl"
VALIDATION_SUMMARY_PATH = EXTRACT_REPORT_DIR / "validation_summary.csv"
OUT_DIR = MARKET_STATS_DIR

DA_NOTE_TEMPLATE = (
    "원티드가 '데이터 분석가'를 경영·비즈니스 직군으로도 분류해 비데이터 공고가 혼입됨. "
    "제목 키워드로 실제 데이터 직무 공고만 필터링한 결과 통과 {pass_}건(입력 {input_}건 중 {mismatch}건 직무불일치 제외). "
    "출현율(%) 분모는 {pass_}을 쓴다. {input_}을 쓰면 실제보다 낮게 계산된다."
)


def load_population(validation_summary_path: Path) -> dict[str, dict[str, int]]:
    """직무별 모집단(통과 공고수)을 validation_summary.csv에서 그대로 읽는다. 숫자를 하드코딩하지 않는다."""
    pop: dict[str, dict[str, int]] = {}
    with open(validation_summary_path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["직무"] not in JOBS:
                continue
            try:
                pop[row["직무"]] = {
                    "입력공고수": int(row["입력공고수"]),
                    "통과": int(row["통과"]),
                    "직무불일치": int(row["직무불일치"]),
                }
            except (TypeError, ValueError) as e:
                raise SystemExit(f"[중단] validation_summary.csv 행이 깨짐 (job={row['직무']}): {dict(row)} — {e}")
    missing = [j for j in JOBS if j not in pop]
    if missing:
        raise SystemExit(f"[중단] validation_summary.csv 에 직무 행 없음: {missing}")
    return pop


def load_tag_freq(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_annual_from(raw_dir: Path, date: str) -> dict[str, int | None]:
    """job_id -> annual_from. 신입vs경력 조인용. career_type은 크롤러 단계 미판정(전건 null)이라 쓰지 않는다."""
    result: dict[str, int | None] = {}
    for job in JOBS:
        path = raw_dir / job / f"wanted_{date}.jsonl"
        if not path.exists():
            continue
        for _lineno, d in iter_jsonl(path, strict=False):
            if d is not None:
                result[d.get("job_id")] = d.get("annual_from")
    return result


def classify_career(annual_from: int | None) -> str:
    if annual_from is None:
        return "미기재"
    return "신입가능" if annual_from == 0 else "경력요구"


def compute_excluded(tag_freq_rows: list[dict]) -> tuple[set[tuple[str, str]], list[dict]]:
    """(canonical, 직무) 쌍 중 출현수가 LOWFREQ_THRESHOLD 미만인 것을 골라낸다."""
    excluded_keys: set[tuple[str, str]] = set()
    excluded_rows: list[dict] = []
    for row in tag_freq_rows:
        canonical = row["canonical"]
        for job in JOBS:
            count = int(row[job])
            if 0 < count < LOWFREQ_THRESHOLD:
                excluded_keys.add((canonical, job))
                excluded_rows.append({
                    "canonical": canonical,
                    "직무": job,
                    "출현수": count,
                    "제외사유": f"출현 {count}건 < {LOWFREQ_THRESHOLD}건, 표본 부족으로 통계 주장 불가 (판단 기준 ①)",
                })
    return excluded_keys, excluded_rows


def compute_rank_by_job(tag_freq_rows: list[dict], population: dict[str, dict[str, int]],
                        excluded_keys: set[tuple[str, str]], da_note: str) -> list[dict]:
    rows_by_job: dict[str, list[dict]] = defaultdict(list)
    for row in tag_freq_rows:
        canonical = row["canonical"]
        for job in JOBS:
            count = int(row[job])
            if count == 0 or (canonical, job) in excluded_keys:
                continue
            denom = population[job]["통과"]
            rows_by_job[job].append({
                "canonical": canonical,
                "직무": job,
                "출현수": count,
                "모집단수": denom,
                "출현율%": round(count / denom * 100, 2) if denom else 0.0,
                "비고": da_note if job == "da" else "",
            })
    out: list[dict] = []
    for job in JOBS:
        ranked = sorted(rows_by_job[job], key=lambda r: (-r["출현율%"], r["canonical"]))
        for i, row in enumerate(ranked, start=1):
            row["순위"] = i
            out.append(row)
    return out


def compute_cooccurrence(extracted_rows: list[dict], excluded_keys: set[tuple[str, str]], top_n: int = 30) -> list[dict]:
    pair_counts: dict[str, Counter] = defaultdict(Counter)
    for rec in extracted_rows:
        job = rec["직무"]
        tags = sorted({t for t in rec.get("tags", []) if (t, job) not in excluded_keys})
        for a, b in combinations(tags, 2):
            pair_counts[job][(a, b)] += 1
    out: list[dict] = []
    for job in JOBS:
        top_pairs = sorted(pair_counts[job].items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        for (a, b), count in top_pairs:
            out.append({"tag_a": a, "tag_b": b, "동시출현수": count, "직무": job})
    return out


def compute_junior_vs_senior(extracted_rows: list[dict], annual_from_map: dict[str, int | None],
                             excluded_keys: set[tuple[str, str]]) -> list[dict]:
    tally: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for rec in extracted_rows:
        job = rec["직무"]
        bucket = classify_career(annual_from_map.get(rec.get("job_id")))
        for tag in rec.get("tags", []):
            if (tag, job) in excluded_keys:
                continue
            tally[(tag, job)][bucket] += 1
    out: list[dict] = []
    for (canonical, job), counts in tally.items():
        total = sum(counts.values())
        out.append({
            "canonical": canonical,
            "직무": job,
            "신입가능": counts["신입가능"],
            "경력요구": counts["경력요구"],
            "미기재": counts["미기재"],
            "신입가능%": round(counts["신입가능"] / total * 100, 1) if total else 0.0,
            "경력요구%": round(counts["경력요구"] / total * 100, 1) if total else 0.0,
        })
    out.sort(key=lambda r: (JOBS.index(r["직무"]), -r["경력요구%"], r["canonical"]))
    return out


def check_inputs_fresh(raw_dir: Path, tag_freq_path: Path, extracted_tags_path: Path,
                       validation_summary_path: Path, date: str) -> None:
    """extractor 산출물이 있는지, 원본 데이터보다 최신인지 확인한다.
    순서 의존성(extractor → analyst)이 깨지면 조용히 틀린 결과를 내는 대신 여기서 멈춘다."""
    missing = [p for p in (tag_freq_path, extracted_tags_path, validation_summary_path) if not p.exists()]
    if missing:
        raise SystemExit(f"[중단] extractor 산출물 없음: {[str(p) for p in missing]}. extractor 먼저 실행 필요.")
    extract_mtime = extracted_tags_path.stat().st_mtime
    for job in JOBS:
        raw_path = raw_dir / job / f"wanted_{date}.jsonl"
        if raw_path.exists() and raw_path.stat().st_mtime > extract_mtime:
            raise SystemExit(
                f"[중단] extractor 산출물이 원본 데이터보다 오래됨 ({raw_path} 이 더 최신). "
                f"date={date}로 extractor 재실행 필요."
            )


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(
    raw_dir: str | Path = RAW_DIR,
    tag_freq_path: str | Path = TAG_FREQ_PATH,
    extracted_tags_path: str | Path = EXTRACTED_TAGS_PATH,
    validation_summary_path: str | Path = VALIDATION_SUMMARY_PATH,
    out_dir: str | Path = OUT_DIR,
    date: str | None = None,
) -> dict:
    """analyst가 호출하는 진입점. 입력 경로를 인자로 받아 하드코딩을 피한다 (CLAUDE.md ④).
    date 를 생략하면 raw_dir 의 최신 배치. 반환: 처리 요약 dict."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    tag_freq_path, extracted_tags_path = Path(tag_freq_path), Path(extracted_tags_path)
    validation_summary_path = Path(validation_summary_path)
    date = resolve_date(date, raw_dir=raw_dir)

    check_inputs_fresh(raw_dir, tag_freq_path, extracted_tags_path, validation_summary_path, date)

    population = load_population(validation_summary_path)
    da_note = DA_NOTE_TEMPLATE.format(pass_=population["da"]["통과"], input_=population["da"]["입력공고수"],
                                      mismatch=population["da"]["직무불일치"])
    tag_freq_rows = load_tag_freq(tag_freq_path)
    extracted_rows = read_jsonl(extracted_tags_path)
    annual_from_map = load_annual_from(raw_dir, date)

    excluded_keys, excluded_rows = compute_excluded(tag_freq_rows)
    rank_rows = compute_rank_by_job(tag_freq_rows, population, excluded_keys, da_note)
    cooc_rows = compute_cooccurrence(extracted_rows, excluded_keys)
    career_rows = compute_junior_vs_senior(extracted_rows, annual_from_map, excluded_keys)

    _write_csv(out_dir / "rank_by_job.csv", rank_rows, ["canonical", "직무", "출현수", "모집단수", "출현율%", "비고", "순위"])
    _write_csv(out_dir / "cooccurrence.csv", cooc_rows, ["tag_a", "tag_b", "동시출현수", "직무"])
    _write_csv(out_dir / "junior_vs_senior.csv", career_rows,
               ["canonical", "직무", "신입가능", "경력요구", "미기재", "신입가능%", "경력요구%"])
    _write_csv(out_dir / "excluded_lowfreq.csv", excluded_rows, ["canonical", "직무", "출현수", "제외사유"])

    return {
        "date": date,
        "모집단": {job: population[job]["통과"] for job in JOBS},
        "da_비고": da_note,
        "순위표_행수": len(rank_rows),
        "동시출현표_행수": len(cooc_rows),
        "신입경력표_행수": len(career_rows),
        "제외건수": len(excluded_rows),
        "직무별_top3": {job: [r["canonical"] for r in rank_rows if r["직무"] == job][:3] for job in JOBS},
        "출력파일": [str(out_dir / f) for f in
                 ("rank_by_job.csv", "cooccurrence.csv", "junior_vs_senior.csv", "excluded_lowfreq.csv")],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--extract-dir", default=str(EXTRACT_REPORT_DIR), help="extractor 산출물 폴더")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--date", default=None, help="extractor 에 쓴 것과 같은 날짜. 생략 시 raw-dir 최신 배치")
    a = ap.parse_args(argv)
    ed = Path(a.extract_dir)
    r = run(raw_dir=a.raw_dir, tag_freq_path=ed / "tag_frequency_by_job.csv",
            extracted_tags_path=ed / "extracted_tags.jsonl",
            validation_summary_path=ed / "validation_summary.csv", out_dir=a.out_dir, date=a.date)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
