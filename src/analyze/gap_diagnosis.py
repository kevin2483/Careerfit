"""
개인 갭 진단 (gap-diagnosis)
data/processed/market_stats/{rank_by_job.csv, junior_vs_senior.csv} + 이력서 텍스트(인자로 받음)
-> 화면에만 출력 (파일에 쓰지 않는다 — diagnoser는 읽기 전용 담당자다. 최종 기록은 리더가 노션에 한다)

diagnoser 담당자가 이 모듈의 run()만 호출한다.

  산식: 우선순위 = 출현율(%) × 미보유가중치 × 신입접근성(0~1)

  미보유가중치는 원래 계획(미보유1/부분보유0.5/보유0)이 3단계였지만, 이력서에 숙련도
  자기표기가 없어(기술 나열식) 부분보유를 판정할 결정적 근거가 없다. 그래서 지금은
  이진(미보유1/보유0)으로 축소했다. 상수로 분리해 뒀으니 이력서에 숙련도 표기가
  추가되면 CAREER_WEIGHT에 "부분보유": 0.5를 더하고 classify_possession()만 바꾸면
  3단계로 복원된다.

  이력서 매칭은 job_skill_extract_report.py와 같은 regex+사전(v1) 방식을 재사용한다
  (같은 태그 정의를 두 번 다르게 구현하면 기준이 어긋난다).
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.extract.job_skill_extract_report import DICT_PATH, compile_patterns, load_dict

JOBS = ["ds", "de", "mle", "da"]
TOP_N = 5  # 판단 기준: 상위 5개만 이번 분기, 6위 이하는 다음 분기 (CLAUDE.md/판단 기준)

# 미보유 가중치. 지금은 이진. 이력서에 숙련도 표기가 생기면 "부분보유": 0.5 를 추가하고
# classify_possession()의 반환값을 그에 맞게 늘린다.
POSSESSION_WEIGHT = {
    "미보유": 1.0,
    "보유": 0.0,
}
POSSESSION_TIER_NOTE = (
    "원 계획(미보유1/부분보유0.5/보유0)의 3단계를 이진(미보유1/보유0)으로 축소함. "
    "이력서가 기술 나열식이라 숙련도 자기표기가 없어 부분보유를 판정할 결정적 근거가 없기 때문. "
    "이력서에 숙련도 표기가 추가되면 POSSESSION_WEIGHT에 부분보유를 더해 3단계로 복원 가능."
)
RESUME_STALENESS_NOTE = (
    "입력 이력서가 구버전이라 최근 경험(LLM, MCP 서버, AI Agent 등)이 반영되지 않았을 수 있다. "
    "입력 문서의 최신성이 진단 정확도를 좌우한다 — 이 진단 결과를 이력서 갱신 없이 그대로 신뢰하지 말 것."
)

RANK_BY_JOB_PATH = Path("data/processed/market_stats/rank_by_job.csv")
JUNIOR_SENIOR_PATH = Path("data/processed/market_stats/junior_vs_senior.csv")

# analyst(market_stats.py)가 이 파일들을 읽어 rank_by_job.csv 등을 만든다.
# 이 파일들이 rank_by_job.csv보다 최신이면 analyst가 재실행되기 전 것이라는 뜻이다.
_MARKET_STATS_SOURCE_PATHS = [
    Path("data/processed/job_skill_extract_report/tag_frequency_by_job.csv"),
    Path("data/processed/job_skill_extract_report/extracted_tags.jsonl"),
]


def check_inputs_fresh(rank_by_job_path: Path, junior_senior_path: Path) -> None:
    """analyst 산출물이 있는지, extractor 산출물보다 최신인지 확인한다.
    순서 의존성(extractor → analyst → diagnoser)이 깨지면 조용히 틀린 결과를 내는
    대신 여기서 멈춘다."""
    missing = [p for p in (rank_by_job_path, junior_senior_path) if not p.exists()]
    if missing:
        raise SystemExit(
            f"[중단] analyst 산출물 없음: {[str(p) for p in missing]}. analyst 먼저 실행 필요."
        )

    rank_mtime = rank_by_job_path.stat().st_mtime
    for src in _MARKET_STATS_SOURCE_PATHS:
        if src.exists() and src.stat().st_mtime > rank_mtime:
            raise SystemExit(
                f"[중단] analyst 산출물이 extractor 산출물보다 오래됨 ({src} 이 더 최신). "
                f"analyst 재실행 필요."
            )


def classify_possession(canonical: str, resume_tags: set[str]) -> str:
    return "보유" if canonical in resume_tags else "미보유"


def extract_resume_tags(resume_text: str, dict_path: Path = DICT_PATH) -> set[str]:
    """이력서 텍스트에서 사전(v1) 기준으로 보유 태그를 뽑는다. extractor와 같은 regex 방식."""
    tags = load_dict(Path(dict_path))
    patterns = compile_patterns(tags)
    found: set[str] = set()
    for canonical, _category, _field, pattern in patterns:
        if pattern.search(resume_text):
            found.add(canonical)
    return found


FIELD3_EXCLUSION_NOTE = (
    "field 3(암묵조건) 태그는 거의 모든 공고에 등장해 변별력이 없고 규칙 기반으로 "
    "문맥 판정이 불가능하다는 설계 결정에 따라 학습 우선순위 산출에서 제외한다 "
    "(extractor의 field_difficulty_summary.csv와 동일 기준). rank_by_job.csv/"
    "junior_vs_senior.csv에는 field 컬럼이 없어 여기서 사전을 다시 읽어 판정한다."
)


def load_field3_tags(dict_path: Path = DICT_PATH) -> set[str]:
    """사전에서 field=3(암묵조건)으로 지정된 canonical 태그 집합을 반환한다."""
    return {t["canonical"] for t in load_dict(Path(dict_path)) if t["field"] == 3}


def load_rank_by_job(path: Path, job: str) -> dict[str, float]:
    """canonical -> 출현율%. 이미 market_stats에서 3건 미만은 빠진 상태다."""
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["직무"] == job:
                out[row["canonical"]] = float(row["출현율%"])
    return out


def load_junior_access(path: Path, job: str) -> dict[str, float]:
    """canonical -> 신입접근성(0~1) = 신입가능% / 100."""
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["직무"] == job:
                out[row["canonical"]] = float(row["신입가능%"]) / 100.0
    return out


def run(
    resume_text: str,
    target_job: str,
    dict_path: Path = DICT_PATH,
    rank_by_job_path: Path = RANK_BY_JOB_PATH,
    junior_senior_path: Path = JUNIOR_SENIOR_PATH,
    top_n: int = TOP_N,
) -> dict:
    if target_job not in JOBS:
        raise SystemExit(f"[중단] target_job은 {JOBS} 중 하나여야 함: {target_job!r}")
    if not resume_text or not resume_text.strip():
        raise SystemExit("[중단] 이력서 텍스트가 비어 있음. 리더가 텍스트를 전달했는지 확인 필요.")
    check_inputs_fresh(Path(rank_by_job_path), Path(junior_senior_path))

    resume_tags = extract_resume_tags(resume_text, dict_path)
    field3_tags = load_field3_tags(dict_path)
    freq = load_rank_by_job(Path(rank_by_job_path), target_job)
    access = load_junior_access(Path(junior_senior_path), target_job)

    scored = []
    possessed = []
    excluded_field3 = []
    for canonical, pct in freq.items():
        if canonical in field3_tags:
            excluded_field3.append({"canonical": canonical, "출현율%": pct})
            continue
        possession = classify_possession(canonical, resume_tags)
        if possession == "보유":
            possessed.append(canonical)
            continue
        weight = POSSESSION_WEIGHT["미보유"]
        junior_access = access.get(canonical, 0.0)
        score = round(pct * weight * junior_access, 4)
        scored.append({
            "canonical": canonical,
            "출현율%": pct,
            "미보유가중치": weight,
            "신입접근성": junior_access,
            "우선순위점수": score,
        })

    scored.sort(key=lambda r: r["우선순위점수"], reverse=True)
    for i, row in enumerate(scored, start=1):
        row["순위"] = i
    excluded_field3.sort(key=lambda r: r["출현율%"], reverse=True)

    this_quarter = scored[:top_n]
    next_quarter = scored[top_n:]

    return {
        "대상직무": target_job,
        "이력서_보유태그수": len(resume_tags),
        "직무_전체태그수": len(freq),
        "보유역량": sorted(possessed),
        "이번분기_학습로드맵(top5)": this_quarter,
        "다음분기_후순위": next_quarter,
        "제외_field3_암묵조건": excluded_field3,
        "field3_제외기준": FIELD3_EXCLUSION_NOTE,
        "미보유가중치_방식": POSSESSION_TIER_NOTE,
        "이력서_최신성_한계": RESUME_STALENESS_NOTE,
    }
