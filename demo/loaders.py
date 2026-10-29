"""
데모가 읽는 커밋 산출물 로더. 숫자를 계산하지 않고 CSV 를 그대로 올린다 — 데모 수치 = README 수치 = CSV 수치.

여기서 새 통계를 만들면 README 와 두 벌이 된다. 가공은 표시용 정렬·필터·이름 치환까지만.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
STATS = PROCESSED / "market_stats"
EXTRACT = PROCESSED / "job_skill_extract_report"

JOB_NAMES = {"ds": "데이터 사이언티스트", "de": "데이터 엔지니어", "mle": "ML 엔지니어", "da": "데이터 분석가"}
METHOD_NAMES = {"regex": "정규식", "haiku": "Claude haiku-4.5", "sonnet": "Claude sonnet-5", "opus": "Claude opus-5"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(x: str) -> float | int | str:
    if x is None or x == "":
        return ""
    try:
        f = float(x)
    except ValueError:
        return x
    return int(f) if f.is_integer() and "." not in x else f


def _typed(rows: list[dict]) -> list[dict]:
    return [{k: _num(v) for k, v in r.items()} for r in rows]


def rank_by_job(job: str, top_n: int = 20) -> list[dict]:
    rows = [r for r in _typed(_read_csv(STATS / "rank_by_job.csv")) if r["직무"] == job]
    rows.sort(key=lambda r: (r["순위"], r["canonical"]))
    return rows[:top_n]


def junior_vs_senior(job: str) -> list[dict]:
    return [r for r in _typed(_read_csv(STATS / "junior_vs_senior.csv")) if r["직무"] == job]


def cooccurrence(job: str, top_n: int = 15) -> list[dict]:
    rows = [r for r in _typed(_read_csv(STATS / "cooccurrence.csv")) if r["직무"] == job]
    rows.sort(key=lambda r: (-r["동시출현수"], r["tag_a"], r["tag_b"]))
    return rows[:top_n]


def excluded_lowfreq(job: str) -> list[dict]:
    return [r for r in _typed(_read_csv(STATS / "excluded_lowfreq.csv")) if r["직무"] == job]


def population() -> dict[str, dict]:
    return {r["직무"]: r for r in _typed(_read_csv(EXTRACT / "validation_summary.csv")) if r["직무"] in JOB_NAMES}


def da_note(job: str) -> str:
    for r in _read_csv(STATS / "rank_by_job.csv"):
        if r["직무"] == job and r["비고"]:
            return r["비고"]
    return ""


def method_comparison() -> list[dict]:
    return _typed(_read_csv(PROCESSED / "method_comparison.csv"))


def bootstrap_ci() -> list[dict]:
    return _typed(_read_csv(PROCESSED / "bootstrap_ci.csv"))


def cascade_sim() -> list[dict]:
    return _typed(_read_csv(PROCESSED / "cascade_sim.csv"))


def failed_rows() -> list[dict]:
    return _typed(_read_csv(PROCESSED / "failed_rows_sensitivity.csv"))


def totals() -> list[dict]:
    """방식별 '전체' 행 + 비용 열 이름 정규화 (전체892건_예상비용USD → 전체비용)."""
    out = []
    for r in method_comparison():
        if r["구분"] != "전체":
            continue
        cost_key = next(k for k in r if k.startswith("전체") and k.endswith("예상비용USD"))
        out.append({**r, "전체비용": r[cost_key], "방식명": METHOD_NAMES.get(r["방식"], r["방식"])})
    return out


def field_f1() -> list[dict]:
    return [{"방식": METHOD_NAMES.get(r["방식"], r["방식"]), "필드": r["구분"], "F1": r["F1"]}
            for r in method_comparison() if r["구분"].startswith(("①", "②"))]


def paired_diffs() -> list[dict]:
    return [r for r in bootstrap_ci() if r["지표"] == "F1 차이"]


def routing_curves() -> list[dict]:
    """cascade_sim 의 라우팅 행을 (정책, 비율, F1, F1_sd) 로 — 그래프용."""
    out = []
    for r in cascade_sim():
        name = r["정책"]
        if name.startswith("라우팅·"):
            policy, frac = name[len("라우팅·"):].rsplit(" ", 1)
            out.append({"정책": policy, "명목비율%": int(frac.rstrip("%")), "실제비율": r["비율"], "F1": r["F1"],
                        "F1_sd": r["F1_sd"] if r["F1_sd"] != "" else 0.0, "전체비용": r["전체비용"]})
    return out


# 교정 이력. 수치 출처는 notes/lab_log.md (해당 날짜 항목) — CSV 에는 최종값만 있어 여기 적는다.
CORRECTIONS = [
    {"차수": 1, "날짜": "2026-08-14", "결함": "정답셋이 정규식 출력을 사람이 확인하는 방식 → 정규식이 못 본 태그가 정답에 없음",
     "조치": "블라인드 풀링 재라벨 (출처 가림, 순서 셔플)", "효과": "regex−LLM 격차 0.091 → 0.024"},
    {"차수": 2, "날짜": "2026-08-15", "결함": "풀링 후보를 regex·haiku 로만 만들어 sonnet·opus 단독 태그가 오탐으로 채점",
     "조치": "sonnet·opus 단독 태그 30건 추가 판정", "효과": "opus 최하위 → 1위"},
    {"차수": 3, "날짜": "2026-10-27", "결함": "사전 alias 충돌 4건, 정답셋 표기 오류 3건, 하위 도구→상위 도구 함의 규칙 부재",
     "조치": "alias 정리, 표기 교정, 사전 v2 implies, 부트스트랩 CI 도입", "효과": "순위 불변. 'opus 우위' → '확정 불가'"},
    {"차수": 4, "날짜": "2026-10-28", "결함": "sonnet 응답 1건이 max_tokens=512 에서 잘려 빈 예측으로 채점 (sonnet FN 의 31%)",
     "조치": "실패 행 경고·민감도 표 추가, 해당 1건 재호출", "효과": "sonnet 0.845 → 0.854, 파레토 최적 합류"},
]
