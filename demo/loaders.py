"""
데모가 읽는 커밋 산출물 로더. 숫자를 계산하지 않고 CSV 를 그대로 올린다 — 데모 수치 = README 수치 = CSV 수치.

여기서 새 통계를 만들면 README 와 두 벌이 된다. 가공은 표시용 정렬·필터·이름 치환까지만.
Streamlit Cloud 는 push 시 app.py 만 다시 읽고 이 모듈은 캐시된 채 둘 때가 있다(2026-10-29 AttributeError). 로더를 추가하면 이 파일도 바뀌어야 한다.
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


def prompt_variants() -> list[dict]:
    return _typed(_read_csv(PROCESSED / "prompt_variants.csv"))


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


# ── 공고 검색 (RAG) — rag_comparison.csv · rag_generation.csv 의 문자열을 표시용으로 나눌 뿐, 새로 계산하지 않는다
RAG_METHOD_NAMES = {"bm25": "BM25 (직접 구현)", "vector": "벡터 (e5-small)", "hybrid": "하이브리드 (RRF)",
                    "hybrid_rerank": "하이브리드 + 리랭커", "ctx_prefix": "벡터 + 문맥 접두", "sec3": "벡터, 3개 구역만"}
RAG_PRIMARY_PAIRS = [("hybrid", "vector"), ("hybrid", "bm25"), ("hybrid_rerank", "hybrid"), ("ctx_prefix", "vector"), ("sec3", "vector")]
RAG_METRICS = ["Recall@5", "MRR@5", "nDCG@5"]
# README "공고 검색 (RAG)" 절과 같은 세 건. 생성된 문장 그대로이고 회사명만 가렸다.
RAG_EXAMPLES = [
    {"질문": "RAG 아키텍처 구축 경험을 우대하는 곳",
     "답": "A사의 Software Engineer(AI Agent부문) 공고는 우대사항으로 RAG(Retrieval-Augmented Generation) 시스템 구축 경험을 제시한다.",
     "근거": "A사 공고의 우대사항 문단"},
    {"질문": "", "답": "B사의 AI 제품 엔지니어 공고는 우대사항으로 벡터 DB 등을 활용한 RAG 시스템을 설계/운영해 본 경험을 제시한다.",
     "근거": "B사 공고의 우대사항 문단"},
    {"질문": "랩톱 및 개발 장비를 자유롭게 선택할 수 있는 예산을 지원하는 회사",
     "답": "C사 공고는 근무에 필요한 장비를 500만원 예산 내에서 자유롭게 선택할 수 있도록 지원한다고 밝히고 있다.",
     "근거": "C사 공고의 복지 문단"},
    {"질문": "", "답": "D사 공고는 업무용 장비를 예산 내에서 원하는 것으로 지원하고 3년 후 교체해 준다고 밝히고 있다.",
     "근거": "D사 공고의 복지 문단"},
    {"질문": "신입 사원에게 초봉으로 15만 달러 이상을 보장하는 곳", "답": "(해당하는 공고 없음)",
     "근거": "전달된 문단 3개에 조건을 담은 것이 없어 답하지 않음"},
]
RAG_LIMITS = [
    "질문 30개, 판정자 1명. 다섯 비교 중 넷이 확정 불가인 이유다.",
    "판정이 흔들렸다. 50건 재판정 일치율 78% (A 60% · B 73% · C 100% · D 100%).",
    "판정 범위와 입력 방식을 도중에 바꿨고, 주장 판정 43건 중 1건은 AI 가 시연했다. 경위는 사전 등록 노트 부록에 있다.",
    "답이 질문에 맞는지, 빠뜨린 공고가 있는지는 재지 않았다.",
    "공고 원문이 비공개라 이 데모에서는 검색을 돌릴 수 없다. 여기 있는 것은 커밋된 평가 결과다.",
]


def _rag_table(name: str) -> dict[str, dict]:
    return {r["항목"]: r for r in _read_csv(PROCESSED / name)}


def _point_ci(value: str) -> tuple[float, float, float]:
    """'0.350 [0.307, 0.398] ...' → (0.350, 0.307, 0.398)"""
    point, rest = value.split(" [", 1)
    lo, hi = rest.split("]", 1)[0].split(", ")
    return float(point), float(lo), float(hi)


def rag_methods() -> list[dict]:
    t = _rag_table("rag_comparison.csv")
    out = []
    for m, name in RAG_METHOD_NAMES.items():
        row = {"방식": name}
        for metric in RAG_METRICS:
            row[metric], row[f"{metric}_하한"], row[f"{metric}_상한"] = _point_ci(t[f"{m} · {metric}"]["값"])
        row["지연"] = t[f"{m} · 검색 지연"]["값"]
        row["지연_p90_ms"] = int(row["지연"].split("p90 ")[1].rstrip("ms"))
        out.append(row)
    return out


def rag_pairs() -> list[dict]:
    t = _rag_table("rag_comparison.csv")
    out = []
    for a, b in RAG_PRIMARY_PAIRS:
        value = t[f"{a} − {b} · Recall@5"]["값"]
        point, lo, hi = _point_ci(value)
        out.append({"비교": f"{RAG_METHOD_NAMES[a]} − {RAG_METHOD_NAMES[b]}", "Recall@5 차이": point,
                    "CI_하한": lo, "CI_상한": hi, "판정": value.split("] ", 1)[1]})
    return out


def rag_summary() -> list[dict]:
    t = _rag_table("rag_comparison.csv")
    keys = ["질문 수", "평가 질문 수 (A~C, 풀 안 정답 ≥1)", "판정 풀", "풀 안 정답 비율", "자기 일치율 (재판정)",
            "자기 일치율 · A 유형", "자기 일치율 · B 유형", "자기 일치율 · C 유형", "자기 일치율 · D 유형", "D 유형 중 정답이 발견된 질문"]
    return [t[k] for k in keys]


def rag_generation() -> list[dict]:
    return _read_csv(PROCESSED / "rag_generation.csv")
