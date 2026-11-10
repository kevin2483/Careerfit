"""
메타데이터 필터 — 질문의 구조화 조건(경력·직무)을 모든 검색 방식에 똑같이 건다.

왜 검색이 아니라 필터인가:
  사전 확인에서 BM25·벡터 모두 "신입 가능"을 본문으로 거르지 못했다 (notes/rag_experiment_2026-11-05.md).
규칙 (같은 노트 부록 A2, 실행 전 고정):
  - annual_max: annual_from 이 숫자이고 annual_max 이하인 공고만 통과. None(미기재)은 제외.
  - jobs: 공고의 all_jobs 중 하나라도 목록에 있으면 통과.
  - 필터는 상위 k 를 자르기 전에 건다. 각 색인의 search(query, k, allowed=...) 가 이 마스크를 받는다.
"""
from __future__ import annotations

from src.common.paths import JOBS

FILTER_KEYS = {"annual_max", "jobs"}


def validate(flt: dict) -> None:
    """모르는 키·잘못된 값은 조용히 무시하지 않고 멈춘다 (오타 난 필터가 '필터 없음'으로 돌면 결과가 바뀐다)."""
    if not isinstance(flt, dict):
        raise ValueError(f"filter 는 객체여야 함: {flt!r}")
    unknown = set(flt) - FILTER_KEYS
    if unknown:
        raise ValueError(f"모르는 filter 키: {sorted(unknown)} (허용: {sorted(FILTER_KEYS)})")
    if "annual_max" in flt and (isinstance(flt["annual_max"], bool) or not isinstance(flt["annual_max"], int)
                                or flt["annual_max"] < 0):
        raise ValueError(f"annual_max 는 0 이상 정수: {flt['annual_max']!r}")
    if "jobs" in flt:
        jobs = flt["jobs"]
        if not isinstance(jobs, list) or not jobs or set(jobs) - set(JOBS):
            raise ValueError(f"jobs 는 {JOBS} 중에서 고른 비어 있지 않은 목록: {jobs!r}")


def passes(chunk: dict, flt: dict) -> bool:
    if "annual_max" in flt:
        a = chunk.get("annual_from")
        if a is None or a > flt["annual_max"]:
            return False
    if "jobs" in flt and not set(chunk.get("all_jobs") or []) & set(flt["jobs"]):
        return False
    return True


def allowed_mask(chunks: list[dict], flt: dict | None) -> list[bool] | None:
    """청크별 통과 여부. 필터가 비어 있으면 None (= 전부 통과, 색인이 마스크 계산을 건너뛴다)."""
    if not flt:
        return None
    validate(flt)
    return [passes(c, flt) for c in chunks]


class MaskedIndex:
    """색인의 일부만 검색 대상으로 삼는 껍데기. 변형 sec3(3구역만 색인)에 쓴다.
    벡터 검색에는 코퍼스 통계가 없으므로, 전체 행렬에서 가리는 것과 3구역만 따로 색인하는 것의 결과가 같다."""

    def __init__(self, index, base: list[bool]):
        self.index, self.base = index, base

    def search(self, query: str, k: int = 10, allowed: list[bool] | None = None) -> list[tuple[int, float]]:
        mask = self.base if allowed is None else [b and a for b, a in zip(self.base, allowed)]
        return self.index.search(query, k, allowed=mask)
