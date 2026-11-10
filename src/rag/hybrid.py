"""
hybrid — BM25 와 벡터 검색의 순위를 RRF(Reciprocal Rank Fusion)로 합친다.

왜 점수가 아니라 순위인가:
  e5 코사인 유사도는 0.87~0.91 에 몰려 있고 BM25 점수는 질의 길이에 따라 0~수십까지 움직인다.
  두 점수를 정규화해 더하면 정규화 방식이 결과를 좌우한다. 순위만 쓰면 그 선택이 사라진다.

고정값 (notes/rag_experiment_2026-11-05.md, 결과를 보고 바꾸지 않는다): RRF k=60, 각 방식 상위 50.
필터는 각 방식이 상위 50 을 자르기 전에 건다 (부록 A2).
"""
from __future__ import annotations

RRF_K = 60
DEPTH = 50


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """rankings: 방식별 문서 번호 목록(1위부터). 점수 = Σ 1/(k + 순위). 동점은 문서 번호가 작은 쪽이 앞."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))


class HybridIndex:
    def __init__(self, *indexes, depth: int = DEPTH, k: int = RRF_K):
        self.indexes, self.depth, self.k = indexes, depth, k

    def search(self, query: str, k: int = 10, allowed: list[bool] | None = None) -> list[tuple[int, float]]:
        rankings = [[i for i, _s in ix.search(query, self.depth, allowed=allowed)] for ix in self.indexes]
        return rrf(rankings, self.k)[:k]
