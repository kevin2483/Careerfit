"""
hybrid_rerank — hybrid 상위 20 을 교차 인코더(리랭커)로 다시 정렬한다.

왜 두 단계인가:
  리랭커는 (질문, 청크)를 한 쌍씩 같이 읽어 점수를 내므로 정확하지만 4,573개 전부에 돌릴 수 없다.
  싼 검색(hybrid)으로 20개까지 줄인 뒤 비싼 모델은 그 20개에만 쓴다.

고정값 (notes/rag_experiment_2026-11-05.md 부록 A6, 실행 전 커밋):
  모델 BAAI/bge-reranker-v2-m3 · 후보 hybrid 상위 20 · 입력은 청크 본문만 · 512 토큰 · 배치 4 · MPS fp16
  점수에 NaN 이 나오면 멈춘다 (fp16 수치 오류를 결과로 남기지 않는다).
"""
from __future__ import annotations

import math

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_DEPTH = 20
MAX_TOKENS = 512
BATCH_SIZE = 4
_SCORERS: dict = {}


def cross_encoder_scorer(model: str = RERANK_MODEL):
    """(질문, 본문 목록) → 점수 목록. 무거운 import 와 모델 로딩은 처음 부를 때만."""
    if model not in _SCORERS:
        import torch
        from sentence_transformers import CrossEncoder
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        kw = {"torch_dtype": torch.float16} if device == "mps" else {}
        ce = CrossEncoder(model, device=device, max_length=MAX_TOKENS, model_kwargs=kw)

        def score(query: str, texts: list[str]) -> list[float]:
            out = ce.predict([(query, t) for t in texts], batch_size=BATCH_SIZE, show_progress_bar=False)
            return [float(x) for x in out]
        _SCORERS[model] = score
    return _SCORERS[model]


class RerankIndex:
    def __init__(self, base, chunks: list[dict], scorer=None, depth: int = RERANK_DEPTH):
        self.base, self.chunks, self.depth = base, chunks, depth
        self._scorer = scorer

    def search(self, query: str, k: int = 10, allowed: list[bool] | None = None) -> list[tuple[int, float]]:
        cands = [i for i, _s in self.base.search(query, self.depth, allowed=allowed)]
        if not cands:
            return []
        scorer = self._scorer or cross_encoder_scorer()
        scores = scorer(query, [self.chunks[i]["text"] for i in cands])
        if any(math.isnan(s) or math.isinf(s) for s in scores):
            raise SystemExit("[중단] 리랭커 점수에 NaN/inf — fp16 수치 오류일 수 있다. 실행 기록을 쓰지 않는다.")
        order = sorted(range(len(cands)), key=lambda j: (-scores[j], j))   # 동점은 hybrid 순위가 앞선 쪽
        return [(cands[j], scores[j]) for j in order[:k]]
