"""
벡터 검색 — 청크를 임베딩해 numpy 행렬로 저장하고, 질의 벡터와 코사인 유사도로 순위를 낸다.

설계:
  - 모델: 기본 intfloat/multilingual-e5-small (로컬 실행 → 공고 원문이 외부 API 로 나가지 않음).
    BAAI/bge-m3 는 8.6GB 맥에서 메모리를 거의 다 써 시스템이 멈췄다(2026-11-05) → 비교용으로만 따로 돌린다.
  - e5 계열은 학습 방식상 질의 앞에 "query: ", 문서 앞에 "passage: " 를 붙여야 한다. 빠뜨리면 성능이 떨어진다.
  - 저장소: 청크 4,573개라 벡터 DB 없이 numpy 행렬 하나(.npy)로 충분하다. 정규화된 벡터의 내적 = 코사인 유사도.
  - 캐시: data/processed/rag/<date>/emb_<모델>.npy + 같은 순서의 chunk_id 목록. chunk_id 가 바뀌면 다시 만든다.

사용:
  python -m src.rag.vector [--date ...] [--model intfloat/multilingual-e5-small]   # 임베딩 생성
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.common.paths import resolve_date
from src.rag.chunk import RAG_DIR

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
_MODELS: dict = {}

MAX_SEQ_TOKENS = 512   # e5 는 512 가 상한. bge-m3 기본 8192 는 메모리 부족을 일으켰다. 청크 최대 1,500자.
# 모델별 (질의 접두어, 문서 접두어)
PREFIXES = {"intfloat/multilingual-e5-small": ("query: ", "passage: "),
            "intfloat/multilingual-e5-base": ("query: ", "passage: ")}


def _model(name: str):
    if name not in _MODELS:
        import torch
        from sentence_transformers import SentenceTransformer  # 무거운 import 는 쓸 때만
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        kw = {"torch_dtype": torch.float16} if device == "mps" else {}
        m = SentenceTransformer(name, device=device, model_kwargs=kw)
        m.max_seq_length = min(MAX_SEQ_TOKENS, m.max_seq_length)
        _MODELS[name] = m
    return _MODELS[name]


def embed(texts: list[str], model: str = DEFAULT_MODEL, kind: str = "passage", batch_size: int = 16) -> np.ndarray:
    """kind: "query" 또는 "passage" — 모델이 요구하는 접두어를 붙인다."""
    q_pre, p_pre = PREFIXES.get(model, ("", ""))
    pre = q_pre if kind == "query" else p_pre
    vecs = _model(model).encode([pre + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=len(texts) > 100, convert_to_numpy=True)
    return vecs.astype(np.float32)


# ctx_prefix 변형이 임베딩 앞에 붙이는 구역 이름 (원티드 공고의 구역 제목)
SECTION_LABELS = {"intro": "회사 소개", "main_tasks": "주요업무", "requirements": "자격요건",
                  "preferred_points": "우대사항", "benefits": "혜택 및 복지"}


def ctx_prefix_text(chunk: dict) -> str:
    """변형 ctx_prefix: "제목 · 회사 · 구역명" 을 청크 앞에 붙인다 (사전 등록 노트의 변형 표).
    짧은 우대사항 문단이 어느 공고·구역 것인지 임베딩이 알게 하려는 것이다. 판정 화면과 BM25 에는 쓰지 않는다."""
    head = " · ".join([chunk.get("title", ""), chunk.get("company", ""),
                       SECTION_LABELS.get(chunk.get("section", ""), chunk.get("section", ""))])
    return f"{head}\n{chunk['text']}"


def cache_paths(date: str, model: str, variant: str = "") -> tuple[Path, Path]:
    tag = model.split("/")[-1] + (f"__{variant}" if variant else "")
    d = RAG_DIR / date
    return d / f"emb_{tag}.npy", d / f"emb_{tag}.ids.json"


def load_or_build(chunks: list[dict], date: str, model: str = DEFAULT_MODEL, variant: str = "") -> np.ndarray:
    """variant="" 는 청크 본문 그대로, "ctx_prefix" 는 ctx_prefix_text() 를 임베딩한다. 캐시 파일이 따로다."""
    if variant not in ("", "ctx_prefix"):
        raise ValueError(f"모르는 임베딩 변형: {variant!r}")
    npy, ids_path = cache_paths(date, model, variant)
    ids = [c["chunk_id"] for c in chunks]
    if npy.exists() and ids_path.exists() and json.loads(ids_path.read_text(encoding="utf-8")) == ids:
        return np.load(npy)
    t0 = time.time()
    mat = embed([ctx_prefix_text(c) if variant else c["text"] for c in chunks], model)
    np.save(npy, mat)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"[임베딩] {len(ids)}개 · {mat.shape[1]}차원 · {time.time() - t0:.0f}초 → {npy}")
    return mat


class VectorIndex:
    def __init__(self, matrix: np.ndarray, model: str = DEFAULT_MODEL):
        self.matrix, self.model = matrix, model

    def search(self, query: str, k: int = 10, allowed: list[bool] | None = None) -> list[tuple[int, float]]:
        """allowed: 청크별 필터 통과 여부. 상위 k 를 자르기 전에 가린다 (사전 등록 부록 A2)."""
        q = embed([query], self.model, kind="query")[0]
        sims = self.matrix @ q
        if allowed is not None:
            sims = np.where(np.asarray(allowed, dtype=bool), sims, -np.inf)
        top = np.argsort(-sims, kind="stable")[:k]
        return [(int(i), float(sims[i])) for i in top if np.isfinite(sims[i])]


def main(argv: list[str] | None = None) -> None:
    from src.rag.search import load_chunks
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    date = resolve_date(args.date, prefer="processed")
    load_or_build(load_chunks(date), date, args.model)


if __name__ == "__main__":
    main()
