"""
청크 검색 CLI — 방식별 상위 k 를 눈으로 확인할 때 쓴다 (평가는 별도 스크립트).

사용:
  python -m src.rag.search "쿠버네티스로 모델 서빙해 본 경험" [--method bm25] [-k 5] [--date ...]
"""
from __future__ import annotations

import argparse

from src.common.jsonl import read_jsonl
from src.common.paths import resolve_date
from src.rag.bm25 import BM25
from src.rag.chunk import RAG_DIR


def load_chunks(date: str) -> list[dict]:
    path = RAG_DIR / date / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"[중단] 청크 없음: {path}  → 먼저 python -m src.rag.chunk --date {date}")
    return read_jsonl(path)


def build(method: str, chunks: list[dict], date: str):
    texts = [c["text"] for c in chunks]
    if method == "bm25":
        return BM25(texts)
    if method == "vector":
        from src.rag.vector import VectorIndex, load_or_build
        return VectorIndex(load_or_build(chunks, date))
    raise SystemExit(f"[중단] 아직 없는 방식: {method}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--method", default="bm25")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--date")
    args = ap.parse_args(argv)
    date = resolve_date(args.date, prefer="processed")
    chunks = load_chunks(date)
    index = build(args.method, chunks, date)
    for rank, (i, score) in enumerate(index.search(args.query, args.k), 1):
        c = chunks[i]
        snippet = c["text"].replace("\n", " ")[:120]
        print(f"{rank}. [{score:.2f}] {c['chunk_id']}  {c['title'][:30]} ({c['company']})\n   {snippet}")


if __name__ == "__main__":
    main()
