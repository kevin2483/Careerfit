"""
실행 기록 — 질문 세트를 한 방식으로 돌려 상위 10 을 파일로 남긴다.

왜 파일로 남기나:
  청크 원문과 임베딩은 레포에 없다. (qid, rank, chunk_id, score) 만 커밋해 두면
  판정 파일과 합쳐 원문 없이 지표를 다시 계산할 수 있다 (make reproduce).

안전판 (notes/rag_experiment_2026-11-05.md):
  - 질문이 하나라도 비어 있으면 돌리지 않는다. 질문은 사람이 쓴다.
  - 이미 실행 기록이 있으면 덮어쓰지 않는다. "각 방식 1회"가 사전 등록 조건이다.
    버그로 다시 돌려야 하면 파일을 직접 지우고 그 사유를 노트에 적는다.

사용:
  python -m src.rag.run --method bm25 [--date ...]
  → data/processed/rag_runs/bm25.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.paths import LABELED_DIR, PROCESSED_DIR, resolve_date
from src.rag.filter import allowed_mask, validate

QUERIES_PATH = LABELED_DIR / "rag_queries.jsonl"
RUNS_DIR = PROCESSED_DIR / "rag_runs"
TOP_K = 10
QUERY_TYPES = {"A", "B", "C", "D"}


def load_queries(path: str | Path = QUERIES_PATH) -> list[dict]:
    """질문 파일을 읽고 형식을 검사한다. 빈 질문·중복 qid·모르는 유형·잘못된 필터는 전부 중단."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[중단] 질문 파일 없음: {path}")
    rows = read_jsonl(path)
    problems: list[str] = []
    seen: set[str] = set()
    for r in rows:
        qid = r.get("qid")
        if not qid or qid in seen:
            problems.append(f"qid 없음/중복: {qid!r}")
        seen.add(qid)
        if r.get("type") not in QUERY_TYPES:
            problems.append(f"{qid}: 모르는 유형 {r.get('type')!r}")
        if not str(r.get("question") or "").strip():
            problems.append(f"{qid}: 질문이 비어 있음")
        try:
            validate(r.get("filter", {}))
        except ValueError as e:
            problems.append(f"{qid}: {e}")
    if not rows:
        problems.append("질문이 0개")
    if problems:
        shown = "\n  ".join(problems[:10])
        more = f"\n  … 외 {len(problems) - 10}건" if len(problems) > 10 else ""
        raise SystemExit(f"[중단] 질문 파일 문제 {len(problems)}건 ({path}). 질문은 사람이 채운다.\n  {shown}{more}")
    return rows


def run_method(index, chunks: list[dict], queries: list[dict], k: int = TOP_K) -> list[dict]:
    """질문마다 필터를 걸어 상위 k. 결과가 k 개 미만일 수 있다 (필터 통과 청크가 적거나 BM25 일치가 없을 때)."""
    rows: list[dict] = []
    for q in queries:
        allowed = allowed_mask(chunks, q.get("filter"))
        for rank, (i, score) in enumerate(index.search(q["question"], k, allowed=allowed), 1):
            rows.append({"qid": q["qid"], "rank": rank, "chunk_id": chunks[i]["chunk_id"], "score": round(score, 6)})
    return rows


def main(argv: list[str] | None = None) -> None:
    from src.rag.search import build, load_chunks
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--method", required=True)
    ap.add_argument("--date")
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--out-dir", default=str(RUNS_DIR))
    args = ap.parse_args(argv)

    out = Path(args.out_dir) / f"{args.method}.jsonl"
    if out.exists():
        raise SystemExit(f"[중단] 이미 실행 기록이 있음: {out}  (각 방식 1회. 다시 돌릴 사유가 있으면 파일을 지우고 노트에 적을 것)")
    queries = load_queries(args.queries)
    date = resolve_date(args.date, prefer="processed")
    chunks = load_chunks(date)
    if chunks and "annual_from" not in chunks[0]:
        raise SystemExit(f"[중단] 청크에 annual_from 이 없음 → python -m src.rag.chunk --date {date} 로 다시 만들 것")
    rows = run_method(build(args.method, chunks, date), chunks, queries)
    write_jsonl(out, rows)
    short = sorted({q["qid"] for q in queries} - {r["qid"] for r in rows if r["rank"] == TOP_K})
    print(f"[날짜] {date}\n{args.method}: 질문 {len(queries)}개 → {len(rows)}행  ({out})")
    if short:
        print(f"[주의] 상위 {TOP_K} 를 못 채운 질문 {len(short)}개: {short}")


if __name__ == "__main__":
    main()
