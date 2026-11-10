"""
검색 결과 판정 — 블라인드 풀링, 한 건씩 y/n 강제.

무엇을 하나:
  여섯 방식의 상위 5(D 유형은 상위 3)를 질문별로 합쳐(풀링), 어느 방식이 냈는지 감추고, 질문 안에서 순서를 섞어
  "이 문단이 질문 조건을 직접 담고 있나"를 한 건씩 묻는다. 결과는 (qid, chunk_id, relevant).

지키는 것 (notes/rag_experiment_2026-11-05.md 부록 A1):
  - 여섯 방식의 실행 기록이 모두 있어야 시작한다. 일부만으로 판정하면 나중 방식의 청크가 구별되어 블라인드가 깨진다.
  - 기본값이 없다. 엔터만 치면 다시 묻는다. 일괄 처리·건너뛰기도 없다.
    (2026-10-30: pool_cli --batch 의 '엔터 = 유지'가 수용 편향을 만들어 판정 208건을 폐기했다)
  - 화면에는 방식 이름·순위·점수가 나오지 않는다.
  - 판정할 때마다 저장한다. q 로 끊고 나중에 이어서 할 수 있다.

사용:
  python -m src.rag.judge              # 미판정부터 이어서
  python -m src.rag.judge --stats      # 진행률만
  python -m src.rag.judge --rejudge    # 1차가 끝난 뒤, 10% 를 이전 답을 가린 채 다시 판정 (자기 일치율)
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.paths import LABELED_DIR, resolve_date
from src.rag.run import JUDGE_K, JUDGE_K_D, METHODS, QUERIES_PATH, RUNS_DIR, load_queries

JUDGMENTS_PATH = LABELED_DIR / "rag_judgments.jsonl"
REJUDGMENTS_PATH = LABELED_DIR / "rag_rejudgments.jsonl"
SEED = 42
REJUDGE_FRACTION = 0.10
Pair = tuple[str, str]   # (qid, chunk_id)


def load_runs(runs_dir: str | Path = RUNS_DIR, methods: list[str] = METHODS) -> dict[str, list[dict]]:
    missing = [m for m in methods if not (Path(runs_dir) / f"{m}.jsonl").exists()]
    if missing:
        raise SystemExit(f"[중단] 실행 기록 없는 방식: {missing} ({runs_dir}). 여섯 방식이 모두 끝난 뒤에 판정·평가한다.")
    return {m: read_jsonl(Path(runs_dir) / f"{m}.jsonl") for m in methods}


def depth_by_qid(queries) -> dict[str, int]:
    """질문별 판정 깊이: D 유형은 JUDGE_K_D, 나머지는 JUDGE_K (부록 A7)."""
    return {q["qid"]: JUDGE_K_D if q["type"] == "D" else JUDGE_K for q in queries}


def build_pool(runs: dict[str, list[dict]], k: int = JUDGE_K, depths: dict[str, int] | None = None) -> dict[str, list[str]]:
    """질문별 후보 청크. 방식 정보는 버리고, 고정 시드로 섞어 순서가 힌트가 되지 않게 한다 (재개해도 순서가 같다).
    depths 가 있으면 질문마다 그 깊이까지, 없으면 k 까지 모은다."""
    pool: dict[str, set[str]] = {}
    for rows in runs.values():
        for r in rows:
            if r["rank"] <= (depths or {}).get(r["qid"], k):
                pool.setdefault(r["qid"], set()).add(r["chunk_id"])
    out: dict[str, list[str]] = {}
    for qid in sorted(pool):
        ids = sorted(pool[qid])
        random.Random(f"{SEED}:{qid}").shuffle(ids)
        out[qid] = ids
    return out


def load_judgments(path: str | Path) -> dict[Pair, bool]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[Pair, bool] = {}
    for r in read_jsonl(path):
        if not isinstance(r.get("relevant"), bool):
            raise SystemExit(f"[중단] 판정 값이 true/false 가 아님: {r} ({path})")
        out[(r["qid"], r["chunk_id"])] = r["relevant"]
    return out


def save_judgments(path: str | Path, judgments: dict[Pair, bool]) -> None:
    write_jsonl(path, [{"qid": q, "chunk_id": c, "relevant": v} for (q, c), v in sorted(judgments.items())])


def ask_yn(prompt: str) -> str:
    """y / n / q 만 받는다. 빈 입력과 오타는 다시 묻는다 — 기본값이 없다."""
    while True:
        try:
            a = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"
        if a in ("y", "n", "q"):
            return a
        print("     ! y 또는 n 을 입력하세요 (q = 저장하고 종료). 엔터만으로는 넘어가지 않습니다.")


def rejudge_sample(pairs: list[Pair], fraction: float = REJUDGE_FRACTION, seed: int = SEED) -> list[Pair]:
    pairs = sorted(pairs)
    n = math.ceil(len(pairs) * fraction)
    picked = random.Random(seed).sample(pairs, n)
    random.Random(f"{seed}:order").shuffle(picked)   # 질문 순서로 몰리지 않게
    return picked


def show(query: dict, chunk: dict, pos: int, total: int) -> None:
    flt = query.get("filter") or {}
    print("\n" + "=" * 72)
    print(f"[{pos}/{total}]  질문 {query['qid']}: {query['question']}")
    if flt:
        print(f"필터: {flt}")
    print("-" * 72)
    print(f"{chunk.get('title', '')}  ·  {chunk.get('company', '')}  ·  구역 {chunk.get('section', '')}")
    print("-" * 72)
    print(chunk["text"])
    print("=" * 72)


def session(todo: list[Pair], queries: dict[str, dict], chunks: dict[str, dict],
            judgments: dict[Pair, bool], out_path: str | Path) -> int:
    """todo 를 순서대로 묻고 한 건마다 저장한다. 판정한 건수를 돌려준다."""
    done = 0
    for pos, (qid, cid) in enumerate(todo, 1):
        show(queries[qid], chunks[cid], pos, len(todo))
        a = ask_yn("이 문단이 질문 조건을 직접 담고 있나? [y/n] (q=저장종료): ")
        if a == "q":
            break
        judgments[(qid, cid)] = (a == "y")
        save_judgments(out_path, judgments)
        done += 1
    return done


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--out", default=str(JUDGMENTS_PATH))
    ap.add_argument("--rejudge-out", default=str(REJUDGMENTS_PATH))
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--rejudge", action="store_true")
    args = ap.parse_args(argv)

    queries = {q["qid"]: q for q in load_queries(args.queries)}
    pool = build_pool(load_runs(args.runs_dir), depths=depth_by_qid(queries.values()))
    pairs = [(qid, cid) for qid, ids in pool.items() for cid in ids]
    first = load_judgments(args.out)
    left = [p for p in pairs if p not in first]
    print(f"풀 {len(pairs)}건 (질문 {len(pool)}개) · 1차 판정 {len(pairs) - len(left)}건 · 남음 {len(left)}건")

    if args.rejudge:
        if left:
            raise SystemExit(f"[중단] 1차 판정이 {len(left)}건 남음. 다 끝난 뒤에 재판정한다.")
        second = load_judgments(args.rejudge_out)
        sample = rejudge_sample(pairs)
        todo, out_path, target = [p for p in sample if p not in second], args.rejudge_out, second
        print(f"재판정 표본 {len(sample)}건 (seed {SEED}, {REJUDGE_FRACTION:.0%}) · 남음 {len(todo)}건. 1차 답은 보여 주지 않는다.")
    else:
        todo, out_path, target = left, args.out, first
    if args.stats or not todo:
        return

    from src.rag.search import load_chunks
    chunks = {c["chunk_id"]: c for c in load_chunks(resolve_date(args.date, prefer="processed"))}
    unknown = [c for _q, c in todo if c not in chunks]
    if unknown:
        raise SystemExit(f"[중단] 청크 파일에 없는 chunk_id {len(unknown)}건: {unknown[:3]} — 실행 기록과 같은 --date 인지 확인")
    n = session(todo, queries, chunks, target, out_path)
    print(f"\n이번에 {n}건 판정 → {out_path}  (남음 {len(todo) - n}건)")


if __name__ == "__main__":
    main()
