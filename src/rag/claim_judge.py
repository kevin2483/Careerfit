"""
생성된 주장 판정 — "이 주장이 인용한 문단에 실제로 있는 말인가?" 를 한 건씩 y/n 으로 받는다.

기준과 절차는 notes/rag_generation_2026-11-11.md (판정 전 고정). 기본값 없음, 건마다 저장. --group N 은 N 개씩 한 화면에 보여 주고 한 줄로 받는다 (부록 B2).

사용:
  python -m src.rag.claim_judge              # 미판정부터 이어서
  python -m src.rag.claim_judge --group 10   # 10개씩 묶어서 (재판정에도 쓸 수 있다)
  python -m src.rag.claim_judge --stats      # 진행률만
  python -m src.rag.claim_judge --rejudge    # 1차가 끝난 뒤 10% 를 1차 답을 가린 채 다시
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.paths import resolve_date
from src.rag.generate import (ANSWERS_PATH, CLAIM_JUDGMENTS_PATH, CLAIM_REJUDGMENTS_PATH, all_claims, chunk_header)
from src.rag.judge import REJUDGE_FRACTION, SEED, ask_confirm, ask_group, ask_yn
from src.rag.run import QUERIES_PATH, load_queries


def load_claim_judgments(path: str | Path) -> dict[str, bool]:
    if not Path(path).exists():
        return {}
    out = {}
    for r in read_jsonl(path):
        if not isinstance(r.get("supported"), bool):
            raise SystemExit(f"[중단] 판정 값이 true/false 가 아님: {r} ({path})")
        out[r["claim_id"]] = r["supported"]
    return out


def save_claim_judgments(path: str | Path, judgments: dict[str, bool]) -> None:
    write_jsonl(path, [{"claim_id": k, "supported": v} for k, v in sorted(judgments.items())])


def rejudge_ids(claim_ids: list[str], fraction: float = REJUDGE_FRACTION, seed: int = SEED) -> list[str]:
    ids = sorted(claim_ids)
    return random.Random(seed).sample(ids, math.ceil(len(ids) * fraction))


def show(claim: dict, question: str, chunk: dict, pos: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"[{pos}/{total}]  질문 {claim['qid']}: {question}")
    print("-" * 72)
    print(f"주장: {claim['text']}")
    print("-" * 72)
    print(f"인용 문단: {chunk_header(chunk)}")
    print(chunk["text"])
    print("=" * 72)


def session(todo: list[dict], questions: dict[str, str], chunks: dict[str, dict],
            judgments: dict[str, bool], out_path: str | Path) -> int:
    done = 0
    for pos, c in enumerate(todo, 1):
        show(c, questions[c["qid"]], chunks[c["chunk_id"]], pos, len(todo))
        a = ask_yn("이 주장이 인용한 문단에 실제로 있는 말인가? [y/n] (q=저장종료): ")
        if a == "q":
            break
        judgments[c["claim_id"]] = (a == "y")
        save_claim_judgments(out_path, judgments)
        done += 1
    return done


def group_session(todo: list[dict], questions: dict[str, str], chunks: dict[str, dict],
                  judgments: dict[str, bool], out_path: str | Path, size: int) -> int:
    """size 개씩 한 화면에 보여 주고 y/n 을 한 줄로 받는다. 되보여 준 목록에 ok 를 쳐야 저장한다."""
    done = 0
    for start in range(0, len(todo), size):
        group = todo[start:start + size]
        while True:
            for i, c in enumerate(group, 1):
                show(c, questions[c["qid"]], chunks[c["chunk_id"]], start + i, len(todo))
                print(f"↑ {i}번")
            answers = ask_group(len(group))
            if answers is None:
                return done
            print()
            for i, (c, v) in enumerate(zip(group, answers), 1):
                print(f"  {i}번 · {c['text'][:40]}… → {'y' if v else 'n'}")
            confirm = ask_confirm()
            if confirm == "q":
                return done
            if confirm == "ok":
                break
        judgments.update({c["claim_id"]: v for c, v in zip(group, answers)})
        save_claim_judgments(out_path, judgments)
        done += len(group)
    return done


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--answers", default=str(ANSWERS_PATH))
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--out", default=str(CLAIM_JUDGMENTS_PATH))
    ap.add_argument("--rejudge-out", default=str(CLAIM_REJUDGMENTS_PATH))
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--rejudge", action="store_true")
    ap.add_argument("--group", type=int, default=1, help="이 개수씩 한 화면에 (1 = 한 건씩)")
    a = ap.parse_args(argv)

    claims = all_claims(read_jsonl(a.answers))
    by_id = {c["claim_id"]: c for c in claims}
    first = load_claim_judgments(a.out)
    left = [c for c in claims if c["claim_id"] not in first]
    print(f"주장 {len(claims)}건 · 1차 판정 {len(claims) - len(left)}건 · 남음 {len(left)}건")
    if a.rejudge:
        if left:
            raise SystemExit(f"[중단] 1차 판정이 {len(left)}건 남음. 다 끝난 뒤에 재판정한다.")
        second = load_claim_judgments(a.rejudge_out)
        sample = rejudge_ids(list(by_id))
        todo, out_path, target = [by_id[i] for i in sample if i not in second], a.rejudge_out, second
        print(f"재판정 표본 {len(sample)}건 (seed {SEED}, {REJUDGE_FRACTION:.0%}) · 남음 {len(todo)}건. 1차 답은 보여 주지 않는다.")
    else:
        todo, out_path, target = left, a.out, first
    if a.stats or not todo:
        return
    from src.rag.search import load_chunks
    chunks = {c["chunk_id"]: c for c in load_chunks(resolve_date(a.date, prefer="processed"))}
    questions = {q["qid"]: q["question"] for q in load_queries(a.queries)}
    if a.group > 1:
        n = group_session(todo, questions, chunks, target, out_path, a.group)
    else:
        n = session(todo, questions, chunks, target, out_path)
    print(f"\n이번에 {n}건 판정 → {out_path}  (남음 {len(todo) - n}건)")


if __name__ == "__main__":
    main()
