"""
생성 단계 평가 — 충실도(사람 판정), 거절 2×2, 인용 적합도, 실패 행, 지연·토큰, LLM 판정 일치율.

정의는 notes/rag_generation_2026-11-11.md. 커밋된 파일만 읽는다 (API·원문 불필요):
  rag_queries.jsonl, rag_runs/hybrid.jsonl, rag_judgments.jsonl(검색 판정),
  rag_generation/answers.jsonl, rag_claim_judgments.jsonl(+ rejudgments), rag_generation/llm_judge.jsonl(있으면)
출력: data/processed/rag_generation.csv (항목 · 값 · 근거)
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.common.jsonl import read_jsonl
from src.common.paths import PROCESSED_DIR
from src.eval.bootstrap_ci import percentile
from src.rag.claim_judge import load_claim_judgments
from src.rag.generate import (ANSWERS_PATH, CLAIM_JUDGMENTS_PATH, CLAIM_REJUDGMENTS_PATH, LLM_JUDGE_PATH, all_claims)
from src.rag.judge import JUDGMENTS_PATH, load_judgments
from src.rag.run import QUERIES_PATH, load_queries

OUT_PATH = PROCESSED_DIR / "rag_generation.csv"
SUSPICIOUS = 0.98


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def evaluate(queries: list[dict], answers: list[dict], search_judgments: dict, claim_judgments: dict[str, bool],
             rejudgments: dict[str, bool] | None = None, llm_rows: list[dict] | None = None,
             b: int = 10000, seed: int = 42) -> list[dict]:
    claims = all_claims(answers)
    missing = [c["claim_id"] for c in claims if c["claim_id"] not in claim_judgments]
    if missing:
        raise SystemExit(f"[중단] 미판정 주장 {len(missing)}건: {missing[:5]} — python -m src.rag.claim_judge 로 먼저 판정")
    if {a["qid"] for a in answers} != {q["qid"] for q in queries}:
        raise SystemExit("[중단] answers.jsonl 의 질문이 질문 파일과 다르다")

    rows: list[dict] = []
    add = lambda item, value, basis: rows.append({"항목": item, "값": value, "근거": basis})   # noqa: E731
    failed = [a for a in answers if a["error"]]
    ok = [a for a in answers if not a["error"]]
    add("질문 수", str(len(queries)), f"생성 호출 {len(answers)}회, 모델 {answers[0]['model']}, 검색 {answers[0]['retriever']}")
    add("실패 행", str(len(failed)), ", ".join(f"{a['qid']}({a['error']})" for a in failed) or "없음")
    add("주장 수", str(len(claims)), f"답한 질문 {sum(bool(a['claims']) for a in ok)}개, 답변당 최대 2개")

    # 충실도: 주장이 있는 질문 단위로 리샘플
    per_q = [[claim_judgments[c["claim_id"]] for c in a["claims"]] for a in ok if a["claims"]]
    if claims:
        point = sum(map(sum, per_q)) / len(claims)
        rng, boots = random.Random(seed), []
        for _ in range(b):
            pick = [per_q[rng.randrange(len(per_q))] for _ in per_q]
            boots.append(sum(map(sum, pick)) / sum(map(len, pick)))
        flag = " ← 0.98 이상: 성과로 쓰기 전에 사람 확인 (과제 난이도·누수)" if point >= SUSPICIOUS else ""
        add("충실도 (주 지표)", f"{_fmt(point)} [{_fmt(percentile(boots, 0.025))}, {_fmt(percentile(boots, 0.975))}]",
            f"사람 판정 y {sum(map(sum, per_q))} / 주장 {len(claims)}, 질문 {len(per_q)}개 단위 부트스트랩 B={b} seed={seed}{flag}")
    else:
        add("충실도 (주 지표)", "계산 불가", "주장 0건")

    # 거절 2×2: 전달 문단에 검색 판정 y 가 하나라도 있는가 × 답했는가
    has_rel = {a["qid"]: any(search_judgments[(a["qid"], c)] for c in a["passed"]) for a in ok}
    cell = lambda rel, ans: sum(has_rel[a["qid"]] == rel and a["answerable"] == ans for a in ok)   # noqa: E731
    n_rel, n_no = sum(has_rel.values()), len(ok) - sum(has_rel.values())
    add("관련 문단 있음 · 답함", str(cell(True, True)), f"실패 행 제외 {len(ok)}개 중 관련 문단 있는 질문 {n_rel}개")
    add("관련 문단 있음 · 거절", str(cell(True, False)), "답할 수 있었는데 거절")
    add("관련 문단 없음 · 답함", str(cell(False, True)), "근거 없이 답함: " + (", ".join(
        a["qid"] for a in ok if not has_rel[a["qid"]] and a["answerable"]) or "없음"))
    add("관련 문단 없음 · 거절", str(cell(False, False)), f"관련 문단 없는 질문 {n_no}개: " + (", ".join(
        sorted(q for q, r in has_rel.items() if not r)) or "없음"))
    add("답변율", _fmt(cell(True, True) / n_rel) if n_rel else "계산 불가", f"{cell(True, True)} / {n_rel}")
    add("올바른 거절률", _fmt(cell(False, False) / n_no) if n_no else "계산 불가",
        f"{cell(False, False)} / {n_no}. 분모가 작아 건수로 읽는다")

    cited = sum(search_judgments[(c["qid"], c["chunk_id"])] for c in claims)
    add("인용 적합도 (보조)", _fmt(cited / len(claims)) if claims else "계산 불가",
        f"인용 문단이 검색 판정 y 인 주장 {cited} / {len(claims)}. 사람 추가 판정 없음")

    if rejudgments:
        both = [k for k in rejudgments if k in claim_judgments]
        same = sum(rejudgments[k] == claim_judgments[k] for k in both)
        add("자기 일치율 (재판정)", _fmt(same / len(both)) if both else "미실시", f"재판정 {len(both)}건 중 {same}건 일치")
    else:
        add("자기 일치율 (재판정)", "미실시", "rag_claim_rejudgments.jsonl 없음")

    if llm_rows:
        valid = [r for r in llm_rows if r["supported"] is not None and r["claim_id"] in claim_judgments]
        pair = lambda h, m: sum(claim_judgments[r["claim_id"]] == h and r["supported"] == m for r in valid)   # noqa: E731
        agree = pair(True, True) + pair(False, False)
        add("LLM 판정 일치율 (보조)", _fmt(agree / len(valid)) if valid else "계산 불가",
            f"{llm_rows[0]['model']}, 주장 {len(valid)}건 중 {agree}건 일치, 실패 {len(llm_rows) - len(valid)}건. 점수에는 쓰지 않는다")
        add("LLM 판정 2×2 (사람 y/n × LLM y/n)", f"yy {pair(True, True)} · yn {pair(True, False)} · ny {pair(False, True)} · nn {pair(False, False)}",
            "앞 글자 사람, 뒷 글자 LLM")
    else:
        add("LLM 판정 일치율 (보조)", "미실시", "llm_judge.jsonl 없음")

    ms = sorted(a["ms"] for a in answers)
    add("생성 지연", f"중앙값 {percentile(ms, 0.5):.0f}ms · p90 {percentile(ms, 0.9):.0f}ms", f"호출 {len(ms)}회 각 1회 실측")
    add("토큰", f"입력 {sum(a['input_tokens'] for a in answers)} · 출력 {sum(a['output_tokens'] for a in answers)}",
        "응답 usage 합계 (출력에는 사고 토큰 포함)")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--answers", default=str(ANSWERS_PATH))
    ap.add_argument("--search-judgments", default=str(JUDGMENTS_PATH))
    ap.add_argument("--judgments", default=str(CLAIM_JUDGMENTS_PATH))
    ap.add_argument("--rejudgments", default=str(CLAIM_REJUDGMENTS_PATH))
    ap.add_argument("--llm-judge", default=str(LLM_JUDGE_PATH))
    ap.add_argument("--b", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args(argv)
    rows = evaluate(load_queries(a.queries), read_jsonl(a.answers), load_judgments(a.search_judgments),
                    load_claim_judgments(a.judgments), load_claim_judgments(a.rejudgments),
                    read_jsonl(a.llm_judge) if Path(a.llm_judge).exists() else None, a.b, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["항목", "값", "근거"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['항목']:<34}{r['값']}")
    print(f"저장 -> {a.out}")


if __name__ == "__main__":
    main()
