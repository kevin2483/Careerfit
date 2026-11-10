"""
검색 방식 비교 — 커밋된 실행 기록과 사람 판정만으로 지표와 짝지은 신뢰구간을 낸다 (원문·모델 불필요).

입력: data/labeled/rag_queries.jsonl, data/labeled/rag_judgments.jsonl(+ rag_rejudgments.jsonl),
      data/processed/rag_runs/<방식>.jsonl
산출: data/processed/rag_comparison.csv  (항목 · 값 · 근거)

정의 (notes/rag_experiment_2026-11-05.md 본문 + 부록 A3, 실행 전 고정):
  - Recall@10 (주 지표) = 상위 10 안의 정답 청크 수 / 풀 안 정답 청크 수. 풀 밖 정답은 모른다.
  - MRR@10 = 1 / 첫 정답 순위 (상위 10 에 없으면 0).  nDCG@10 = 이진 이득, 이상 순위는 min(정답 수, 10) 개가 앞에.
  - 공고 단위 Recall@10 = 정답 청크가 속한 공고 중 상위 10 에 하나라도 나온 공고 비율.
  - A~C 인데 풀 안 정답이 0 개인 질문은 평균에서 빼고 수를 적는다. D 는 넣지 않고 상위 1 점수 분포만 낸다.
  - 차이는 질문 단위 부트스트랩(B=10,000, seed 42)의 짝지은 95% CI. 0 을 포함하면 "확정 불가".
    주 비교 5쌍만 판정하고 나머지 쌍은 참고로만 싣는다.
  - 상위 10 에 미판정 청크가 하나라도 있으면 수치를 내지 않고 멈춘다. 미판정을 오답으로 치지 않는다.

    python -m src.eval.rag_eval
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from src.common.paths import PROCESSED_DIR
from src.eval.bootstrap_ci import percentile
from src.rag.judge import JUDGMENTS_PATH, REJUDGMENTS_PATH, build_pool, load_judgments, load_runs
from src.rag.run import QUERIES_PATH, RUNS_DIR, TOP_K, load_queries

METRICS = ["Recall@10", "MRR@10", "nDCG@10", "공고 Recall@10"]
PRIMARY_PAIRS = [("hybrid", "bm25"), ("hybrid", "vector"), ("ctx_prefix", "vector"),
                 ("sec3", "vector"), ("hybrid_rerank", "hybrid")]
SCORE_METHODS = ["bm25", "vector", "ctx_prefix", "sec3"]   # RRF·리랭커 점수는 질문 간 비교가 안 된다
SUSPICIOUS = 0.98   # CLAUDE.md ⑤: 비정상적으로 높은 지표는 누수를 의심


def query_metrics(ranked: list[str], relevant: set[str], k: int = TOP_K) -> dict[str, float]:
    """ranked: 1위부터의 chunk_id. relevant: 풀 안 정답 chunk_id (비어 있으면 호출하지 않는다)."""
    top = ranked[:k]
    hits = [c in relevant for c in top]
    first = next((i for i, h in enumerate(hits, 1) if h), None)
    dcg = sum(1 / math.log2(i + 1) for i, h in enumerate(hits, 1) if h)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    job = lambda c: c.split(":", 1)[0]   # noqa: E731  chunk_id = "<job_id>:<구역>:<번호>"
    rel_jobs = {job(c) for c in relevant}
    return {"Recall@10": sum(hits) / len(relevant), "MRR@10": 1 / first if first else 0.0,
            "nDCG@10": dcg / idcg, "공고 Recall@10": len({job(c) for c in top} & rel_jobs) / len(rel_jobs)}


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _ci(xs: list[float]) -> tuple[float, float]:
    return percentile(xs, 0.025), percentile(xs, 0.975)


def evaluate(queries: list[dict], runs: dict[str, list[dict]], judgments: dict, rejudgments: dict | None = None,
             b: int = 10_000, seed: int = 42) -> list[dict]:
    """rag_comparison.csv 의 행(항목·값·근거)을 만든다."""
    pool = build_pool(runs)
    pairs = [(q, c) for q, ids in pool.items() for c in ids]
    unjudged = [p for p in pairs if p not in judgments]
    if unjudged:
        raise SystemExit(f"[중단] 상위 {TOP_K} 안에 미판정 청크 {len(unjudged)}건 (예: {unjudged[:3]}). "
                         "판정을 끝낸 뒤 평가한다. 미판정을 오답으로 치지 않는다.")
    known = {q["qid"] for q in queries}
    stray = sorted({q for q, _c in pairs} - known)
    if stray:
        raise SystemExit(f"[중단] 질문 파일에 없는 qid 가 실행 기록에 있음: {stray[:5]}")

    relevant = {q["qid"]: {c for c in pool.get(q["qid"], []) if judgments[(q["qid"], c)]} for q in queries}
    ranked = {m: {} for m in runs}
    top1 = {m: {} for m in runs}
    for m, rows in runs.items():
        for r in sorted(rows, key=lambda r: (r["qid"], r["rank"])):
            ranked[m].setdefault(r["qid"], []).append(r["chunk_id"])
            if r["rank"] == 1:
                top1[m][r["qid"]] = r["score"]

    scored = [q for q in queries if q["type"] != "D"]
    no_gold = [q["qid"] for q in scored if not relevant[q["qid"]]]
    eval_q = [q["qid"] for q in scored if relevant[q["qid"]]]
    d_q = [q["qid"] for q in queries if q["type"] == "D"]
    if not eval_q:
        raise SystemExit("[중단] 정답이 있는 A~C 질문이 0개라 지표를 낼 수 없음")
    n = len(eval_q)
    per = {m: [query_metrics(ranked[m].get(q, []), relevant[q]) for q in eval_q] for m in runs}

    rows: list[dict] = []
    add = lambda item, value, basis: rows.append({"항목": item, "값": value, "근거": basis})   # noqa: E731
    n_rel = sum(judgments[p] for p in pairs)
    add("질문 수", str(len(queries)), "유형별 " + " · ".join(
        f"{t} {sum(q['type'] == t for q in queries)}" for t in "ABCD"))
    add("평가 질문 수 (A~C, 풀 안 정답 ≥1)", str(n), "Recall·MRR·nDCG 의 분모")
    add("정답 0개라 제외한 A~C 질문", str(len(no_gold)), ", ".join(no_gold) or "없음")
    add("판정 풀", f"{len(pairs)}건", f"{len(runs)}개 방식 상위 {TOP_K} 합집합, 질문 {len(pool)}개")
    add("풀 안 정답 비율", _fmt(n_rel / len(pairs)), f"정답 {n_rel} / 판정 {len(pairs)}")
    if rejudgments:
        both = [p for p in rejudgments if p in judgments]
        same = sum(rejudgments[p] == judgments[p] for p in both)
        add("자기 일치율 (재판정)", _fmt(same / len(both)) if both else "미실시",
            f"재판정 {len(both)}건 중 {same}건 일치. 점수에는 1차 판정만 사용")
    else:
        add("자기 일치율 (재판정)", "미실시", "rag_rejudgments.jsonl 없음")

    rng = random.Random(seed)
    samples = [[rng.randrange(n) for _ in range(n)] for _ in range(b)]
    mean = lambda xs, idx: sum(xs[i] for i in idx) / len(idx)   # noqa: E731
    full = list(range(n))
    basis = f"질문 {n}개 평균, 질문 단위 부트스트랩 B={b} seed={seed} 95% CI"
    boot: dict[tuple[str, str], list[float]] = {}
    point: dict[tuple[str, str], float] = {}
    for m in runs:
        for metric in METRICS:
            xs = [p[metric] for p in per[m]]
            point[m, metric] = mean(xs, full)
            boot[m, metric] = [mean(xs, s) for s in samples]
            lo, hi = _ci(boot[m, metric])
            add(f"{m} · {metric}", f"{_fmt(point[m, metric])} [{_fmt(lo)}, {_fmt(hi)}]", basis)

    names = list(runs)
    all_pairs = [(a, c) for a, c in PRIMARY_PAIRS if a in runs and c in runs]
    all_pairs += [(a, c) for i, a in enumerate(names) for c in names[i + 1:]
                  if (a, c) not in PRIMARY_PAIRS and (c, a) not in PRIMARY_PAIRS]
    for a, c in all_pairs:
        primary = (a, c) in PRIMARY_PAIRS
        for metric in METRICS:
            diffs = [x - y for x, y in zip(boot[a, metric], boot[c, metric])]
            lo, hi = _ci(diffs)
            verdict = ("확정" if lo > 0 or hi < 0 else "확정 불가") if primary else "참고 (판정 안 함)"
            add(f"{a} − {c} · {metric}", f"{point[a, metric] - point[c, metric]:+.3f} [{lo:+.3f}, {hi:+.3f}] {verdict}",
                ("주 비교, " if primary else "사전 등록한 주 비교가 아님, ") + f"같은 리샘플의 짝지은 차이 (질문 {n}개, B={b})")

    d_found = [q for q in d_q if relevant[q]]
    add("D 유형 중 정답이 발견된 질문", f"{len(d_found)} / {len(d_q)}", ", ".join(d_found) or "없음")
    for m in SCORE_METHODS:
        if m not in runs:
            continue
        xs = sorted(top1[m][q] for q in d_q if q in top1[m])
        value = f"중앙값 {_fmt(percentile(xs, 0.5))} (최소 {_fmt(xs[0])}, 최대 {_fmt(xs[-1])})" if xs else "결과 없음"
        add(f"{m} · D 유형 상위 1 점수", value, f"D 질문 {len(d_q)}개 중 결과가 있는 {len(xs)}개. 임계값 거절은 범위 밖")

    high = [f"{m} {metric}" for (m, metric), v in point.items() if v >= SUSPICIOUS]
    if high:
        print(f"[경고] 지표가 {SUSPICIOUS} 이상: {high} — 질문 누수·판정 편향을 먼저 확인할 것 (CLAUDE.md ⑤)")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--judgments", default=str(JUDGMENTS_PATH))
    ap.add_argument("--rejudgments", default=str(REJUDGMENTS_PATH))
    ap.add_argument("--b", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROCESSED_DIR / "rag_comparison.csv"))
    a = ap.parse_args(argv)

    if not Path(a.judgments).exists():
        raise SystemExit(f"[중단] 판정 파일 없음: {a.judgments}")
    rows = evaluate(load_queries(a.queries), load_runs(a.runs_dir), load_judgments(a.judgments),
                    load_judgments(a.rejudgments), a.b, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["항목", "값", "근거"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['항목']:<40}{r['값']}")
    print(f"저장 -> {out}")


if __name__ == "__main__":
    main()
