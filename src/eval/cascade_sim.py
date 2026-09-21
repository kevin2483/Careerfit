"""
CareerFit — 캐스케이드 라우팅 시뮬레이션 (계획서 5번)

위치: src/eval/cascade_sim.py

무엇을 하나
-----------
API 를 새로 부르지 않는다. 이미 있는 regex 예측과 opus 예측을 섞어서
"어떤 공고를 opus 로 보낼까" 라우팅 정책별 정확도·비용을 계산한다.

  regex 만        : 전부 regex. 비용 0
  opus 만         : 전부 opus
  조합(전부)       : ①명시 기술은 regex, 그 외는 opus. 비용은 opus 와 같다
  라우팅(k%)       : '어려워 보이는' 상위 k% 공고만 조합, 나머지는 regex.
                    비용 = k% × opus 건당 비용

라우팅 기준 두 가지
  length  : 공고 본문 길이
  f2count : regex 가 찾은 ②서술형(field 2) 태그 수

주의
----
정답셋이 38건이다. 여기서 k 를 골라 "최적 운영점"이라 부르면 과적합이다.
곡선의 모양(어디서 꺾이는가)을 보는 용도로 쓸 것.

    python -m src.eval.cascade_sim
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from src.extract.job_skill_extract_report import compile_patterns, load_dict

DICT = "data/raw/dict/skill_tags_v1.yaml"
LABELS = "data/labeled/label_draft.jsonl"
OPUS_PRED = "data/processed/2026-07-30/llm_extract_claude-opus-5.jsonl"
OPUS_SUM = "data/processed/2026-07-30/llm_extract_claude-opus-5_summary.json"


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def field_map(path: str) -> dict[str, int]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {e["canonical"]: e.get("field")
            for es in raw.values() if isinstance(es, list) for e in es}


def score(pairs: list[tuple[set, set]], fm: dict) -> dict:
    """evaluate_extraction.py 와 같은 규칙: field 3 은 양쪽에서 제외, 사전 밖 태그는 포함."""
    tp = fp = fn = 0
    for gold, pred in pairs:
        g = {t for t in gold if fm.get(t) != 3}
        p = {t for t in pred if fm.get(t) != 3}
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"P": P, "R": R, "F1": F, "TP": tp, "FP": fp, "FN": fn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=892, help="전체 코퍼스 고유 공고 수 (비용 환산용)")
    ap.add_argument("--out", default="data/processed/cascade_sim.csv")
    a = ap.parse_args()

    fm = field_map(DICT)
    patterns = compile_patterns(load_dict(Path(DICT)))
    rows = [r for r in read_jsonl(LABELS) if r.get("reviewed")]

    opus = {str(r["job_id"]): set(r.get("tags") or []) for r in read_jsonl(OPUS_PRED)}
    opus_cpd = json.loads(Path(OPUS_SUM).read_text(encoding="utf-8"))["cost_per_doc_usd"]

    missing = [str(r["job_id"]) for r in rows if str(r["job_id"]) not in opus]
    if missing:
        raise SystemExit(f"[중단] opus 예측 없는 정답 {len(missing)}건: {missing[:5]}")

    docs = []
    for r in rows:
        jid = str(r["job_id"])
        body = r.get("body_raw") or ""
        rx = {c for c, _cat, _f, pat in patterns if pat.search(body)}
        op = opus[jid]
        combo = {t for t in rx if fm.get(t) == 1} | {t for t in op if fm.get(t) != 1}
        docs.append({
            "jid": jid,
            "gold": set(r.get("gold_tags") or []),
            "regex": rx, "opus": op, "combo": combo,
            "length": len(body),
            "f2count": sum(1 for t in rx if fm.get(t) == 2),
        })
    n = len(docs)

    results = []

    def add(name: str, pairs, frac: float):
        s = score(pairs, fm)
        s.update(정책=name, 비율=frac, 건당비용=opus_cpd * frac,
                 전체비용=opus_cpd * frac * a.n_docs)
        results.append(s)

    add("regex 만", [(d["gold"], d["regex"]) for d in docs], 0.0)
    add("opus 만", [(d["gold"], d["opus"]) for d in docs], 1.0)
    add("조합(전부)", [(d["gold"], d["combo"]) for d in docs], 1.0)

    for key in ("length", "f2count"):
        order = sorted(docs, key=lambda d: d[key], reverse=True)
        for pct in range(10, 100, 10):
            k = round(n * pct / 100)
            routed = {d["jid"] for d in order[:k]}
            pairs = [(d["gold"], d["combo"] if d["jid"] in routed else d["regex"]) for d in docs]
            add(f"라우팅·{key} {pct}%", pairs, k / n)

    base = results[0]["F1"]
    print("=" * 80)
    print(f"캐스케이드 시뮬레이션 — 정답셋 {n}건 · opus ${opus_cpd:.4f}/건 · 전체 {a.n_docs}건 환산")
    print("=" * 80)
    print(f"{'정책':<22}{'F1':>7}{'Δregex':>8}{'P':>7}{'R':>7}{'opus비율':>9}{'전체비용':>10}")
    for s in results:
        print(f"{s['정책']:<22}{s['F1']:>7.3f}{s['F1'] - base:>+8.3f}{s['P']:>7.3f}{s['R']:>7.3f}"
              f"{s['비율']:>8.0%}{'$' + format(s['전체비용'], '.2f'):>10}")

    # 파레토: 더 싸면서(<=) 더 정확한(>=) 정책이 있으면 지배당함
    print(f"\n{'─' * 80}\n파레토 최적 정책")
    for s in sorted(results, key=lambda x: x["전체비용"]):
        dominated = any(
            o is not s and o["F1"] >= s["F1"] and o["전체비용"] <= s["전체비용"]
            and (o["F1"] > s["F1"] or o["전체비용"] < s["전체비용"])
            for o in results
        )
        if not dominated:
            print(f"  {s['정책']:<22} F1 {s['F1']:.3f}  ${s['전체비용']:.2f}")

    print(f"\n※ 정답셋 {n}건 기준. 여기서 비율 k 를 골라 운영점으로 확정하면 과적합이다.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["정책", "비율", "F1", "P", "R", "TP", "FP", "FN", "건당비용", "전체비용"])
        w.writeheader()
        for s in results:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    print(f"저장 -> {out}")


if __name__ == "__main__":
    main()
