"""
캐스케이드 라우팅 시뮬레이션 — API 를 새로 부르지 않는다.

이미 있는 regex 예측과 opus 예측을 섞어서 "어떤 공고를 opus 로 보낼까" 라우팅 정책별
정확도·비용을 계산한다.

  regex 만        : 전부 regex. 비용 0
  opus 만         : 전부 opus
  조합(전부)       : ①명시 기술은 regex, 그 외는 opus. 비용은 opus 와 같다
  라우팅(k%)       : '어려워 보이는' 상위 k% 공고만 조합, 나머지는 regex. 비용 = k% × opus 건당 비용

라우팅 기준
  length  : 공고 본문 길이 (regex 예측 파일의 _meta.n_chars)
  f2count : regex 가 찾은 ②서술형(field 2) 태그 수
  random  : 무작위 k% (시드 N회 평균·표준편차) — 위 두 기준이 '무작위보다 나은가'의 베이스라인

주의
  정답셋이 38건이다. 여기서 k 를 골라 "최적 운영점"이라 부르면 과적합이다.
  곡선의 모양(어디서 꺾이는가)과 random 대비 차이를 보는 용도로 쓸 것.

입력은 전부 레포에 커밋된 파일이라 공개 데이터만으로 재현된다:
  data/labeled/label_draft.public.jsonl, data/processed/<date>/{regex_extract,llm_extract_claude-opus-5}.jsonl

    python -m src.eval.cascade_sim --date 2026-07-30
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from src.common.jsonl import read_jsonl
from src.common.paths import DICT_PATH, PROCESSED_DIR, labels_path, resolve_date
from src.common.skill_dict import close_tags, field_map, implies_map


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


def build_docs(labels: str | Path, regex_pred: str | Path, opus_pred: str | Path, fm: dict,
               implies: dict[str, list[str]] | None = None) -> list[dict]:
    rows = [r for r in read_jsonl(labels) if r.get("reviewed")]
    regex = {str(r["job_id"]): r for r in read_jsonl(regex_pred)}
    opus = {str(r["job_id"]): close_tags(r.get("tags") or [], implies) for r in read_jsonl(opus_pred)}
    for name, m in (("regex", regex), ("opus", opus)):
        missing = [str(r["job_id"]) for r in rows if str(r["job_id"]) not in m]
        if missing:
            raise SystemExit(f"[중단] {name} 예측 없는 정답 {len(missing)}건: {missing[:5]}")
    docs = []
    for r in rows:
        jid = str(r["job_id"])
        rx = close_tags(regex[jid].get("tags") or [], implies)
        op = opus[jid]
        length = regex[jid].get("_meta", {}).get("n_chars") or len(r.get("body_raw") or "")
        docs.append({
            "jid": jid,
            "gold": close_tags(r.get("gold_tags") or [], implies),
            "regex": rx, "opus": op,
            "combo": {t for t in rx if fm.get(t) == 1} | {t for t in op if fm.get(t) != 1},
            "length": length,
            "f2count": sum(1 for t in rx if fm.get(t) == 2),
        })
    return docs


def simulate(docs: list[dict], fm: dict, opus_cpd: float, n_docs: int,
             random_trials: int, seed: int) -> list[dict]:
    n = len(docs)
    results: list[dict] = []

    def add(name: str, pairs, frac: float, **extra):
        s = score(pairs, fm)
        s.update(정책=name, 비율=frac, 건당비용=opus_cpd * frac, 전체비용=opus_cpd * frac * n_docs, **extra)
        results.append(s)

    add("regex 만", [(d["gold"], d["regex"]) for d in docs], 0.0)
    add("opus 만", [(d["gold"], d["opus"]) for d in docs], 1.0)
    add("조합(전부)", [(d["gold"], d["combo"]) for d in docs], 1.0)

    def routed_pairs(routed: set[str]):
        return [(d["gold"], d["combo"] if d["jid"] in routed else d["regex"]) for d in docs]

    for key in ("length", "f2count"):
        order = sorted(docs, key=lambda d: (-d[key], d["jid"]))  # 동률은 jid 로 고정 → 결정적
        for pct in range(10, 100, 10):
            k = round(n * pct / 100)
            add(f"라우팅·{key} {pct}%", routed_pairs({d["jid"] for d in order[:k]}), k / n)

    # 무작위 라우팅 베이스라인: 같은 k 로 random_trials 회 평균
    rng = random.Random(seed)
    jids = [d["jid"] for d in docs]
    for pct in range(10, 100, 10):
        k = round(n * pct / 100)
        f1s = [score(routed_pairs(set(rng.sample(jids, k))), fm)["F1"] for _ in range(random_trials)]
        mean = statistics.fmean(f1s)
        sd = statistics.pstdev(f1s) if len(f1s) > 1 else 0.0
        results.append({
            "정책": f"라우팅·random {pct}%", "비율": k / n, "F1": mean, "P": None, "R": None,
            "TP": None, "FP": None, "FN": None,
            "건당비용": opus_cpd * k / n, "전체비용": opus_cpd * k / n * n_docs,
            "F1_sd": sd, "trials": random_trials,
        })
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None)
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--dict", dest="dict_path", default=str(DICT_PATH))
    ap.add_argument("--regex-pred", default=None, help="기본: <processed>/<date>/regex_extract.jsonl")
    ap.add_argument("--opus-pred", default=None, help="기본: <processed>/<date>/llm_extract_claude-opus-5.jsonl")
    ap.add_argument("--opus-summary", default=None)
    ap.add_argument("--n-docs", type=int, default=892, help="전체 코퍼스 고유 공고 수 (비용 환산용)")
    ap.add_argument("--random-trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="기본: <processed>/cascade_sim.csv")
    a = ap.parse_args(argv)

    date = resolve_date(a.date, processed_dir=a.processed_dir)
    pdir = Path(a.processed_dir)
    regex_pred = a.regex_pred or pdir / date / "regex_extract.jsonl"
    opus_pred = a.opus_pred or pdir / date / "llm_extract_claude-opus-5.jsonl"
    opus_sum = a.opus_summary or pdir / date / "llm_extract_claude-opus-5_summary.json"
    out = Path(a.out) if a.out else pdir / "cascade_sim.csv"

    fm = field_map(a.dict_path)
    docs = build_docs(labels_path(a.labels), regex_pred, opus_pred, fm, implies=implies_map(a.dict_path))
    opus_cpd = json.loads(Path(opus_sum).read_text(encoding="utf-8"))["cost_per_doc_usd"]
    results = simulate(docs, fm, opus_cpd, a.n_docs, a.random_trials, a.seed)
    n = len(docs)

    base = results[0]["F1"]
    print("=" * 84)
    print(f"캐스케이드 시뮬레이션 — 정답셋 {n}건 · opus ${opus_cpd:.4f}/건 · 전체 {a.n_docs}건 환산")
    print("=" * 84)
    print(f"{'정책':<24}{'F1':>7}{'Δregex':>8}{'P':>7}{'R':>7}{'opus비율':>9}{'전체비용':>10}")
    for s in results:
        p = f"{s['P']:>7.3f}" if s["P"] is not None else f"{'±' + format(s['F1_sd'], '.3f'):>7}"
        r = f"{s['R']:>7.3f}" if s["R"] is not None else " " * 7
        print(f"{s['정책']:<24}{s['F1']:>7.3f}{s['F1'] - base:>+8.3f}{p}{r}"
              f"{s['비율']:>8.0%}{'$' + format(s['전체비용'], '.2f'):>10}")

    print(f"\n{'─' * 84}\n길이 기준 vs 무작위 (같은 비율에서 F1 차이)")
    rand = {s["정책"].split()[-1]: s for s in results if s["정책"].startswith("라우팅·random")}
    for s in results:
        if s["정책"].startswith("라우팅·length"):
            pct = s["정책"].split()[-1]
            r = rand[pct]
            z = (s["F1"] - r["F1"]) / r["F1_sd"] if r["F1_sd"] else float("nan")
            print(f"  {pct:>4}  length {s['F1']:.3f}  random {r['F1']:.3f}±{r['F1_sd']:.3f}  Δ {s['F1'] - r['F1']:+.3f}  ({z:+.1f}σ)")

    print(f"\n{'─' * 84}\n파레토 최적 정책 (random 제외)")
    det = [s for s in results if s["P"] is not None]
    for s in sorted(det, key=lambda x: x["전체비용"]):
        dominated = any(
            o is not s and o["F1"] >= s["F1"] and o["전체비용"] <= s["전체비용"]
            and (o["F1"] > s["F1"] or o["전체비용"] < s["전체비용"])
            for o in det
        )
        if not dominated:
            print(f"  {s['정책']:<24} F1 {s['F1']:.3f}  ${s['전체비용']:.2f}")
    print(f"\n※ 정답셋 {n}건 기준. 10% 구간 = 공고 약 {round(n / 10)}개. 비율 k 를 골라 운영점으로 확정하면 과적합이다.")

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["정책", "비율", "F1", "P", "R", "TP", "FP", "FN", "건당비용", "전체비용", "F1_sd", "trials"]
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in results:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else ("" if v is None else v))
                        for k, v in s.items() if k in cols})
    print(f"저장 -> {out}")


if __name__ == "__main__":
    main()
