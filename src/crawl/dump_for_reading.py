"""
수집한 공고를 '눈으로 훑기 좋은' 형태로 뽑아낸다.
1주차 2단계(역량 태그 사전 v0)는 이 출력물을 읽으면서 손으로 만든다.

사용:
  python -m src.crawl.dump_for_reading data/raw/wanted_2026-07-24.jsonl -n 30
  python -m src.crawl.dump_for_reading data/raw/*.jsonl --sections requirements,preferred_points
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SECTION_LABELS = {
    "intro": "회사소개",
    "main_tasks": "주요업무",
    "requirements": "자격요건",
    "preferred_points": "우대사항",
    "benefits": "혜택및복지",
}


def load(paths: list[str]) -> list[dict]:
    recs: list[dict] = []
    for p in paths:
        # 파일 객체를 직접 순회한다. splitlines()는 \u2028 등 유니코드 줄바꿈까지
        # 잘라내서 한 줄짜리 JSON 레코드를 두 동강 내므로 쓰면 안 된다.
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    # 같은 job_id 스냅샷이 여러 개면 최신 1건만
    latest: dict[str, dict] = {}
    for r in recs:
        key = f"{r['source']}:{r['job_id']}"
        if key not in latest or r["collected_at"] > latest[key]["collected_at"]:
            latest[key] = r
    return list(latest.values())


def summarize(recs: list[dict]) -> str:
    """읽기 전에 표본이 어디로 치우쳤는지부터 본다."""
    lines = [f"# 수집 요약 (고유 공고 {len(recs)}건)", ""]

    def dist(name: str, counter: Counter, top: int = 12) -> None:
        lines.append(f"## {name}")
        for k, v in counter.most_common(top):
            lines.append(f"- {k}: {v}")
        lines.append("")

    ann = Counter()
    for r in recs:
        lo, hi = r.get("annual_from"), r.get("annual_to")
        ann[f"{lo}~{hi}"] += 1
    dist("경력 조건 (annual_from~annual_to)", ann)
    dist("회사", Counter(r.get("company", "") for r in recs))
    dist("원티드 skill_tags", Counter(t for r in recs for t in (r.get("skill_tags") or [])), top=40)

    lens = sorted(len(r.get("body_raw", "")) for r in recs)
    if lens:
        mid = lens[len(lens) // 2]
        lines += [f"## 본문 길이(자): min {lens[0]} / 중앙값 {mid} / max {lens[-1]}", ""]
    return "\n".join(lines)


def render(recs: list[dict], sections: list[str]) -> str:
    out: list[str] = []
    for i, r in enumerate(recs, 1):
        out.append(f"\n\n{'='*78}\n[{i}] {r.get('title','')} — {r.get('company','')}")
        out.append(f"경력: {r.get('annual_from')}~{r.get('annual_to')} | "
                   f"원티드태그: {', '.join(r.get('skill_tags') or []) or '없음'}")
        out.append(f"{r.get('url','')}")
        for key in sections:
            text = (r.get("sections") or {}).get(key, "").strip()
            if text:
                out.append(f"\n--- {SECTION_LABELS.get(key, key)} ---\n{text}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("-n", type=int, default=30, help="훑어볼 공고 수")
    p.add_argument("--seed", type=int, default=42, help="같은 표본을 다시 보려면 고정")
    p.add_argument("--sections", default="main_tasks,requirements,preferred_points",
                   type=lambda s: [x.strip() for x in s.split(",")])
    p.add_argument("-o", "--out", default="notes/sample_read.md")
    args = p.parse_args()

    recs = load(args.paths)
    random.Random(args.seed).shuffle(recs)
    sample = recs[: args.n]

    body = summarize(recs) + "\n" + render(sample, args.sections)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print(f"{len(sample)}건 -> {out_path}")


if __name__ == "__main__":
    main()
