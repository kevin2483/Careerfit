"""
라벨링 뷰어 — 공고를 한 건씩 읽기 좋게 출력한다.
JSONL 을 눈으로 보는 고통을 없애기 위한 도구. 판단만 하도록 돕는다.

사용:
  python -m src.label.view_labels              # 전체 순서대로
  python -m src.label.view_labels --only-flagged   # 확인필요(flag)만
  python -m src.label.view_labels --n 1         # 1번 공고 하나만
  python -m src.label.view_labels --start 5 --count 3   # 5번부터 3건
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

SECTION_LABELS = {
    "회사소개": "🏢 회사소개",
    "주요업무": "📋 주요업무",
    "자격요건": "✅ 자격요건",
    "우대사항": "⭐ 우대사항",
    "혜택및복지": "🎁 혜택및복지",
}


def split_sections(body: str) -> list[tuple[str, str]]:
    """body_raw 의 [섹션명] 마커로 나눠서 (라벨, 내용) 리스트를 만든다."""
    parts = re.split(r"\[([^\]]+)\]", body)
    # parts = ['', '회사소개', '내용...', '주요업무', '내용...', ...]
    out = []
    i = 1
    while i < len(parts):
        name = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        out.append((SECTION_LABELS.get(name, f"◾ {name}"), content))
        i += 2
    return out


def bulletify(text: str) -> str:
    """줄바꿈·중점(•)을 살려 읽기 좋게 들여쓴다."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\n".join("   " + ln for ln in lines)


def show_one(idx: int, total: int, row: dict, show_body: bool = True) -> None:
    print("\n" + "═" * 72)
    print(f"[{idx}/{total}]  {row.get('title','')}   ({row.get('직무','')})")
    if len(row.get("all_jobs", [])) > 1:
        print(f"   ※ 복수직무 공고: {row['all_jobs']}")
    print(f"   job_id: {row.get('job_id')}")
    print("─" * 72)
    print(f"초벌 태그: {', '.join(row.get('gold_tags', [])) or '(없음)'}")

    flags = row.get("review_flags", [])
    if flags:
        print("⚠️  확인필요:")
        for f in flags:
            print(f"     - {f}")
    else:
        print("   (확인 플래그 없음 — regex 결과 그대로 검토)")

    if show_body:
        print()
        for label, content in split_sections(row.get("body_raw", "")):
            # 회사소개는 길고 역량과 무관한 경우 많음 → 앞부분만
            if "회사소개" in label:
                content = content[:200] + (" …(생략)" if len(content) > 200 else "")
            print(label)
            print(bulletify(content))
            print()
    print("═" * 72)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/labeled/label_draft.jsonl")
    p.add_argument("--only-flagged", action="store_true", help="확인필요 건만 보기")
    p.add_argument("--n", type=int, help="특정 번호 하나만 (1부터)")
    p.add_argument("--start", type=int, default=1, help="시작 번호")
    p.add_argument("--count", type=int, help="몇 건 볼지")
    p.add_argument("--no-body", action="store_true", help="본문 빼고 태그만")
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.infile).read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(rows)

    if args.n:
        show_one(args.n, total, rows[args.n - 1], show_body=not args.no_body)
        return

    subset = list(enumerate(rows, 1))
    if args.only_flagged:
        subset = [(i, r) for i, r in subset if r.get("review_flags")]
        print(f"※ 확인필요 {len(subset)}건만 표시합니다.")

    start = args.start
    subset = [x for x in subset if x[0] >= start]
    if args.count:
        subset = subset[: args.count]

    for i, r in subset:
        show_one(i, total, r, show_body=not args.no_body)

    done = sum(1 for r in rows if r.get("reviewed"))
    print(f"\n진행: {done}/{total} 검수 완료 (reviewed=true)")


if __name__ == "__main__":
    main()
