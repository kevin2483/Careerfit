"""
대화형 라벨링 도구 — 묻고, 답하면, 저장한다.

사람이 하는 일: 본문을 읽고 y/n 로 판단. (판단은 사람 몫 — 순환논리 회피)
도구가 하는 일: 화면 출력, 입력 받기, 파일 저장.

사용:
  python -m src.label.label_cli                 # 검수 안 한 것부터 이어서
  python -m src.label.label_cli --n 1           # 1번만
  python -m src.label.label_cli --redo          # 이미 검수한 것도 다시

조작:
  y / 엔터  = 유지        n = 제외        s = 이 공고 통째로 건너뛰기
  q         = 저장하고 종료
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.label.view_labels import split_sections, bulletify


def ask(prompt: str, default: str = "y") -> str:
    try:
        a = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"
    return a or default


def show_body(row: dict) -> None:
    for label, content in split_sections(row.get("body_raw", "")):
        if "회사소개" in label:
            continue  # 역량과 무관한 경우가 대부분 → 생략
        print(f"\n{label}")
        print(bulletify(content))


def review_one(idx: int, total: int, row: dict) -> str:
    print("\n" + "═" * 70)
    print(f"[{idx}/{total}]  {row.get('title','')}   ({row.get('직무','')})")
    print("═" * 70)
    show_body(row)

    print("\n" + "─" * 70)
    print(f"현재 태그: {', '.join(row['gold_tags']) or '(없음)'}")
    print("─" * 70)

    # 1) 확인필요 태그를 하나씩 물어본다
    flagged_tags = []
    for f in row.get("review_flags", []):
        if f.startswith("근거확인:"):
            tag = f.split("근거확인:")[1].split(" (")[0].strip()
            flagged_tags.append(tag)

    keep = list(row["gold_tags"])
    for tag in flagged_tags:
        if tag not in keep:
            continue
        a = ask(f"  ⚠️  '{tag}' — 본문에 근거 있나요? [Y=유지 / n=제외 / q=종료]: ")
        if a == "q":
            return "quit"
        if a == "n":
            keep.remove(tag)
            print(f"     → 제외됨")
        else:
            print(f"     → 유지")

    # 2) regex 가 놓친 태그 추가
    print("\n  regex가 놓친 역량이 본문에 있나요?")
    print("  (있으면 쉼표로 입력, 없으면 그냥 엔터)")
    a = ask("  추가할 태그: ", default="")
    if a == "q":
        return "quit"
    if a:
        added = [t.strip() for t in a.split(",") if t.strip()]
        keep.extend(added)
        print(f"     → 추가됨: {', '.join(added)}")

    # 3) 사전에 없는 새 역량 (v1 후보)
    print("\n  사전에 없는 새 역량이 있나요? (v1 후보로 기록)")
    a = ask("  새 역량: ", default="")
    if a == "q":
        return "quit"
    if a:
        cands = [t.strip() for t in a.split(",") if t.strip()]
        row["new_tag_candidates"] = cands
        print(f"     → 기록됨: {', '.join(cands)}")

    row["gold_tags"] = sorted(set(keep))
    row["reviewed"] = True
    print(f"\n  ✅ 확정: {', '.join(row['gold_tags']) or '(없음)'}")
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/labeled/label_draft.jsonl")
    p.add_argument("--n", type=int, help="특정 번호만")
    p.add_argument("--redo", action="store_true", help="검수한 것도 다시")
    args = p.parse_args()

    path = Path(args.infile)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(rows)

    if args.n:
        targets = [(args.n, rows[args.n - 1])]
    else:
        targets = [(i, r) for i, r in enumerate(rows, 1)
                   if args.redo or not r.get("reviewed")]

    if not targets:
        print("검수할 항목이 없습니다. (--redo 로 다시 볼 수 있어요)")
        return

    print(f"검수 대상 {len(targets)}건. (q 입력하면 저장하고 종료)")

    done = 0
    for i, row in targets:
        res = review_one(i, total, row)
        if res == "quit":
            break
        done += 1
        # 매 건마다 저장 (중간에 끊겨도 안전)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    reviewed = sum(1 for r in rows if r.get("reviewed"))
    print(f"\n{'='*70}")
    print(f"이번 세션 {done}건 검수. 전체 진행: {reviewed}/{total}")
    print(f"저장됨 -> {path}")


if __name__ == "__main__":
    main()
