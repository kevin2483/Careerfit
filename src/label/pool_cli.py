"""
블라인드 풀링 재검수 — 정답셋 편향 제거

위치: src/label/pool_cli.py

왜 필요한가
-----------
기존 label_cli.py 는 비대칭이었다.
  regex 가 뽑은 태그 -> 화면에 '현재 태그'로 제시하고 y/n 으로 확인
  regex 가 놓친 태그 -> 사람이 기억해내서 손으로 타이핑
그래서 regex 가 못 본 것은 정답셋에도 못 들어갔다. 실측 증거:
  da/282256 자격요건 "SQL과 Python을 활용해 ..." -> gold 에 SQL/Python 없음

이 도구는 그 비대칭을 없앤다.
  - regex ∪ LLM 후보를 한 통에 넣고 (pooling)
  - 출처를 감추고 (blind)
  - 문서 안에서 순서를 섞어 (shuffle)
  - 전부 같은 방식으로 y/n 을 묻는다

한계도 분명히 한다: 두 시스템이 모두 못 찾은 태그는 후보에 오르지 않는다.
그래서 마지막에 '둘 다 놓친 것' 자유 입력을 두되, 그것이 남은 비대칭임을 안다.

사용
----
    python -m src.label.pool_cli                 # 미판정부터 이어서
    python -m src.label.pool_cli --doc 355021    # 특정 공고만
    python -m src.label.pool_cli --redo          # 판정한 것도 다시
    python -m src.label.pool_cli --stats         # 진행률만
    python -m src.label.pool_cli --apply         # 판정 결과를 gold_tags 에 반영

조작
----
    y / 엔터 = 요구 역량이 맞다      n = 아니다
    s        = 이 공고 통째로 건너뜀   q = 저장하고 종료

pool_key.jsonl(어느 시스템이 냈는지)은 --apply 전까지 열지 말 것. 블라인드가 깨진다.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from src.label.view_labels import bulletify, split_sections

POOL = "data/labeled/pool_candidates.jsonl"
LABELS = "data/labeled/label_draft.jsonl"
SEED = 42


def read_jsonl(path: str | Path) -> list[dict]:
    # splitlines() 쓰지 말 것 — 본문의 \u2028 등에서 레코드가 절단된다 (lab_log 기록됨)
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def ask(prompt: str, default: str = "y") -> str:
    try:
        a = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"
    return a or default


def show_doc(row: dict, n_cand: int) -> None:
    print("\n" + "=" * 72)
    print(f"{row.get('title', '')}   ({row.get('직무', '')})   job_id={row.get('job_id')}")
    print("=" * 72)
    for label, content in split_sections(row.get("body_raw", "")):
        if "회사소개" in label:
            continue  # 요구 역량과 무관한 경우가 대부분
        print(f"\n{label}")
        print(bulletify(content))
    print("\n" + "-" * 72)
    print(f"판정할 후보 {n_cand}개. 이 공고가 '지원자에게 요구'하는 역량인지만 보세요.")
    print("-" * 72)


def review(args: argparse.Namespace) -> None:
    cands = read_jsonl(args.pool)
    labels = {str(r["job_id"]): r for r in read_jsonl(args.labels) if r.get("reviewed")}

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_doc[str(c["job_id"])].append(c)

    # 문서 안 순서를 고정 시드로 섞는다 -> 출처 순서가 힌트가 되지 않고, 재개해도 순서가 같다
    for jid, group in by_doc.items():
        random.Random(f"{SEED}:{jid}").shuffle(group)

    doc_ids = sorted(by_doc)
    if args.doc:
        doc_ids = [d for d in doc_ids if d == str(args.doc)]
        if not doc_ids:
            raise SystemExit(f"[중단] job_id={args.doc} 후보 없음")

    todo = [d for d in doc_ids
            if args.redo or any(c["verdict"] is None for c in by_doc[d])]
    if not todo:
        print("판정할 후보가 없습니다. (--redo 로 다시 볼 수 있습니다)")
        return

    total_left = sum(1 for d in todo for c in by_doc[d]
                     if args.redo or c["verdict"] is None)
    print(f"미판정 {total_left}건 / 공고 {len(todo)}건. (q 입력 시 저장 후 종료)")

    quit_now = False
    for k, jid in enumerate(todo, 1):
        if quit_now:
            break
        group = [c for c in by_doc[jid] if args.redo or c["verdict"] is None]
        if not group:
            continue
        row = labels.get(jid)
        if row is None:
            print(f"[경고] job_id={jid} 가 정답셋에 없음. 건너뜀")
            continue

        print(f"\n\n########## 공고 {k}/{len(todo)} ##########")
        show_doc(row, len(group))

        for i, c in enumerate(group, 1):
            a = ask(f"  [{i}/{len(group)}] {c['tag']}  요구 역량? [Y/n/s=공고건너뜀/q]: ")
            if a == "q":
                quit_now = True
                break
            if a == "s":
                print("  → 이 공고 건너뜀")
                break
            c["verdict"] = (a != "n")
            print("     → " + ("유지" if c["verdict"] else "제외"))

        if not quit_now:
            kept = sorted(c["tag"] for c in by_doc[jid] if c["verdict"])
            print(f"\n  확정 {len(kept)}개: {', '.join(kept) or '(없음)'}")
            print("  둘 다 놓친 역량이 본문에 있나요? (쉼표 구분, 없으면 엔터)")
            a = ask("  추가: ", default="")
            if a == "q":
                quit_now = True
            elif a:
                for t in [x.strip() for x in a.split(",") if x.strip()]:
                    cands.append({"job_id": jid, "tag": t, "verdict": True, "source": "manual"})
                    print(f"     → 추가: {t}")

        write_jsonl(args.pool, cands)  # 공고 단위 저장 — 중간에 끊겨도 안전

    write_jsonl(args.pool, cands)
    done = sum(1 for c in cands if c["verdict"] is not None)
    print(f"\n{'=' * 72}")
    print(f"진행: {done}/{len(cands)} 판정 완료 -> {args.pool}")


def stats(args: argparse.Namespace) -> None:
    cands = read_jsonl(args.pool)
    done = [c for c in cands if c["verdict"] is not None]
    docs = {str(c["job_id"]) for c in cands}
    docs_done = {str(c["job_id"]) for c in cands} - {
        str(c["job_id"]) for c in cands if c["verdict"] is None
    }
    print(f"후보 {len(cands)}건 / 판정 {len(done)}건 ({len(done) / max(len(cands), 1) * 100:.1f}%)")
    print(f"공고 {len(docs)}건 / 완료 {len(docs_done)}건")
    if done:
        y = sum(1 for c in done if c["verdict"])
        print(f"판정 내역: 유지 {y} / 제외 {len(done) - y}")
        print(f"수기 추가: {sum(1 for c in cands if c.get('source') == 'manual')}건")


def apply(args: argparse.Namespace) -> None:
    cands = read_jsonl(args.pool)
    pending = [c for c in cands if c["verdict"] is None]
    if pending and not args.force:
        raise SystemExit(
            f"[중단] 미판정 {len(pending)}건이 남아 있다. "
            f"전부 끝낸 뒤 실행하거나 --force 를 쓸 것."
        )

    new_gold: dict[str, set[str]] = defaultdict(set)
    for c in cands:
        if c["verdict"]:
            new_gold[str(c["job_id"])].add(c["tag"])

    path = Path(args.labels)
    shutil.copy(path, path.with_name(path.name + ".bak_prepool"))
    rows = read_jsonl(path)

    n_doc = added = removed = 0
    for r in rows:
        jid = str(r["job_id"])
        if not r.get("reviewed") or jid not in new_gold:
            continue
        old, new = set(r.get("gold_tags") or []), new_gold[jid]
        added += len(new - old)
        removed += len(old - new)
        if old != new:
            n_doc += 1
        r["gold_tags_prepool"] = sorted(old)   # 이전 라벨을 남긴다 — 되돌릴 수 있어야 한다
        r["gold_tags"] = sorted(new)
        r["pooled"] = True
    write_jsonl(path, rows)

    print(f"반영 완료 -> {path}  (백업: {path.name}.bak_prepool)")
    print(f"바뀐 공고 {n_doc}건 / 태그 추가 {added} · 제거 {removed}")
    print("\n이제 같은 정답셋으로 regex 와 LLM 을 다시 채점하라.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default=POOL)
    p.add_argument("--labels", default=LABELS)
    p.add_argument("--doc", help="특정 job_id 만")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--force", action="store_true", help="--apply 시 미판정 무시")
    a = p.parse_args()

    if a.stats:
        stats(a)
    elif a.apply:
        apply(a)
    else:
        review(a)


if __name__ == "__main__":
    main()
