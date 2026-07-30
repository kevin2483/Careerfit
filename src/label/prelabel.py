"""
초벌 라벨링 생성기 (판단 기준 4를 규칙으로 적용)

무엇을 하나:
  label_template.jsonl 의 각 공고에 대해, regex_prefilled 를 기준으로
  '초벌 gold_tags'를 만들되, 판단이 갈리는 태그는 review_flags 로 표시한다.
  사람은 review_flags 가 붙은 것만 집중 확인하면 된다 (백지 라벨링보다 빠름).

판단 기준(면접 방어 문서와 일치):
  ① 명시 기술/도구 → 유지
  ② 의미 명확한 서술형 → 유지
  ③ 근거 약한 추측 → drop 후보로 flag (사람이 최종 결정)

이 도구는 '초벌'이다. 최종 확정은 사람이 review_flags 를 보고 gold_tags 를 손본 뒤
reviewed=true 로 바꾸는 것으로 완료된다. (순환논리 회피: 기준은 사람, 확정도 사람)

사용:
  python -m src.label.prelabel
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# 판단 3(약한 추측)로 자주 오탐되는 태그 -> 이 태그가 붙으면 "근거 확인 요망" flag.
# regex 가 단일·범용 단어로 잡기 쉬워 문맥 확인이 필요한 태그들.
WEAK_MATCH_TAGS = {
    "ABテスト",        # "실험" 한 단어로 오탐되기 쉬움
    "통계분석",         # "분석" 범용어
    "대시보드",         # "리포트" 범용어
    "협업커뮤니케이션",  # field 3, 어차피 평가 제외지만 표시
}


def prelabel_row(row: dict) -> dict:
    prefilled = row.get("regex_prefilled", [])
    draft = list(prefilled)  # 기본: regex 결과를 초벌 정답으로
    flags: list[str] = []

    for tag in prefilled:
        if tag in WEAK_MATCH_TAGS:
            flags.append(f"근거확인:{tag} (본문에 명확한 근거 없으면 제외)")

    # regex 가 0개면 사전 밖 표현 가능성 -> 사람이 본문에서 직접 태그 발굴 필요
    if not prefilled:
        flags.append("regex 0개: 본문 읽고 직접 태그 발굴 (사전 v1 후보 가능성)")

    row["gold_tags"] = draft
    row["review_flags"] = flags
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", default="data/labeled/label_template.jsonl")
    p.add_argument("--out", default="data/labeled/label_draft.jsonl")
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.infile).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        prelabel_row(r)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_flagged = sum(1 for r in rows if r["review_flags"])
    n_zero = sum(1 for r in rows if not r["regex_prefilled"])
    print(f"초벌 라벨링 생성: {len(rows)}건 -> {out}")
    print(f"집중 확인 필요(review_flags 있음): {n_flagged}건")
    print(f"  - 그중 regex 0개 공고: {n_zero}건")
    print(f"나머지 {len(rows)-n_flagged}건은 regex 결과가 그대로 초벌 정답 (빠르게 훑기).")
    print("\n사람이 할 일:")
    print("  1. review_flags 붙은 건 본문 보고 gold_tags 확정")
    print("  2. 모든 건에서 regex가 놓친 서술형 역량 추가 (판단4-②)")
    print("  3. 사전에 없는 새 역량은 new_tag_candidates 에 기록")
    print("  4. reviewed=true 로 변경")


if __name__ == "__main__":
    main()
