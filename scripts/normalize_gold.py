"""
정답셋 태그명 정규화 (사전 v0 -> v1 개정에 따른 보정)

문제:
  사전 v1 에서 canonical 이름을 고쳤다(예: ABテスト -> A/B테스트).
  그런데 정답셋(gold_tags)에는 옛 이름이 남아 있어, 같은 역량이
  FN(옛이름)과 FP(새이름)로 이중 계산된다.

이 스크립트가 하는 일:
  - 이름 변경 매핑을 적용해 gold_tags 를 v1 canonical 로 통일
  - 대소문자·표기 흔들림(kafka -> Kafka)도 사전의 canonical/alias 로 맞춤
  - 사람이 자유 입력한 값 중 사전 alias 에 해당하면 canonical 로 승격

주의:
  판단을 바꾸는 게 아니다. '같은 것을 같은 이름으로' 부르게만 한다.
  사람이 넣은 태그를 삭제하거나 새로 추가하지 않는다.

사용:
  python -m scripts.normalize_gold data/labeled/label_draft.jsonl data/raw/dict/skill_tags_v1.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.skill_dict import alias_index

# 사전 v1 개정 시 이름이 바뀐 것 (수동 매핑)
RENAMES = {
    "ABテスト": "A/B테스트",
}


def main() -> None:
    label_path = Path(sys.argv[1])
    dict_path = Path(sys.argv[2])
    idx = alias_index(dict_path)

    rows = read_jsonl(label_path)

    changed = 0
    detail: dict[str, str] = {}
    for r in rows:
        new_tags = []
        for t in r.get("gold_tags", []):
            # 1) 명시적 이름 변경
            t2 = RENAMES.get(t, t)
            # 2) 사전 alias/canonical 로 정규화 (kafka -> Kafka)
            t3 = idx.get(t2.lower(), t2)
            if t3 != t:
                detail[t] = t3
                changed += 1
            new_tags.append(t3)
        r["gold_tags"] = sorted(set(new_tags))

    write_jsonl(label_path, rows)

    print(f"정규화 완료: {changed}건 변경 -> {label_path}")
    if detail:
        print("변경 매핑:")
        for a, b in sorted(detail.items()):
            print(f"  {a}  ->  {b}")


if __name__ == "__main__":
    main()
