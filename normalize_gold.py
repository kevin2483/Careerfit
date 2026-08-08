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
  python normalize_gold.py data/labeled/label_draft.jsonl data/raw/dict/skill_tags_v1.yaml
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import yaml

# 사전 v1 개정 시 이름이 바뀐 것 (수동 매핑)
RENAMES = {
    "ABテスト": "A/B테스트",
}


def build_alias_index(dict_path: Path) -> dict[str, str]:
    """소문자 표기 -> canonical. alias 와 canonical 자신을 모두 포함."""
    raw = yaml.safe_load(dict_path.read_text(encoding="utf-8"))
    idx: dict[str, str] = {}
    for _cat, entries in raw.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            canon = e["canonical"]
            idx[canon.lower()] = canon
            for a in (e.get("aliases") or []):
                idx[str(a).lower()] = canon
    return idx


def main() -> None:
    label_path = Path(sys.argv[1])
    dict_path = Path(sys.argv[2])
    idx = build_alias_index(dict_path)

    rows = [json.loads(l) for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]

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

    with label_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"정규화 완료: {changed}건 변경 -> {label_path}")
    if detail:
        print("변경 매핑:")
        for a, b in sorted(detail.items()):
            print(f"  {a}  ->  {b}")


if __name__ == "__main__":
    main()
