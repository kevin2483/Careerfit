"""
정답셋 교정 2026-10-27 — 사람 판정은 바꾸지 않고 '표기'만 고친다. 일회성. 감사 추적용으로 남긴다.

교정 내용 (전체 코드 검증에서 발견):
  1. 374560: 풀링에서 후보 "FastAPI" 를 n 으로 판정한 뒤 수기 추가란에 "fastapi" 를 넣었다.
     수기 추가가 판정을 덮는 의도였으므로(같은 역량) 후보 행의 n 판정을 제거하고 수기 행을 canonical
     "FastAPI" 로 맞춘다. 고치지 않으면 네 방식 모두 FP(FastAPI)+FN(fastapi) 로 이중 감점된다.
  2. 292454: gold "코드품 품질관리" 는 오타이며 사전에 없는 역량이다 → gold 에서 빼고
     new_tag_candidates 에 "코드품질관리" 로 기록한다 (사전 v2 후보).
  3. 348118: new_tag_candidates "인과추ㄴ론" → "인과추론", 다른 공고의 "기계ㄱ공학 전공" → "기계공학 전공" (오타).

이후 scripts/normalize_gold.py 와 scripts/make_public_labels.py 를 돌려 공개판을 맞춘다.

    python -m scripts.fix_gold_2026_10_27
"""
from __future__ import annotations

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.paths import LABELED_DIR, LABELS_PRIVATE

POOL = LABELED_DIR / "pool_candidates.jsonl"


def main() -> None:
    rows = read_jsonl(LABELS_PRIVATE)
    changed = []
    for r in rows:
        jid = str(r["job_id"])
        if jid == "292454" and "코드품 품질관리" in r["gold_tags"]:
            r["gold_tags"] = sorted(t for t in r["gold_tags"] if t != "코드품 품질관리")
            r["new_tag_candidates"] = sorted(set(r.get("new_tag_candidates") or []) | {"코드품질관리"})
            changed.append((jid, "gold '코드품 품질관리' → new_tag_candidates '코드품질관리'"))
        if jid == "348118" and "인과추ㄴ론" in (r.get("new_tag_candidates") or []):
            r["new_tag_candidates"] = ["인과추론" if c == "인과추ㄴ론" else c for c in r["new_tag_candidates"]]
            changed.append((jid, "new_tag_candidates '인과추ㄴ론' → '인과추론'"))
        if "기계ㄱ공학 전공" in (r.get("new_tag_candidates") or []):
            r["new_tag_candidates"] = ["기계공학 전공" if c == "기계ㄱ공학 전공" else c for c in r["new_tag_candidates"]]
            changed.append((jid, "new_tag_candidates '기계ㄱ공학 전공' → '기계공학 전공'"))
        if jid == "374560" and "fastapi" in r["gold_tags"]:
            r["gold_tags"] = sorted((set(r["gold_tags"]) - {"fastapi"}) | {"FastAPI"})
            changed.append((jid, "gold 'fastapi' → 'FastAPI'"))
    write_jsonl(LABELS_PRIVATE, rows)

    pool = read_jsonl(POOL)
    before = len(pool)
    pool = [c for c in pool if not (str(c["job_id"]) == "374560" and c["tag"] == "FastAPI" and c["verdict"] is False)]
    for c in pool:
        if str(c["job_id"]) == "374560" and c["tag"] == "fastapi" and c.get("source") == "manual":
            c["tag"] = "FastAPI"
            c["note"] = "2026-10-27 표기 정규화 (fastapi → FastAPI). 중복된 n 판정 행 제거"
    write_jsonl(POOL, pool)
    changed.append(("pool", f"{before} → {len(pool)}건 (374560 FastAPI n 판정 행 제거, 수기 행 canonical 화)"))

    for jid, what in changed:
        print(f"  {jid}: {what}")
    print(f"교정 {len(changed)}건 완료")


if __name__ == "__main__":
    main()
