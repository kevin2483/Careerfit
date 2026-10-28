"""
예측 파일 행의 실패 여부 — 세 채점기(evaluate_extraction · bootstrap_ci · cascade_sim)가 같은 기준을 쓴다.

API 호출이 실패했거나 출력이 잘려 파싱이 안 된 행은 tags=[] 로 저장된다. 그걸 그대로 "아무것도 못 찾음"으로
채점하면 모델 성능이 아니라 호출 설정(max_tokens 등)을 재게 된다. 2026-10-28 리뷰에서 sonnet 1건(361662,
max_tokens=512 에서 잘림)이 sonnet 전체 FN 의 31% 를 만들고 있었다. 채점기는 이 행을 빼지 않되(정답셋 수를
바꾸지 않기 위해) 반드시 소리 내어 알리고, failed_rows.py 가 제외했을 때의 수치를 따로 낸다.
"""
from __future__ import annotations


def pred_failure(rec: dict) -> str | None:
    """실패 사유 문자열, 정상이면 None. regex 예측처럼 _meta 가 단순한 행은 항상 정상."""
    m = rec.get("_meta") or {}
    if m.get("error"):
        return f"error={m['error']}"
    if m.get("stop_reason") == "max_tokens":
        return "max_tokens(출력 잘림)"
    if m.get("parse_ok") is False:
        return "parse_fail"
    return None


def failed_ids(recs: list[dict], ids: list[str] | None = None) -> dict[str, str]:
    """job_id -> 실패 사유. ids 를 주면 그 문서만 본다."""
    want = set(ids) if ids is not None else None
    out: dict[str, str] = {}
    for r in recs:
        jid = str(r.get("job_id"))
        if want is not None and jid not in want:
            continue
        why = pred_failure(r)
        if why:
            out[jid] = why
    return out


def report_failed_rows(name: str, recs: list[dict], ids: list[str] | None = None) -> dict[str, str]:
    bad = failed_ids(recs, ids)
    if bad:
        shown = ", ".join(f"{j}({w})" for j, w in sorted(bad.items()))
        print(f"[경고] {name}: 예측 실패 행 {len(bad)}건이 빈 예측으로 채점됨 — {shown}")
    return bad
