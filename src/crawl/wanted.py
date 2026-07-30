"""
CareerFit - Wanted 공고 수집기

설계 원칙:
  1. 수집은 멍청하게. 정규화/판단은 전부 정제(clean) 단계로 미룬다.
  2. 원문은 무조건 통째로 보존한다. 추출은 앞으로 수십 번 다시 돌린다.
  3. 한 줄 = 한 스냅샷. (source, job_id, collected_at) 이 유일키.

사용:
  python -m src.crawl.wanted --job-ids 1024,1025 --target 100
  python -m src.crawl.wanted --job-group-id 518 --target 500 --sleep 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

KST = timezone(timedelta(hours=9))

LIST_URL = "https://www.wanted.co.kr/api/chaos/navigation/v1/results"
DETAIL_URL = "https://www.wanted.co.kr/api/v4/jobs/{job_id}"
WD_URL = "https://www.wanted.co.kr/wd/{job_id}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.wanted.co.kr/",
}

# 본문 섹션. 순서와 라벨을 고정해두면 4번 평가에서 필드별 난이도 분리가 가능해진다.
#   requirements/preferred_points -> ② 서술형 요구 경험
#   main_tasks/intro              -> ③ 암묵적 조건이 주로 숨는 곳
SECTIONS: list[tuple[str, str]] = [
    ("intro", "회사소개"),
    ("main_tasks", "주요업무"),
    ("requirements", "자격요건"),
    ("preferred_points", "우대사항"),
    ("benefits", "혜택및복지"),
]


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def get_json(session: requests.Session, url: str, params: dict | None = None,
             retries: int = 3, backoff: float = 2.0) -> dict | None:
    """실패를 조용히 삼키지 않는다. 재시도 후에도 실패하면 None + 로그."""
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = backoff ** attempt
                print(f"  [{r.status_code}] {url} -> {wait:.0f}s 후 재시도 ({attempt}/{retries})",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  [{r.status_code}] {url} 포기", file=sys.stderr)
            return None
        except requests.RequestException as e:
            wait = backoff ** attempt
            print(f"  [ERR] {type(e).__name__} {url} -> {wait:.0f}s 후 재시도 ({attempt}/{retries})",
                  file=sys.stderr)
            time.sleep(wait)
    return None


def iter_job_list(session: requests.Session, target: int, job_group_id: int | None,
                  job_ids: list[int] | None, years: int, page_size: int,
                  sleep: float) -> Iterable[tuple[dict, dict]]:
    """목록 API를 offset 페이지네이션으로 훑는다. (list_item, query_meta) 를 yield."""
    offset = 0
    yielded = 0
    while yielded < target:
        params: dict[str, Any] = {
            "country": "kr",
            "job_sort": "job.latest_order",   # chaos 엔드포인트 필수
            "years": years,
            "locations": "all",
            "limit": min(page_size, target - yielded),
            "offset": offset,
        }
        # chaos/navigation 엔드포인트는 job_group_id + job_ids 를 세트로 요구한다.
        # job_ids 만 넘기면 필터가 무시되고 전 직군 최신순이 나온다. (2026-07-25 실패기록)
        if job_group_id is not None:
            params["job_group_id"] = job_group_id
        if job_ids:
            params["job_ids"] = ",".join(str(j) for j in job_ids)

        payload = get_json(session, LIST_URL, params=params)
        if not payload:
            print("  목록 요청 실패. 중단.", file=sys.stderr)
            return
        items = payload.get("data") or []
        if not items:
            print(f"  offset={offset} 에서 결과 없음. 목록 끝.")
            return

        query_meta = {k: v for k, v in params.items()}
        for it in items:
            yield it, query_meta
            yielded += 1
            if yielded >= target:
                return

        offset += len(items)
        time.sleep(sleep)


# --------------------------------------------------------------------------
# 레코드 조립 (순수 함수 - 테스트 대상)
# --------------------------------------------------------------------------
def build_body_raw(detail: dict) -> tuple[dict, str]:
    """섹션별 원문 dict 와, 섹션 마커가 붙은 단일 문자열을 함께 만든다."""
    sections: dict[str, str] = {}
    parts: list[str] = []
    for key, label in SECTIONS:
        text = (detail.get(key) or "").strip()
        sections[key] = text
        if text:
            parts.append(f"[{label}]\n{text}")
    return sections, "\n\n".join(parts)


def build_record(list_item: dict, detail_job: dict, query_meta: dict,
                 collected_at: str) -> dict:
    job_id = str(detail_job.get("id") or list_item.get("id"))
    detail = detail_job.get("detail") or {}
    company = detail_job.get("company") or {}
    address = detail_job.get("address") or {}
    sections, body_raw = build_body_raw(detail)

    return {
        # --- 식별 ---
        "source": "wanted",
        "job_id": job_id,
        "collected_at": collected_at,
        "url": WD_URL.format(job_id=job_id),

        # --- 표층 필드 (가공 없음) ---
        "title": detail_job.get("position") or detail.get("position") or "",
        "company": company.get("name", ""),
        "company_id": company.get("id"),
        "address_raw": {
            "country": address.get("country"),
            "location": address.get("location"),
            "district": address.get("district"),
            "full_location": address.get("full_location"),
        },
        "due_time": detail_job.get("due_time"),
        "status": detail_job.get("status"),

        # --- 경력 조건: 원본 그대로. 신입/경력 판정은 정제 단계에서 ---
        "annual_from": detail_job.get("annual_from"),
        "annual_to": detail_job.get("annual_to"),
        "career_type": None,        # clean 단계에서 채운다

        # --- 원티드가 이미 붙여둔 태그: 내 추출 결과와 비교할 참고군 ---
        "skill_tags": [t.get("title") for t in (detail_job.get("skill_tags") or [])],
        "category_tags": [t.get("title") for t in (detail_job.get("category_tags") or [])],

        # --- 본문 ---
        "sections": sections,
        "body_raw": body_raw,

        # --- 출처 추적 (표본 편향 진단용) ---
        "query_meta": query_meta,

        # --- 보험: 지금 안 쓰는 필드도 통째로 보관 ---
        "api_raw": detail_job,

        # --- 정제 단계에서 채울 자리 ---
        "first_seen_at": None,
        "is_closed": None,
    }


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------
def load_today_seen(out_path: Path) -> set[str]:
    """같은 날 중간에 끊겼을 때 이어서 받기 위한 resume 용."""
    seen: set[str] = set()
    if not out_path.exists():
        return seen
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["job_id"])
            except Exception:
                continue
    return seen


def run(args: argparse.Namespace) -> int:
    now = datetime.now(KST)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wanted_{now:%Y-%m-%d}.jsonl"
    fail_path = out_dir / f"wanted_{now:%Y-%m-%d}.failed.txt"

    seen = load_today_seen(out_path)
    if seen:
        print(f"오늘 이미 수집된 {len(seen)}건은 건너뜁니다.")

    session = requests.Session()
    saved, failed, skipped = 0, 0, 0

    with out_path.open("a", encoding="utf-8") as out, \
         fail_path.open("a", encoding="utf-8") as flog:

        for list_item, query_meta in iter_job_list(
            session,
            target=args.target,
            job_group_id=args.job_group_id,
            job_ids=args.job_ids,
            years=args.years,
            page_size=args.page_size,
            sleep=args.sleep,
        ):
            job_id = str(list_item.get("id"))
            if job_id in seen:
                skipped += 1
                continue

            payload = get_json(session, DETAIL_URL.format(job_id=job_id))
            time.sleep(args.sleep)

            detail_job = (payload or {}).get("job")
            if not detail_job:
                failed += 1
                flog.write(f"{job_id}\n")
                flog.flush()
                continue

            rec = build_record(list_item, detail_job, query_meta,
                               collected_at=now.isoformat())
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            seen.add(job_id)
            saved += 1
            if saved % 10 == 0:
                print(f"  ... {saved}건 저장")

    print(f"\n저장 {saved} / 스킵 {skipped} / 실패 {failed}")
    print(f"-> {out_path}")
    if failed:
        print(f"-> 실패 ID 목록: {fail_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wanted 공고 수집기 (CareerFit)")
    p.add_argument("--target", type=int, default=100, help="수집할 공고 수")
    p.add_argument("--job-group-id", type=int, default=518,
                   help="직군 ID (개발 518, 경영·비즈니스 507). job_ids 와 세트 필수")
    p.add_argument("--job-ids", type=lambda s: [int(x) for x in s.split(",") if x.strip()],
                   default=None, help="세부 직무 ID 쉼표구분 (예: 1024,1025)")
    p.add_argument("--years", type=int, default=-1, help="-1=전체, 0=신입")
    p.add_argument("--page-size", type=int, default=20)
    p.add_argument("--sleep", type=float, default=0.7)
    p.add_argument("--out-dir", default="data/raw")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
