"""
RAG 1단계 — 공고를 검색 단위(청크)로 자른다.

청킹이 왜 필요한가:
  공고 한 건은 약 1만 자다. 통째로 임베딩하면 "회사 소개 + 업무 + 자격 + 복지"가 벡터 하나에 섞여
  "Spark 경험 요구" 같은 질문과의 유사도가 흐려진다. 반대로 너무 잘게 자르면 문맥이 끊긴다.
  원티드 공고는 이미 sections 5개(intro/main_tasks/requirements/preferred_points/benefits)로 나뉘어 있으니
  그 경계를 1차 단위로 쓰고, MAX_CHARS 를 넘는 구역만 빈 줄 → 줄 단위로 더 나눈다.

중복 처리:
  같은 공고가 여러 직무 폴더에 들어 있다(통과 1,132행 중 고유 공고는 그보다 적다).
  검색 대상은 '고유 공고'여야 같은 문단이 상위 k 를 중복으로 채우지 않는다. job_id 로 합치고 all_jobs 에 직무를 모은다.
  da 의 비데이터 공고는 추출 보고서와 같은 is_data_job() 기준으로 뺀다.

사용:
  python -m src.rag.chunk [--date YYYY-MM-DD]
  → data/processed/rag/<date>/chunks.jsonl  (원문 포함 → 레포 미포함)
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.common.jsonl import iter_jsonl, write_jsonl
from src.common.paths import JOBS, PROCESSED_DIR, RAW_DIR, resolve_date
from src.extract.job_skill_extract_report import is_data_job

# 추출(SECTION_KEYS)은 역량이 적히는 3구역만 보지만, 검색은 "재택 가능?" "회사가 뭘 하나" 같은 질문도 받으므로 5구역 전부.
# intro 는 회사 홍보문이라 잡음이 될 수 있다 → 정답셋 비교에서 '3구역만' 변형으로 확인한다.
RAG_SECTIONS = ["intro", "main_tasks", "requirements", "preferred_points", "benefits"]
MAX_CHARS = 1500   # 구역이 이보다 길면 더 나눈다 (구역 중앙값 260~540자, 최대 5,575자)
MIN_CHARS = 30     # 이보다 짧은 조각은 앞 조각에 붙인다 ("- 우대" 같은 머리글만 남는 것 방지)
RAG_DIR = PROCESSED_DIR / "rag"


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """긴 구역을 빈 줄(문단) 경계로, 그래도 길면 줄 경계로 묶어 max_chars 이하 조각으로 만든다.
    문장 중간은 자르지 않는다. 한 줄이 max_chars 를 넘으면 그 줄은 그대로 둔다."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    units: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        units.extend(para.split("\n") if len(para) > max_chars else [para])
    pieces: list[str] = []
    buf = ""
    for u in (u.strip() for u in units):
        if not u:
            continue
        if buf and len(buf) + 1 + len(u) > max_chars:
            pieces.append(buf)
            buf = u
        else:
            buf = f"{buf}\n{u}" if buf else u
    if buf:
        pieces.append(buf)
    # 너무 짧은 조각은 앞에 붙인다
    merged: list[str] = []
    for p in pieces:
        if merged and len(p) < MIN_CHARS:
            merged[-1] = f"{merged[-1]}\n{p}"
        else:
            merged.append(p)
    return merged


def load_postings(date: str, raw_dir: Path = RAW_DIR) -> dict[str, dict]:
    """직무 폴더 4개를 읽어 job_id 로 합친 고유 공고. 값: title, company, all_jobs, sections."""
    postings: dict[str, dict] = {}
    for job in JOBS:
        path = Path(raw_dir) / job / f"wanted_{date}.jsonl"
        if not path.exists():
            print(f"[주의] 없음: {path}")
            continue
        for _ln, rec in iter_jsonl(path, strict=False):
            if not rec or not rec.get("job_id"):
                continue
            if job == "da" and not is_data_job(rec.get("title", "")):
                continue
            jid = str(rec["job_id"])
            p = postings.setdefault(jid, {
                "title": rec.get("title", ""), "company": rec.get("company", ""),
                "all_jobs": [], "sections": rec.get("sections") or {},
            })
            if job not in p["all_jobs"]:
                p["all_jobs"].append(job)
    return postings


def make_chunks(postings: dict[str, dict]) -> list[dict]:
    chunks: list[dict] = []
    for jid in sorted(postings, key=int):
        p = postings[jid]
        for sec in RAG_SECTIONS:
            for i, piece in enumerate(split_long(p["sections"].get(sec) or "")):
                chunks.append({
                    "chunk_id": f"{jid}:{sec}:{i}", "job_id": jid, "section": sec,
                    "title": p["title"], "company": p["company"], "all_jobs": p["all_jobs"],
                    "text": piece,
                })
    return chunks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--date")
    args = ap.parse_args(argv)
    date = resolve_date(args.date)
    postings = load_postings(date)
    chunks = make_chunks(postings)
    out = RAG_DIR / date / "chunks.jsonl"
    write_jsonl(out, chunks)

    lens = sorted(len(c["text"]) for c in chunks)
    by_sec = Counter(c["section"] for c in chunks)
    print(f"[날짜] {date}")
    print(f"고유 공고 {len(postings)}건 → 청크 {len(chunks)}개  ({out})")
    print(f"구역별 청크: {dict(by_sec)}")
    print(f"청크 길이: 중앙값 {lens[len(lens)//2]}자 · 최대 {lens[-1]}자 · {MAX_CHARS}자 초과 {sum(l > MAX_CHARS for l in lens)}개")


if __name__ == "__main__":
    main()
