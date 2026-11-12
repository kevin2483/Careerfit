"""
생성 — hybrid 검색 상위 문단만 근거로 답을 만들고, 주장마다 인용 문단을 붙인다.

고정값은 notes/rag_generation_2026-11-11.md (실행 전 커밋). 프롬프트 문구를 여기서 고치지 않는다.
  - 검색을 다시 돌리지 않는다. 커밋된 실행 기록(rag_runs/hybrid.jsonl)의 상위 5(D 유형 3)를 그대로 넘긴다.
  - 질문당 1회 호출. 형식이 틀린 응답은 고치지 않고 실패 행으로 남긴다 (재호출은 사람이 정한다).
  - LLM 판정(--llm-judge)은 사람 판정이 전부 끝난 뒤에만 돈다. 점수에는 쓰이지 않는 참고 수치다.

사용:
  python -m src.rag.generate                 # 답 생성 (이미 있으면 덮어쓰지 않고 멈춘다)
  python -m src.rag.generate --llm-judge     # 사람 판정 완료 후, 같은 주장을 LLM 이 판정
  python -m src.rag.generate --make-public   # 주장 문장을 뺀 공개판 다시 만들기
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.common.jsonl import read_jsonl, write_jsonl
from src.common.paths import LABELED_DIR, PROCESSED_DIR, resolve_date
from src.rag.judge import depth_by_qid
from src.rag.run import QUERIES_PATH, RUNS_DIR, load_queries

RETRIEVER = "hybrid"
GEN_MODEL = "claude-sonnet-5-5"
JUDGE_MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 8000
MAX_CLAIMS = 2
GEN_DIR = PROCESSED_DIR / "rag_generation"
ANSWERS_PATH = GEN_DIR / "answers.jsonl"                 # 주장 문장 포함. 레포 미포함 (공고 내용이 들어간다)
ANSWERS_PUBLIC_PATH = GEN_DIR / "answers.public.jsonl"   # 주장 문장만 뺀 공개판. 평가는 이것만 읽는다
LLM_JUDGE_PATH = GEN_DIR / "llm_judge.jsonl"
CLAIM_JUDGMENTS_PATH = LABELED_DIR / "rag_claim_judgments.jsonl"
CLAIM_REJUDGMENTS_PATH = LABELED_DIR / "rag_claim_rejudgments.jsonl"

SYSTEM_PROMPT = """당신은 채용공고 검색 결과를 요약해 답하는 도우미다.
사용자 질문과 번호가 붙은 채용공고 문단들이 주어진다. 문단은 검색기가 찾은 것이라 질문과 무관한 것이 섞여 있을 수 있다.

규칙:
- 주어진 문단에 적힌 내용만 쓴다. 문단에 없는 사실, 일반 상식, 추측을 더하지 않는다.
- 질문의 조건을 직접 담은 문단이 하나도 없으면 answerable 을 false 로 하고 claims 를 비운다. 억지로 답하지 않는다.
- 답할 수 있으면 answerable 을 true 로 하고 claims 에 주장을 1개 또는 2개 쓴다.
- 주장 하나는 한 문장이고, 어느 회사의 공고가 무엇을 요구·우대·제공하는지를 문단의 표현에 맞춰 쓴다. "우대"를 "필수"로 바꾸는 식으로 강도를 바꾸지 않는다.
- 주장마다 근거 문단 번호 하나를 source 에 적는다. 그 문단만 읽고도 주장이 확인되어야 한다.
- 한국어로 쓴다."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "source": {"type": "integer"}},
            "required": ["text", "source"], "additionalProperties": False}},
    },
    "required": ["answerable", "claims"], "additionalProperties": False,
}

JUDGE_SYSTEM = """채용공고 문단 하나와 그 문단을 근거로 쓴 주장 한 문장이 주어진다. 주장이 그 문단에 실제로 있는 말인지 판정한다.

기준:
- 주장의 내용이 전부 그 문단(제목·회사·구역 표시 포함)에서 확인되면 supported 는 true.
- 일부만 확인되거나, 강도가 바뀌었거나(우대 → 필수, 일부 → 전부), 문단에 없는 내용이 섞였으면 false.
- 표현이 달라도 뜻이 같으면 true. 애매하면 false.
- 질문에 맞는 답인지는 따지지 않는다. 문단에 있는 말인지만 본다."""

JUDGE_SCHEMA = {"type": "object", "properties": {"supported": {"type": "boolean"}},
                "required": ["supported"], "additionalProperties": False}


def chunk_header(chunk: dict) -> str:
    return f"{chunk.get('title', '')} · {chunk.get('company', '')} · {chunk.get('section', '')}"


def passed_chunks(queries: list[dict], run_rows: list[dict]) -> dict[str, list[str]]:
    """질문별로 생성에 넘길 chunk_id (순위 순). 깊이는 검색 판정과 같다: A~C 5, D 3."""
    depth = depth_by_qid(queries)
    out: dict[str, list[str]] = {q["qid"]: [] for q in queries}
    for r in sorted(run_rows, key=lambda r: (r["qid"], r["rank"])):
        if r["qid"] in out and r["rank"] <= depth[r["qid"]]:
            out[r["qid"]].append(r["chunk_id"])
    return out


def user_message(question: str, chunks: list[dict]) -> str:
    parts = [f"질문: {question}"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk_header(c)}\n{c['text']}")
    return "\n\n".join(parts)


def call_json(client, model: str, system: str, user: str, schema: dict) -> dict:
    """한 번 호출해 {data, error, stop_reason, tokens, ms} 를 돌려준다. 내용을 보고 다시 부르지 않는다."""
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model, max_tokens=MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
    )
    out = {"data": None, "error": None, "stop_reason": resp.stop_reason,
           "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
           "ms": round((time.perf_counter() - t0) * 1000, 1)}
    if resp.stop_reason != "end_turn":
        out["error"] = f"stop_reason={resp.stop_reason}"
        return out
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        out["data"] = json.loads(text)
    except json.JSONDecodeError:
        out["error"] = "json_parse"
    return out


def validate_answer(data, n_chunks: int) -> str | None:
    """사전 등록한 실패 사유. 통과하면 None."""
    if not isinstance(data, dict) or not isinstance(data.get("answerable"), bool) or not isinstance(data.get("claims"), list):
        return "bad_shape"
    claims = data["claims"]
    if len(claims) > MAX_CLAIMS:
        return "too_many_claims"
    if data["answerable"] and not claims:
        return "answerable_without_claims"
    if not data["answerable"] and claims:
        return "claims_when_unanswerable"
    for c in claims:
        if not isinstance(c, dict) or not isinstance(c.get("text"), str) or not c["text"].strip():
            return "bad_shape"
        if isinstance(c.get("source"), bool) or not isinstance(c.get("source"), int) or not 1 <= c["source"] <= n_chunks:
            return "source_out_of_range"
    return None


def generate(client, queries: list[dict], passed: dict[str, list[str]], chunks: dict[str, dict],
             model: str = GEN_MODEL) -> list[dict]:
    rows = []
    for q in queries:
        ids = passed[q["qid"]]
        r = call_json(client, model, SYSTEM_PROMPT, user_message(q["question"], [chunks[c] for c in ids]), ANSWER_SCHEMA)
        error = r["error"] or validate_answer(r["data"], len(ids))
        ok = error is None
        rows.append({
            "qid": q["qid"], "model": model, "retriever": RETRIEVER, "passed": ids,
            "answerable": r["data"]["answerable"] if ok else None,
            "claims": [{"claim_id": f"{q['qid']}:{i}", "text": c["text"].strip(), "chunk_id": ids[c["source"] - 1]}
                       for i, c in enumerate(r["data"]["claims"], 1)] if ok else [],
            "error": error, "stop_reason": r["stop_reason"],
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"], "ms": r["ms"],
        })
        print(f"  {q['qid']}: " + (f"실패 {error}" if error else f"answerable={rows[-1]['answerable']} 주장 {len(rows[-1]['claims'])}개"))
    return rows


def public_rows(answers: list[dict]) -> list[dict]:
    """주장 문장(text)만 뺀다. claim_id·chunk_id·판정에 필요한 나머지는 그대로."""
    return [dict(a, claims=[{k: v for k, v in c.items() if k != "text"} for c in a["claims"]]) for a in answers]


def all_claims(answers: list[dict]) -> list[dict]:
    return [dict(c, qid=a["qid"]) for a in answers for c in a["claims"]]


def llm_judge(client, answers: list[dict], chunks: dict[str, dict], model: str = JUDGE_MODEL) -> list[dict]:
    rows = []
    for c in all_claims(answers):
        ch = chunks[c["chunk_id"]]
        r = call_json(client, model, JUDGE_SYSTEM, f"문단: {chunk_header(ch)}\n{ch['text']}\n\n주장: {c['text']}", JUDGE_SCHEMA)
        ok = r["error"] is None and isinstance(r["data"], dict) and isinstance(r["data"].get("supported"), bool)
        rows.append({"claim_id": c["claim_id"], "model": model, "supported": r["data"]["supported"] if ok else None,
                     "error": None if ok else (r["error"] or "bad_shape")})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--out", default=str(ANSWERS_PATH))
    ap.add_argument("--llm-judge", action="store_true")
    ap.add_argument("--make-public", action="store_true", help="answers.jsonl 에서 공개판만 다시 만든다 (API 호출 없음)")
    ap.add_argument("--public-out", default=str(ANSWERS_PUBLIC_PATH))
    ap.add_argument("--judgments", default=str(CLAIM_JUDGMENTS_PATH))
    ap.add_argument("--llm-out", default=str(LLM_JUDGE_PATH))
    a = ap.parse_args(argv)

    if a.make_public:
        write_jsonl(a.public_out, public_rows(read_jsonl(a.out)))
        print(f"저장 -> {a.public_out}")
        return
    out = Path(a.llm_out if a.llm_judge else a.out)
    if out.exists():
        raise SystemExit(f"[중단] {out} 이 이미 있다. 한 번만 돌린다 (사전 등록). 다시 하려면 새 사전 등록 문서를 쓴다.")
    if a.llm_judge:
        answers = read_jsonl(a.out)
        human = {r["claim_id"] for r in read_jsonl(a.judgments)} if Path(a.judgments).exists() else set()
        left = [c["claim_id"] for c in all_claims(answers) if c["claim_id"] not in human]
        if left:
            raise SystemExit(f"[중단] 사람 판정이 {len(left)}건 남음. LLM 판정은 사람 판정이 끝난 뒤에 돌린다.")
    from anthropic import Anthropic
    from src.rag.search import load_chunks
    chunks = {c["chunk_id"]: c for c in load_chunks(resolve_date(a.date, prefer="processed"))}
    client = Anthropic()
    if a.llm_judge:
        rows = llm_judge(client, answers, chunks)
    else:
        queries = load_queries(a.queries)
        rows = generate(client, queries, passed_chunks(queries, read_jsonl(Path(a.runs_dir) / f"{RETRIEVER}.jsonl")), chunks)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    if not a.llm_judge:
        write_jsonl(a.public_out, public_rows(rows))
    failed = [r for r in rows if r["error"]]
    print(f"저장 -> {out} ({len(rows)}행)")
    if failed:
        print(f"[경고] 실패 행 {len(failed)}건: {[(r.get('qid') or r.get('claim_id'), r['error']) for r in failed]} — 재호출 여부는 사람이 정한다.")


if __name__ == "__main__":
    main()
