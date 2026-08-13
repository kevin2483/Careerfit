"""
CareerFit — LLM 기반 스킬 추출기 (regex baseline과 동일 인터페이스)

위치: src/extract/llm_skill_extract.py

설계 원칙
---------
1. 닫힌 라벨 공간(closed set). 사전 v1의 canonical 106개만 출력하게 강제한다.
   regex와 같은 라벨 공간이어야 evaluate_extraction.py로 직접 비교가 된다.
   자유 생성을 허용하면 "LLM이 더 많이 찾았다"가 성능인지 라벨 불일치인지 구분 불가.
2. 가설을 프롬프트에 명시. regex는 문자열이 있으면 무조건 잡는다.
   LLM의 기대 우위는 "회사 소개·제품 설명에 등장할 뿐 요구 역량이 아닌 기술"을
   버리는 것 = precision. 이 가설이 맞는지가 4단계의 실험 내용이다.
3. 호출당 usage/latency/cost를 전부 남긴다. 5단계(캐스케이드 Pareto)는
   여기서 남긴 로그가 없으면 다시 돌려야 한다. 지금 남긴다.

스키마 맞추기 (레포에 붙일 때 여기만 확인)
------------------------------------------
- RAW_ID_KEYS / RAW_TEXT_KEYS / RAW_ROLE_KEYS: 원본 JSONL 키 후보. 실제 키 추가.
- PRED_TAGS_KEY: evaluate_extraction.py가 예측 태그를 읽는 키 이름.
- load_canonical_tags(): 사전 YAML 구조에 맞게 확인.

실행
----
    export ANTHROPIC_API_KEY=...
    # 정답셋 38건만 (평가 루프용, 싸다)
    python -m src.extract.llm_skill_extract \
        --raw-dir data/raw --dict data/raw/dict/skill_tags_v1.yaml \
        --out-dir data/processed --date 2026-08-12 \
        --model claude-haiku-4-5-20251001 --ids-file data/gold/gold_ids.txt

    # 전체
    python -m src.extract.llm_skill_extract ... --model claude-sonnet-5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import yaml
from anthropic import Anthropic

# ---------------------------------------------------------------- 스키마 어댑터

RAW_ID_KEYS = ("job_id", "id", "position_id", "wanted_id")
RAW_TEXT_KEYS = ("full_text", "detail", "description", "requirements", "content")
RAW_ROLE_KEYS = ("role", "job_role", "category")
PRED_TAGS_KEY = "tags"

MAX_CHARS = 12_000  # 원티드 공고 본문은 대부분 이 안에 들어옴. 초과분은 잘리고 플래그가 남음.

# USD / 1M tokens. 2026-08-12 기준. Sonnet 5는 2026-09-01부터 3/15로 바뀜 —
# 리포트에 숫자 쓸 때 rate_date를 같이 박아둘 것.
PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}
PRICING_AS_OF = "2026-08-12"

SYSTEM_PROMPT = """당신은 채용공고에서 '지원자에게 요구되는 기술 역량'만 골라내는 추출기다.

허용 태그 목록 (이 목록 밖의 것은 절대 출력하지 않는다):
{tag_list}

규칙:
1. 공고가 자격요건·우대사항·업무내용에서 지원자에게 요구하거나 우대하는 기술만 뽑는다.
2. 회사 소개, 서비스·제품 설명, 기술 블로그 홍보, 사내 인프라 자랑에만 등장하고
   지원자에게 요구하지 않는 기술은 제외한다.
3. 표기가 달라도 같은 기술이면 위 목록의 이름으로 정규화한다.
   (예: "파이썬", "Python3", "python으로" -> Python)
4. 목록에 없는 기술은 버린다. 비슷한 것으로 억지로 매핑하지 않는다.
5. 확실하지 않으면 넣지 않는다.

출력은 JSON 배열 하나만. 설명·코드펜스·다른 텍스트 금지.
예: ["Python", "SQL", "Airflow"]
빈 경우: []"""

USER_TEMPLATE = """다음 채용공고에서 요구 기술 태그를 추출하라.

<job_posting>
{text}
</job_posting>"""


# ---------------------------------------------------------------- 로딩


def load_canonical_tags(dict_path: str | Path) -> list[str]:
    """사전 YAML에서 canonical 이름만 뽑는다.

    다음 구조를 모두 허용:
      A) {canonical: [alias, ...], ...}
      B) {tags: [{canonical: "Python", aliases: [...]}, ...]}
      C) [{name: "Python", aliases: [...]}, ...]
    구조가 다르면 여기서 터지게 둔다. 조용히 빈 목록을 돌려주면
    LLM이 아무 태그도 못 뽑고 그게 '성능'으로 기록된다.
    """
    raw = yaml.safe_load(Path(dict_path).read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "tags" in raw:
        raw = raw["tags"]

    names: list[str] = []
    if isinstance(raw, dict):
        names = [str(k) for k in raw.keys()]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                for key in ("canonical", "name", "tag"):
                    if key in item:
                        names.append(str(item[key]))
                        break
                else:
                    raise ValueError(f"canonical 키를 못 찾음: {item!r}")
    else:
        raise ValueError(f"지원하지 않는 사전 구조: {type(raw)}")

    names = sorted(set(names))
    if not names:
        raise ValueError(f"{dict_path} 에서 canonical 태그를 하나도 못 읽음")
    return names


def _pick(rec: dict, keys: Iterable[str]) -> Any:
    for k in keys:
        v = rec.get(k)
        if v:
            return v
    return None


def iter_raw_records(raw_dir: str | Path, date: str | None) -> Iterable[dict]:
    """raw_dir 아래 *.jsonl 을 재귀로 읽는다. date가 주어지면 경로에 포함된 것만."""
    root = Path(raw_dir)
    paths = sorted(p for p in root.rglob("*.jsonl") if date is None or date in str(p))
    if not paths:
        raise FileNotFoundError(f"{root} (date={date}) 에서 jsonl을 못 찾음")

    for path in paths:
        # splitlines() 쓰지 말 것 — 본문 안 \u2028 등에서 줄이 깨진다. (lab_log 기록됨)
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] {path}:{lineno} JSON 파싱 실패 스킵: {e}", file=sys.stderr)
                    continue
                rec.setdefault("_src_file", str(path))
                yield rec


# ---------------------------------------------------------------- 추출


_ARRAY_RE = re.compile(r"\[.*?\]", re.S)


def parse_tags(text: str, canonical: set[str]) -> tuple[list[str], list[str]]:
    """모델 출력 -> (허용목록 안 태그, 버려진 환각 태그)"""
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", body).strip()

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        m = _ARRAY_RE.search(body)
        if not m:
            return [], []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return [], []

    if not isinstance(parsed, list):
        return [], []

    kept, dropped = [], []
    for item in parsed:
        s = str(item).strip()
        if s in canonical:
            kept.append(s)
        else:
            dropped.append(s)
    return sorted(set(kept)), sorted(set(dropped))


def extract_one(
    client: Anthropic,
    model: str,
    system_prompt: str,
    canonical: set[str],
    rec: dict,
    max_retries: int = 5,
) -> dict:
    job_id = _pick(rec, RAW_ID_KEYS)
    text = _pick(rec, RAW_TEXT_KEYS) or ""
    truncated = len(text) > MAX_CHARS

    result = {
        "job_id": job_id,
        "role": _pick(rec, RAW_ROLE_KEYS),
        PRED_TAGS_KEY: [],
        "_meta": {
            "model": model,
            "truncated": truncated,
            "n_chars": len(text),
            "hallucinated_tags": [],
            "error": None,
        },
    }
    if not text:
        result["_meta"]["error"] = "empty_text"
        return result

    delay = 1.0
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": USER_TEMPLATE.format(text=text[:MAX_CHARS])}],
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            out_text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            tags, dropped = parse_tags(out_text, canonical)

            in_tok = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
            p_in, p_out = PRICING.get(model, (0.0, 0.0))

            result[PRED_TAGS_KEY] = tags
            result["_meta"].update(
                {
                    "hallucinated_tags": dropped,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "latency_ms": latency_ms,
                    "cost_usd": (in_tok * p_in + out_tok * p_out) / 1_000_000,
                    "stop_reason": resp.stop_reason,
                    "raw_output": out_text if not tags else None,  # 파싱 실패만 원문 보존
                }
            )
            return result

        except Exception as e:  # noqa: BLE001 — 재시도 후 실패는 행 단위로 기록하고 진행
            if attempt == max_retries - 1:
                result["_meta"]["error"] = f"{type(e).__name__}: {e}"
                return result
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 30)

    return result


# ---------------------------------------------------------------- run


def _slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-")


def run(
    raw_dir: str,
    dict_path: str,
    out_dir: str,
    date: str,
    model: str = "claude-haiku-4-5-20251001",
    limit: int | None = None,
    ids: set[str] | None = None,
    workers: int = 4,
    resume: bool = True,
) -> dict:
    """regex 추출기와 동일한 시그니처. 요약 dict 반환."""
    canonical_list = load_canonical_tags(dict_path)
    canonical = set(canonical_list)
    system_prompt = SYSTEM_PROMPT.format(tag_list="\n".join(f"- {t}" for t in canonical_list))

    out_root = Path(out_dir) / date
    out_root.mkdir(parents=True, exist_ok=True)
    pred_path = out_root / f"llm_extract_{_slug(model)}.jsonl"
    summary_path = out_root / f"llm_extract_{_slug(model)}_summary.json"

    done: set[str] = set()
    if resume and pred_path.exists():
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(str(json.loads(line)["job_id"]))
                except Exception:  # noqa: BLE001
                    continue
        print(f"[resume] 기존 {len(done)}건 스킵")

    targets: list[dict] = []
    for rec in iter_raw_records(raw_dir, date):
        jid = str(_pick(rec, RAW_ID_KEYS))
        if ids is not None and jid not in ids:
            continue
        if jid in done:
            continue
        targets.append(rec)
        if limit and len(targets) >= limit:
            break

    print(f"[run] model={model} 대상 {len(targets)}건 workers={workers}")
    if not targets:
        print("[run] 처리할 건 없음")

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rows: list[dict] = []

    with pred_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(extract_one, client, model, system_prompt, canonical, rec): rec
                for rec in targets
            }
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                if i % 25 == 0 or i == len(targets):
                    print(f"  {i}/{len(targets)}")

    ok = [r for r in rows if not r["_meta"]["error"]]
    n_err = len(rows) - len(ok)
    total_cost = sum(r["_meta"].get("cost_usd", 0.0) for r in ok)
    lat = sorted(r["_meta"].get("latency_ms", 0) for r in ok)
    halluc = sum(len(r["_meta"]["hallucinated_tags"]) for r in ok)
    parse_fail = sum(1 for r in ok if r["_meta"].get("raw_output"))

    summary = {
        "model": model,
        "date": date,
        "pricing_as_of": PRICING_AS_OF,
        "n_processed": len(rows),
        "n_error": n_err,
        "n_parse_fail": parse_fail,
        "n_truncated": sum(1 for r in rows if r["_meta"]["truncated"]),
        "n_hallucinated_tags": halluc,
        "avg_tags": round(sum(len(r[PRED_TAGS_KEY]) for r in ok) / max(len(ok), 1), 3),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_doc_usd": round(total_cost / max(len(ok), 1), 8),
        "latency_ms_p50": lat[len(lat) // 2] if lat else None,
        "latency_ms_p95": lat[int(len(lat) * 0.95)] if lat else None,
        "pred_path": str(pred_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--dict", dest="dict_path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", default=None, help="한 줄에 job_id 하나. 정답셋만 돌릴 때.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    ids = None
    if a.ids_file:
        ids = {ln.strip() for ln in Path(a.ids_file).read_text(encoding="utf-8").split("\n") if ln.strip()}

    run(
        raw_dir=a.raw_dir,
        dict_path=a.dict_path,
        out_dir=a.out_dir,
        date=a.date,
        model=a.model,
        limit=a.limit,
        ids=ids,
        workers=a.workers,
        resume=not a.no_resume,
    )


if __name__ == "__main__":
    main()
