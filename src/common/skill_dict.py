"""
역량 태그 사전 로딩 + alias 정규식 컴파일 — 추출·평가·진단이 전부 이 한 곳을 쓴다.

사전 구조: {카테고리: [{canonical, aliases, field, parent?}, ...]}
  field 1 = 명시 기술 / 2 = 서술형 역량 / 3 = 암묵 조건(집계·평가 제외)

같은 태그 정의를 두 번 다르게 구현하면 기준이 어긋나므로(가령 평가는 v0, 추출은 v1),
여기 외의 곳에서 YAML 을 직접 읽지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.extract.boundary import build_alias_pattern

Pattern = tuple[str, str, int | None, re.Pattern]  # (canonical, category, field, 정규식)


def load_dict(path: str | Path) -> list[dict]:
    """사전 없음/파싱 실패는 공용 자원 문제 → 개별 스킵이 아니라 전체 중단."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[전체중단] 태그 사전 없음: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SystemExit(f"[전체중단] 태그 사전 파싱 실패: {e}")

    tags: list[dict] = []
    seen: set[str] = set()
    alias_owner: dict[str, str] = {}
    for category, entries in raw.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            canonical = e["canonical"]
            if canonical in seen:
                print(f"[경고] canonical 중복 정의: {canonical}")
            seen.add(canonical)
            aliases = [str(a) for a in (e.get("aliases") or [])]
            # 같은 alias 가 두 canonical 에 걸리면 그 문자열은 항상 두 태그를 동시에 만든다.
            # (2026-10-27 발견: mariadb→MySQL+MariaDB 등 4건이 정답셋 FP 5건을 만들었다)
            for a in [canonical.lower(), *[a.lower() for a in aliases]]:
                if a in alias_owner and alias_owner[a] != canonical:
                    print(f"[경고] alias 충돌: {a!r} → {alias_owner[a]} / {canonical}")
                alias_owner.setdefault(a, canonical)
            tags.append({
                "canonical": canonical,
                "aliases": aliases,
                "field": e.get("field"),
                "category": category,
                "parent": e.get("parent"),
            })
    return tags


def compile_patterns(tags: list[dict]) -> list[Pattern]:
    patterns: list[Pattern] = []
    for t in tags:
        alts = sorted({a for a in [*t["aliases"], t["canonical"]] if a}, key=len, reverse=True)
        if not alts:
            continue
        pat = re.compile("|".join(build_alias_pattern(a) for a in alts), re.IGNORECASE)
        patterns.append((t["canonical"], t["category"], t["field"], pat))
    return patterns


def load_patterns(path: str | Path) -> list[Pattern]:
    return compile_patterns(load_dict(path))


def field_map(path: str | Path) -> dict[str, int | None]:
    """canonical -> field."""
    return {t["canonical"]: t["field"] for t in load_dict(path)}


def canonical_names(path: str | Path) -> list[str]:
    return sorted(t["canonical"] for t in load_dict(path))


def alias_index(path: str | Path) -> dict[str, str]:
    """소문자 표기 -> canonical. canonical 자신도 포함. 사람이 자유 입력한 태그를 정규화할 때 쓴다."""
    idx: dict[str, str] = {}
    for t in load_dict(path):
        idx.setdefault(t["canonical"].lower(), t["canonical"])
        for a in t["aliases"]:
            idx.setdefault(a.lower(), t["canonical"])
    return idx


def extract_tags(text: str, patterns: list[Pattern]) -> set[str]:
    return {c for c, _cat, _f, pat in patterns if pat.search(text)}
