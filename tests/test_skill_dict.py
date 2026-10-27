"""사전 v2 의 구조적 불변식: canonical 중복 없음, alias 충돌 없음, 정규식은 함의를 모르고 close_tags 만 함의를 안다."""
from collections import Counter

import pytest

from src.common.jsonl import read_jsonl
from src.common.paths import DICT_PATH, LABELS_PUBLIC, RAW_DIR
from src.common.skill_dict import alias_index, close_tags, extract_tags, implies_map, load_dict, load_patterns


def test_v2_size_and_uniqueness():
    tags = load_dict(DICT_PATH)
    assert len(tags) == 106
    assert max(Counter(t["canonical"] for t in tags).values()) == 1


def test_no_alias_collision(capsys):
    """같은 alias 가 두 canonical 에 걸리면 그 문자열은 항상 두 태그를 동시에 만든다 (2026-10-27 수정)."""
    owners = {}
    for t in load_dict(DICT_PATH):
        for a in [t["canonical"].lower(), *[x.lower() for x in t["aliases"]]]:
            assert owners.setdefault(a, t["canonical"]) == t["canonical"], f"alias 충돌: {a}"
    assert "alias 충돌" not in capsys.readouterr().out


def test_regex_alone_does_not_imply_parent_tool():
    """정규식 단계는 하위 도구만 낸다. 함의는 close_tags 가 따로 적용한다 (두 단계를 섞지 않아야 v1 재현이 된다)."""
    pats = load_patterns(DICT_PATH)
    assert extract_tags("PySpark 기반 배치 처리", pats) == {"PySpark"}
    assert extract_tags("MariaDB 운영 경험", pats) == {"MariaDB"}
    assert extract_tags("ArgoCD 로 배포", pats) == {"ArgoCD"}
    assert extract_tags("Spark, MySQL, CI/CD 파이프라인", pats) >= {"Spark", "MySQL", "CI/CD"}


def test_alias_index_normalizes_free_text():
    idx = alias_index(DICT_PATH)
    assert idx["fastapi"] == "FastAPI"
    assert idx["파이썬"] == "Python"
    assert idx["python"] == "Python"


def test_v2_implies_is_only_pyspark_to_spark():
    """v2 의 함의 규칙은 PySpark→Spark 하나. 정답셋이 하위-only 를 정답으로 인정한 쌍은 넣지 않는다."""
    assert implies_map(DICT_PATH) == {"PySpark": ["Spark"]}
    assert implies_map(RAW_DIR / "dict" / "skill_tags_v1.yaml") == {}          # v1 은 규칙 없음 → 닫기가 항등


def test_close_tags_transitive_and_identity():
    imp = {"A": ["B"], "B": ["C"]}
    assert close_tags({"A"}, imp) == {"A", "B", "C"}
    assert close_tags({"C"}, imp) == {"C"}
    assert close_tags({"PySpark"}, implies_map(DICT_PATH)) == {"PySpark", "Spark"}
    assert close_tags({"MariaDB"}, implies_map(DICT_PATH)) == {"MariaDB"}
    assert close_tags(["X"], None) == {"X"} and close_tags([], {}) == set()


def test_implies_target_must_be_canonical(tmp_path):
    bad = tmp_path / "d.yaml"
    bad.write_text("cat:\n  - canonical: A\n    aliases: [a]\n    field: 1\n    implies: [없는태그]\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="implies 대상이 사전에 없음"):
        load_dict(bad)


def test_public_gold_is_closed_under_implies():
    """사람 정답셋이 이미 함의 규칙을 만족한다 → 평가 시 닫기는 정답을 바꾸지 않는다 (AI 가 정답을 채우지 않음)."""
    imp = implies_map(DICT_PATH)
    for r in read_jsonl(LABELS_PUBLIC):
        if r.get("reviewed"):
            assert close_tags(r["gold_tags"], imp) == set(r["gold_tags"]), r["job_id"]
