"""사전 v1 의 구조적 불변식: canonical 중복 없음, alias 충돌 없음, 하위 도구가 상위 도구를 자동 생성하지 않음."""
from collections import Counter

from src.common.paths import DICT_PATH
from src.common.skill_dict import alias_index, extract_tags, load_dict, load_patterns


def test_v1_size_and_uniqueness():
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


def test_subtool_does_not_imply_parent_tool():
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
