"""한국어 조사·하이픈·C++/C# 경계 처리. boundary.py 의 셀프테스트 케이스를 그대로 pytest 로 돌린다."""
import pytest

from src.extract.boundary import _CASES, compile_alias


@pytest.mark.parametrize("alias,text,want", _CASES, ids=[f"{a}|{t[:24]}" for a, t, _ in _CASES])
def test_boundary_case(alias, text, want):
    assert (compile_alias(alias).search(text) is not None) == want


def test_particle_list_longest_first():
    """alternation 은 앞에 있는 것을 먼저 시도하므로 '으로' 가 '으로써' 보다 앞에 오면 안 된다."""
    from src.extract.boundary import _PARTICLES
    assert _PARTICLES.index("으로써") < _PARTICLES.index("으로")
    assert _PARTICLES.index("에서는") < _PARTICLES.index("에서")
