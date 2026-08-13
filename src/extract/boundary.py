r"""
CareerFit — alias 매칭 경계 패턴 v2 (한글 조사 대응)

위치: src/extract/boundary.py
교체 지점: job_skill_extract_report.py 의 compile_patterns()

    # before
    pat = re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)
    # after
    from src.extract.boundary import build_alias_pattern
    pat = re.compile("|".join(build_alias_pattern(a) for a in alts), re.IGNORECASE)

    ※ alts 는 re.escape 하지 말 것. build_alias_pattern 이 내부에서 한다.

현행 \b 방식의 실측 결함 (정답셋 38건 기준)
--------------------------------------------
1. 조사 결합 미매칭 — "SQL을", "대시보드를", "대용량 데이터를" 이 전부 씹힌다.
   \b 는 \w 기준이고 파이썬 \w 는 한글을 포함하므로 조사가 붙으면 경계가 사라진다.
   한글 alias(대시보드, 대용량 데이터 …)가 특히 심하게 당한다 = field 2 집중 타격.
2. C++ / C# 미매칭 — r"\bc\+\+\b" 는 뒤쪽 \b 가 '+' 다음에 단어문자를 요구한다.
   따라서 "C++ 기반", "MISRA C++\n" 처럼 뒤에 공백·개행이 오면 매칭되지 않는다.
3. 오탐 — "Node.js" 안의 js 가 JavaScript 로, "C++" 안의 C 가 C 로 잡힌다.

규칙
----
  len(alias) >= 3 : 뒤에 한글이 오면 무조건 허용        (Python으로, MLOps에서)
  len(alias) <= 2 : 뒤에 한글이 오면 '조사'일 때만 허용  (R을 O / R&D X / C레벨 X)

경계 문자 집합 설계 (v1에서 수정한 부분)
  '-' 제외  : 하이픈 복합어를 살린다 (TensorRT-LLM, docker-swarm, LLM-as-a-judge)
  '.' 은 경계 문자로 쓰지 않고 _DOT_GUARD 로 대체 : "Node.js"의 js, "llama.cpp"의 cpp 는
     막되 "2.Python", "End-to-End." 같은 문장부호·번호는 통과시킨다
  '+' '#' 유지 : C++ / C# 과 그 안의 단일문자 C 를 구분하기 위해 필수
  영숫자로 끝나지 않는 alias는 후행 숫자 허용 : C++17/20 이 C++ 로 잡혀야 한다

"C언어", "Go언어" 처럼 조사가 아닌 명사가 붙는 복합어는 정규식으로 풀지 말고
사전에 alias 로 직접 추가할 것. alias 추가는 canonical 이름을 바꾸지 않으므로
정답셋 태그명을 갱신할 필요가 없다 (canonical 개정과 다르다).

    python -m src.extract.boundary      # 셀프테스트
"""

from __future__ import annotations

import re

_BEFORE = r"A-Za-z0-9+#_"
_AFTER = r"A-Za-z0-9+#_"
_BEFORE_SHORT = r"A-Za-z0-9+#_&"    # 짧은 alias는 &도 막는다 (R&D 방어)
_AFTER_SHORT = r"A-Za-z0-9+#_&"
# 식별자 내부의 점만 막는다: "Node.js"의 js, "llama.cpp"의 cpp 는 차단하고
# "2.Python" 같은 번호 목록은 통과시킨다 (점 앞이 '문자'일 때만 경계로 본다).
_DOT_GUARD = r"(?<![A-Za-z]\.)"

# 긴 것부터 나열해야 alternation이 최장 일치한다 (으로 < 으로써)
_PARTICLES = [
    "으로써", "으로서", "이라고", "이라는", "이라면", "에서는", "에서도",
    "으로", "이라", "이랑", "이나", "이며", "부터", "까지", "처럼", "보다",
    "에서", "에게", "라고", "라는", "와의", "과의", "만을", "만이",
    "은", "는", "이", "가", "을", "를", "와", "과", "로", "에", "의",
    "도", "만", "랑", "나", "며", "및", "등", "라", "임",
]
_PARTICLE_ALT = "(?:" + "|".join(_PARTICLES) + ")"

SHORT_ALIAS_LEN = 2


def build_alias_pattern(alias: str) -> str:
    """alias 하나에 대한 정규식 문자열. re.IGNORECASE 로 컴파일해서 쓸 것."""
    body = re.escape(alias)
    # C++ / C# 처럼 alias가 영숫자로 끝나지 않으면 뒤에 숫자가 와도 같은 토큰이다.
    # (C++17, C#9 …) 이 경우에만 후행 경계에서 숫자를 뺀다.
    digits = "0-9" if alias[-1].isalnum() else ""
    if len(alias) > SHORT_ALIAS_LEN:
        after = f"A-Za-z{digits}+#_"
        return rf"(?<![{_BEFORE}]){_DOT_GUARD}{body}(?![{after}])"
    after = f"A-Za-z{digits}+#_&"
    return (
        rf"(?<![{_BEFORE_SHORT}]){_DOT_GUARD}{body}(?![{after}])"
        rf"(?:(?![가-힣])|(?={_PARTICLE_ALT}))"
    )


def compile_alias(alias: str) -> re.Pattern:
    return re.compile(build_alias_pattern(alias), re.IGNORECASE)


# ---------------------------------------------------------------- 셀프테스트
# 실제 공고 원문에서 뽑은 케이스 포함 (출처: data/labeled/label_draft.jsonl)

_CASES = [
    # 조사 결합 — 이번 패치의 주목적
    ("Python", "Python으로 개발합니다", True),
    ("SQL", "SQL을 활용한 데이터 추출 및 분석 경험", True),          # ds/373409 실제 원문
    ("SQL", "SQL과 Python을 활용해 대용량 데이터를 추출", True),      # da/282256 실제 원문
    ("대시보드", "대시보드를 설계하고 운영합니다", True),               # da/351569 실제 원문
    ("대용량 데이터", "대용량 데이터를 추출·정제·가공", True),
    ("MLOps", "MLOps에서의 경험", True),
    ("임베딩", "임베딩을 활용한 검색", True),
    # 하이픈 복합어 — v1에서 깨졌던 것
    ("llm", "TensorRT-LLM. Promote the results", True),
    ("llm", "LLM-as-a-judge 기반의 자동화된 평가", True),
    ("docker", "kubernetes/docker-swarm 등 환경에서의 운영", True),
    ("end-to-end", "운영 고도화까지 End-to-End.)", True),
    # C++ / C# — 현행 \b 가 놓치던 것
    ("c++", "C++ 기반 임베디드 및 실시간 소프트웨어 개발", True),
    ("c++", "ISO 26262, MISRA C++ 등 Automotive", True),
    ("c++", "C++17/20 · Python · CUDA", True),
    ("c#", "C# 개발 경험", True),
    # 번호 목록의 점 — 경계로 오인하면 안 된다
    ("Python", "2.Python 기반 표준 ML 스택 능숙", True),
    # 회귀 방지
    ("Python", "Python", True), ("Python", "(Python)", True), ("Python", "Python, SQL", True),
    ("파이썬", "파이썬을 사용", True),
    ("R", "R을 활용한 통계 분석", True), ("R", "R,", True),
    ("Go", "Go로 작성된 서버", True), ("C", "C 언어", True),
    # 막아야 하는 것
    ("Python", "Python3", False), ("Python", "MicroPython", False),
    ("js", "API, Django, Flask, Node.js 등으로", False),   # JavaScript 오탐 차단
    ("cpp", "vLLM, SGLang, llama.cpp 등 LLM Inference", False),  # C++ 오탐 차단
    ("C", "C++ 기반 임베디드", False),                      # C 오탐 차단
    ("Go", "Google Cloud", False), ("Go", "Django", False),
    ("R", "R&D 부서", False), ("R", "Redis", False),
    ("C", "C레벨 임원", False), ("AI", "AIML", False),
    # 알려진 미매칭: 조사가 아닌 명사 결합 -> 사전 alias 로 처리
    ("C", "C언어 경험", False), ("Go", "Go언어", False),
]


def _selftest() -> int:
    fails = []
    for alias, text, want in _CASES:
        got = compile_alias(alias).search(text) is not None
        if got != want:
            fails.append((alias, text, want, got))
            print(f"[FAIL] alias={alias!r} text={text!r} want={want} got={got}")
    if fails:
        print(f"\n{len(fails)}/{len(_CASES)}건 실패")
        return 1
    print(f"{len(_CASES)}건 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
