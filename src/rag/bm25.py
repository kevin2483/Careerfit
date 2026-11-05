"""
BM25 키워드 검색 (패키지 없이 직접 구현).

토큰화:
  한국어는 조사가 붙어 "파이프라인을"·"파이프라인의"가 공백 단위로는 다른 단어가 된다.
  형태소 분석기 대신 한글 연속열은 글자 2-gram 으로, 영문·숫자 연속열은 소문자 단어 하나로 쪼갠다.
  ("Spark를 활용한" → spark, 활용, 용한)  — 형태소 분석기와의 비교는 실험 노트에서 변형으로 다룬다.

점수: Okapi BM25, k1=1.5, b=0.75 (일반적 기본값; 정답셋을 보며 튜닝하지 않는다).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[가-힣]+|[a-zA-Z][a-zA-Z0-9+#.]*[a-zA-Z0-9+#]|[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group(0)
        if "가" <= s[0] <= "힣":
            # 한 글자 한글은 대부분 떨어져 나온 조사("Spark를"의 '를')라 버린다. 대가: "앱"·"웹" 단독 질의는 못 찾음
            toks.extend(s[i:i + 2] for i in range(len(s) - 1))
        else:
            toks.append(s.lower())
    return toks


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens) if self.lens else 0.0
        n = len(docs)
        df = Counter(t for tf in self.tfs for t in tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        # 역색인: 질의 토큰이 든 문서만 점수 계산
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tfs[i][t]
                norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
                out[i] = out.get(i, 0.0) + idf * norm
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        return sorted(s.items(), key=lambda x: (-x[1], x[0]))[:k]
