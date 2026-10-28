# CareerFit — 재현 명령 모음
#   make reproduce   공개 데이터만으로 평가·비교·신뢰구간·라우팅 시뮬레이션을 다시 만든다 (API 호출 없음)
#   make test        단위 테스트
#   make extract     원문(data/raw, 레포 미포함)이 있을 때 추출 → 시장통계
PY ?= python3
DATE ?= 2026-07-30
N_DOCS ?= 892
P := data/processed

.PHONY: install test lint reproduce extract stats check clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

lint:
	$(PY) -m pyflakes src scripts tests

# 레포에 커밋된 예측 파일 4개 + 공개 정답셋만 쓴다. 결과는 커밋된 CSV 와 바이트 단위로 같아야 한다 (CI 가 diff).
reproduce:
	$(PY) -m src.eval.evaluate_extraction --pred-file $(P)/$(DATE)/regex_extract.jsonl --out-dir $(P)/evaluation_regex_pooled --label regex
	$(PY) -m src.eval.evaluate_extraction --pred-file $(P)/$(DATE)/llm_extract_claude-haiku-4-5-20251001.jsonl --out-dir $(P)/evaluation_llm_haiku --label haiku
	$(PY) -m src.eval.evaluate_extraction --pred-file $(P)/$(DATE)/llm_extract_claude-sonnet-5.jsonl --out-dir $(P)/evaluation_llm_sonnet --label sonnet
	$(PY) -m src.eval.evaluate_extraction --pred-file $(P)/$(DATE)/llm_extract_claude-opus-5.jsonl --out-dir $(P)/evaluation_llm_opus --label opus
	$(PY) -m src.eval.bootstrap_ci --date $(DATE)
	$(PY) -m src.eval.compare_methods --date $(DATE) --n-docs $(N_DOCS)
	$(PY) -m src.eval.cascade_sim --date $(DATE) --n-docs $(N_DOCS)
	$(PY) -m src.eval.failed_rows --date $(DATE)

# 재현 결과가 커밋본과 같은지 확인 (CI 와 동일)
check: reproduce
	git diff --exit-code -- $(P)/evaluation_regex_pooled $(P)/evaluation_llm_haiku $(P)/evaluation_llm_sonnet \
		$(P)/evaluation_llm_opus $(P)/bootstrap_ci.csv $(P)/method_comparison.csv $(P)/cascade_sim.csv $(P)/failed_rows_sensitivity.csv
	@echo "재현 일치"

# 아래 둘은 공고 원문(data/raw/*/wanted_$(DATE).jsonl)이 있어야 한다
extract:
	$(PY) -m src.extract.job_skill_extract_report --date $(DATE)
	$(PY) -m src.extract.regex_predict --date $(DATE)

stats:
	$(PY) -m src.analyze.market_stats --date $(DATE)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
