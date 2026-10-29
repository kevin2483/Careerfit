# CLAUDE.md — CareerFit 에이전트 팀 운영 규칙

## ① 이 폴더는 무엇을 하는 곳인가
채용공고 1,190건(데이터 사이언티스트/엔지니어/ML엔지니어/데이터분석가)에서 요구 역량을 추출해
직무별 시장 지형도를 만들고, 개인 이력과 대조해 학습 우선순위를 진단한다.
핵심 원칙: 추출 결과를 정성적으로 믿지 않고 자체 정답셋으로 정량 평가한 뒤 판단한다.

## ② 자주 쓰는 자료는 어디에 있나
- 공고 원문: `data/raw/{ds,de,mle,da}/wanted_YYYY-MM-DD.jsonl` (레포 미포함, 최신 2026-07-30)
- 역량 태그 사전: `data/raw/dict/skill_tags_v2.yaml` (106 canonical / 288 alias / 함의 규칙 `implies` 1건: PySpark→Spark) ← v0·v1 아님
  `parent` 필드는 개념 그룹 메타데이터일 뿐 함의 규칙이 아니다. 정답·정규식·LLM 예측은 평가 전에 `skill_dict.close_tags()` 로 같이 닫는다.
- 정답셋: `data/labeled/label_draft.jsonl` (비공개본, 본문 포함, reviewed=true 38건만 유효)
  공개판 `label_draft.public.jsonl` 은 본문만 뺀 것. 비공개본을 고치면 `python -m scripts.make_public_labels` 로 맞춘다.
- 예측 파일: `data/processed/<date>/{regex_extract,llm_extract_*}.jsonl` — 평가는 이 파일만 있으면 된다.
- 경력 구분: career_type은 크롤러 단계 미판정(null). 정제 시 annual_from 사용
  (0=신입가능 / ≥1=경력요구 / None=미기재). annual 상한 100은 "이상" 관례값.
- 공용 코드: `src/common/` — JSONL 읽기(`jsonl.py`), 사전 로딩·정규식(`skill_dict.py`), 경로·날짜(`paths.py`).
  YAML 이나 JSONL 을 다른 곳에서 직접 파싱하지 않는다.

## ③ 결과는 어디에 어떤 형식으로 남기나
- 추출 산출물: `data/processed/job_skill_extract_report/` (CSV 4종, 컬럼·행 순서 고정)
- 평가 산출물: `data/processed/evaluation_{regex_pooled,llm_haiku,llm_sonnet,llm_opus}/eval_summary.csv`
  비교·CI·라우팅·실패민감도·프롬프트변형: `data/processed/{method_comparison,bootstrap_ci,cascade_sim,failed_rows_sensitivity,prompt_variants}.csv`
- 프롬프트를 바꿔 다시 재려면 먼저 `notes/prompt_experiment_*.md` 에 변형·판정 기준을 적고 커밋한 뒤 한 번만 돌린다. 정답셋 38건을 보며 반복 수정하지 않는다.
- `make reproduce` 가 공개 데이터만으로 위 전부를 다시 만들고, `make check` 가 커밋본과 diff 한다. CI 가 매번 돈다.
- 데모(`demo/app.py`, `make demo`)는 위 CSV 와 `gap_diagnosis.run()` 만 읽는다. 데모 안에서 새 통계를 계산하지 않는다(README 와 두 벌이 됨).
- 모든 표는 **항목 · 값 · 근거** 세 칸을 갖춘다. 근거 없는 수치는 내지 않는다.

## ④ 절대 하지 말 것
- **정답셋(gold_tags)을 AI가 판단해 채우지 않는다.** 사람이 확정한 값만 유효하다.
  (AI가 만든 정답으로 AI를 평가하면 순환논리 → 평가 전체가 무효)
- **날짜·파일명을 코드에 하드코딩하지 않는다.** `--date` 인자로 받고, 생략 시 `src/common/paths.resolve_date()` 가 최신 배치를 고른다.
  (2026-07-25 고정 사고 재발 방지)
- **JSONL을 `splitlines()`로 읽지 않는다.** `src/common/jsonl.read_jsonl()` 만 쓴다. 유니코드 줄바꿈에서 레코드가 절단된다.
  (2026-07-25 사고, 이후 같은 위반을 4곳에서 다시 발견해 2026-10-27 통일)
- **커밋된 산출물을 손으로 고치지 않는다.** 스크립트를 고치고 `make reproduce` 로 다시 만든다.

## ⑤ 언제 멈추고 사람에게 물을 것인가
- **사전(skill_tags)을 고쳐야 할 때** — canonical 이름이 바뀌면 정답셋 태그명도 같이 바꿔야 하므로,
  둘을 함께 처리할지 사람에게 확인받고 진행한다. (v1 개정 시 F1이 0.897→0.797로 하락한 사고)
  alias 만 추가·삭제하는 것도 점수를 바꾼다(2026-10-27 alias 충돌 4건 제거로 regex F1 0.843→0.849). 사람 확인 후 전 평가 재실행.
  `implies` 를 추가할 때는 정답셋에 "하위만 태깅된 공고"가 0건인지 먼저 센다. 1건이라도 있으면 사람 판정과 충돌하므로 넣지 않는다
  (2026-10-28: 후보 5쌍 중 PySpark→Spark 만 통과). `tests/test_skill_dict.py` 가 공개 정답셋이 규칙으로 닫혀 있는지 검사한다.
- **검증 표의 산수가 안 맞을 때** — `입력공고수 = 통과 + 모든 스킵/분리 사유`가 깨지면 즉시 중단하고 보고한다.
- **성능 지표가 비정상적으로 높을 때(F1 ≥ 0.98)** — 라벨링 편향이나 평가 누수를 의심하고,
  결과를 쓰기 전에 사람에게 원인 확인을 요청한다.
- **정답셋에 사전 밖 태그가 새로 생겼을 때** — 오타인지 새 역량인지 사람이 가린다. `pool_cli --apply` 가 "[주의] 사전 밖 태그" 로 알린다.
- **채점기가 "[경고] 예측 실패 행" 을 찍을 때** — 잘린 응답·파싱 실패는 모델 성능이 아니라 호출 설정이다. 그 수치를 결론에 쓰기 전에
  `failed_rows_sensitivity.csv` 의 제외 수치와 함께 보고하고, 재호출 여부를 사람이 정한다. (2026-10-28: sonnet 361662 가 max_tokens=512 에서 잘림)

## 판단 기준 (analyst·diagnoser가 쓰는 숫자 기준)
- 어떤 역량의 (canonical, 직무) 출현 건수가 3건 미만이면 → 그 직무의 순위·비교·동시출현·
  신입경력 통계에서 빼고 별도 목록(`excluded_lowfreq.csv`)으로 뺀다. (표본 부족)
- 갭 우선순위 = 출현빈도 × 미보유도(미보유 1 / 부분보유 0.5 / 보유 0) × 신입접근성.
  이 점수 상위 5개만 학습 로드맵에 넣고, 6위 이하는 "다음 분기"로 분리한다.
- 비교하려는 두 직무의 모집단(통과 공고수)이 3배 이상 차이 나면
  (실제: mle 498 vs da 63, 7.9배 — da는 원티드 직군 오염으로 입력 121 중 63만 유효)
  → 절대건수 비교를 금지하고 출현율(%)로만 비교한다.
- 방식 간 F1 차이는 `bootstrap_ci.csv` 의 짝지은 차이 CI 가 0 을 포함하면 "확정 불가"로 쓴다. 점추정만으로 순위를 주장하지 않는다.
