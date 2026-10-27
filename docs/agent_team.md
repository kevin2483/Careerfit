# 에이전트 팀 설계와 검증 기록

Claude Code 서브에이전트 3명이 직렬로 동작한다. 이 문서는 "왜 이렇게 나눴는가"와 "실제로 재현되는가"를 적는다.
팀 운영 절차는 [`.claude/skills/careerfit팀-돌리기/SKILL.md`](../.claude/skills/careerfit팀-돌리기/SKILL.md), 각 담당자 정의는 [`.claude/agents/`](../.claude/agents/)에 있다.

## 담당자와 손의 크기

| 담당자 | 하는 일 | 호출하는 스크립트 | 쓰는 곳 | tools |
|---|---|---|---|---|
| extractor | 공고에서 역량 태그 추출, 4종 산출물 | `src/extract/job_skill_extract_report.py` | `data/processed/job_skill_extract_report/` | Read, Bash |
| analyst | 직무별 순위·동시출현·신입vs경력 통계 | `src/analyze/market_stats.py` | `data/processed/market_stats/` | Read, Bash |
| diagnoser | 시장 통계와 이력서 대조, 학습 우선순위 top5 | `src/analyze/gap_diagnosis.py` | 파일 미기록, 표준출력만 | Read, Bash |
| 리더(메인 세션) | 이력서 읽기(Google Drive MCP), 결과 기록(Notion MCP) | — | Notion | 전부 |

세 담당자 모두 `Write`/`Edit` 도구가 없다. **다만 이는 도구 권한이지 강제 규약이 아니다.** Bash가 있으므로 프롬프트를 벗어난 임의 쓰기를 도구 차원에서 막지는 못한다. 실제 안전판은 둘이다.

1. 각 담당자는 정해진 스크립트의 `run()` 하나만 호출한다. 담당자 정의에 그 호출문이 그대로 적혀 있다.
2. 그 스크립트는 자기 `out_dir`에만 쓴다. 판단 기준(3건 미만 제외, 출현율 비교, 우선순위 산식, top5 분리)은 담당자의 프롬프트가 아니라 **스크립트 코드에 상수로 고정**돼 있어, 담당자가 "감으로" 순위를 바꿀 여지가 없다.

## 순서 의존성은 코드가 강제한다

extractor → analyst → diagnoser 순서가 깨지면 조용히 틀린 결과가 나온다. 담당자에게 "먼저 확인하라"고 시키는 대신 스크립트가 멈춘다.

- `market_stats.check_inputs_fresh()`: extractor 산출물이 없거나 원본 공고보다 오래되면 `SystemExit`
- `gap_diagnosis.check_inputs_fresh()`: analyst 산출물이 없거나 extractor 산출물보다 오래되면 `SystemExit`
- 담당자는 `SystemExit` 메시지를 지어내지 않고 그대로 리더에게 전달하도록 정의돼 있다.

## 재현성 검증 (2026-08-21 실측, 2026-10-27 재확인)

같은 입력으로 두 번 돌려 바이트 단위로 같은지 봤다. 말로 확인하지 않고 실제로 두 번 돌렸다.

| 대상 | 1회 | 2회 | 차이 |
|---|---|---|---|
| extractor (date=2026-07-30) | 통과 1,132 · 직무불일치 58 | 동일 | 4개 산출물 바이트 일치 |
| analyst (date=2026-07-30) | 모집단 ds218·de353·mle498·da63, 제외 52건 | 동일 | 4개 산출물 바이트 일치 |
| diagnoser (target_job=da, 같은 이력서) | top5·점수·후순위·보유역량 | 동일 | 완전 일치 |

이상 입력은 합성 데이터로 넣어 봤다. 정상 1 · 필드누락 1 · 본문결측 1 · JSON 파싱실패 1 · 직무불일치 1 · 원본 파일 자체가 없는 직무 2 — 크래시 없이 각 스킵 사유로 집계되고 불변식 `입력 = 통과 + 모든 스킵 사유`가 유지된다. 이 케이스는 [`tests/test_extract_report.py`](../tests/test_extract_report.py)에 있어 CI에서 매번 돈다.

## 사고에서 추가된 기준

- **field 3(암묵조건) 태그는 학습 로드맵에서 제외.** 2026-08-12 재현성 점검에서 "협업커뮤니케이션"이 로드맵 2위로 올라왔다. 거의 모든 공고에 등장해 변별력이 없고 "배울 수 있는 항목"도 아니다. `gap_diagnosis.py`가 사전의 field를 읽어 제외한다.
- **top5에 배울 수 없는 항목(도메인 지식·자격증·근무조건)이 오르면 노션 기록 전에 사람이 확인한다.** 태그로는 잡히지만 학습 대상이 아닌 것이 상위에 뜰 수 있다. 이건 코드로 못 거르므로 멈춤 지점으로 둔다.
- **사전을 고칠 때는 정답셋 태그명을 함께 본다.** v0→v1 개정 때 정답셋을 갱신하지 않아 개선이 하락(0.897→0.797)으로 보였다.

## 한계

- 이력서가 외부 드라이브에 있어 레포만으로는 갭 진단이 재현되지 않는다. `python -m src.analyze.gap_diagnosis --job de --resume-file <파일>`로 로컬 텍스트를 넣으면 같은 산식으로 돈다.
- `check_inputs_fresh()`는 파일 mtime을 본다. git clone 직후에는 mtime이 체크아웃 시각이라 순서 보장이 약하다. 원문을 다시 수집해 extractor부터 돌리는 흐름에서는 문제없다.
- 담당자 간 데이터 전달은 파일이다. 같은 `date`를 셋이 공유해야 하며, 다르면 analyst가 다른 배치의 공고에 annual_from을 조인한다. `--date`를 생략하면 세 스크립트 모두 `data/raw`의 최신 배치를 같은 규칙으로 고르므로 기본값으로는 어긋나지 않는다.
