"""
CareerFit 데모 — 커밋된 산출물만으로 동작한다 (공고 원문·API 키 불필요).

    make demo            # 로컬
    streamlit run demo/app.py

탭 1 시장 지형도: data/processed/market_stats/*.csv
탭 2 갭 진단:     src.analyze.gap_diagnosis.run() 을 그대로 호출 (산식 중복 없음)
탭 3 평가 결과:   method_comparison · bootstrap_ci · cascade_sim · failed_rows_sensitivity
탭 4 공고 검색:   rag_comparison · rag_generation (검색 자체는 원문이 비공개라 여기서 돌지 않는다)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # src/common/paths 의 상대 경로 기준

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from demo import loaders as L  # noqa: E402
from demo.pdf_text import extract_text  # noqa: E402
from src.analyze import gap_diagnosis as gd  # noqa: E402

st.set_page_config(page_title="CareerFit", page_icon="📊", layout="wide")

SAMPLE_RESUME = (ROOT / "demo" / "sample_resume.txt").read_text(encoding="utf-8")
JOB_OPTIONS = list(L.JOB_NAMES)


def job_label(j: str) -> str:
    return f"{L.JOB_NAMES[j]} ({j})"


@st.cache_data
def df(name: str, *args) -> pd.DataFrame:
    return pd.DataFrame(getattr(L, name)(*args))


st.title("CareerFit")
st.caption("채용공고 1,190건에서 직무별로 요구하는 역량을 뽑았습니다. 탭 순서대로 ① 어떤 역량이 많이 요구되는지 ② 내 이력서에 무엇이 없는지 "
           "③ 그 추출이 얼마나 정확한지 ④ 공고 검색(RAG)을 어떻게 평가했는지 볼 수 있습니다. 모든 숫자는 레포에 커밋된 CSV 그대로입니다.")

tab_market, tab_gap, tab_eval, tab_rag = st.tabs(["시장 지형도", "갭 진단", "평가 결과", "공고 검색 (RAG)"])

# ───────────────────────────────────────────── 탭 1
with tab_market:
    pop = L.population()
    job = st.radio("직무", JOB_OPTIONS, format_func=job_label, horizontal=True, key="job_market")
    p = pop[job]
    c1, c2, c3 = st.columns(3)
    c1.metric("수집한 공고", f"{p['입력공고수']:,}")
    c2.metric("통계에 쓴 공고 (통과)", f"{p['통과']:,}")
    c3.metric("직무가 달라 뺀 공고", f"{p['직무불일치']:,}")
    note = L.da_note(job)
    if note:
        st.info(note)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("요구 역량 상위 20")
        rank = df("rank_by_job", job, 20)
        chart = alt.Chart(rank).mark_bar().encode(
            x=alt.X("출현율%:Q", title="이 역량을 요구하는 공고 (%)"),
            y=alt.Y("canonical:N", sort="-x", title=None),
            tooltip=["순위", "canonical", "출현수", "모집단수", "출현율%"],
        ).properties(height=520)
        st.altair_chart(chart, use_container_width=True)
        st.caption("막대 길이 = 이 직무 공고 100건 중 그 역량을 요구한 건수. 분모는 위의 '통과' 공고 수이고, 3건 미만으로 나온 역량은 표본이 부족해 뺐습니다(아래 목록).")
    with right:
        st.subheader("그 역량을 요구한 공고에 신입도 지원할 수 있나")
        jvs = df("junior_vs_senior", job)
        top_names = list(rank["canonical"])
        jvs = jvs[jvs["canonical"].isin(top_names)].copy()
        jvs["순위"] = jvs["canonical"].map({n: i for i, n in enumerate(top_names)})
        sc = alt.Chart(jvs.sort_values("순위")).mark_bar().encode(
            x=alt.X("신입가능%:Q", title="신입가능 공고 비율 (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("canonical:N", sort=top_names, title=None),
            color=alt.Color("신입가능%:Q", legend=None, scale=alt.Scale(scheme="blues")),
            tooltip=["canonical", "신입가능", "경력요구", "미기재", "신입가능%", "경력요구%"],
        ).properties(height=520)
        st.altair_chart(sc, use_container_width=True)
        st.caption("그 역량을 요구한 공고 중 신입 지원이 가능한(요구 경력 0년) 공고의 비율. 갭 진단 탭이 순서를 정할 때 이 값을 씁니다.")

        st.subheader("함께 요구되는 역량 쌍 상위 15")
        st.dataframe(df("cooccurrence", job, 15)[["tag_a", "tag_b", "동시출현수"]].rename(columns={"tag_a": "역량 A", "tag_b": "역량 B", "동시출현수": "함께 나온 공고 수"}), hide_index=True, use_container_width=True)

    with st.expander(f"표본 부족으로 제외한 역량 ({len(L.excluded_lowfreq(job))}건)"):
        st.dataframe(df("excluded_lowfreq", job)[["canonical", "출현수", "제외사유"]].rename(columns={"canonical": "역량"}), hide_index=True, use_container_width=True)

# ───────────────────────────────────────────── 탭 2
RENAME = {"순위": "순위", "canonical": "역량", "출현율%": "요구하는 공고 (%)", "신입가능%": "그중 신입 가능 (%)", "우선순위점수": "우선순위 점수"}


def roadmap_table(rows: list[dict]) -> pd.DataFrame:
    """run() 결과를 읽기 쉬운 열 이름으로 바꾼다. 신입접근성(0~1)은 % 로만 바꿔 보여 준다 (값은 그대로)."""
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    t["신입가능%"] = (t["신입접근성"] * 100).round(1)
    return t[list(RENAME)].rename(columns=RENAME)


with tab_gap:
    st.subheader("내 이력서에 없는데 공고가 자주 요구하는 역량은?")
    st.markdown("**① 지원할 직무를 고르고 ② 이력서를 넣으면 ③ 먼저 채울 역량 5개를 알려 줍니다.** "
                "이력서는 서버에 저장하지 않습니다.")
    job2 = st.radio("① 지원할 직무", JOB_OPTIONS, format_func=job_label, horizontal=True, key="job_gap", index=1)
    if "resume" not in st.session_state:
        st.session_state["resume"] = ""
    b1, b2 = st.columns([1, 3])
    if b1.button("예시 이력서로 해 보기", type="primary"):
        st.session_state["resume"] = SAMPLE_RESUME
    up = b2.file_uploader("② 이력서·포트폴리오 PDF 또는 txt 올리기", type=["pdf", "txt"], label_visibility="visible")
    if up is not None and st.session_state.get("_uploaded") != up.name:
        text, note = extract_text(up.getvalue(), up.name)
        st.session_state["_uploaded"] = up.name
        if text:
            st.session_state["resume"] = text
            st.success(note)
        else:
            st.warning(note)
    resume = st.text_area("또는 여기에 붙여 넣기", key="resume", height=180, placeholder="경력·기술·프로젝트를 붙여 넣으세요.")
    if not resume.strip():
        st.info("이력서를 넣으면 결과가 여기에 나옵니다. 먼저 보고 싶으면 '예시 이력서로 해 보기'를 누르세요.")
    else:
        r = gd.run(resume_text=resume, target_job=job2, check_fresh=False)
        name = L.JOB_NAMES[job2]
        have = set(r["보유역량"])
        skipped = {x["canonical"] for x in r["제외_field3_암묵조건"]}
        top = r["이번분기_학습로드맵(top5)"]

        # 상위 20 역량에 보유 여부만 붙인다. 순위·비율은 시장 지형도 탭과 같은 CSV 값이다.
        top20 = df("rank_by_job", job2, 20).copy()
        top20["상태"] = ["공통 소양 (판정 안 함)" if c in skipped else "이력서에 있음" if c in have else "이력서에 없음"
                         for c in top20["canonical"]]
        judged = top20[top20["상태"] != "공통 소양 (판정 안 함)"]
        n_have = int((judged["상태"] == "이력서에 있음").sum())

        st.divider()
        st.markdown(f"### ③ {name} 공고가 자주 요구하는 역량 {len(judged)}개 중 **{n_have}개**가 이력서에 있습니다")
        if not have:
            st.warning("이력서에서 역량을 하나도 읽지 못했습니다. 기술 이름이 글자로 적혀 있는지 확인하세요 (이미지로 된 PDF 는 읽지 못합니다).")

        st.markdown("**이력서에서 읽은 역량**")
        st.markdown(" ".join(f"`{c}`" for c in r["보유역량"]) or "없음")
        st.caption("갖고 있는데 여기 없다면 이력서에 그 단어가 없거나 다르게 적힌 것입니다. 단어를 찾는 방식이라 숙련도는 구분하지 못합니다.")

        st.markdown(f"**{name} 공고가 많이 요구하는 역량 20개와 내 이력서**")
        status_color = alt.Scale(domain=["이력서에 있음", "이력서에 없음", "공통 소양 (판정 안 함)"], range=["#2563eb", "#f97316", "#cbd5e1"])
        gap_chart = alt.Chart(top20).mark_bar().encode(
            x=alt.X("출현율%:Q", title="이 역량을 요구하는 공고 (%)"),
            y=alt.Y("canonical:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=160)),
            color=alt.Color("상태:N", scale=status_color, legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("canonical", title="역량"), alt.Tooltip("출현율%", title="요구하는 공고 (%)"), "상태"],
        ).properties(height=alt.Step(26))
        st.altair_chart(gap_chart, use_container_width=True)
        st.caption("주황색 막대가 길수록 많이 요구되는데 이력서에 없는 역량입니다. 회색은 협업·커뮤니케이션처럼 거의 모든 공고에 나와 따로 판정하지 않는 항목입니다.")

        st.markdown("### 먼저 채울 역량 5개")
        st.caption("많은 공고가 요구하고, 그 공고에 신입도 지원할 수 있을수록 앞에 옵니다.")
        if not top:
            st.success("이 직무에서 통계에 잡힌 역량은 이력서에 모두 있습니다.")
        for row in top:
            with st.container(border=True):
                a, b = st.columns([3, 2])
                a.markdown(f"**{row['순위']}. {row['canonical']}**")
                a.markdown(f"{name} 공고 100건 중 **{row['출현율%']:.0f}건**이 요구합니다. "
                           f"그 공고 가운데 신입이 지원할 수 있는 비율은 **{row['신입접근성'] * 100:.0f}%** 입니다.")
                b.progress(min(row["출현율%"] / 100, 1.0), text=f"요구하는 공고 {row['출현율%']:.1f}%")

        with st.expander(f"그다음에 볼 역량 ({len(r['다음분기_후순위'])}개)"):
            st.dataframe(roadmap_table(r["다음분기_후순위"]), hide_index=True, use_container_width=True)
        with st.expander("순서는 어떻게 정하나요?"):
            st.markdown("우선순위 점수 = **요구하는 공고 비율(%) × 그중 신입 가능 비율**. 이력서에 이미 있는 역량은 0점으로 빠집니다. "
                        "점수가 높은 5개만 '먼저 채울 역량'으로 보여 주고 나머지는 '그다음'으로 넘깁니다.")
            st.dataframe(roadmap_table(top), hide_index=True, use_container_width=True)
            st.caption(r["미보유가중치_방식"])
        with st.expander("순서에서 뺀 공통 소양"):
            st.write(r["field3_제외기준"])
            st.dataframe(pd.DataFrame(r["제외_field3_암묵조건"]).rename(columns=RENAME), hide_index=True, use_container_width=True)

        st.markdown("**이 결과를 읽을 때 알아 둘 점**")
        st.markdown("- 이력서에 적히지 않은 역량은 없는 것으로 봅니다. 최근 경험을 안 적었다면 결과도 그만큼 어긋납니다.\n"
                    "- 단어가 한 번이라도 나오면 '있음'입니다. 써 본 정도와 깊이는 구분하지 못합니다.\n"
                    f"- 기준은 2026년 7월에 모은 {name} 공고입니다. 공고 수와 제외 기준은 '시장 지형도' 탭에 있습니다.")

# ───────────────────────────────────────────── 탭 3
with tab_eval:
    st.subheader("추출이 맞는지 잰 결과 — 정답셋 38건, 공고 단위 부트스트랩 10,000회")
    tot = df("totals")
    ci = pd.DataFrame([r for r in L.bootstrap_ci() if r["지표"] == "F1"])
    ci["방식명"] = ci["방식"].map(L.METHOD_NAMES)
    order = list(tot.sort_values("F1", ascending=False)["방식명"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**F1과 95% 신뢰구간**")
        base = alt.Chart(ci).encode(y=alt.Y("방식명:N", sort=order, title=None))
        bars = base.mark_rule(strokeWidth=3).encode(x=alt.X("CI_하한:Q", title="F1", scale=alt.Scale(domain=[0.75, 0.92])), x2="CI_상한:Q")
        pts = base.mark_point(filled=True, size=120).encode(x="점추정:Q", tooltip=["방식명", "점추정", "CI_하한", "CI_상한"])
        st.altair_chart((bars + pts).properties(height=180), use_container_width=True)
        st.caption("구간이 겹친다. 확정되는 것은 'haiku 는 opus·sonnet 보다 낮다'뿐이다 (아래 짝지은 차이).")
    with c2:
        st.markdown("**정확도 대비 비용 (892건 전체 처리 환산)**")
        par = alt.Chart(tot).mark_point(filled=True, size=160).encode(
            x=alt.X("전체비용:Q", title="전체 비용 (USD)"), y=alt.Y("F1:Q", scale=alt.Scale(domain=[0.80, 0.88])),
            color=alt.Color("방식명:N", legend=None), tooltip=["방식명", "F1", "precision", "recall", "전체비용"],
        )
        txt = par.mark_text(align="left", dx=8).encode(text="방식명:N")
        st.altair_chart((par + txt).properties(height=180), use_container_width=True)
        st.caption("파레토 최적: 정규식($0) · sonnet · opus. haiku 는 더 싸고 더 정확한 정규식에 지배당한다.")

    st.markdown("**방식별 요약**")
    show = tot[["방식명", "F1", "F1_CI95_하한", "F1_CI95_상한", "precision", "recall", "TP", "FP", "FN", "전체비용"]]
    st.dataframe(show, hide_index=True, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**필드 난이도별 F1** — 명시 기술은 네 방식이 동률, 차이는 서술형에서")
        ff = df("field_f1")
        g = alt.Chart(ff).mark_bar().encode(
            x=alt.X("방식:N", title=None, sort=order), y=alt.Y("F1:Q", scale=alt.Scale(domain=[0.6, 1.0])),
            column=alt.Column("필드:N", title=None), color=alt.Color("방식:N", legend=None), tooltip=["방식", "필드", "F1"],
        ).properties(height=200, width=180)
        st.altair_chart(g)
    with c4:
        st.markdown("**짝지은 차이 (같은 리샘플)** — CI 가 0 을 포함하면 확정 불가")
        pdiff = pd.DataFrame(L.paired_diffs())[["방식", "점추정", "CI_하한", "CI_상한", "P(차이>0)"]]
        pdiff["확정"] = ["예" if (lo > 0 or hi < 0) else "아니오" for lo, hi in zip(pdiff["CI_하한"], pdiff["CI_상한"])]
        st.dataframe(pdiff, hide_index=True, use_container_width=True)

    st.markdown("**라우팅 시뮬레이션** — 일부 공고만 opus 로 보내면? (무작위 라우팅 1,000회와 비교)")
    rc = df("routing_curves")
    rc["lo"], rc["hi"] = rc["F1"] - rc["F1_sd"], rc["F1"] + rc["F1_sd"]
    y_scale = alt.Scale(domain=[0.84, 0.87])
    line = alt.Chart(rc).mark_line(point=True).encode(
        x=alt.X("명목비율%:Q", title="opus 로 보낸 비율 (%)"), y=alt.Y("F1:Q", scale=y_scale),
        color=alt.Color("정책:N", title="라우팅 기준"), tooltip=["정책", "명목비율%", "F1", "F1_sd", "전체비용"],
    )
    band = alt.Chart(rc[rc["정책"] == "random"]).mark_area(opacity=0.15, color="#dc2626").encode(
        x="명목비율%:Q", y=alt.Y("lo:Q", scale=y_scale, title="F1"), y2="hi:Q",
    )
    st.altair_chart((band + line).properties(height=260), use_container_width=True)
    st.caption("길이 기준 40%는 무작위 40%보다 +1.1σ. 38건으로는 '긴 공고를 고르는 것'의 효과를 구분하지 못한다. 음영 = 무작위 ±1σ.")

    st.markdown("**haiku 프롬프트 변형 실험 — 사전 등록 후 각 1회, 전부 실패**")
    pv = df("prompt_variants")
    pv_f1 = pv[pv["지표"] == "F1"][["변형", "F1", "CI_하한", "CI_상한", "precision", "recall", "F1_①명시", "F1_②서술형", "환각태그", "파싱실패", "FP"]]
    pv_d = pv[pv["지표"] == "F1 차이"][["변형", "F1", "CI_하한", "CI_상한", "P(차이>0)"]].rename(columns={"F1": "차이"})
    cc1, cc2 = st.columns([3, 2])
    cc1.dataframe(pv_f1, hide_index=True, use_container_width=True)
    cc2.dataframe(pv_d, hide_index=True, use_container_width=True)
    st.caption("alias = 서술형 태그에 사전 표현 예 추가 · criteria = 사람이 쓴 라벨링 기준 삽입 · schema = 구조화 출력(enum 106). "
               "세 변형 모두 CI 가 0 을 포함해 확정 불가이며 점추정은 하락. schema 는 환각 0 이지만 목록 안 오탐이 늘었다. 변형을 고쳐 재실행하지 않았다.")

    st.markdown("**평가 장치 자체를 네 번 고쳤다**")
    st.dataframe(pd.DataFrame(L.CORRECTIONS), hide_index=True, use_container_width=True)
    fr = df("failed_rows")
    n_failed = sum(1 for r in L.failed_rows() if r["구분"] == "전체" and r["제외문서"])
    st.caption(f"예측 실패 행(출력 잘림·파싱 실패) 민감도 검사: 현재 실패 행 {n_failed}건. 자세한 경위는 레포 README 와 notes/lab_log.md.")

# ───────────────────────────────────────────── 탭 4
with tab_rag:
    st.subheader("공고 892건을 문단 4,573개로 나눠 검색하고, 찾은 문단만 근거로 답한다")
    st.info("이 탭은 검색창이 아니라 평가 결과다. 공고 원문이 비공개라 공개 데모에서는 검색을 돌릴 수 없다. "
            "방식·파라미터·판정 기준을 실행 전에 문서로 고정하고 방식마다 한 번만 돌렸다 (notes/rag_experiment_2026-11-05.md).")
    rm = df("rag_methods")
    rag_order = list(rm.sort_values("Recall@5", ascending=False)["방식"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Recall@5 와 95% 신뢰구간** — 질문 24개, 직접 판정 493건")
        base = alt.Chart(rm).encode(y=alt.Y("방식:N", sort=rag_order, title=None))
        rule = base.mark_rule(strokeWidth=3).encode(x=alt.X("Recall@5_하한:Q", title="Recall@5", scale=alt.Scale(domain=[0.15, 0.45])), x2="Recall@5_상한:Q")
        dot = base.mark_point(filled=True, size=120).encode(x="Recall@5:Q", tooltip=["방식", "Recall@5", "Recall@5_하한", "Recall@5_상한"])
        st.altair_chart((rule + dot).properties(height=220), use_container_width=True)
        st.caption("판정한 문단의 65%가 정답이라 질문당 정답이 5개를 넘는 경우가 많다. 절대값보다 방식 간 차이로 읽는다.")
    with c2:
        st.markdown("**검색 지연 p90 (로그 눈금)** — 목표 1,000ms")
        lat = alt.Chart(rm).mark_bar().encode(
            y=alt.Y("방식:N", sort=rag_order, title=None),
            x=alt.X("지연_p90_ms:Q", title="p90 (ms)", scale=alt.Scale(type="log", domain=[5, 10000])),
            tooltip=["방식", "지연"])
        goal = alt.Chart(pd.DataFrame({"x": [1000]})).mark_rule(color="#dc2626", strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((lat + goal).properties(height=220), use_container_width=True)
        st.caption("BM25 는 13ms 로 하이브리드와 비슷한 점수를 낸다. 리랭커는 8GB 맥에서 6.1초라 생성 단계에 쓰지 않았다.")

    st.markdown("**방식별 요약**")
    st.dataframe(rm[["방식", "Recall@5", "MRR@5", "nDCG@5", "지연"]], hide_index=True, use_container_width=True)

    c3, c4 = st.columns([3, 2])
    with c3:
        st.markdown("**사전 등록한 비교 5개** — CI 가 0 을 포함하면 확정 불가")
        st.dataframe(df("rag_pairs"), hide_index=True, use_container_width=True)
        st.caption("확정된 것은 '하이브리드가 벡터 단독보다 낫다' 하나다. 하이브리드가 BM25 보다 낫다고는 말할 수 없다.")
    with c4:
        st.markdown("**판정 요약**")
        st.dataframe(df("rag_summary"), hide_index=True, use_container_width=True)

    st.markdown("**답변 생성** — 하이브리드 상위 5개 문단 → claude-sonnet-5-5, 주장 최대 2개, 주장마다 근거 문단 1개")
    st.dataframe(df("rag_generation"), hide_index=True, use_container_width=True)
    st.caption("충실도 1.000 은 성과로 쓰지 않는다. 주장을 2개로 제한하고 문단 표현에 맞춰 쓰게 한 발췌형 과제라 틀릴 여지가 거의 없었다. "
               "정보가 있는 수치는 거절 쪽이다: 근거 없는 질문 5개는 전부 거절했고, 답할 수 있던 25개 중 3개도 거절했다.")

    st.markdown("**실제 답 세 건** — 생성된 문장 그대로, 회사명만 가림")
    st.dataframe(pd.DataFrame(L.RAG_EXAMPLES), hide_index=True, use_container_width=True)

    st.markdown("**한계**")
    for line in L.RAG_LIMITS:
        st.markdown(f"- {line}")
