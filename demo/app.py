"""
CareerFit 데모 — 커밋된 산출물만으로 동작한다 (공고 원문·API 키 불필요).

    make demo            # 로컬
    streamlit run demo/app.py

탭 1 시장 지형도: data/processed/market_stats/*.csv
탭 2 갭 진단:     src.analyze.gap_diagnosis.run() 을 그대로 호출 (산식 중복 없음)
탭 3 평가 결과:   method_comparison · bootstrap_ci · cascade_sim · failed_rows_sensitivity
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
st.caption("채용공고 1,190건에서 뽑은 직무별 요구 역량, 이력서 대조 학습 우선순위, 그리고 그 추출이 맞는지 잰 결과. "
           "모든 숫자는 레포에 커밋된 CSV 그대로이며 `make reproduce` 로 다시 만들어진다.")

tab_market, tab_gap, tab_eval = st.tabs(["시장 지형도", "갭 진단", "평가 결과"])

# ───────────────────────────────────────────── 탭 1
with tab_market:
    pop = L.population()
    job = st.radio("직무", JOB_OPTIONS, format_func=job_label, horizontal=True, key="job_market")
    p = pop[job]
    c1, c2, c3 = st.columns(3)
    c1.metric("입력 공고", f"{p['입력공고수']:,}")
    c2.metric("통과(모집단)", f"{p['통과']:,}")
    c3.metric("직무불일치 제외", f"{p['직무불일치']:,}")
    note = L.da_note(job)
    if note:
        st.info(note)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("요구 역량 상위 20")
        rank = df("rank_by_job", job, 20)
        chart = alt.Chart(rank).mark_bar().encode(
            x=alt.X("출현율%:Q", title="출현율 (%)"),
            y=alt.Y("canonical:N", sort="-x", title=None),
            tooltip=["순위", "canonical", "출현수", "모집단수", "출현율%"],
        ).properties(height=520)
        st.altair_chart(chart, use_container_width=True)
        st.caption("출현율 분모는 통과 공고 수. 3건 미만 역량은 표본 부족으로 제외(아래 목록).")
    with right:
        st.subheader("상위 20 역량의 신입 접근성")
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
        st.caption("그 역량을 요구한 공고 중 신입 지원 가능(annual_from 0)인 비율. 갭 진단의 '신입접근성' 가중치가 이 값이다.")

        st.subheader("함께 요구되는 역량 쌍 상위 15")
        st.dataframe(df("cooccurrence", job, 15)[["tag_a", "tag_b", "동시출현수"]], hide_index=True, use_container_width=True)

    with st.expander(f"표본 부족으로 제외한 역량 ({len(L.excluded_lowfreq(job))}건)"):
        st.dataframe(df("excluded_lowfreq", job)[["canonical", "출현수", "제외사유"]], hide_index=True, use_container_width=True)

# ───────────────────────────────────────────── 탭 2
with tab_gap:
    st.subheader("이력서와 시장 요구의 차이")
    st.caption("우선순위 = 직무 출현율 × 미보유(1) × 신입접근성. 상위 5개만 이번 분기, 나머지는 다음 분기. "
               "이력서는 서버에 저장하지 않으며 정규식으로 역량 태그만 뽑는다.")
    job2 = st.radio("대상 직무", JOB_OPTIONS, format_func=job_label, horizontal=True, key="job_gap", index=1)
    if "resume" not in st.session_state:
        st.session_state["resume"] = ""
    b1, b2 = st.columns([1, 6])
    if b1.button("예시 이력서 넣기"):
        st.session_state["resume"] = SAMPLE_RESUME
    resume = st.text_area("이력서 텍스트", key="resume", height=220, placeholder="경력·기술·프로젝트를 붙여 넣으세요.")
    if resume.strip():
        r = gd.run(resume_text=resume, target_job=job2, check_fresh=False)
        m1, m2, m3 = st.columns(3)
        m1.metric("이력서에서 찾은 역량", r["이력서_보유태그수"])
        m2.metric(f"{job2} 시장 역량 수", r["직무_전체태그수"])
        m3.metric("이번 분기 학습 항목", len(r["이번분기_학습로드맵(top5)"]))

        st.markdown("**이번 분기 학습 로드맵 (top 5)**")
        top = pd.DataFrame(r["이번분기_학습로드맵(top5)"])
        if len(top):
            bar = alt.Chart(top).mark_bar().encode(
                x=alt.X("우선순위점수:Q"), y=alt.Y("canonical:N", sort="-x", title=None),
                tooltip=["순위", "canonical", "출현율%", "신입접근성", "우선순위점수"],
            ).properties(height=200)
            st.altair_chart(bar, use_container_width=True)
            st.dataframe(top[["순위", "canonical", "출현율%", "신입접근성", "우선순위점수"]], hide_index=True, use_container_width=True)
        st.markdown("**이미 보유한 역량** · " + (", ".join(r["보유역량"]) or "없음"))
        with st.expander(f"다음 분기 후순위 ({len(r['다음분기_후순위'])}건)"):
            st.dataframe(pd.DataFrame(r["다음분기_후순위"])[["순위", "canonical", "출현율%", "신입접근성", "우선순위점수"]]
                         if r["다음분기_후순위"] else pd.DataFrame(), hide_index=True, use_container_width=True)
        with st.expander("로드맵에서 뺀 암묵 조건 (field 3)"):
            st.write(r["field3_제외기준"])
            st.dataframe(pd.DataFrame(r["제외_field3_암묵조건"]), hide_index=True, use_container_width=True)
        st.caption("이력서에 적히지 않은 역량은 미보유로 계산된다. 최근 경험이 빠진 이력서면 로드맵도 그만큼 어긋난다.")

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

    st.markdown("**평가 장치 자체를 네 번 고쳤다**")
    st.dataframe(pd.DataFrame(L.CORRECTIONS), hide_index=True, use_container_width=True)
    fr = df("failed_rows")
    n_failed = sum(1 for r in L.failed_rows() if r["구분"] == "전체" and r["제외문서"])
    st.caption(f"예측 실패 행(출력 잘림·파싱 실패) 민감도 검사: 현재 실패 행 {n_failed}건. 자세한 경위는 레포 README 와 notes/lab_log.md.")
