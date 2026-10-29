# CareerFit 데모 이미지 — 커밋된 산출물만 담는다 (공고 원문·API 키·비공개 정답셋 없음)
#   docker build -t careerfit-demo .
#   docker run --rm -p 8501:8501 careerfit-demo
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-demo.txt ./
RUN pip install -r requirements-demo.txt

COPY src ./src
COPY demo ./demo
COPY data/raw/dict ./data/raw/dict
COPY data/processed/market_stats ./data/processed/market_stats
COPY data/processed/job_skill_extract_report/validation_summary.csv ./data/processed/job_skill_extract_report/
COPY data/processed/job_skill_extract_report/tag_frequency_by_job.csv ./data/processed/job_skill_extract_report/
COPY data/processed/method_comparison.csv data/processed/bootstrap_ci.csv data/processed/cascade_sim.csv \
     data/processed/failed_rows_sensitivity.csv ./data/processed/
COPY .streamlit ./.streamlit

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["python", "-m", "streamlit", "run", "demo/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
