"""사전 v0 -> v1: FN 분석 결과 반영. 기존 항목은 건드리지 않고 추가/오타수정만."""
import sys, yaml
from pathlib import Path

src = Path(sys.argv[1]); dst = Path(sys.argv[2])
d = yaml.safe_load(src.read_text(encoding="utf-8"))

def add(cat, canonical, aliases, field, parent=None):
    d.setdefault(cat, [])
    for e in d[cat]:
        if e["canonical"] == canonical:
            # 이미 있으면 alias 만 보강
            e["aliases"] = sorted(set((e.get("aliases") or []) + aliases))
            return
    item = {"canonical": canonical, "aliases": aliases, "field": field}
    if parent: item["parent"] = parent
    d[cat].append(item)

# ── 오타 수정: ABテスト -> A/B테스트 ────────────────────────────
for cat, entries in d.items():
    if not isinstance(entries, list): continue
    for e in entries:
        if e["canonical"] == "ABテスト":
            e["canonical"] = "A/B테스트"
            e["aliases"] = sorted(set((e.get("aliases") or []) + ["a/b test", "ab test", "a/b 테스트", "가설 검정", "가설검정"]))

# ── ① 명시 기술 (field 1) ──────────────────────────────────────
add("data_pipeline", "Kafka", ["kafka", "apache kafka", "카프카"], 1, "스트리밍")
add("data_pipeline", "PySpark", ["pyspark"], 1, "분산처리")
add("data_pipeline", "Flink", ["flink", "apache flink"], 1, "스트리밍")
add("data_pipeline", "Airflow", ["airflow", "apache airflow"], 1, "워크플로우오케스트레이션")
add("ml_frameworks", "Pandas", ["pandas", "판다스"], 1)
add("ml_frameworks", "NumPy", ["numpy", "넘파이"], 1)

add("bi_viz", "Tableau", ["tableau", "태블로"], 1, "BI")
add("bi_viz", "PowerBI", ["power bi", "powerbi", "파워비아이"], 1, "BI")
add("bi_viz", "QuickSight", ["quicksight"], 1, "BI")
add("bi_viz", "Grafana", ["grafana", "그라파나"], 1, "모니터링")
add("bi_viz", "Excel", ["excel", "엑셀"], 1)

add("backend", "FastAPI", ["fastapi"], 1, "웹프레임워크")
add("backend", "SpringBoot", ["spring boot", "springboot", "스프링부트", "spring"], 1, "웹프레임워크")
add("backend", "Node.js", ["node.js", "nodejs", "노드js"], 1, "웹프레임워크")
add("backend", "REST API", ["rest api", "restful", "rest"], 1)
add("backend", "MSA", ["msa", "마이크로서비스"], 1)

add("infra", "ArgoCD", ["argocd", "argo cd"], 1, "CI/CD")
add("infra", "Prometheus", ["prometheus", "프로메테우스"], 1, "모니터링")
add("infra", "ELK", ["elk", "loki", "elk/loki"], 1, "모니터링")
add("infra", "Athena", ["athena"], 1)

add("datastores", "MSSQL", ["mssql", "sql server"], 1, "RDBMS")
add("datastores", "MariaDB", ["mariadb"], 1, "RDBMS")
add("languages", "C", ["c언어"], 1)

# ── ② ML/AI 개념 (field 2) ─────────────────────────────────────
add("ml_concepts", "분류", ["분류", "classification"], 2)
add("ml_concepts", "회귀", ["회귀", "regression"], 2)
add("ml_concepts", "CNN", ["cnn", "합성곱"], 2, "딥러닝")
add("ml_concepts", "RNN", ["rnn", "lstm"], 2, "딥러닝")
add("ml_concepts", "EdgeAI", ["edge ai", "efficient ai", "ai accelerator", "엣지 ai"], 2)
add("ml_concepts", "임베디드", ["임베디드", "embedded"], 2)
add("ml_concepts", "IoT", ["iot", "사물인터넷"], 2)
add("llm_genai", "LLM API", ["llm api", "openai api", "claude api"], 2, "LLM응용")

# 모델경량화는 기존 experience_narrative 에 있으므로 alias 보강
add("experience_narrative", "모델서빙배포",
    ["model compression", "low-bit", "경량화", "양자화"], 2, "MLOps")

# ── ③ 자격증 (별도 카테고리, field 1) ──────────────────────────
add("certifications", "SQLD", ["sqld"], 1, "자격증")
add("certifications", "SQLP", ["sqlp"], 1, "자격증")
add("certifications", "DAP", ["dap"], 1, "자격증")
add("certifications", "DASP", ["dasp"], 1, "자격증")
add("certifications", "ADsP", ["adsp"], 1, "자격증")

dst.write_text(yaml.dump(d, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")

n_tags = sum(len(v) for v in d.values() if isinstance(v, list))
n_alias = sum(len(t.get("aliases") or []) for v in d.values() if isinstance(v, list) for t in v)
print(f"v1 생성: 카테고리 {len(d)} / canonical {n_tags} / alias {n_alias}")
for k, v in d.items():
    if isinstance(v, list): print(f"  {k}: {len(v)}")
