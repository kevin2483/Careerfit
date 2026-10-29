"""2차 정답셋 확장 도구 — N 시스템 풀링(결정적·출처 숨김), 판정된 미검수 행의 reviewed 승격, PDF 본문 추출."""
import argparse
import json

from src.common.jsonl import read_jsonl
from src.common.paths import DICT_PATH
from src.label.make_pool import build_pool
from src.label.pool_cli import apply


def test_build_pool_union_blind_and_seeded():
    preds = {"regex": {"1": {"Python", "SQL"}}, "opus": {"1": {"SQL", "Airflow"}}, "haiku": {"1": set()}}
    c1, k1 = build_pool(preds, ["1"], seed=7)
    c2, k2 = build_pool(preds, ["1"], seed=7)
    assert c1 == c2 and k1 == k2                                       # 같은 seed → 같은 순서
    assert {c["tag"] for c in c1} == {"Python", "SQL", "Airflow"} and all(c["verdict"] is None for c in c1)
    assert all("sources" not in c for c in c1)                         # 후보 파일에는 출처가 없다
    by = {k["tag"]: k["sources"] for k in k1}
    assert by["SQL"] == ["regex", "opus"] and by["Airflow"] == ["opus"]


def test_build_pool_refuses_missing_system():
    import pytest
    with pytest.raises(SystemExit, match="예측 없는 공고"):
        build_pool({"regex": {"1": {"Python"}}, "opus": {}}, ["1"], seed=1)


def test_apply_promotes_judged_unreviewed_row(tmp_path, capsys):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"job_id": "9", "reviewed": False, "gold_tags": ["Python", "Excel"]}) + "\n", encoding="utf-8")
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(c) + "\n" for c in [
        {"job_id": "9", "tag": "Python", "verdict": True},
        {"job_id": "9", "tag": "Excel", "verdict": False},
        {"job_id": "9", "tag": "Airflow", "verdict": True},
    ]), encoding="utf-8")
    apply(argparse.Namespace(pool=str(pool), labels=str(labels), force=False, dict_path=str(DICT_PATH)))
    row = read_jsonl(labels)[0]
    assert row["reviewed"] is True and row["gold_tags"] == ["Airflow", "Python"] and row["gold_tags_prepool"] == ["Excel", "Python"]
    assert "신규 확정: 9" in capsys.readouterr().out


def _tiny_pdf(text: str) -> bytes:
    """글자 하나 들어간 최소 PDF. pypdf 가 extract_text 로 읽을 수 있는 형태."""
    stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_pdf_and_text_extraction():
    from demo.pdf_text import extract_text
    text, note = extract_text(_tiny_pdf("Python SQL Airflow"), "resume.pdf")
    assert "Python" in text and "Airflow" in text and "1쪽" in note
    text, note = extract_text("파이썬과 SQL".encode("utf-8"), "resume.txt")
    assert text == "파이썬과 SQL"
    text, note = extract_text(b"not a pdf", "x.pdf")
    assert text == "" and "읽지 못했습니다" in note
    assert extract_text(b"x" * (6 * 1024 * 1024), "big.pdf")[0] == ""
