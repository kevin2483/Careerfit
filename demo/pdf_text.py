"""업로드한 PDF/텍스트 파일에서 본문만 꺼낸다. 메모리에서만 처리하고 어디에도 저장하지 않는다."""
from __future__ import annotations

import io

MAX_BYTES = 5 * 1024 * 1024


def extract_text(data: bytes, filename: str) -> tuple[str, str]:
    """(본문, 경고). PDF 는 pypdf 로 페이지별 텍스트를 잇는다. 스캔본(이미지)은 글자가 없어 빈 문자열이 온다."""
    if len(data) > MAX_BYTES:
        return "", f"파일이 {MAX_BYTES // (1024 * 1024)}MB 를 넘습니다."
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as e:  # noqa: BLE001 — 깨진 PDF 는 메시지로 알린다
            return "", f"PDF 를 읽지 못했습니다: {e}"
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            return "", "PDF 에서 글자를 찾지 못했습니다. 스캔 이미지 PDF 는 지원하지 않습니다. 텍스트를 직접 붙여 넣어 주세요."
        return text, f"PDF {len(pages)}쪽에서 {len(text):,}자를 읽었습니다."
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("cp949", errors="replace")
    return text.strip(), f"텍스트 {len(text):,}자를 읽었습니다."
