"""
load_data.py  (원래 예시 구조의 load_pdf.py → load_data.py 로 변경)
---------------------------------------------------
RAG 파이프라인의 1단계: 원본 데이터를 로드하고 chunk로 분할한다.

지원 입력
  - JSON : fetch_law.py가 만든 조문 데이터(예: data/laws_all.json).
           이미 조문 단위로 정제 + 메타데이터가 붙어 있으므로 그대로 활용.
  - PDF  : 원본 PDF. 텍스트를 추출해 분할.
  - 폴더 : 폴더 경로를 주면 안의 .json/.pdf 를 모두 로드.

핵심 파라미터 (실험 변수)
  - chunk_size   : 한 chunk의 최대 글자 수
  - overlap_size : chunk 간 겹치는 글자 수

반환: LangChain Document 리스트 (page_content + metadata)
      → 다음 단계 build_vectorstore.py 가 그대로 임베딩에 사용.

설치: pip install langchain langchain-community pypdf
"""

import json
from pathlib import Path

# --- 버전에 따라 위치가 다른 import 들 (신버전 우선, 구버전 폴백) ---
try:
    from langchain_core.documents import Document          # langchain >= 0.1
except ImportError:
    from langchain.schema import Document                  # 구버전

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # 신버전
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter    # 구버전

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader


# =====================================================================
# 분할기 생성
# =====================================================================
def make_splitter(chunk_size: int, overlap_size: int) -> RecursiveCharacterTextSplitter:
    """법령/문서용 재귀 분할기. 문단→줄→문장→어절 순으로 자연스럽게 자른다."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


# =====================================================================
# JSON 로드 (fetch_law.py 산출물)
# =====================================================================
def load_json(path: Path, splitter: RecursiveCharacterTextSplitter | None) -> list[Document]:
    """
    조문 JSON을 Document로 변환.
    각 조문의 '본문'(헤더 포함)을 chunk 대상 텍스트로 쓰고,
    나머지 필드는 metadata로 보존한다.

    splitter가 있으면: 본문이 chunk_size보다 '길 때만' 다시 나눈다.
                       (짧은 조문은 그대로 통과 → 중복 분할 아님)
    splitter가 None이면: 조문 1개 = chunk 1개 (조문 단위 그대로 보존).
                       ※ 단, 초대형 별표는 임베딩 한도를 넘을 수 있으니 주의.
    """
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    docs: list[Document] = []

    for it in items:
        text = it.get("본문") or it.get("내용") or ""
        if not text.strip():
            continue
        meta = {
            "source": str(path.name),
            "법령명": it.get("법령명", ""),
            "법령종류": it.get("법령종류", ""),
            "장": it.get("장", ""),
            "조문번호": it.get("조문번호", ""),
            "조문가지번호": it.get("조문가지번호", ""),   # 제619조'의2'
            "조문표시": it.get("조문표시", ""),           # '제619조의2' 전체 표기
            "조문제목": it.get("조문제목", ""),
            "시행일자": it.get("시행일자", ""),
        }
        chunks = splitter.split_text(text) if splitter else [text]
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata=dict(meta)))
    return docs


# =====================================================================
# PDF 로드
# =====================================================================
def load_pdf(path: Path, splitter: RecursiveCharacterTextSplitter | None) -> list[Document]:
    """
    PDF를 페이지별로 읽어 텍스트를 추출한 뒤 chunk로 분할.
    splitter가 None이면 페이지 1개 = chunk 1개(분할 없음).
    """
    loader = PyPDFLoader(str(path))
    pages = loader.load()  # 페이지당 Document

    docs: list[Document] = []
    for page in pages:
        base_meta = {
            "source": str(path.name),
            "page": page.metadata.get("page", None),
            "법령명": Path(path).stem,  # 파일명을 법령명 대용으로
        }
        chunks = splitter.split_text(page.page_content) if splitter else [page.page_content]
        for chunk in chunks:
            if chunk.strip():
                docs.append(Document(page_content=chunk, metadata=dict(base_meta)))
    return docs


# =====================================================================
# 통합 진입점
# =====================================================================
# 파일 형식별 처리 함수 매핑 (범용성: 형식이 늘어나면 여기에 추가만 하면 됨)
LOADERS = {
    "json": load_json,
    "pdf": load_pdf,
}


def load_data(
    path: str,
    file_type: str,
    chunk_size: int = 500,
    overlap_size: int = 50,
    split_long: bool = True,
) -> list[Document]:
    """
    파일 경로와 파일 형식을 받아 Document 리스트를 반환.

    Args:
        path         : 데이터 파일 경로 (예: ../data/laws_all.json)
        file_type    : 파일 형식. "json" 또는 "pdf"  ← 형식에 맞는 로더가 실행됨
        chunk_size   : chunk 최대 글자 수 (실험 변수)
        overlap_size : chunk 간 겹침 글자 수 (실험 변수)
        split_long   : True(기본)  → chunk_size보다 '긴' 항목만 재분할.
                                     짧은 조문은 그대로 통과(중복 분할 없음).
                       False       → 재분할 없이 항목 1개 = chunk 1개.
                                     (JSON은 조문 단위 그대로 보존. 조문단위 vs
                                      고정길이 비교 실험용. 초대형 별표 주의)
    """
    ftype = file_type.lower().lstrip(".")  # ".json" / "JSON" 등도 허용
    if ftype not in LOADERS:
        raise ValueError(
            f"지원하지 않는 file_type: '{file_type}'. "
            f"사용 가능: {list(LOADERS.keys())}"
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    splitter = make_splitter(chunk_size, overlap_size) if split_long else None
    docs = LOADERS[ftype](p, splitter)  # 형식에 맞는 함수만 수행

    mode = f"chunk_size={chunk_size}, overlap_size={overlap_size}" if split_long \
        else "조문 단위 그대로(split_long=False)"
    print(f"[load_data] ({ftype}) {p.name} → {len(docs)}개 chunk ({mode})")
    return docs


# =====================================================================
# 단독 실행 테스트
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF/JSON 로드 + 청킹")
    parser.add_argument("path", help="데이터 파일 경로 (예: ../data/laws_all.json)")
    parser.add_argument("--file_type", required=True, choices=["json", "pdf"],
                        help="파일 형식 지정")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap_size", type=int, default=50)
    args = parser.parse_args()

    docs = load_data(args.path, args.file_type, args.chunk_size, args.overlap_size)
    if docs:
        print("\n--- 첫 chunk 미리보기 ---")
        print("metadata:", docs[0].metadata)
        print("content :", docs[0].page_content[:200], "...")
