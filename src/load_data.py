"""
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

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:  # 구버전 호환
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
def load_json(path: Path, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    """
    조문 JSON을 Document로 변환.
    각 조문의 '본문'(헤더 포함)을 chunk 대상 텍스트로 쓰고,
    나머지 필드는 metadata로 보존한다.
    본문이 chunk_size보다 길면 splitter가 다시 나눈다(헤더는 첫 조각에 유지).
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
            "조문제목": it.get("조문제목", ""),
            "시행일자": it.get("시행일자", ""),
        }
        # 조문 단위로 먼저 Document를 만든 뒤 chunk_size 기준 재분할
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata=dict(meta)))
    return docs


# =====================================================================
# PDF 로드
# =====================================================================
def load_pdf(path: Path, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    """PDF를 페이지별로 읽어 텍스트를 추출한 뒤 chunk로 분할."""
    loader = PyPDFLoader(str(path))
    pages = loader.load()  # 페이지당 Document

    docs: list[Document] = []
    for page in pages:
        base_meta = {
            "source": str(path.name),
            "page": page.metadata.get("page", None),
            "법령명": Path(path).stem,  # 파일명을 법령명 대용으로
        }
        for chunk in splitter.split_text(page.page_content):
            if chunk.strip():
                docs.append(Document(page_content=chunk, metadata=dict(base_meta)))
    return docs


# =====================================================================
# 통합 진입점
# =====================================================================
def load_data(
    path: str,
    chunk_size: int = 500,
    overlap_size: int = 50,
) -> list[Document]:
    """
    파일(또는 폴더) 경로를 받아 Document 리스트를 반환.

    Args:
        path         : .json / .pdf 파일 경로, 또는 폴더 경로
        chunk_size   : chunk 최대 글자 수 (실험 변수)
        overlap_size : chunk 간 겹침 글자 수 (실험 변수)
    """
    p = Path(path)
    splitter = make_splitter(chunk_size, overlap_size)

    # 대상 파일 목록 결정
    if p.is_dir():
        targets = sorted(list(p.glob("*.json")) + list(p.glob("*.pdf")))
    else:
        targets = [p]

    if not targets:
        raise FileNotFoundError(f"로드할 .json/.pdf 파일이 없습니다: {path}")

    all_docs: list[Document] = []
    for f in targets:
        ext = f.suffix.lower()
        if ext == ".json":
            all_docs.extend(load_json(f, splitter))
        elif ext == ".pdf":
            all_docs.extend(load_pdf(f, splitter))
        else:
            print(f"  (건너뜀) 지원하지 않는 형식: {f.name}")

    print(f"[load_data] {len(targets)}개 파일 → {len(all_docs)}개 chunk "
          f"(chunk_size={chunk_size}, overlap_size={overlap_size})")
    return all_docs


# =====================================================================
# 단독 실행 테스트
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF/JSON 로드 + 청킹")
    parser.add_argument("path", help=".json/.pdf 파일 또는 폴더 경로 (예: ../data/laws_all.json)")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap_size", type=int, default=50)
    args = parser.parse_args()

    docs = load_data(args.path, args.chunk_size, args.overlap_size)
    if docs:
        print("\n--- 첫 chunk 미리보기 ---")
        print("metadata:", docs[0].metadata)
        print("content :", docs[0].page_content[:200], "...")
