"""
ksj_vectorstore.py
---------------------------------------------------
KSJ 전용 벡터스토어 모듈입니다.

팀 공용 build_vectorstore.py는 그대로 두고, 8GB GPU 메모리를 Mistral 7B에
양보하기 위해 임베딩 모델의 실행 장치를 CPU로 명시할 수 있게 분리했습니다.

범용성 설계
  - 저장소 종류를 store_type 매개변수로 선택:
      · 벡터DB : "faiss", "chroma"
      · 그래프DB: "neo4j"        (Neo4j 벡터 인덱스 기반)
  - 임베딩 모델도 embedding_name 으로 교체 가능:
      · "hf"     : HuggingFace 문장 임베딩(기본: 한국어 ko-sbert)
      · "openai" : OpenAI 임베딩 (API 키 필요)
  - 반환 객체는 모두 LangChain VectorStore 인터페이스를 따르므로,
    다음 단계 ksj_rag_chain.py에서 store.as_retriever(...)로 동일하게 사용 가능.

새 저장소/임베딩을 추가하려면 아래 레지스트리에 함수만 등록하면 된다.

설치(사용하는 것만):
  pip install langchain langchain-community
  pip install faiss-cpu                         # FAISS
  pip install chromadb                          # Chroma
  pip install neo4j langchain-neo4j             # Neo4j (그래프DB)
  pip install sentence-transformers             # HuggingFace 임베딩
"""

from pathlib import Path

# .env 파일이 있으면 자동 로드 (UPSTAGE_API_KEY 등)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =====================================================================
# 1) 임베딩 모델 선택
# =====================================================================
def get_embeddings(
    embedding_name: str = "hf",
    model: str | None = None,
    device: str = "cpu",
):
    """
    embedding_name:
      - "hf"      : HuggingFace (기본 모델: jhgan/ko-sbert-nli, 한국어)
      - "openai"  : OpenAI (기본: text-embedding-3-small, OPENAI_API_KEY)
      - "upstage" : Upstage Solar 임베딩 (UPSTAGE_API_KEY)
    model: 특정 모델명을 직접 지정하고 싶을 때
    device: HF 임베딩 계산 장치. 8GB GPU에서 7B LLM을 실행할 때는 "cpu" 권장
    """
    name = embedding_name.lower()

    if name == "hf":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings   # 신버전
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # 구버전
        return HuggingFaceEmbeddings(
            model_name=model or "jhgan/ko-sbert-nli",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if name == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")

    if name == "upstage":
        from langchain_upstage import UpstageEmbeddings
        return UpstageEmbeddings(model=model or "solar-embedding-1-large")

    raise ValueError(f"지원하지 않는 embedding_name: '{embedding_name}' "
                     f"(hf, openai, upstage)")


# =====================================================================
# 2) 저장소별 빌더 (docs → store)
#    각 함수는 (docs, embeddings, persist_dir, **kwargs) → store 를 반환
# =====================================================================
def _build_faiss(docs, embeddings, persist_dir, **kw):
    from langchain_community.vectorstores import FAISS
    store = FAISS.from_documents(docs, embeddings)
    if persist_dir:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        store.save_local(persist_dir)
    return store


def _build_chroma(docs, embeddings, persist_dir, **kw):
    from langchain_community.vectorstores import Chroma
    store = Chroma.from_documents(
        docs, embeddings,
        persist_directory=persist_dir or None,
        collection_name=kw.get("collection_name", "law_rag"),
    )
    return store


def _build_neo4j(docs, embeddings, persist_dir, **kw):
    """
    그래프DB(Neo4j) 벡터 인덱스에 저장.
    연결 정보는 kwargs 또는 환경변수(NEO4J_URI/USERNAME/PASSWORD)로 전달.
    """
    import os
    try:
        from langchain_neo4j import Neo4jVector
    except ImportError:
        from langchain_community.vectorstores import Neo4jVector
    return Neo4jVector.from_documents(
        docs, embeddings,
        url=kw.get("url") or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=kw.get("username") or os.getenv("NEO4J_USERNAME", "neo4j"),
        password=kw.get("password") or os.getenv("NEO4J_PASSWORD", ""),
        index_name=kw.get("index_name", "law_rag"),
        node_label=kw.get("node_label", "LawChunk"),
    )


# 저장소 종류 → (빌더, 분류) 레지스트리
STORE_BUILDERS = {
    "faiss": (_build_faiss, "vector"),
    "chroma": (_build_chroma, "vector"),
    "neo4j": (_build_neo4j, "graph"),
}


# =====================================================================
# 3) 저장소별 로더 (기존 저장소 다시 불러오기)
# =====================================================================
def _load_faiss(embeddings, persist_dir, **kw):
    from langchain_community.vectorstores import FAISS
    return FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True
    )


def _load_chroma(embeddings, persist_dir, **kw):
    from langchain_community.vectorstores import Chroma
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=kw.get("collection_name", "law_rag"),
    )


def _load_neo4j(embeddings, persist_dir, **kw):
    import os
    try:
        from langchain_neo4j import Neo4jVector
    except ImportError:
        from langchain_community.vectorstores import Neo4jVector
    return Neo4jVector.from_existing_index(
        embeddings,
        url=kw.get("url") or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=kw.get("username") or os.getenv("NEO4J_USERNAME", "neo4j"),
        password=kw.get("password") or os.getenv("NEO4J_PASSWORD", ""),
        index_name=kw.get("index_name", "law_rag"),
    )


STORE_LOADERS = {
    "faiss": _load_faiss,
    "chroma": _load_chroma,
    "neo4j": _load_neo4j,
}


# =====================================================================
# 4) 통합 진입점
# =====================================================================
def build_vectorstore(
    docs,
    store_type: str = "faiss",
    embedding_name: str = "hf",
    embedding_model: str | None = None,
    embedding_device: str = "cpu",
    persist_dir: str | None = "./stores/faiss",
    **store_kwargs,
):
    """
    Document 리스트를 임베딩해 지정한 저장소에 저장.

    Args:
        docs           : load_data() 가 반환한 Document 리스트
        store_type     : "faiss" / "chroma" / "neo4j"  (실험 변수)
        embedding_name : "hf" / "openai"               (실험 변수)
        embedding_model: 특정 임베딩 모델명 (선택)
        embedding_device: HF 임베딩 장치. 7B LLM과 함께 쓸 때 "cpu" 권장
        persist_dir    : 벡터DB 저장 경로 (neo4j는 무시, DB에 저장)
        store_kwargs   : neo4j 연결정보(url/username/password) 등
    Returns:
        LangChain VectorStore (store.as_retriever() 사용 가능)
    """
    st = store_type.lower()
    if st not in STORE_BUILDERS:
        raise ValueError(
            f"지원하지 않는 store_type: '{store_type}'. "
            f"사용 가능: {list(STORE_BUILDERS.keys())}"
        )

    embeddings = get_embeddings(
        embedding_name,
        embedding_model,
        device=embedding_device,
    )
    builder, kind = STORE_BUILDERS[st]
    store = builder(docs, embeddings, persist_dir, **store_kwargs)

    print(f"[build_vectorstore] {kind}DB '{st}' 생성 완료 "
          f"(문서 {len(docs)}개, 임베딩={embedding_name})")
    return store


def load_vectorstore(
    store_type: str = "faiss",
    embedding_name: str = "hf",
    embedding_model: str | None = None,
    embedding_device: str = "cpu",
    persist_dir: str | None = "./stores/faiss",
    **store_kwargs,
):
    """이미 만들어 둔 저장소를 다시 로드한다(재임베딩 없이 검색용)."""
    st = store_type.lower()
    if st not in STORE_LOADERS:
        raise ValueError(f"지원하지 않는 store_type: '{store_type}'")
    embeddings = get_embeddings(
        embedding_name,
        embedding_model,
        device=embedding_device,
    )
    store = STORE_LOADERS[st](embeddings, persist_dir, **store_kwargs)
    print(f"[load_vectorstore] '{st}' 로드 완료")
    return store


# =====================================================================
# 단독 실행 테스트
# =====================================================================
if __name__ == "__main__":
    import argparse
    from load_data import load_data

    parser = argparse.ArgumentParser(description="문서 임베딩 + 저장소 구축")
    parser.add_argument("path", help="데이터 파일 경로 (예: ../data/laws_all.json)")
    parser.add_argument("--file_type", required=True, choices=["json", "pdf"])
    parser.add_argument("--store_type", default="faiss", choices=list(STORE_BUILDERS.keys()))
    parser.add_argument("--embedding_name", default="hf", choices=["hf", "openai"])
    parser.add_argument(
        "--embedding_model",
        default=None,
        help=(
            "구체 임베딩 모델명. 예: BAAI/bge-m3. "
            "생략하면 선택한 제공자의 기본 모델을 사용"
        ),
    )
    parser.add_argument(
        "--embedding_device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="HF 임베딩 계산 장치. 8GB GPU에서 Mistral 7B와 같이 쓸 때는 cpu 권장",
    )
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap_size", type=int, default=50)
    parser.add_argument("--persist_dir", default="./stores/test")
    args = parser.parse_args()

    docs = load_data(args.path, args.file_type, args.chunk_size, args.overlap_size)
    store = build_vectorstore(
        docs,
        store_type=args.store_type,
        embedding_name=args.embedding_name,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        persist_dir=args.persist_dir,
    )

    # 간단 검색 확인
    q = "타워크레인은 순간풍속 얼마에서 운전을 멈춰야 하나요?"
    hits = store.as_retriever(search_kwargs={"k": 3}).invoke(q)
    print(f"\n--- '{q}' 검색 상위 3개 ---")
    for i, d in enumerate(hits, 1):
        print(f"[{i}] {d.metadata.get('법령명','')} {d.metadata.get('조문번호','')}: "
              f"{d.page_content[:80]}...")
