"""
고정된 영문 실험 조합의 단일 실행 파일.

실행 모드
---------
1. --build-store        : 검증된 1:1 영문 청크로 Upstage FAISS 최초 생성
2. --ask "question"     : 질문 하나 실행
3. --interactive        : Mistral을 한 번 로드하고 반복 질문
4. --evaluate dev       : 동일 14문항 + RAGAS 평가
5. --evaluate holdout   : 별도 28문항 + RAGAS 최종 확인
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "laws_chunks_en_1to1.jsonl"

# 기존 3,747개 영문 재청킹 DB와 섞이지 않도록 저장 경로도 새로 분리합니다.
STORE_PATH = (
    PROJECT_ROOT
    / "stores"
    / "english_upstage_faiss_kochunk500_o50_1to1"
)


def parse_args():
    """한 번에 하나의 실행 모드만 선택하도록 명령행 인자를 정의합니다."""
    parser = argparse.ArgumentParser(
        description=(
            "Mistral-7B + Upstage embedding + Hybrid + FAISS + "
            "top_k=5 + threshold=0.4 English RAG"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--build-store",
        action="store_true",
        help="영문 법령을 Upstage로 임베딩하여 FAISS를 생성",
    )
    mode.add_argument("--ask", help="질문 하나를 실행하고 종료")
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="모델을 한 번만 로드하고 질문을 반복 입력",
    )
    mode.add_argument(
        "--evaluate",
        choices=("dev", "holdout"),
        help="dev 14문항 또는 holdout 28문항 평가",
    )
    mode.add_argument(
        "--search-only",
        help="LLM 생성 없이 질문 하나의 검색 결과만 확인",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="평가 결과 파일명. 생략하면 고정 조건으로 자동 생성",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="설치·디버깅 때만 RAGAS를 생략(최종 비교에는 사용하지 말 것)",
    )
    parser.add_argument(
        "--rebuild-store",
        action="store_true",
        help="평가·질문 실행 전에 기존 FAISS를 다시 생성",
    )
    return parser.parse_args()


def build_store():
    """검증된 1:1 영문 청크를 재분할 없이 Upstage+FAISS로 저장합니다."""
    from load_data_english import load_parallel_english_chunks
    from validate_parallel_chunks import validate
    from vectorstore_upstage import build_vectorstore

    # 벡터DB를 만들기 직전에 ID·순서·숫자 보존을 다시 검사합니다.
    report = validate()
    if report["errors"]:
        raise ValueError(
            f"1:1 번역 검증 오류가 {report['error_count']}개 있습니다. "
            "`python src/validate_parallel_chunks.py`의 보고서를 확인하세요."
        )

    documents = load_parallel_english_chunks(str(DATA_PATH))
    return build_vectorstore(documents, str(STORE_PATH))


def get_store(rebuild: bool = False):
    """요청에 따라 벡터DB를 새로 만들거나 저장된 DB를 불러옵니다."""
    if rebuild:
        return build_store()

    from vectorstore_upstage import load_vectorstore

    return load_vectorstore(str(STORE_PATH))


def print_answer(result: dict) -> None:
    """질문 1개 결과에서 사용자가 확인할 핵심 정보만 출력합니다."""
    print("\n[Question]", result["question"])
    print("\n[Answer]\n" + result["answer"])
    print("\n[Retrieved sources]", result.get("sources", []))
    print("[Actually cited sources]", result.get("used_sources", []))
    print("[Top score]", result.get("top_score"))
    print("[Latency]", result["latency"], "seconds")


def main() -> int:
    args = parse_args()

    # 프로젝트 루트의 .env에서 UPSTAGE_API_KEY를 읽습니다.
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        # --help는 패키지 설치 전에도 볼 수 있게 하고, 실제 실행에서만
        # requirements.txt 설치 필요성을 명확히 안내합니다.
        raise RuntimeError(
            "python-dotenv가 없습니다. "
            "`python -m pip install -r requirements.txt`를 먼저 실행하세요."
        )

    if args.build_store:
        build_store()
        print("\n다음 단계: python src/main_english_upstage.py --evaluate dev")
        return 0

    store = get_store(rebuild=args.rebuild_store)

    # 검색만 볼 때는 Mistral을 로드하지 않아 빠르게 확인할 수 있습니다.
    if args.search_only:
        from rag_chain_english import EnglishRagChain

        chain = EnglishRagChain.__new__(EnglishRagChain)
        # 검색 인덱스 초기화에는 LLM이 필요 없으므로 임시로 None을 둡니다.
        chain.store = store
        chain.top_k = 5
        chain.score_threshold = 0.4
        chain.llm = None
        chain.all_documents = list(getattr(store.docstore, "_dict", {}).values())
        from rank_bm25 import BM25Okapi
        from rag_chain_english import _tokenize

        corpus = [
            _tokenize(
                f"{doc.metadata.get('law_name_en', '')} "
                f"{doc.metadata.get('article_title_en', '')} "
                f"{doc.page_content}"
            )
            for doc in chain.all_documents
        ]
        chain.bm25 = BM25Okapi(corpus)
        result = chain.search(args.search_only)
        print("\n[Question]", result["question"])
        for index, (source, key) in enumerate(
            zip(result["sources"], result["source_keys"]),
            1,
        ):
            print(f"{index}. {source} ({key})")
        print("[Top score]", result["top_score"])
        return 0

    from rag_chain_english import (
        MODEL_NAME,
        SCORE_THRESHOLD,
        SINGLE_MAX_TOKENS,
        MULTI_MAX_TOKENS,
        TOP_K,
        build_rag_chain,
    )

    # ask/interactive/evaluate 모두 같은 체인 객체를 재사용합니다.
    chain = build_rag_chain(store)

    if args.ask:
        print_answer(chain.ask(args.ask))
        return 0

    if args.interactive:
        print("\nEnglish interactive mode. Type exit or quit to stop.")
        while True:
            try:
                question = input("\nQuestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped.")
                return 0
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("Stopped.")
                return 0
            print_answer(chain.ask(question))

    if args.evaluate:
        from evaluate_english import evaluate

        evaluation_dir = PROJECT_ROOT / "eval" / args.evaluate
        run_name = args.run_name or (
            f"mistral7b_english_upstage_{args.evaluate}"
            f"_hybrid_faiss_k{TOP_K}_t{str(SCORE_THRESHOLD).replace('.', '')}"
            f"{'_ragas' if not args.no_ragas else '_debug_no_ragas'}"
        )
        output_csv = PROJECT_ROOT / "eval" / "results" / f"{run_name}.csv"
        experiment = {
            "run_name": run_name,
            "language": "en",
            "model": MODEL_NAME,
            "prompt": "strict",
            "embedding_provider": "upstage",
            "embedding_model": "solar-embedding-1-large",
            "store": "faiss",
            "search": "hybrid",
            "top_k": TOP_K,
            "score_threshold": SCORE_THRESHOLD,
            # 청크는 한국어에서 500/50으로 먼저 만든 뒤 1:1 번역했습니다.
            "source_chunk_language": "ko",
            "chunk_alignment": "ko_to_en_1to1",
            "source_chunk_count": 1953,
            "chunk_size": 500,
            "overlap_size": 50,
            "english_rechunk": False,
            "quantization": "nf4_4bit",
            "device": "cuda",
            "single_max_tokens": SINGLE_MAX_TOKENS,
            "multi_max_tokens": MULTI_MAX_TOKENS,
            "ragas_judge": "upstage" if not args.no_ragas else "disabled",
        }
        evaluate(
            chain,
            questions_csv=str(evaluation_dir / "questions.csv"),
            references_csv=str(evaluation_dir / "references.csv"),
            output_csv=str(output_csv),
            experiment=experiment,
            use_ragas=not args.no_ragas,
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
