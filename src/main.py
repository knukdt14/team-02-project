"""
main.py
---------------------------------------------------
전체 RAG 파이프라인 '실행 진입점'.
앞에서 만든 모듈들을 순서대로 호출만 한다.

    load_data → build_vectorstore → build_rag_chain → evaluate

실험은 이 파일을 뜯어고치는 게 아니라,
아래 CONFIG 값만 바꾸거나 커맨드라인 인자로 넘겨서 '여러 번 실행'하면 된다.
(모델·프롬프트·top_k 등을 바꿔 돌릴 때마다 결과가 results.csv에 저장됨)

실행 예)
  # 기본 설정으로 실행
  python main.py

  # Qwen 파인튜닝 모델로 실행
  python main.py --llm_type hf --model_name ./models/qwen2.5-7b-ft

  # 프롬프트/top_k 바꿔서 실행
  python main.py --prompt_name cite --top_k 5 --search_type mmr

  # 답변 1개만 빠르게 확인(평가 생략)
  python main.py --ask "타워크레인은 순간풍속 얼마에서 멈춰야 하나요?"
"""

import argparse
import re
from pathlib import Path

from load_data import load_data
from build_vectorstore import build_vectorstore
from rag_chain import build_rag_chain
from evaluate import evaluate


def _safe(s) -> str:
    """파일명에 못 쓰는 문자를 _로 치환. 경로(모델명)는 마지막 요소만 사용."""
    s = str(s).replace("\\", "/").rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9A-Za-z가-힣._-]", "_", s)


def make_run_name(args) -> str:
    """run_name이 비어 있으면 설정값으로 자동 파일명 생성."""
    if args.run_name:
        return args.run_name
    model = args.model_name or args.llm_type
    return (f"{_safe(model)}_{args.prompt_name}_k{args.top_k}"
            f"_{args.search_type}_{args.store_type}_{args.embedding_name}")


# =====================================================================
# 기본 설정 (커맨드라인 인자로 덮어쓸 수 있음)
# =====================================================================
CONFIG = dict(
    # 데이터
    data_path="../data/laws_all.json",
    file_type="json",
    chunk_size=500,
    overlap_size=50,
    # 저장소 / 임베딩
    store_type="faiss",
    embedding_name="hf",
    # LLM
    llm_type="hf",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    # 검색 / 프롬프트
    prompt_name="basic",
    top_k=3,
    search_type="similarity",
    # 평가
    questions_csv="../eval/questions.csv",
    results_dir="../results",   # 결과 CSV들을 모아둘 폴더
    run_name="",                # 결과 파일 이름(확장자 제외). 비우면 설정값으로 자동 생성
    use_ragas=False,
)


def parse_args():
    p = argparse.ArgumentParser(description="RAG 파이프라인 실행")
    for k, v in CONFIG.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif isinstance(v, int):
            p.add_argument(f"--{k}", type=int, default=v)
        else:
            p.add_argument(f"--{k}", default=v)
    p.add_argument("--ask", default=None,
                   help="질문 1개만 실행하고 종료(평가 생략)")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) 데이터 로드 + 청킹
    docs = load_data(args.data_path, args.file_type,
                     chunk_size=args.chunk_size, overlap_size=args.overlap_size)

    # 2) 임베딩 + 저장소 구축
    store = build_vectorstore(docs, store_type=args.store_type,
                              embedding_name=args.embedding_name, persist_dir=None)

    # 3) RAG 체인 구성
    chain = build_rag_chain(
        store,
        llm_type=args.llm_type, model_name=args.model_name,
        prompt_name=args.prompt_name, top_k=args.top_k,
        search_type=args.search_type,
    )

    # 4-a) 질문 1개만 확인하고 끝
    if args.ask:
        r = chain.ask(args.ask)
        print("\n[질문]", r["question"])
        print("[답변]", r["answer"])
        print("[출처]", r["sources"])
        print("[응답시간]", r["latency"], "초")
        return

    # 4-b) 평가셋 전체 실행 → results/<run_name>.csv 저장
    run_name = make_run_name(args)
    out_csv = str(Path(args.results_dir) / f"{run_name}.csv")

    exp_config = {
        "run_name": run_name,
        "model": args.model_name or args.llm_type,
        "prompt": args.prompt_name,
        "top_k": args.top_k,
        "search": args.search_type,
        "store": args.store_type,
        "embed": args.embedding_name,
        "chunk": args.chunk_size,
    }
    evaluate(
        chain,
        questions_csv=args.questions_csv,
        out_csv=out_csv,
        exp_config=exp_config,
        use_ragas=args.use_ragas,
    )


if __name__ == "__main__":
    main()
