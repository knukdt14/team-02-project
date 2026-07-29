"""
main_copy_ksj.py
---------------------------------------------------
김상준(KSJ)의 Mistral-7B RAG 비교 실험 전용 실행 파일입니다.

팀 공용 main.py와 공용 모듈은 수정하지 않습니다. 이 파일은 아래 KSJ 전용
모듈만 불러오기 때문에 다른 팀원이 Qwen, EXAONE, Llama 등을 실행할 때
Mistral 설정이나 KSJ 검색 로직이 개입하지 않습니다.

    load_data(공용)
      → ksj_vectorstore
      → ksj_rag_chain
      → ksj_evaluate

기본 실험 조건:
  - LLM       : mistralai/Mistral-7B-Instruct-v0.3
  - 양자화    : NF4 4비트
  - 실행 장치 : CUDA 강제
  - 임베딩    : solar-embedding-1-large(Upstage API)
  - 검색      : FAISS + hybrid, top_k=5
  - 검색 게이트: 비활성화(Upstage 점수 분포 확인 후 별도 보정)
  - 프롬프트  : strict
  - 생성 길이 : 단순질문 180 / 복합질문 300 토큰

실행 예:
  python src/main_copy_ksj.py --ask "타워크레인 작업 중지 기준은?"
  python src/main_copy_ksj.py --interactive
  python src/main_copy_ksj.py
"""

import argparse
import re
from pathlib import Path


# 이 파일을 어느 폴더에서 실행하더라도 같은 data/eval/stores 경로를 사용합니다.
# 예: 프로젝트 루트와 src 폴더 중 어디에서 실행해도 경로가 달라지지 않습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe(s) -> str:
    """파일명에 못 쓰는 문자를 _로 치환. 경로(모델명)는 마지막 요소만 사용."""
    s = str(s).replace("\\", "/").rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9A-Za-z가-힣._-]", "_", s)


def make_run_name(args) -> str:
    """run_name이 비어 있으면 설정값으로 자동 파일명 생성."""
    if args.run_name:
        return args.run_name
    model = args.model_name or args.llm_type
    # 같은 HF 통합 이름을 사용해도 실제 임베딩 모델은 다를 수 있습니다.
    # 자동 결과 파일명에 구체 모델명을 넣어 실험 결과가 덮어써지지 않게 합니다.
    embedding_label = args.embedding_model or args.embedding_name
    return (f"{_safe(model)}_{args.prompt_name}_k{args.top_k}"
            f"_{args.search_type}_{args.store_type}_{_safe(embedding_label)}")


# =====================================================================
# 기본 설정 (커맨드라인 인자로 덮어쓸 수 있음)
# =====================================================================
CONFIG = dict(
    # 데이터
    data_path=str(PROJECT_ROOT / "data" / "laws_all.json"),
    file_type="json",
    chunk_size=500,
    overlap_size=50,
    # 저장소 / 임베딩
    store_type="faiss",
    # 한국어 법령 원문을 Upstage의 원격 임베딩 API로 변환합니다.
    # 기존 ko-sbert FAISS와 섞이지 않도록 제공자와 모델명이 저장 경로에 포함됩니다.
    embedding_name="upstage",
    embedding_model="solar-embedding-1-large",
    # Upstage 임베딩은 원격 API에서 계산되므로 아래 장치값은 사용되지 않습니다.
    # Mistral-7B 답변 생성은 기존 설정대로 로컬 CUDA를 사용합니다.
    embedding_device="cpu",
    store_dir=str(PROJECT_ROOT / "stores"),  # 저장된 벡터DB를 다음 실행에서 재사용
    rebuild_store=False,       # True면 저장된 것 무시하고 새로 생성
    # 사용할 RAG 체인을 선택합니다.
    # "public": 팀 공용 rag_chain.py
    # "ksj": 기존 14문항으로 튜닝한 ksj_rag_chain.py
    # "generic": 정답 조문 강제 규칙을 제거한 ksj_rag_chain_generic.py
    chain_variant="generic",
    # 검색 게이트
    # 임베딩을 바꾸면 유사도 점수 분포도 달라지므로 ko-sbert에서 정한 0.50을
    # 그대로 사용하면 안 됩니다. 첫 Upstage 비교에서는 게이트를 끄고 평가한 뒤,
    # 답변 가능/불가능 문항의 실제 점수 분포를 보고 전용 임계값을 정합니다.
    score_threshold=0.4,
    # LLM
    llm_type="hf",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True,         # 7B FP16은 8GB에 안 들어가므로 NF4 4비트가 필수
    force_cuda=True,           # CPU로 조용히 폴백하지 않고 CUDA 미지원 시 즉시 오류
    temperature=0.0,           # 법령 답변은 무작위성을 끄고 재현성을 확보
    # 속도 실험에서는 프롬프트를 간결하게 만들고 단일/복합 질문의 생성 상한을
    # 각각 180/300 토큰으로 제한합니다. 평가 후 truncation_rate도 함께 확인합니다.
    max_new_tokens=180,
    multi_max_new_tokens=300,
    repetition_penalty=1.05,   # 같은 문장 반복을 약하게 억제
    # 검색 / 프롬프트
    prompt_name="strict",
    top_k=5,
    # dense 임베딩 + 한국어 법률 용어 BM25를 합쳐 조문 누락을 줄입니다.
    search_type="hybrid",
    # 평가
    questions_csv=str(PROJECT_ROOT / "eval" / "questions.csv"),
    results_dir=str(PROJECT_ROOT / "eval" / "results"),
    run_name="",                 # 비워 두면 모델·설정으로 중복 없는 이름을 자동 생성
    use_ragas=False,             # API 키와 비용이 필요한 RAGAS는 명시적으로 켤 때만 실행
    ragas_judge="upstage",      # RAGAS 심판: "upstage"(UPSTAGE_API_KEY) 또는 "openai"
)


def parse_args():
    p = argparse.ArgumentParser(description="RAG 파이프라인 실행")
    for k, v in CONFIG.items():
        if isinstance(v, bool):
            # True/False 기본값 모두 명령행에서 뒤집을 수 있게 합니다.
            # 예: --load_in_4bit / --no-load_in_4bit, --use_ragas / --no-use_ragas
            p.add_argument(
                f"--{k}",
                action=argparse.BooleanOptionalAction,
                default=v,
            )
        elif isinstance(v, int):
            p.add_argument(f"--{k}", type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(f"--{k}", type=float, default=v)
        else:
            p.add_argument(f"--{k}", default=v)
    p.add_argument("--ask", default=None,
                   help="질문 1개만 실행하고 종료(평가 생략)")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="모델을 한 번만 로드한 뒤 질문을 반복 입력하는 대화형 모드",
    )
    return p.parse_args()


def get_or_build_store(args):
    """
    벡터DB 저장/재사용:
      같은 (데이터, chunk, overlap, store_type, embedding model) 조합이면
      저장된 스토어를 로드해 재임베딩을 건너뛴다. (--rebuild_store 로 강제 재생성)

    중요:
      임베딩 모델이 다르면 벡터 차원과 값이 달라지므로 기존 FAISS를 재사용하면
      안 됩니다. 구체 모델명을 저장 경로 키에 넣어 모델별 DB를 분리합니다.
    """
    # --help는 패키지가 설치되기 전에도 확인할 수 있도록 무거운 프로젝트 모듈은
    # 실제 벡터스토어가 필요할 때 지연 import합니다.
    from load_data import load_data
    from ksj_vectorstore import build_vectorstore, load_vectorstore

    # 기존 기본 모델은 예전 stores/..._hf 경로를 그대로 사용해 호환성을 유지합니다.
    # BGE-M3처럼 구체 모델을 지정한 실험만 별도 경로를 만듭니다.
    embedding_key = args.embedding_name
    if args.embedding_model:
        embedding_key = f"{embedding_key}_{_safe(args.embedding_model)}"

    key = (f"{_safe(Path(args.data_path).stem)}_c{args.chunk_size}"
           f"_o{args.overlap_size}_{args.store_type}_{embedding_key}")
    persist_dir = str(Path(args.store_dir) / key)
    embedding_model = args.embedding_model or None

    can_persist = args.store_type in ("faiss", "chroma")  # neo4j는 DB 자체 저장
    exists = Path(persist_dir).exists() and any(Path(persist_dir).iterdir()) \
        if Path(persist_dir).exists() else False

    if can_persist and exists and not args.rebuild_store:
        print(f"[store] 저장된 벡터DB 재사용: {persist_dir} (재임베딩 생략)")
        return load_vectorstore(store_type=args.store_type,
                                embedding_name=args.embedding_name,
                                embedding_model=embedding_model,
                                persist_dir=persist_dir,
                                embedding_device=args.embedding_device)

    # 1) 데이터 로드 + 청킹 → 2) 임베딩 + 저장
    docs = load_data(args.data_path, args.file_type,
                     chunk_size=args.chunk_size, overlap_size=args.overlap_size)
    store = build_vectorstore(docs, store_type=args.store_type,
                              embedding_name=args.embedding_name,
                              embedding_model=embedding_model,
                              persist_dir=persist_dir if can_persist else None,
                              embedding_device=args.embedding_device)
    if can_persist:
        print(f"[store] 벡터DB 저장 완료: {persist_dir} (다음 실행부터 재사용)")
    return store


def main():
    args = parse_args()

    # 이 파일은 KSJ 전용 모듈만 불러오므로 팀 공용 파이프라인에는 영향이 없습니다.
    # 평가 코드는 두 체인 모두 같은 ksj_evaluate.py를 사용합니다.
    # 그래야 평가 방식까지 달라지는 것을 방지할 수 있습니다.
    from ksj_evaluate import evaluate

    # 명령행 옵션에 따라 공용 체인과 KSJ 체인 중 하나를 선택합니다.
    if args.chain_variant == "public":
        from rag_chain import build_rag_chain
    elif args.chain_variant == "ksj":
        from ksj_rag_chain import build_rag_chain
    elif args.chain_variant == "generic":
        from ksj_rag_chain_generic import build_rag_chain
    else:
        raise ValueError(
            f"지원하지 않는 chain_variant입니다: {args.chain_variant}"
        )

    # 1~2) 벡터DB 로드 또는 생성 (저장/재사용)
    store = get_or_build_store(args)

    # 3) RAG 체인 구성
    # 두 체인에서 공통으로 사용할 실험 조건입니다.
    common_chain_args = {
        "llm_type": args.llm_type,
        "model_name": args.model_name,
        "prompt_name": args.prompt_name,
        "top_k": args.top_k,
        "search_type": args.search_type,
        "load_in_4bit": args.load_in_4bit,
        "score_threshold": args.score_threshold,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }

    if args.chain_variant == "public":
        # 공용 rag_chain.py는 hybrid를 지원하지 않습니다.
        if args.search_type == "hybrid":
            raise ValueError(
                "공용 rag_chain.py는 hybrid를 지원하지 않습니다. "
                "--search_type similarity 또는 mmr을 사용하세요."
            )

        # 공용 체인이 지원하는 인자만 전달합니다.
        chain = build_rag_chain(
            store,
            **common_chain_args,
        )

    else:
        # KSJ 체인에는 동적 출력 길이, 반복 억제, CUDA 강제 옵션이 있습니다.
        chain = build_rag_chain(
            store,
            **common_chain_args,
            multi_max_new_tokens=args.multi_max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            force_cuda=args.force_cuda,
        )

    # 4-a) 질문 1개만 확인하고 끝
    if args.ask:
        r = chain.ask(args.ask)
        print("\n[질문]", r["question"])
        print("[답변]", r["answer"])
        print("[모델 원문]", r.get("raw_answer", r["answer"]))
        print("[검색된 근거]", r["sources"])
        print("[LLM에 전달한 근거]", r.get("prompt_sources", r["sources"]))
        print("[모델 원문 인용]", r.get("model_used_sources", []))
        print("[검증된 사용 출처]", r.get("used_sources", []))
        print("[인용 검증 후 변경]", r.get("citation_repaired", False))
        print("[인용 상태]", r.get("citation_status"))
        print(
            "[질문 유형/생성 한도]",
            r.get("question_mode"),
            r.get("max_new_tokens_used"),
        )
        print(
            "[초기 잘림/자동 재생성/최종 잘림]",
            r.get("initial_truncation_detected"),
            r.get("generation_retried"),
            r.get("truncation_detected"),
        )
        print(
            "[초기 생성토큰/최종 생성토큰/최종 한도도달]",
            r.get("initial_generated_tokens"),
            r.get("generated_tokens"),
            r.get("hit_token_limit"),
        )
        print("[품질 재작성 사유]", r.get("quality_retry_reasons", []))
        print("[최종 잔여 품질 문제]", r.get("quality_issues_remaining", []))
        print("[최고 관련도]", r.get("top_score"), "| 검색통과:", r.get("retrieved"))
        print("[응답시간]", r["latency"], "초")
        return

    # 4-b) 대화형 모드:
    # 모델과 벡터DB를 메모리에 올린 상태로 계속 재사용하므로 질문할 때마다
    # Python 프로세스와 Mistral 모델을 다시 로드하는 시간을 없앨 수 있습니다.
    if args.interactive:
        print("\n[대화형 모드 시작]")
        print("질문을 입력하세요. 종료하려면 exit 또는 quit를 입력하세요.")

        while True:
            try:
                question = input("\n질문> ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl+Z/Enter 또는 Ctrl+C로도 안전하게 종료합니다.
                print("\n대화형 모드를 종료합니다.")
                return

            # 빈 입력은 모델에 전달하지 않고 다음 입력을 기다립니다.
            if not question:
                continue

            # 영문 대소문자와 관계없이 exit/quit를 종료 명령으로 처리합니다.
            if question.lower() in {"exit", "quit"}:
                print("대화형 모드를 종료합니다.")
                return

            # 이미 로드된 동일한 RAG 체인과 Mistral 모델로 답변합니다.
            result = chain.ask(question)
            print("\n[답변]", result["answer"])
            print(
                "[사용 출처]",
                result.get("used_sources")
                or result.get("prompt_sources")
                or result.get("sources", []),
            )
            print("[응답시간]", result["latency"], "초")

    # 4-c) 평가셋 전체 실행 → results/<run_name>.csv 저장
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
        # CSV에 구체 임베딩 모델을 남겨 hf 실험끼리도 구분할 수 있게 합니다.
        "embedding_model": (
            args.embedding_model or "jhgan/ko-sbert-nli"
        ),
        "chunk": args.chunk_size,
        "single_max_tokens": args.max_new_tokens,
        "multi_max_tokens": args.multi_max_new_tokens,
        "quantization": "nf4_4bit" if args.load_in_4bit else "none",
        "device": "cuda" if args.force_cuda else "auto",
        "chain_variant": args.chain_variant,
    }
    evaluate(
        chain,
        questions_csv=args.questions_csv,
        out_csv=out_csv,
        exp_config=exp_config,
        use_ragas=args.use_ragas,
        ragas_judge=args.ragas_judge,
    )


if __name__ == "__main__":
    main()
