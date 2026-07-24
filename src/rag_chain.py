"""
rag_chain.py
---------------------------------------------------
RAG 파이프라인 3단계: 검색(retriever) + LLM 을 연결해 답변을 생성한다.

범용성 설계 (전부 실험 변수로 교체 가능)
  - LLM 선택 (llm_type):
      · "hf"        : HuggingFace 로컬 모델 (Qwen2.5-7B, Llama-3.1-8B, 파인튜닝 모델 등)
      · "openai"    : OpenAI (ChatGPT)
      · "anthropic" : Claude
      · "upstage"   : Upstage Solar
  - 프롬프트 선택 (prompt_name): "basic" / "cot" / "cite" / "strict"
  - 검색 방식 (search_type): "similarity" / "mmr"
  - top_k: 검색해 LLM에 넘길 chunk 수

반환: RagChain 객체.  chain.ask(question) → {answer, contexts, sources, latency}
      (evaluate.py 가 이 결과를 그대로 채점에 사용)

설치(사용하는 것만):
  pip install langchain langchain-core
  pip install transformers accelerate torch      # HuggingFace 로컬 LLM
  pip install langchain-openai                    # OpenAI
  pip install langchain-anthropic                 # Claude
  pip install langchain-upstage                   # Upstage
"""

import time

# .env 파일이 있으면 자동 로드 (UPSTAGE_API_KEY 등을 환경변수로 읽음)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =====================================================================
# 1) 프롬프트 레지스트리 (실험 변수: prompt_name)
#    모든 프롬프트는 {context} 와 {question} 자리표시자를 가진다.
# =====================================================================
_SYSTEM = "당신은 산업안전보건 법령 전문 상담 챗봇입니다."

PROMPTS = {
    # 기본형
    "basic": _SYSTEM + """
아래 법령 근거를 참고하여 질문에 답하세요.

[근거]
{context}

[질문]
{question}

[답변]""",

    # 단계적 사고형(Chain-of-Thought)
    "cot": _SYSTEM + """
아래 법령 근거를 바탕으로, 관련 조문을 먼저 짚고 단계적으로 생각한 뒤 결론을 내리세요.

[근거]
{context}

[질문]
{question}

[답변] (관련 조문 확인 → 판단 → 결론 순서로)""",

    # 근거 인용 강제형
    "cite": _SYSTEM + """
아래 법령 근거에만 기반해 답하고, 답변 끝에 반드시 근거 조문을 (근거: 법령명 제○조) 형식으로 표기하세요.

[근거]
{context}

[질문]
{question}

[답변]""",

    # 엄격형: 근거 없으면 모른다고 답변 (할루시네이션 억제)
    "strict": _SYSTEM + """
아래 법령 근거에만 기반해 답하세요. 근거에서 답을 찾을 수 없으면
"산업안전보건법령에서 해당 내용을 찾을 수 없습니다."라고만 답하세요.
추측하거나 지어내지 마세요.

[근거]
{context}

[질문]
{question}

[답변]""",
}


def get_prompt(prompt_name: str = "basic") -> str:
    if prompt_name not in PROMPTS:
        raise ValueError(f"지원하지 않는 prompt_name: '{prompt_name}'. "
                         f"사용 가능: {list(PROMPTS.keys())}")
    return PROMPTS[prompt_name]


# =====================================================================
# 2) LLM 선택 (실험 변수: llm_type, model_name)
#    반환 객체는 .invoke(text) 를 지원 → _to_text 로 문자열로 정규화
# =====================================================================
def get_llm(llm_type: str = "hf", model_name: str | None = None,
            temperature: float = 0.2, max_new_tokens: int = 512,
            api_key: str | None = None, base_url: str | None = None):
    """
    api_key / base_url 을 넘기지 않으면 각 라이브러리가 환경변수에서 자동으로 읽는다.
      · Upstage    : UPSTAGE_API_KEY
      · OpenAI     : OPENAI_API_KEY   (base_url 지정 시 OpenAI 호환 커스텀 엔드포인트)
      · Anthropic  : ANTHROPIC_API_KEY
    """
    t = llm_type.lower()

    # 지정된 값만 kwargs에 넣는다(None이면 환경변수 사용)
    def _auth(**extra):
        kw = dict(extra)
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        return kw

    # --- HuggingFace 로컬 모델 (Qwen / Llama / 파인튜닝 모델) ---
    if t == "hf":
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        try:
            from langchain_huggingface import HuggingFacePipeline
        except ImportError:
            from langchain_community.llms import HuggingFacePipeline
        import torch

        name = model_name or "Qwen/Qwen2.5-7B-Instruct"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map="auto",              # GPU 있으면 자동 사용
        )
        pipe = pipeline(
            "text-generation", model=mdl, tokenizer=tok,
            max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=temperature > 0, return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=pipe)

    # --- Upstage (Solar) ---
    if t == "upstage":
        from langchain_upstage import ChatUpstage
        return ChatUpstage(**_auth(model=model_name or "solar-pro",
                                    temperature=temperature))

    # --- OpenAI (ChatGPT) / OpenAI 호환 커스텀(base_url) ---
    if t == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**_auth(model=model_name or "gpt-4o-mini",
                                  temperature=temperature))

    # --- Anthropic (Claude) ---
    if t == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**_auth(model=model_name or "claude-3-5-sonnet-latest",
                                     temperature=temperature))

    raise ValueError(f"지원하지 않는 llm_type: '{llm_type}' "
                     f"(hf, openai, anthropic, upstage)")


def _to_text(resp) -> str:
    """LLM 응답을 문자열로 정규화 (ChatModel은 .content, LLM은 문자열)."""
    if hasattr(resp, "content"):
        return resp.content
    return str(resp)


# =====================================================================
# 3) 검색 문서 → 프롬프트용 컨텍스트 문자열
# =====================================================================
def format_contexts(docs) -> str:
    blocks = []
    for d in docs:
        law = d.metadata.get("법령명", "")
        jo = d.metadata.get("조문번호", "")
        tag = f"[{law} {jo}]".strip()
        blocks.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(blocks)


# =====================================================================
# 4) RAG 체인 본체
# =====================================================================
class RagChain:
    def __init__(self, store, llm, prompt_template: str,
                 top_k: int = 3, search_type: str = "similarity"):
        self.llm = llm
        self.prompt_template = prompt_template
        # 검색기 구성 (search_type / top_k 실험 변수)
        search_kwargs = {"k": top_k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = max(top_k * 4, 20)  # MMR 후보군
        self.retriever = store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    def ask(self, question: str) -> dict:
        t0 = time.time()

        # ① 검색
        docs = self.retriever.invoke(question)
        context = format_contexts(docs)

        # ② 프롬프트 완성 후 LLM 호출
        prompt = self.prompt_template.format(context=context, question=question)
        answer = _to_text(self.llm.invoke(prompt)).strip()

        latency = round(time.time() - t0, 3)

        return {
            "question": question,
            "answer": answer,
            "contexts": [d.page_content for d in docs],          # 평가용
            "sources": [f"{d.metadata.get('법령명','')} "
                        f"{d.metadata.get('조문번호','')}".strip() for d in docs],
            "latency": latency,                                  # 응답시간(초)
        }


# =====================================================================
# 5) 통합 진입점
# =====================================================================
def build_rag_chain(
    store,
    llm_type: str = "hf",
    model_name: str | None = None,
    prompt_name: str = "basic",
    top_k: int = 3,
    search_type: str = "similarity",
    temperature: float = 0.2,
    api_key: str | None = None,
    base_url: str | None = None,
) -> RagChain:
    """
    Args:
        store       : build_vectorstore() 가 만든 저장소
        llm_type    : "hf" / "openai" / "anthropic" / "upstage"   (실험 변수)
        model_name  : 구체적 모델명 또는 파인튜닝 모델 경로         (실험 변수)
        prompt_name : "basic" / "cot" / "cite" / "strict"          (실험 변수)
        top_k       : 검색 chunk 수                                (실험 변수)
        search_type : "similarity" / "mmr"                         (실험 변수)
        api_key     : 미지정 시 환경변수에서 자동 로드(UPSTAGE_API_KEY 등)
        base_url    : OpenAI 호환 커스텀 엔드포인트용
    """
    llm = get_llm(llm_type, model_name, temperature=temperature,
                  api_key=api_key, base_url=base_url)
    prompt = get_prompt(prompt_name)
    chain = RagChain(store, llm, prompt, top_k=top_k, search_type=search_type)
    print(f"[build_rag_chain] LLM={llm_type}({model_name or '기본'}), "
          f"prompt={prompt_name}, top_k={top_k}, search={search_type}")
    return chain


# =====================================================================
# 단독 실행 테스트
# =====================================================================
if __name__ == "__main__":
    import argparse
    from load_data import load_data
    from build_vectorstore import build_vectorstore

    parser = argparse.ArgumentParser(description="RAG 체인 테스트")
    parser.add_argument("path")
    parser.add_argument("--file_type", required=True, choices=["json", "pdf"])
    parser.add_argument("--store_type", default="faiss")
    parser.add_argument("--embedding_name", default="hf")
    parser.add_argument("--llm_type", default="openai")   # 테스트는 API가 빠름
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--prompt_name", default="cite")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--search_type", default="similarity")
    args = parser.parse_args()

    docs = load_data(args.path, args.file_type)
    store = build_vectorstore(docs, store_type=args.store_type,
                              embedding_name=args.embedding_name, persist_dir=None)
    chain = build_rag_chain(
        store, llm_type=args.llm_type, model_name=args.model_name,
        prompt_name=args.prompt_name, top_k=args.top_k, search_type=args.search_type,
    )

    q = "타워크레인은 순간풍속 얼마를 초과하면 운전을 멈춰야 하나요?"
    r = chain.ask(q)
    print("\n[질문]", r["question"])
    print("[답변]", r["answer"])
    print("[출처]", r["sources"])
    print("[응답시간]", r["latency"], "초")
