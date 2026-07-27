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

[Chat Template 적용]
  모든 LLM 호출은 (system, user) 메시지 구조로 전달된다.
  - HF 로컬 모델 : tokenizer.apply_chat_template() 로 모델별 대화 형식을 자동 적용
                   (Qwen의 <|im_start|>, Llama의 헤더 토큰 등을 토크나이저가 알아서 처리
                    → 어떤 HF Instruct 모델을 써도 코드 수정 불필요)
  - API 모델     : LangChain 메시지 [SystemMessage, HumanMessage] 로 전달
  ※ chat template이 없는 구형 모델은 자동으로 일반 문자열 방식으로 폴백.

반환: RagChain 객체.  chain.ask(question) → {answer, contexts, sources, latency}
      (evaluate.py 가 이 결과를 그대로 채점에 사용)

설치(사용하는 것만):
  pip install langchain langchain-core
  pip install transformers accelerate torch      # HuggingFace 로컬 LLM
  pip install langchain-openai                    # OpenAI
  pip install langchain-anthropic                 # Claude
  pip install langchain-upstage                   # Upstage
"""

import re
import time

# .env 파일이 있으면 자동 로드 (UPSTAGE_API_KEY 등을 환경변수로 읽음)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =====================================================================
# 1) 프롬프트 레지스트리 (실험 변수: prompt_name)
#    chat template 적용을 위해 system / user 메시지로 분리.
#    user 템플릿은 {context} 와 {question} 자리표시자를 가진다.
# =====================================================================
_SYSTEM = "당신은 산업안전보건 법령 전문 상담 챗봇입니다."

# 모든 프롬프트 공통: 근거는 [근거 1], [근거 2]… 로 번호가 붙어 제공되며,
# 모델은 실제로 사용한 근거만 (근거 N) 형식으로 표기한다.
# → 프로그램이 (근거 N)을 실제 법령명·조문으로 변환 = 검증된 출처(used_sources)
_CITE_RULE = (" 근거 조각들은 [근거 1], [근거 2]처럼 번호가 붙어 있습니다."
              " 답변에 실제로 사용한 근거만 문장 끝에 (근거 1) 또는 (근거 1, 근거 3)"
              " 형식으로 번호로 표기하세요. 사용하지 않은 근거 번호는 표기하지 마세요.")

REFUSAL_MSG = "산업안전보건법령에서 해당 내용을 찾을 수 없습니다."

PROMPTS = {
    # 기본형
    "basic": {
        "system": _SYSTEM + " 아래 제공되는 법령 근거를 참고하여 질문에 답하세요."
                  + _CITE_RULE,
        "user": "[근거]\n{context}\n\n[질문]\n{question}",
    },

    # 단계적 사고형(Chain-of-Thought)
    "cot": {
        "system": _SYSTEM + " 법령 근거를 바탕으로, 관련 조문을 먼저 짚고 "
                  "단계적으로 생각한 뒤 결론을 내리세요. "
                  "(관련 조문 확인 → 판단 → 결론 순서로)" + _CITE_RULE,
        "user": "[근거]\n{context}\n\n[질문]\n{question}",
    },

    # 근거 인용 강조형
    "cite": {
        "system": _SYSTEM + " 제공된 법령 근거에만 기반해 답하세요."
                  + _CITE_RULE +
                  " 근거 표기가 없는 답변은 무효입니다.",
        "user": "[근거]\n{context}\n\n[질문]\n{question}",
    },

    # 엄격형: 근거 없으면 모른다고 답변 (할루시네이션 억제)
    "strict": {
        "system": _SYSTEM + " 제공된 법령 근거에만 기반해 답하세요. "
                  "근거에서 답을 찾을 수 없으면 "
                  f"\"{REFUSAL_MSG}\"라고만 답하세요. "
                  "추측하거나 지어내지 마세요." + _CITE_RULE,
        "user": "[근거]\n{context}\n\n[질문]\n{question}",
    },
}


def get_prompt(prompt_name: str = "basic") -> dict:
    if prompt_name not in PROMPTS:
        raise ValueError(f"지원하지 않는 prompt_name: '{prompt_name}'. "
                         f"사용 가능: {list(PROMPTS.keys())}")
    return PROMPTS[prompt_name]


# =====================================================================
# 2) LLM 래퍼 — 어떤 모델이든 .chat(system, user) -> str 로 통일
# =====================================================================
class HFChatLLM:
    """
    HuggingFace 로컬 모델 래퍼.
    tokenizer.apply_chat_template() 으로 모델 고유의 대화 형식을 자동 적용한다.
    (Qwen / Llama / EXAONE 등 어떤 Instruct 모델이든 동일 코드로 동작)
    """

    def __init__(self, model_name: str, temperature: float = 0.2,
                 max_new_tokens: int = 512, load_in_4bit: bool = False,
                 trust_remote_code: bool = False):
        """
        trust_remote_code: EXAONE 등 저장소에 모델 정의 코드가 포함된 모델용.
          기본 False → 기존 모델(Qwen/Llama 등)은 동작 변화 없음.
          내장 구조 모델은 True여도 옵션이 무시되므로 무해함.
          ※ 신뢰 가능한 공식 저장소의 모델에만 사용할 것.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)

        # 4비트 양자화: VRAM 부족(예: 8GB GPU에 7B 모델) 시 사용
        quant_cfg = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_cfg,
            trust_remote_code=trust_remote_code,
        )
        self.pipe = pipeline(
            "text-generation", model=model, tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            return_full_text=False,
        )

        # 이 모델이 chat template을 갖고 있는지 (Instruct 모델은 대부분 있음)
        self.has_template = getattr(self.tokenizer, "chat_template", None) is not None
        if not self.has_template:
            print(f"  [경고] '{model_name}' 에 chat template이 없어 "
                  f"일반 문자열 프롬프트로 폴백합니다. (Base 모델일 가능성)")

    def chat(self, system: str, user: str) -> str:
        if self.has_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            # 모델별 형식(<|im_start|> 등)을 토크나이저가 자동 적용
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # chat template 없는 구형/Base 모델 폴백
            prompt = f"{system}\n\n{user}\n\n[답변]\n"

        out = self.pipe(prompt)
        return out[0]["generated_text"].strip()


class APIChatLLM:
    """
    API 모델(LangChain ChatModel) 래퍼.
    SystemMessage / HumanMessage 로 전달 → 각 API가 자체 대화 형식으로 처리.
    """

    def __init__(self, lc_chat_model):
        self.model = lc_chat_model

    def chat(self, system: str, user: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        resp = self.model.invoke([SystemMessage(content=system),
                                  HumanMessage(content=user)])
        return resp.content.strip()


# =====================================================================
# 3) LLM 선택 (실험 변수: llm_type, model_name)
# =====================================================================
def get_llm(llm_type: str = "hf", model_name: str | None = None,
            temperature: float = 0.2, max_new_tokens: int = 512,
            api_key: str | None = None, base_url: str | None = None,
            load_in_4bit: bool = False, trust_remote_code: bool = False):
    """
    반환: .chat(system, user) -> str 을 지원하는 래퍼 객체.

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

    # --- HuggingFace 로컬 모델 (Qwen / Llama / EXAONE / 파인튜닝 모델) ---
    if t == "hf":
        return HFChatLLM(
            model_name or "Qwen/Qwen2.5-7B-Instruct",
            temperature=temperature, max_new_tokens=max_new_tokens,
            load_in_4bit=load_in_4bit, trust_remote_code=trust_remote_code,
        )

    # --- Upstage (Solar) ---
    if t == "upstage":
        from langchain_upstage import ChatUpstage
        return APIChatLLM(ChatUpstage(**_auth(
            model=model_name or "solar-pro", temperature=temperature)))

    # --- OpenAI (ChatGPT) / OpenAI 호환 커스텀(base_url) ---
    if t == "openai":
        from langchain_openai import ChatOpenAI
        return APIChatLLM(ChatOpenAI(**_auth(
            model=model_name or "gpt-4o-mini", temperature=temperature)))

    # --- Anthropic (Claude) ---
    if t == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return APIChatLLM(ChatAnthropic(**_auth(
            model=model_name or "claude-3-5-sonnet-latest",
            temperature=temperature)))

    raise ValueError(f"지원하지 않는 llm_type: '{llm_type}' "
                     f"(hf, openai, anthropic, upstage)")


# =====================================================================
# 4) 검색 문서 → 프롬프트용 컨텍스트 문자열
# =====================================================================
def _jo_label(meta: dict) -> str:
    """조문 표기: 조문표시('제619조의2')가 있으면 우선, 없으면 조문번호."""
    return meta.get("조문표시") or str(meta.get("조문번호", ""))


def _source_name(doc) -> str:
    """문서의 출처 표기: '법령명 제37조'"""
    return f"{doc.metadata.get('법령명','')} {_jo_label(doc.metadata)}".strip()


def format_contexts(docs) -> str:
    """검색 문서에 [근거 N] ID를 붙여 컨텍스트 구성 (citation-ID 방식)."""
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[근거 {i}] {_source_name(d)}\n{d.page_content}")
    return "\n\n".join(blocks)


_CITE_ID_RE = re.compile(r"\(\s*근거\s*([\d,\s근거]+)\)")


def resolve_citations(answer: str, docs) -> tuple[str, list[str]]:
    """
    답변 속 (근거 N) 표기를 해석해:
      1) 실제 사용된 출처 목록(used_sources)을 만들고
      2) 답변의 (근거 N)을 실제 법령명·조문으로 치환한다.
    → sources(검색된 전체)와 달리, used_sources는 '답변이 실제 인용한' 출처.
    """
    used_ids = []
    for m in _CITE_ID_RE.finditer(answer):
        for n in re.findall(r"\d+", m.group(1)):
            i = int(n)
            if 1 <= i <= len(docs) and i not in used_ids:
                used_ids.append(i)

    used_sources = [_source_name(docs[i - 1]) for i in used_ids]

    def _replace(m):
        names = []
        for n in re.findall(r"\d+", m.group(1)):
            i = int(n)
            if 1 <= i <= len(docs):
                nm = _source_name(docs[i - 1])
                if nm not in names:
                    names.append(nm)
        return f"(근거: {', '.join(names)})" if names else m.group(0)

    resolved = _CITE_ID_RE.sub(_replace, answer)
    return resolved, used_sources


# =====================================================================
# 5) RAG 체인 본체
# =====================================================================
class RagChain:
    def __init__(self, store, llm, prompt: dict,
                 top_k: int = 3, search_type: str = "similarity",
                 score_threshold: float = 0.0):
        self.store = store
        self.llm = llm                      # .chat(system, user) 지원 래퍼
        self.system = prompt["system"]
        self.user_template = prompt["user"]
        self.top_k = top_k
        self.search_type = search_type
        self.score_threshold = score_threshold  # 0.0이면 게이트 비활성화
        # MMR용 검색기 (similarity는 점수 조회로 대체)
        search_kwargs = {"k": top_k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = max(top_k * 4, 20)
        self.retriever = store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    def _retrieve_with_scores(self, question: str):
        """검색 + 관련도 점수(0~1, 높을수록 관련). 점수 미지원 스토어는 게이트 생략."""
        try:
            pairs = self.store.similarity_search_with_relevance_scores(
                question, k=self.top_k)
            docs = [d for d, s in pairs]
            top_score = max((s for _, s in pairs), default=0.0)
            return docs, round(float(top_score), 4)
        except Exception:
            return self.retriever.invoke(question), None

    def ask(self, question: str) -> dict:
        t0 = time.time()

        # ① 검색 + 관련도 점수
        docs, top_score = self._retrieve_with_scores(question)

        # ② 점수 게이트: 최고 점수가 임계값 미만이면 LLM을 호출하지 않고 거부
        #    (관련 없는 질문에 억지 근거가 붙는 것을 시스템 차원에서 차단)
        if (self.score_threshold > 0 and top_score is not None
                and top_score < self.score_threshold):
            return {
                "question": question,
                "answer": REFUSAL_MSG,
                "contexts": [],
                "sources": [_source_name(d) for d in docs],  # 디버깅용 기록
                "used_sources": [],
                "retrieved": False,
                "top_score": top_score,
                "latency": round(time.time() - t0, 3),
            }

        # ③ MMR이면 다양성 반영된 문서로 교체 (점수는 게이트용으로만 사용)
        if self.search_type == "mmr":
            docs = self.retriever.invoke(question)

        # ④ [근거 N] ID 부착 컨텍스트 → LLM 호출
        context = format_contexts(docs)
        user_msg = self.user_template.format(context=context, question=question)
        raw_answer = self.llm.chat(self.system, user_msg)

        # ⑤ (근거 N) 해석 → 실제 사용 출처(used_sources) + 조문명 치환
        answer, used_sources = resolve_citations(raw_answer, docs)

        return {
            "question": question,
            "answer": answer,
            "contexts": [d.page_content for d in docs],       # 평가용
            "sources": [_source_name(d) for d in docs],        # 검색된 전체
            "used_sources": used_sources,                      # 답변이 실제 인용한 출처
            "retrieved": True,
            "top_score": top_score,
            "latency": round(time.time() - t0, 3),
        }


# =====================================================================
# 6) 통합 진입점
# =====================================================================
def build_rag_chain(
    store,
    llm_type: str = "hf",
    model_name: str | None = None,
    prompt_name: str = "basic",
    top_k: int = 3,
    search_type: str = "similarity",
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    api_key: str | None = None,
    base_url: str | None = None,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    score_threshold: float = 0.0,
) -> RagChain:
    """
    Args:
        store         : build_vectorstore() 가 만든 저장소
        llm_type      : "hf" / "openai" / "anthropic" / "upstage"   (실험 변수)
        model_name    : 구체적 모델명 또는 파인튜닝 모델 경로         (실험 변수)
        prompt_name   : "basic" / "cot" / "cite" / "strict"          (실험 변수)
        top_k         : 검색 chunk 수                                (실험 변수)
        search_type   : "similarity" / "mmr"                         (실험 변수)
        max_new_tokens: 생성 최대 토큰 수
        api_key       : 미지정 시 환경변수에서 자동 로드(UPSTAGE_API_KEY 등)
        base_url      : OpenAI 호환 커스텀 엔드포인트용
        load_in_4bit  : HF 모델 4비트 양자화 (VRAM 부족 시)
    """
    llm = get_llm(llm_type, model_name, temperature=temperature,
                  max_new_tokens=max_new_tokens,
                  api_key=api_key, base_url=base_url, load_in_4bit=load_in_4bit,
                  trust_remote_code=trust_remote_code)
    prompt = get_prompt(prompt_name)
    chain = RagChain(store, llm, prompt, top_k=top_k, search_type=search_type,
                     score_threshold=score_threshold)
    print(f"[build_rag_chain] LLM={llm_type}({model_name or '기본'}), "
          f"prompt={prompt_name}, top_k={top_k}, search={search_type}, "
          f"threshold={score_threshold or '끔'}")
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
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    docs = load_data(args.path, args.file_type)
    store = build_vectorstore(docs, store_type=args.store_type,
                              embedding_name=args.embedding_name, persist_dir=None)
    chain = build_rag_chain(
        store, llm_type=args.llm_type, model_name=args.model_name,
        prompt_name=args.prompt_name, top_k=args.top_k,
        search_type=args.search_type, load_in_4bit=args.load_in_4bit,
    )

    q = "타워크레인은 순간풍속 얼마를 초과하면 운전을 멈춰야 하나요?"
    r = chain.ask(q)
    print("\n[질문]", r["question"])
    print("[답변]", r["answer"])
    print("[출처]", r["sources"])
    print("[응답시간]", r["latency"], "초")
