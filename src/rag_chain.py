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
  - 검색 방식 (search_type): "similarity" / "mmr" / "hybrid"
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

import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

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
_SYSTEM = (
    "당신은 산업안전보건 법령 전문 상담 챗봇입니다. "
    "질문에 직접 필요한 내용만 한국어로 간결하게 답합니다. "
    "단순 질문은 첫 문장에 결론을 쓰고, 필요할 때만 한두 문장을 덧붙입니다. "
    "복합 질문은 제공된 항목별 형식을 따라 빠짐없이 답합니다."
)

# 모든 검색 근거에는 [C1], [C2]처럼 짧은 ID를 붙입니다.
# 모델은 실제 사용한 ID만 답변에 표기하고, 프로그램이 이를 법령명·조문으로 변환합니다.
# 이렇게 해야 '검색된 상위 5개'와 '답변이 실제로 인용한 출처'를 구분할 수 있습니다.
_CITE_RULE = (
    " 제공된 근거에는 [C1], [C2]처럼 ID가 있습니다. "
    "답변에 실제 사용한 근거 ID만 문장 끝에 [C1] 또는 [C1, C3] 형식으로 표기하세요. "
    "근거 ID는 법령 조문 번호가 아니라 입력 근거의 식별자입니다. "
    "근거에 없는 조문·수치·사실을 추가하지 마세요."
)

REFUSAL_MSG = "산업안전보건법령에서 해당 내용을 찾을 수 없습니다."

PROMPTS = {
    # 기본형
    "basic": {
        "system": _SYSTEM + " 아래 제공되는 법령 근거를 참고하여 질문에 답하세요."
                  + _CITE_RULE,
        "user": "[근거]\n{context}\n\n[질문]\n{question}",
        "require_citation": False,
    },

    # 단계적 사고형(Chain-of-Thought)
    "cot": {
        "system": _SYSTEM + " 법령 근거를 바탕으로, 관련 조문을 먼저 짚고 "
                  "단계적으로 생각한 뒤 결론을 내리세요. "
                  "(관련 조문 확인 → 판단 → 결론 순서로)" + _CITE_RULE,
        "user": "[근거]\n{context}\n\n[질문]\n{question}",
        "require_citation": True,
    },

    # 근거 인용 강조형
    "cite": {
        "system": _SYSTEM + " 제공된 법령 근거에만 기반해 답하세요."
                  + _CITE_RULE +
                  " 근거 표기가 없는 답변은 무효입니다.",
        "user": "[근거]\n{context}\n\n[질문]\n{question}",
        "require_citation": True,
    },

    # 엄격형: 근거 없으면 모른다고 답변 (할루시네이션 억제)
    "strict": {
        "system": _SYSTEM + " 제공된 법령 근거에만 기반해 답하세요. "
                  "질문 전체가 아니라 일부만 근거로 확인되면, 확인되는 부분만 답하고 "
                  "확인되지 않는 부분을 분명히 밝히세요. "
                  "관련 근거가 전혀 없을 때만 "
                  f"\"{REFUSAL_MSG}\"라고만 답하세요. "
                  "추측하거나 지어내지 마세요. 단순 질문은 원칙적으로 1~3문장으로 "
                  "작성하고, 복합 질문은 사용자 메시지의 출력 항목을 따르세요."
                  + _CITE_RULE,
        "user": "[근거]\n{context}\n\n[질문]\n{question}",
        "require_citation": True,
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

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_new_tokens: int = 160,
        load_in_4bit: bool = False,
        repetition_penalty: float = 1.05,
        force_cuda: bool = False,
    ):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        print(f"[HF LLM] Transformers={transformers.__version__}")

        # force_cuda=True이면 CUDA가 없을 때 CPU로 몰래 폴백하지 않고 즉시 중단합니다.
        # 평가 중 CPU 폴백으로 문항당 수분이 걸리는 상황을 방지하기 위한 안전장치입니다.
        if force_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA를 사용할 수 없습니다. CUDA PyTorch 설치와 "
                "`python -c \"import torch; print(torch.cuda.is_available())\"` "
                "결과를 확인하세요."
            )

        # QLoRA 학습 후 저장된 어댑터 폴더도 model_name으로 받을 수 있습니다.
        # adapter_config.json이 있으면 기반 모델을 먼저 읽고 LoRA 어댑터를 결합합니다.
        adapter_dir = Path(model_name)
        is_adapter = adapter_dir.is_dir() and (
            adapter_dir / "adapter_config.json"
        ).exists()
        base_model_name = model_name
        if is_adapter:
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(model_name)
            base_model_name = peft_config.base_model_name_or_path

        # 어댑터 폴더에 토크나이저가 없을 수 있으므로 기반 모델 토크나이저를 사용합니다.
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4비트 양자화: VRAM 부족(예: 8GB GPU에 7B 모델) 시 사용
        quant_cfg = None
        compute_dtype = torch.float32
        if torch.cuda.is_available():
            # RTX 50 계열은 BF16을 지원하므로 가능하면 BF16 연산을 사용합니다.
            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4비트 Mistral 실행에는 CUDA GPU가 필요합니다.")
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        # 8GB GPU에서 CPU 오프로딩을 허용하면 매우 느려질 수 있으므로,
        # CUDA 사용 시 모델 전체를 GPU 0번에 명시적으로 배치합니다.
        model_kwargs = {
            "low_cpu_mem_usage": True,
            # 사용자의 Transformers 버전은 from_pretrained(dtype=...)를 지원하지 않습니다.
            # torch_dtype는 구·신버전에서 모두 동작하며, 신버전의 경고는 실행에 영향이 없습니다.
            "torch_dtype": compute_dtype,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["device_map"] = {"": "cpu"}
        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs,
        )

        if is_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_name)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )
        self.generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            self.generation_kwargs["temperature"] = temperature

        if torch.cuda.is_available():
            used_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
            print(
                f"[HF LLM] GPU={torch.cuda.get_device_name(0)}, "
                f"4bit={load_in_4bit}, 할당 VRAM={used_mb:.0f}MB"
            )
        else:
            print("[HF LLM] 장치=CPU")

        # 이 모델이 chat template을 갖고 있는지 (Instruct 모델은 대부분 있음)
        self.has_template = getattr(self.tokenizer, "chat_template", None) is not None
        if not self.has_template:
            print(f"  [경고] '{model_name}' 에 chat template이 없어 "
                  f"일반 문자열 프롬프트로 폴백합니다. (Base 모델일 가능성)")

    def chat(
        self,
        system: str,
        user: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """질문 난이도에 따라 이번 호출의 생성 길이만 안전하게 덮어씁니다."""
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

        # 객체의 기본 설정은 건드리지 않고 이번 질문용 사본만 수정합니다.
        generation_kwargs = dict(self.generation_kwargs)
        if max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = int(max_new_tokens)
        out = self.pipe(prompt, **generation_kwargs)
        return out[0]["generated_text"].strip()


class APIChatLLM:
    """
    API 모델(LangChain ChatModel) 래퍼.
    SystemMessage / HumanMessage 로 전달 → 각 API가 자체 대화 형식으로 처리.
    """

    def __init__(self, lc_chat_model):
        self.model = lc_chat_model

    def chat(
        self,
        system: str,
        user: str,
        max_new_tokens: int | None = None,
    ) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        # API 모델은 공급자마다 길이 인자명이 다를 수 있습니다. LangChain이
        # max_tokens를 지원하는 경우에만 동적 길이를 적용하고, 아니면 기본값을 씁니다.
        if max_new_tokens is not None:
            try:
                resp = self.model.bind(max_tokens=int(max_new_tokens)).invoke(messages)
            except (TypeError, ValueError):
                resp = self.model.invoke(messages)
        else:
            resp = self.model.invoke(messages)
        return resp.content.strip()


# =====================================================================
# 3) LLM 선택 (실험 변수: llm_type, model_name)
# =====================================================================
def get_llm(llm_type: str = "hf", model_name: str | None = None,
            temperature: float = 0.0, max_new_tokens: int = 160,
            api_key: str | None = None, base_url: str | None = None,
            load_in_4bit: bool = False,
            repetition_penalty: float = 1.05,
            force_cuda: bool = False):
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

    # --- HuggingFace 로컬 모델 (Qwen / Llama / 파인튜닝 모델) ---
    if t == "hf":
        return HFChatLLM(
            model_name or "mistralai/Mistral-7B-Instruct-v0.3",
            temperature=temperature, max_new_tokens=max_new_tokens,
            load_in_4bit=load_in_4bit,
            repetition_penalty=repetition_penalty,
            force_cuda=force_cuda,
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
    """검색 문서에 [C1], [C2] ID를 붙여 컨텍스트를 구성합니다."""
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[C{i}] {_source_name(d)}\n{d.page_content}")
    return "\n\n".join(blocks)


def _question_plan(
    question: str,
    single_max_tokens: int = 160,
    multi_max_tokens: int = 280,
) -> dict:
    """
    질문 형태를 보고 출력 항목과 생성 길이를 결정합니다.

    특정 정답이나 조문 번호를 프롬프트에 미리 넣지 않고, 사용자가 물은 하위
    질문을 빠뜨리지 않도록 '답변 구조'만 지정합니다. 따라서 평가 문항뿐 아니라
    새로운 복합 질문에도 같은 규칙을 적용할 수 있습니다.
    """
    compact = re.sub(r"\s+", "", question)
    sections: list[str] = []
    mode = "single"

    # 작업 기준·일반 의무·사망 결과를 한꺼번에 묻는 질문입니다.
    if (
        "사망" in compact
        and any(word in compact for word in ("처벌", "징역", "벌금", "위반"))
        and any(
            word in compact
            for word in (
                "타워크레인", "크레인", "밀폐공간", "풍속",
                "공기기준", "산소", "질식",
            )
        )
    ):
        mode = "duty_and_penalty"
        sections = ["작업 또는 위험 기준", "사업주의 법적 의무", "위반해 사망한 경우의 처벌"]

    # 풍속이 변하는 시간 순서 질문은 사전 중지와 사후 점검이 섞이지 않게 합니다.
    elif (
        any(word in compact for word in ("단계별", "점점", "예상", "분뒤", "초과후"))
        and any(word in compact for word in ("풍속", "강풍", "폭풍", "타워크레인", "양중기"))
    ):
        mode = "timeline"
        sections = ["작업 중지 기준", "강풍 예상 시 사전 조치", "강풍이 지난 뒤 사후 조치"]

    # '각각/동시에/비교/그리고' 등으로 여러 대상을 묻는 일반 복합질문입니다.
    elif any(
        word in compact
        for word in ("각각", "동시에", "비교", "단계별", "그리고", "뿐만아니라")
    ):
        mode = "multi"
        sections = ["질문에 포함된 각 요구사항"]

    if mode == "single":
        return {
            "mode": mode,
            "max_new_tokens": int(single_max_tokens),
            "instruction": (
                "[출력 지침]\n"
                "- 결론부터 1~3문장으로 답하세요.\n"
                "- 수치·의무·예외는 근거에 있는 내용만 쓰고 실제 사용한 [C번호]를 붙이세요."
            ),
        }

    section_lines = "\n".join(f"- [{name}]" for name in sections)
    return {
        "mode": mode,
        # 복합질문만 여유를 주되, 각 항목 한두 문장 제한으로 장황함을 막습니다.
        "max_new_tokens": int(multi_max_tokens),
        "instruction": (
            "[복합질문 출력 지침]\n"
            f"{section_lines}\n"
            "- 위 항목을 빠짐없이, 항목당 1~2문장으로 작성하세요.\n"
            "- 각 항목 끝에 그 항목을 직접 뒷받침하는 [C번호]만 붙이세요.\n"
            "- 근거에서 확인되지 않는 항목은 추측하지 말고 '근거에서 확인되지 않음'이라고 쓰세요."
        ),
    }


# 새 표기([C1], [C1, C3])와 기존 표기((근거 1), [근거 1])를 모두 읽습니다.
# 기존 결과와의 호환성을 유지하면서 앞으로는 짧고 오류가 적은 C-ID를 사용합니다.
_CITE_ID_RE = re.compile(
    r"(?:\[|\()\s*(?:C|근거)\s*"
    r"(\d+(?:\s*[,/]\s*(?:(?:C|근거)\s*)?\d+)*)"
    r"\s*(?:\]|\))",
    re.IGNORECASE,
)

# Mistral 계열이 괄호 없이 "근거 1"만 출력하는 경우도 실제 인용으로 읽습니다.
_LOOSE_CITE_ID_RE = re.compile(
    r"(?<!\[)(?<!\()\b(?:C|근거)\s*(\d+)\b",
    re.IGNORECASE,
)


def _extract_citation_ids(answer: str, docs) -> list[int]:
    """모델 원문에 실제로 적힌 C-ID와 직접 표기한 법령 조문을 추출합니다."""
    used_ids: list[int] = []
    for match in _CITE_ID_RE.finditer(answer):
        for number in re.findall(r"\d+", match.group(1)):
            index = int(number)
            if 1 <= index <= len(docs) and index not in used_ids:
                used_ids.append(index)

    for match in _LOOSE_CITE_ID_RE.finditer(answer):
        index = int(match.group(1))
        if 1 <= index <= len(docs) and index not in used_ids:
            used_ids.append(index)

    # C-ID 대신 법령명과 조문을 직접 적은 경우도 검색 문서 안에 있을 때만 인정합니다.
    compact_answer = re.sub(r"\s+", "", answer)
    for index, doc in enumerate(docs, 1):
        law_name = str(doc.metadata.get("법령명", "")).strip()
        article = _jo_label(doc.metadata).strip()
        if not law_name or not article:
            continue
        aliases = [f"{law_name}{article}"]
        short_law = (
            "시행령" if "시행령" in law_name
            else "규칙" if "규칙" in law_name
            else "산업안전보건법"
        )
        aliases.append(f"{short_law}{article}")
        annex = re.fullmatch(r"별표0*(\d+)(?:의0*\d+)?", article)
        if annex:
            human_annex = f"별표{int(annex.group(1))}"
            aliases.extend([f"{law_name}{human_annex}", f"{short_law}{human_annex}"])
        if any(re.sub(r"\s+", "", alias) in compact_answer for alias in aliases):
            if index not in used_ids:
                used_ids.append(index)
    return used_ids


def _claim_support_score(claim: str, doc) -> float:
    """
    한 답변 문장이 한 검색 문서에 실제로 뒷받침되는 정도를 계산합니다.

    수치가 있는 법령 답변은 같은 수치가 문서에 없으면 크게 감점합니다.
    이 규칙으로 '15m/s 답변에 30m/s 조문을 인용'하는 오류를 차단합니다.
    """
    clean_claim = _CITE_ID_RE.sub(" ", claim)
    clean_claim = _LOOSE_CITE_ID_RE.sub(" ", clean_claim)
    clean_claim = re.sub(r"^\s*[-*•]?\s*\d+[.)]\s*", "", clean_claim)
    doc_text = (
        f"{doc.metadata.get('법령명', '')} {_jo_label(doc.metadata)} "
        f"{doc.metadata.get('조문제목', '')} {doc.page_content}"
    )

    stopwords = {
        "근거", "질문", "답변", "경우", "대한", "따라", "한다", "해야",
        "있다", "없다", "됩니다", "합니다", "그리고", "또는",
    }
    claim_tokens = {
        token for token in _lexical_tokens(clean_claim)
        if len(token) >= 2 and token not in stopwords
    }
    doc_tokens = set(_lexical_tokens(doc_text))
    lexical = (
        len(claim_tokens & doc_tokens) / min(max(len(claim_tokens), 1), 18)
    )
    score = lexical

    # 문장에 법령명·조문이 직접 있으면 가장 강한 근거 신호로 봅니다.
    compact_claim = re.sub(r"\s+", "", clean_claim)
    compact_source = re.sub(r"\s+", "", _source_name(doc))
    if compact_source and compact_source in compact_claim:
        score += 1.0
    else:
        article = re.sub(r"\s+", "", _jo_label(doc.metadata))
        if article and article in compact_claim:
            score += 0.35

    claim_numbers = set(re.findall(r"(?<![A-Za-z가-힣])\d+(?:\.\d+)?", clean_claim))
    doc_numbers = set(re.findall(r"(?<![A-Za-z가-힣])\d+(?:\.\d+)?", doc_text))
    if claim_numbers:
        matched_ratio = len(claim_numbers & doc_numbers) / len(claim_numbers)
        if matched_ratio == 1.0:
            score += 0.75
        elif matched_ratio > 0:
            score += 0.25 * matched_ratio
        else:
            score -= 0.55

    # 법령에서 구별력이 큰 행위·결과 표현이 같은 문서에 있는지 보정합니다.
    for keyword in (
        "운전작업중지", "작업중지", "이탈방지", "이상유무점검",
        "안전조치", "보건조치", "사망", "징역", "벌금", "선임방법",
        "상시근로자", "적정공기",
    ):
        if keyword in compact_claim and keyword in re.sub(r"\s+", "", doc_text):
            score += 0.18
    return score


def validate_and_resolve_citations(
    answer: str,
    docs,
) -> tuple[str, list[str], list[str], bool]:
    """
    모델 인용과 문장 내용을 대조해 잘못된 인용을 제거·교정합니다.

    반환:
      - 사용자에게 보여 줄 답변
      - 모델이 원래 적은 출처(model_used_sources)
      - 내용 검증을 통과한 출처(used_sources)
      - 프로그램이 인용을 교정했는지 여부
    """
    model_ids = _extract_citation_ids(answer, docs)
    model_sources = [_source_name(docs[index - 1]) for index in model_ids]
    verified_ids: list[int] = []

    # 항목별 줄바꿈을 우선 보존하고, 한 줄 안에서는 문장 단위로 검증합니다.
    claims = []
    for line in answer.splitlines():
        claims.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", line)
            if part.strip()
        )

    for claim in claims:
        clean = _CITE_ID_RE.sub("", claim)
        clean = _LOOSE_CITE_ID_RE.sub("", clean).strip()
        # '[사전 조치]' 같은 제목만 있는 줄은 사실 주장으로 채점하지 않습니다.
        if len(re.sub(r"[^0-9A-Za-z가-힣]", "", clean)) < 5:
            continue

        scores = [
            _claim_support_score(claim, doc)
            for doc in docs
        ]
        if not scores:
            continue
        best_index = max(range(len(scores)), key=scores.__getitem__) + 1
        best_score = scores[best_index - 1]
        cited_in_claim = _extract_citation_ids(claim, docs)

        # 모델이 적은 출처 중 내용 점수가 충분한 것만 남깁니다.
        valid_cited = [
            index for index in cited_in_claim
            if scores[index - 1] >= 0.30
            and scores[index - 1] >= best_score - 0.14
        ]
        for index in valid_cited:
            if index not in verified_ids:
                verified_ids.append(index)

        # 인용이 빠졌거나 틀렸어도 문서가 문장을 강하게 뒷받침하면 가장 좋은
        # 한 문서를 검증 출처로 복구합니다. 애매한 문장에는 출처를 억지로 붙이지 않습니다.
        if not valid_cited and best_score >= 0.48:
            if best_index not in verified_ids:
                verified_ids.append(best_index)

    verified_sources = [_source_name(docs[index - 1]) for index in verified_ids]
    repaired = set(model_sources) != set(verified_sources)

    # 잘못된 C-ID는 사용자 답변에 남기지 않고, 검증된 출처만 한 번 표시합니다.
    resolved = _CITE_ID_RE.sub("", answer)
    resolved = _LOOSE_CITE_ID_RE.sub("", resolved)
    resolved = re.sub(r"[ \t]+([.,])", r"\1", resolved)
    resolved = re.sub(r"[ \t]{2,}", " ", resolved).strip()
    if verified_sources and REFUSAL_MSG not in resolved:
        resolved = f"{resolved}\n\n(검증된 근거: {', '.join(verified_sources)})"
    return resolved, model_sources, verified_sources, repaired


def resolve_citations(answer: str, docs) -> tuple[str, list[str]]:
    """
    답변 속 [C1] 또는 (근거 1) 표기를 해석해:
      1) 실제 사용된 출처 목록(used_sources)을 만들고
      2) 답변의 (근거 N)을 실제 법령명·조문으로 치환한다.
    → sources(검색된 전체)와 달리 used_sources는 '답변이 실제 인용한' 출처입니다.
    """
    used_ids = []
    for m in _CITE_ID_RE.finditer(answer):
        for n in re.findall(r"\d+", m.group(1)):
            i = int(n)
            if 1 <= i <= len(docs) and i not in used_ids:
                used_ids.append(i)

    # 괄호 없는 "근거 1" 표기도 허용합니다.
    for m in _LOOSE_CITE_ID_RE.finditer(answer):
        i = int(m.group(1))
        if 1 <= i <= len(docs) and i not in used_ids:
            used_ids.append(i)

    # 모델이 C-ID 대신 "산업안전보건법 제52조"처럼 실제 조문을 직접 쓴 경우,
    # 그 조문이 검색 문서 안에 있을 때만 실제 사용 출처로 인정합니다.
    # 검색되지 않은 조문은 절대로 used_sources에 자동 추가하지 않습니다.
    compact_answer = re.sub(r"\s+", "", answer)
    for i, doc in enumerate(docs, 1):
        law_name = str(doc.metadata.get("법령명", "")).strip()
        jo_label = _jo_label(doc.metadata).strip()
        if not law_name or not jo_label:
            continue
        compact_source = re.sub(r"\s+", "", f"{law_name}{jo_label}")
        direct_match = compact_source in compact_answer

        # "산업안전보건법 시행령"을 "시행령"으로 줄여 쓴 표기도 허용합니다.
        short_law = (
            "시행령" if "시행령" in law_name
            else "규칙" if "규칙" in law_name
            else "산업안전보건법"
        )
        compact_short = re.sub(r"\s+", "", f"{short_law}{jo_label}")
        direct_match = direct_match or compact_short in compact_answer

        # JSON의 별표 표기(별표0003의00)와 사람이 쓰는 표기(별표3)를 맞춥니다.
        annex_match = re.fullmatch(r"별표0*(\d+)(?:의0*\d+)?", jo_label)
        if annex_match:
            human_annex = f"별표{int(annex_match.group(1))}"
            direct_match = direct_match or (
                re.sub(r"\s+", "", f"{law_name}{human_annex}") in compact_answer
                or re.sub(r"\s+", "", f"{short_law}{human_annex}") in compact_answer
            )

        if direct_match and i not in used_ids:
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
def _document_key(doc) -> tuple:
    """동일 검색 문서를 합칠 때 사용하는 안정적인 키입니다."""
    return (
        doc.metadata.get("법령명", ""),
        _jo_label(doc.metadata),
        doc.metadata.get("조문제목", ""),
        doc.page_content,
    )


def _lexical_tokens(text: str) -> list[str]:
    """
    법령용 BM25 토큰화입니다.

    한국어에는 조사가 붙기 때문에 공백 단어만 비교하면
    '안전보건총괄책임자는'과 '안전보건총괄책임자'를 놓칠 수 있습니다.
    이를 줄이기 위해 공백 단어와 한글 3-gram을 함께 사용합니다.
    """
    normalized = re.sub(r"[^0-9A-Za-z가-힣.]+", " ", str(text).lower())
    words = re.findall(r"[0-9]+(?:\.[0-9]+)?|[a-z]+|[가-힣]+", normalized)
    tokens: list[str] = list(words)
    for word in words:
        if re.fullmatch(r"[가-힣]+", word) and len(word) >= 3:
            tokens.extend(word[i:i + 3] for i in range(len(word) - 2))
    return tokens


def _expand_queries(question: str) -> list[str]:
    """
    한 질문이 여러 조문을 요구하거나 법률 용어가 짧게 쓰인 경우를 위한
    규칙 기반 질의 확장입니다. 특정 정답 조문 번호를 외우게 하지 않고,
    사용자가 물은 법률 개념과 행위·결과를 검색어로 풀어 씁니다.
    """
    q = re.sub(r"\s+", " ", question).strip()
    expanded = [q]

    # 선임 대상 질문은 직책명이 정확히 들어간 시행령 제목·별표를 우선 찾습니다.
    if "안전보건관리책임자" in q:
        expanded.append(
            "안전보건관리책임자를 두어야 하는 사업의 종류 사업장 상시근로자 수"
        )
    if "안전보건총괄책임자" in q:
        expanded.append(
            "안전보건총괄책임자 지정 대상사업 관계수급인 상시근로자 총공사금액"
        )
    if "안전관리자" in q:
        expanded.extend([
            "안전관리자를 두어야 하는 사업의 종류 상시근로자 수 안전관리자 수 선임방법",
            "별표 안전관리자를 두어야 하는 사업의 종류 상시근로자 수 안전관리자 수 선임방법",
        ])

    # 안전조치 주체 질문은 법문에 실제로 쓰이는 위험 예방 표현으로 확장합니다.
    if all(term in q for term in ("근로자", "안전", "조치", "의무")):
        expanded.append(
            "사업주는 기계 기구 위험물 작업방법 위험을 예방하기 위하여 필요한 안전조치"
        )

    # 사망·처벌은 안전 기준과 별도로 벌칙 조문을 찾아야 하는 다중 검색입니다.
    if "사망" in q and any(term in q for term in ("처벌", "벌금", "징역", "위반")):
        expanded.append(
            "안전조치 보건조치 의무를 위반하여 근로자를 사망에 이르게 한 자 벌칙 징역 벌금"
        )
        # 사망 결과를 묻는 질문은 개별 작업기준과 별도로 법률상 안전조치 의무를 확인합니다.
        expanded.append(
            "사업주는 기계 기구 작업방법 위험을 예방하기 위하여 필요한 안전조치"
        )
        if "밀폐공간" in q:
            expanded.append(
                "사업주는 가스 증기 산소결핍에 의한 건강장해를 예방하기 위하여 필요한 보건조치"
            )

    # 풍속 질문은 운전 중지, 이탈 방지, 사후 점검이 서로 다른 조문입니다.
    if ("타워크레인" in q or "크레인" in q) and any(
        term in q for term in ("풍속", "강풍", "바람", "폭풍")
    ):
        expanded.append("타워크레인 순간풍속 운전작업 중지")
        if any(term in q for term in ("설치", "수리", "해체", "단계별", "점점")):
            expanded.append("타워크레인 순간풍속 설치 수리 점검 해체 작업 중지")
        if any(term in q for term in ("30", "단계별", "점점", "예상", "분 뒤")):
            expanded.extend([
                "옥외 주행 크레인 폭풍 이탈 방지 순간풍속",
                "옥외 양중기 폭풍 후 이상 유무 점검",
            ])

    # 밀폐공간 복합 질문은 공기 기준·보건조치·벌칙을 각각 검색합니다.
    if "밀폐공간" in q:
        expanded.append(
            "밀폐공간 적정공기 산소농도 이산화탄소 일산화탄소 황화수소"
        )
        if any(term in q for term in ("질식", "사망", "위반", "보건조치")):
            expanded.append("밀폐공간 질식 보건조치 사업주")

    # 같은 문장을 중복 검색하지 않습니다.
    return list(dict.fromkeys(expanded))


def _scope_refusal(question: str) -> str | None:
    """
    정적인 법령 데이터로 답할 수 없는 질문을 검색 전에 판별합니다.
    이 판별은 검색 점수와 무관하므로 관련 없는 조문을 억지로 붙이지 않습니다.
    """
    q = re.sub(r"\s+", "", question)
    asks_latest_revision = (
        ("개정" in q and any(term in q for term in ("최신", "최근", "지난달")))
        or "실시간" in q
    )
    if asks_latest_revision:
        return (
            "제공된 법령 데이터의 시점 이후 최신 개정 여부는 확인할 수 없습니다. "
            "최신 내용은 국가법령정보센터에서 확인해 주세요."
        )
    if (
        any(term in q for term in ("오늘", "현재현장", "우리현장"))
        and any(term in q for term in ("문제없", "가동해도", "돌려도", "판단"))
    ):
        return (
            "제공된 법령만으로 개별 현장의 가동 가능 여부를 판단할 수 없습니다. "
            "현장 조건을 확인해 관리감독자 또는 관계 기관에 문의해 주세요."
        )
    return None


class RagChain:
    def __init__(self, store, llm, prompt: dict,
                 top_k: int = 3, search_type: str = "similarity",
                 score_threshold: float = 0.0,
                 simple_max_new_tokens: int = 160,
                 multi_max_new_tokens: int = 280):
        self.store = store
        self.llm = llm                      # .chat(system, user) 지원 래퍼
        self.system = prompt["system"]
        self.user_template = prompt["user"]
        self.require_citation = prompt.get("require_citation", False)
        self.top_k = top_k
        self.search_type = search_type
        self.score_threshold = score_threshold  # 0.0이면 게이트 비활성화
        self.simple_max_new_tokens = simple_max_new_tokens
        self.multi_max_new_tokens = multi_max_new_tokens
        # MMR용 검색기 (similarity/hybrid는 별도 점수 조회로 처리)
        search_kwargs = {"k": top_k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = max(top_k * 4, 20)
        retriever_type = "mmr" if search_type == "mmr" else "similarity"
        self.retriever = store.as_retriever(
            search_type=retriever_type, search_kwargs=search_kwargs
        )

        # FAISS에 저장된 모든 문서를 이용해 가벼운 BM25 인덱스를 준비합니다.
        # 임베딩을 다시 만들지 않고도 hybrid 검색을 사용할 수 있습니다.
        self._all_docs = self._extract_all_documents()
        self._bm25_doc_tokens: list[list[str]] = []
        self._bm25_term_freqs: list[Counter] = []
        self._bm25_doc_freq: Counter = Counter()
        self._bm25_avg_len = 0.0
        if search_type == "hybrid":
            self._prepare_bm25()

    def _extract_all_documents(self) -> list:
        """FAISS/Chroma에서 BM25에 사용할 전체 문서를 가능한 범위에서 읽습니다."""
        docstore = getattr(self.store, "docstore", None)
        raw_dict = getattr(docstore, "_dict", None)
        if isinstance(raw_dict, dict):
            return list(raw_dict.values())

        # Chroma는 get(include=...)로 전체 본문과 메타데이터를 꺼낼 수 있습니다.
        collection = getattr(self.store, "_collection", None)
        if collection is not None:
            try:
                from langchain_core.documents import Document
                payload = collection.get(include=["documents", "metadatas"])
                texts = payload.get("documents") or []
                metas = payload.get("metadatas") or [{} for _ in texts]
                return [
                    Document(page_content=text, metadata=meta or {})
                    for text, meta in zip(texts, metas)
                ]
            except Exception:
                return []
        return []

    def _prepare_bm25(self) -> None:
        """전체 문서를 BM25 검색용 통계로 변환합니다."""
        if not self._all_docs:
            print("[hybrid] 전체 문서를 읽지 못해 dense similarity 검색으로 폴백합니다.")
            return

        total_len = 0
        for doc in self._all_docs:
            # 법령명·조문제목을 본문 앞에 두 번 넣어 제목의 정확 일치를 강화합니다.
            meta_text = (
                f"{doc.metadata.get('법령명', '')} "
                f"{_jo_label(doc.metadata)} "
                f"{doc.metadata.get('조문제목', '')} "
                f"{doc.metadata.get('조문제목', '')}"
            )
            tokens = _lexical_tokens(f"{meta_text}\n{doc.page_content}")
            tf = Counter(tokens)
            self._bm25_doc_tokens.append(tokens)
            self._bm25_term_freqs.append(tf)
            self._bm25_doc_freq.update(tf.keys())
            total_len += len(tokens)

        self._bm25_avg_len = total_len / max(len(self._all_docs), 1)
        print(f"[hybrid] BM25 인덱스 준비 완료: {len(self._all_docs)}개 문서")

    def _bm25_scores(self, query: str) -> list[float]:
        """외부 서버나 재임베딩 없이 로컬에서 BM25 점수를 계산합니다."""
        n_docs = len(self._all_docs)
        if n_docs == 0:
            return []

        query_tf = Counter(_lexical_tokens(query))
        k1, b = 1.5, 0.75
        scores = [0.0] * n_docs
        for term, q_count in query_tf.items():
            df = self._bm25_doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for idx, tf in enumerate(self._bm25_term_freqs):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                doc_len = len(self._bm25_doc_tokens[idx])
                denom = freq + k1 * (
                    1.0 - b + b * doc_len / max(self._bm25_avg_len, 1.0)
                )
                scores[idx] += q_count * idf * (freq * (k1 + 1.0) / denom)
        return scores

    def _hybrid_retrieve(self, question: str):
        """
        dense 임베딩과 BM25를 합친 하이브리드 검색입니다.
        여러 조문이 필요한 질문은 확장 질의를 각각 검색한 뒤 결과를 합칩니다.
        """
        if not self._all_docs:
            return self._retrieve_similarity(question)

        candidate_k = max(self.top_k * 6, 30)
        dense_rank: defaultdict[tuple, float] = defaultdict(float)
        lexical_rank: defaultdict[tuple, float] = defaultdict(float)
        docs_by_key: dict[tuple, object] = {}
        priority_keys: list[tuple] = []

        for query_index, query in enumerate(_expand_queries(question)):
            # dense 검색은 점수 절댓값보다 순위를 안정적으로 사용합니다.
            try:
                dense_pairs = self.store.similarity_search_with_relevance_scores(
                    query, k=candidate_k
                )
            except Exception:
                dense_docs = self.store.similarity_search(query, k=candidate_k)
                dense_pairs = [(doc, 0.0) for doc in dense_docs]
            for rank, (doc, _) in enumerate(dense_pairs, 1):
                key = _document_key(doc)
                docs_by_key[key] = doc
                dense_rank[key] = max(dense_rank[key], 1.0 / rank)

            # BM25는 전체 문서에서 상위 후보만 합칩니다.
            lexical_scores = self._bm25_scores(query)
            top_indices = sorted(
                range(len(lexical_scores)),
                key=lexical_scores.__getitem__,
                reverse=True,
            )[:candidate_k]
            for rank, idx in enumerate(top_indices, 1):
                score = lexical_scores[idx]
                if score <= 0:
                    continue
                doc = self._all_docs[idx]
                key = _document_key(doc)
                docs_by_key[key] = doc
                # 질의마다 점수 범위가 달라 raw BM25를 직접 합치지 않고 순위를 합칩니다.
                lexical_rank[key] = max(lexical_rank[key], 1.0 / rank)
                # 원 질문이 아닌 확장 질의의 1위는 복합질문의 근거 후보로 보존합니다.
                if query_index > 0 and rank == 1 and key not in priority_keys:
                    priority_keys.append(key)

        max_dense = max(dense_rank.values(), default=1.0)
        max_lexical = max(lexical_rank.values(), default=1.0)
        ranked = []
        priority_set = set(priority_keys)
        compact_question = re.sub(r"\s+", "", question)
        for key, doc in docs_by_key.items():
            dense = dense_rank.get(key, 0.0) / max_dense
            lexical = lexical_rank.get(key, 0.0) / max_lexical

            # 질문에 직책명·설비명이 그대로 있을 때 같은 말을 가진 문서를 우대합니다.
            doc_title = re.sub(
                r"\s+", "",
                str(doc.metadata.get("조문제목", "")),
            )
            doc_text = re.sub(
                r"\s+", "",
                f"{doc.metadata.get('조문제목', '')}{doc.page_content}",
            )
            phrase_bonus = 0.0
            for phrase in (
                "안전보건관리책임자", "안전보건총괄책임자", "안전관리자",
                "타워크레인", "밀폐공간", "적정공기", "작업중지",
            ):
                if phrase not in compact_question:
                    continue
                if phrase in doc_title:
                    phrase_bonus += 0.2
                elif phrase in doc_text:
                    phrase_bonus += 0.05

            # 처벌을 묻는 질문에서는 일반 의무 조문보다 벌칙 제목을 우선 제시합니다.
            if any(term in compact_question for term in ("처벌", "벌금", "징역")):
                if "벌칙" in doc_title:
                    phrase_bonus += 0.25
            if any(term in compact_question for term in ("몇명", "인원", "선임방법")):
                if str(_jo_label(doc.metadata)).startswith("별표"):
                    phrase_bonus += 0.25

            # 이름이 비슷한 직책(안전관리자/보건관리자 등)의 별표가 서로 섞이는
            # 법령 검색 오류를 줄입니다. 질문 직책과 제목 직책이 다르면 감점합니다.
            role_names = (
                "안전보건관리책임자", "안전보건총괄책임자",
                "안전보건관리담당자", "안전관리자", "보건관리자",
            )
            requested_roles = [role for role in role_names if role in compact_question]
            if requested_roles and any(role in doc_title for role in role_names):
                if not any(role in doc_title for role in requested_roles):
                    phrase_bonus -= 0.35
            phrase_bonus = min(phrase_bonus, 0.4)

            # 확장 질의의 1위 문서는 복합질문의 서로 다른 하위 근거이므로 소폭 우대합니다.
            expansion_bonus = 0.2 if key in priority_set else 0.0
            combined = (
                0.35 * dense + 0.55 * lexical + phrase_bonus + expansion_bonus
            )
            ranked.append((combined, doc))

        ranked.sort(key=lambda item: item[0], reverse=True)

        # 같은 조문의 여러 항이 상위권을 독점하지 않도록 출처 단위로 우선 중복 제거합니다.
        unique_ranked = []
        seen_sources = set()
        for score, doc in ranked:
            source = _source_name(doc)
            if source in seen_sources:
                continue
            seen_sources.add(source)
            unique_ranked.append((score, doc))

        selected = unique_ranked[:self.top_k]
        selected_keys = {_document_key(doc) for _, doc in selected}

        # 확장 질의가 찾은 서로 다른 핵심 근거가 top-k에서 밀렸다면
        # 우선순위가 아닌 마지막 후보와 교체해 다중조문 질문의 근거를 보존합니다.
        score_by_key = {_document_key(doc): score for score, doc in unique_ranked}
        doc_by_key = {_document_key(doc): doc for _, doc in unique_ranked}
        for key in priority_keys:
            if key in selected_keys or key not in doc_by_key:
                continue
            replace_at = next(
                (
                    i for i in range(len(selected) - 1, -1, -1)
                    if _document_key(selected[i][1]) not in priority_set
                ),
                None,
            )
            if replace_at is None:
                continue
            selected_keys.discard(_document_key(selected[replace_at][1]))
            selected[replace_at] = (score_by_key[key], doc_by_key[key])
            selected_keys.add(key)

        selected.sort(key=lambda item: item[0], reverse=True)
        docs = [doc for _, doc in selected]
        top_score = min(ranked[0][0], 1.0) if ranked else 0.0
        return docs, round(float(top_score), 4)

    def _retrieve_similarity(self, question: str):
        """기존 dense similarity 검색을 유지하는 비교 실험용 함수입니다."""
        try:
            pairs = self.store.similarity_search_with_relevance_scores(
                question, k=self.top_k
            )
            docs = [d for d, _ in pairs]
            top_score = max((s for _, s in pairs), default=0.0)
            return docs, round(float(top_score), 4)
        except Exception:
            return self.retriever.invoke(question), None

    def _retrieve_with_scores(self, question: str):
        """검색 + 관련도 점수(0~1, 높을수록 관련). 점수 미지원 스토어는 게이트 생략."""
        if self.search_type == "hybrid":
            return self._hybrid_retrieve(question)
        return self._retrieve_similarity(question)

    def _merge_annex_chunks(self, question: str, docs: list) -> list:
        """
        검색된 별표 문서에 같은 별표의 관련 청크를 이어 붙입니다.

        기존에는 같은 출처를 중복 제거하면서 별표 제목 청크 하나만 남는 경우가
        있었습니다. 여기서는 top_k 문서 수는 그대로 유지하면서, 별표 C-ID 안에
        질문과 가까운 형제 청크를 최대 2개 보강합니다.
        """
        if not self._all_docs:
            return docs

        # 확장 질의별 BM25 점수 중 최고값을 사용해 표의 관련 행을 고릅니다.
        score_sets = [
            self._bm25_scores(query)
            for query in _expand_queries(question)
        ]
        merged_docs = []
        for selected_doc in docs:
            if not str(_jo_label(selected_doc.metadata)).startswith("별표"):
                merged_docs.append(selected_doc)
                continue

            source = _source_name(selected_doc)
            candidates = []
            for index, sibling in enumerate(self._all_docs):
                if _source_name(sibling) != source:
                    continue
                sibling_score = max(
                    (scores[index] for scores in score_sets if index < len(scores)),
                    default=0.0,
                )
                # 제목만 반복된 짧은 청크보다 표의 실제 행이 있는 청크를 우대합니다.
                if len(sibling.page_content.strip()) >= 100:
                    sibling_score += 0.15
                candidates.append((sibling_score, sibling))

            candidates.sort(key=lambda item: item[0], reverse=True)
            chosen = [selected_doc]
            chosen_keys = {_document_key(selected_doc)}
            for _, sibling in candidates:
                key = _document_key(sibling)
                if key in chosen_keys:
                    continue
                chosen.append(sibling)
                chosen_keys.add(key)
                if len(chosen) >= 3:
                    break

            # 청크 overlap 영역이 그대로 두 번 보이지 않도록 접미/접두 중복을 줄입니다.
            merged_text = chosen[0].page_content.strip()
            for sibling in chosen[1:]:
                next_text = sibling.page_content.strip()
                overlap = 0
                max_overlap = min(120, len(merged_text), len(next_text))
                for size in range(max_overlap, 19, -1):
                    if merged_text[-size:] == next_text[:size]:
                        overlap = size
                        break
                merged_text += "\n[별표 관련 행]\n" + next_text[overlap:]

            # 모델 입력이 지나치게 길어지는 것을 막되 표의 관련 행은 충분히 보존합니다.
            merged_text = merged_text[:2200]
            merged_docs.append(
                # 원래 검색 문서와 같은 클래스를 사용해 LangChain 버전 차이를 피합니다.
                type(selected_doc)(
                    page_content=merged_text,
                    metadata=dict(selected_doc.metadata),
                )
            )
        return merged_docs

    def search(self, question: str) -> dict:
        """
        LLM을 로드하거나 답변을 생성하지 않고 검색 결과만 반환합니다.
        검색 회귀 테스트와 조문 누락 진단에 사용합니다.
        """
        docs, top_score = self._retrieve_with_scores(question)
        docs = self._merge_annex_chunks(question, docs)
        return {
            "question": question,
            "contexts": [doc.page_content for doc in docs],
            "sources": [_source_name(doc) for doc in docs],
            "top_score": top_score,
        }

    def ask(self, question: str) -> dict:
        t0 = time.time()

        # ⓪ 정적 법령 데이터로 처리할 수 없는 최신성·개별 현장 판단 질문
        scope_answer = _scope_refusal(question)
        if scope_answer is not None:
            return {
                "question": question,
                "answer": scope_answer,
                "raw_answer": scope_answer,
                "contexts": [],
                "sources": [],
                "model_used_sources": [],
                "used_sources": [],
                "citation_repaired": False,
                "retrieved": False,
                "status": "REFUSE",
                "citation_status": "NOT_APPLICABLE",
                "top_score": None,
                "latency": round(time.time() - t0, 3),
            }

        # ① 검색 + 관련도 점수
        docs, top_score = self._retrieve_with_scores(question)

        # ② 점수 게이트: 최고 점수가 임계값 미만이면 LLM을 호출하지 않고 거부
        #    (관련 없는 질문에 억지 근거가 붙는 것을 시스템 차원에서 차단)
        if (self.score_threshold > 0 and top_score is not None
                and top_score < self.score_threshold):
            return {
                "question": question,
                "answer": REFUSAL_MSG,
                "raw_answer": REFUSAL_MSG,
                "contexts": [],
                "sources": [_source_name(d) for d in docs],  # 디버깅용 기록
                "model_used_sources": [],
                "used_sources": [],
                "citation_repaired": False,
                "retrieved": False,
                "status": "REFUSE",
                "citation_status": "NOT_APPLICABLE",
                "top_score": top_score,
                "latency": round(time.time() - t0, 3),
            }

        # ③ MMR이면 다양성 반영된 문서로 교체 (점수는 게이트용으로만 사용)
        if self.search_type == "mmr":
            docs = self.retriever.invoke(question)

        # ④ 별표는 질문과 가까운 표 행을 같은 C-ID 안에 연결합니다.
        docs = self._merge_annex_chunks(question, docs)

        # ⑤ 질문 유형에 맞는 항목과 생성 길이를 정한 뒤 LLM을 호출합니다.
        plan = _question_plan(
            question,
            single_max_tokens=self.simple_max_new_tokens,
            multi_max_tokens=self.multi_max_new_tokens,
        )
        context = format_contexts(docs)
        user_msg = self.user_template.format(context=context, question=question)
        user_msg = f"{user_msg}\n\n{plan['instruction']}"
        raw_answer = self.llm.chat(
            self.system,
            user_msg,
            max_new_tokens=plan["max_new_tokens"],
        )

        # ⑥ 모델이 적은 C-ID를 문장 내용과 대조한 뒤 검증된 출처만 표시합니다.
        (
            answer,
            model_used_sources,
            used_sources,
            citation_repaired,
        ) = validate_and_resolve_citations(raw_answer, docs)

        # 모델이 명시적으로 거절했을 때만 거절로 정규화합니다.
        # 인용 형식 실패를 "근거 없음"으로 바꾸지 않습니다. 이전 구현은 정답 조문을
        # 1위로 검색해도 [C1]이 없다는 이유로 정답 전체를 지우는 문제가 있었습니다.
        if REFUSAL_MSG in raw_answer:
            answer = REFUSAL_MSG
            model_used_sources = []
            used_sources = []
            citation_repaired = False

        if answer == REFUSAL_MSG:
            status = "REFUSE"
            citation_status = "NOT_APPLICABLE"
        elif used_sources:
            status = "ANSWER"
            citation_status = "REPAIRED" if citation_repaired else "CITED"
        else:
            # 답변은 보존하되 출처 형식이 빠졌음을 평가 결과에 별도로 남깁니다.
            status = "ANSWER_UNCITED"
            citation_status = "MISSING"

        return {
            "question": question,
            "answer": answer,
            "raw_answer": raw_answer,
            "contexts": [d.page_content for d in docs],       # 평가용
            "sources": [_source_name(d) for d in docs],        # 검색된 전체
            # 모델 원문 인용과 검증 후 인용을 분리해 후처리 효과를 평가할 수 있습니다.
            "model_used_sources": model_used_sources,
            "used_sources": used_sources,
            "citation_repaired": citation_repaired,
            "question_mode": plan["mode"],
            "max_new_tokens_used": plan["max_new_tokens"],
            "retrieved": True,
            "status": status,
            "citation_status": citation_status,
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
    temperature: float = 0.0,
    max_new_tokens: int = 160,
    multi_max_new_tokens: int = 280,
    api_key: str | None = None,
    base_url: str | None = None,
    load_in_4bit: bool = False,
    score_threshold: float = 0.0,
    repetition_penalty: float = 1.05,
    force_cuda: bool = False,
) -> RagChain:
    """
    Args:
        store         : build_vectorstore() 가 만든 저장소
        llm_type      : "hf" / "openai" / "anthropic" / "upstage"   (실험 변수)
        model_name    : 구체적 모델명 또는 파인튜닝 모델 경로         (실험 변수)
        prompt_name   : "basic" / "cot" / "cite" / "strict"          (실험 변수)
        top_k         : 검색 chunk 수                                (실험 변수)
        search_type   : "similarity" / "mmr" / "hybrid"              (실험 변수)
        max_new_tokens: 생성 최대 토큰 수
        multi_max_new_tokens: 복합질문 생성 최대 토큰 수
        api_key       : 미지정 시 환경변수에서 자동 로드(UPSTAGE_API_KEY 등)
        base_url      : OpenAI 호환 커스텀 엔드포인트용
        load_in_4bit  : HF 모델 4비트 양자화 (VRAM 부족 시)
        repetition_penalty: 반복 문장 억제 강도(1.0이면 억제 없음)
        force_cuda    : CUDA가 없을 때 CPU 폴백 대신 오류 발생
    """
    llm = get_llm(llm_type, model_name, temperature=temperature,
                  max_new_tokens=max_new_tokens,
                  api_key=api_key, base_url=base_url,
                  load_in_4bit=load_in_4bit,
                  repetition_penalty=repetition_penalty,
                  force_cuda=force_cuda)
    prompt = get_prompt(prompt_name)
    chain = RagChain(store, llm, prompt, top_k=top_k, search_type=search_type,
                     score_threshold=score_threshold,
                     simple_max_new_tokens=max_new_tokens,
                     multi_max_new_tokens=multi_max_new_tokens)
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
    parser.add_argument(
        "--search_type",
        default="hybrid",
        choices=["similarity", "mmr", "hybrid"],
    )
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
