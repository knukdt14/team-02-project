"""
영문 산업안전보건 법령 RAG 체인.

고정 실험 조건
----------------
- LLM: mistralai/Mistral-7B-Instruct-v0.3
- 양자화: NF4 4비트
- 장치: CUDA 강제
- 검색: Upstage 임베딩 + BM25 하이브리드
- 저장소: FAISS
- top_k: 5
- 검색 임계값: 0.4
- 프롬프트: 영문 strict

평가 질문이나 정답 조문을 코드에 넣지 않습니다. 검색 결과에 [C1]~[C5]를
붙여 모델이 실제 사용한 근거만 인용하도록 하고, 해당 ID만 출처로 기록합니다.
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TOP_K = 5
SCORE_THRESHOLD = 0.4
SINGLE_MAX_TOKENS = 180
MULTI_MAX_TOKENS = 300
REFUSAL_MESSAGE = (
    "The provided occupational safety and health statutes do not contain "
    "sufficient grounds to answer this question."
)


STRICT_SYSTEM_PROMPT = """
You are a legal question-answering assistant for Korean occupational safety
and health statutes translated into English.

Rules:
1. Answer only from the supplied evidence blocks.
2. If the evidence supports only part of the question, answer that part and
   explicitly state which part is unsupported.
3. If no supplied evidence supports the question, output exactly:
   "The provided occupational safety and health statutes do not contain
   sufficient grounds to answer this question."
4. Do not invent article numbers, duties, exceptions, amounts, or penalties.
5. Start with the conclusion. Keep a simple answer to 1-3 sentences.
6. For a multi-part question, use a numbered list and answer every part in
   1-2 sentences.
7. Cite only evidence actually used. Put [C1] or [C1, C3] at the end of the
   sentence it supports.
8. Stop immediately after the requested answer and citations. Do not add
   background information that was not requested.
""".strip()


def _normalize_article(value: str) -> str:
    """한국어 JSON의 조문·별표 표기를 평가용 키로 정규화합니다."""
    compact = re.sub(r"\s+", "", str(value or ""))

    annex = re.fullmatch(r"별표0*(\d+)(?:의0*(\d+))?", compact)
    if annex:
        main = str(int(annex.group(1)))
        sub = annex.group(2)
        return f"annex{main}-{int(sub)}" if sub and int(sub) else f"annex{main}"

    article = re.fullmatch(r"제?0*(\d+)(?:조)?(?:의0*(\d+))?", compact)
    if article:
        main = str(int(article.group(1)))
        sub = article.group(2)
        return f"{main}-{int(sub)}" if sub else main

    return compact.lower()


def _law_kind(metadata: dict) -> str:
    """한국어 원본 법령명을 act/decree/rule 식별자로 변환합니다."""
    law_name = str(metadata.get("법령명", "")).strip()
    if "시행령" in law_name:
        return "decree"
    if "규칙" in law_name:
        return "rule"
    if law_name == "산업안전보건법":
        return "act"
    return "other"


def _article_value(metadata: dict) -> str:
    """조문표시를 우선 사용하고, 없으면 조문번호를 사용합니다."""
    return str(metadata.get("조문표시") or metadata.get("조문번호") or "")


def _source_key(document) -> str:
    """검색·평가에 사용하는 안정적인 출처 키입니다. 예: act:38."""
    return f"{_law_kind(document.metadata)}:{_normalize_article(_article_value(document.metadata))}"


def _source_name(document) -> str:
    """사용자에게 표시할 영문 법령명과 조문명을 만듭니다."""
    metadata = document.metadata
    law_name = metadata.get("law_name_en") or metadata.get("법령명") or "Statute"
    article = metadata.get("article_label_en")
    if not article:
        normalized = _normalize_article(_article_value(metadata))
        if normalized.startswith("annex"):
            article = normalized.replace("annex", "Annex ").replace("-", "-")
        else:
            article = f"Article {normalized.replace('-', '-')}"
    return f"{law_name} {article}".strip()


def _document_key(document) -> tuple[str, str]:
    """동일 청크 중복 제거용 키입니다."""
    return _source_key(document), document.page_content


def _tokenize(text: str) -> list[str]:
    """영문 BM25와 근거 검증에 공통으로 사용하는 가벼운 토크나이저입니다."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", str(text).lower())


def _expand_queries(question: str) -> list[str]:
    """
    복합질문을 일반적인 접속 표현 기준으로만 나눕니다.

    특정 평가 문항·조문 지식을 추가하지 않으므로 평가셋 과적합을 피합니다.
    """
    question = " ".join(question.split())
    if not question:
        return []

    clauses = re.split(
        r"[;?\n]+|"
        r"\b(?:and\s+what|and\s+which|and\s+how|and\s+when|"
        r"and\s+can|as\s+well\s+as|respectively)\b",
        question,
        flags=re.IGNORECASE,
    )

    queries = [question]
    seen = {question.lower()}
    for clause in clauses:
        clause = clause.strip(" ,.;:")
        # "penalties apply"처럼 짧지만 독립적인 요구사항도 보존합니다.
        if len(_tokenize(clause)) < 2:
            continue
        key = clause.lower()
        if key not in seen:
            seen.add(key)
            queries.append(clause)
    return queries[:4]


def _question_plan(question: str) -> dict:
    """질문 형식만 보고 단일/복합 출력 형식과 생성 상한을 정합니다."""
    lowered = question.lower()
    multi = (
        len(_expand_queries(question)) > 1
        or question.count("?") > 1
        or any(term in lowered for term in ("step by step", "respectively"))
    )

    if multi:
        return {
            "mode": "multi",
            "max_new_tokens": MULTI_MAX_TOKENS,
            "instruction": (
                "Answer each requested part in order as a numbered list. "
                "Use 1-2 sentences per item and attach the supporting C-ID "
                "to each item."
            ),
        }

    if any(term in lowered for term in ("who must", "who is obligated", "whose duty")):
        instruction = (
            "State the directly responsible party first. Do not add other "
            "parties unless the evidence makes them directly relevant."
        )
        mode = "responsible_party"
    elif lowered.startswith(("can ", "may ", "is it permissible", "does ")):
        instruction = (
            "Begin with Yes or No, then state the controlling condition in "
            "no more than two sentences."
        )
        mode = "yes_no"
    elif "what criteria" in lowered or "what standard" in lowered:
        instruction = (
            "State only the deciding criteria first. Mention an annex only "
            "when the supplied evidence directly relies on it."
        )
        mode = "criteria"
    else:
        instruction = (
            "Answer with the conclusion first in 1-3 sentences and omit "
            "unrequested background."
        )
        mode = "single"

    return {
        "mode": mode,
        "max_new_tokens": SINGLE_MAX_TOKENS,
        "instruction": instruction,
    }


def _scope_refusal(question: str) -> str | None:
    """정적 3개 법령 데이터로 판단할 수 없는 요청을 생성 전에 차단합니다."""
    lowered = " ".join(question.lower().split())

    future_or_latest = any(
        term in lowered
        for term in ("latest amendment", "future amendment", "next month", "today's law")
    )
    if future_or_latest:
        return (
            "The static legal dataset cannot verify later or real-time "
            "amendments. Please check the official National Law Information "
            "Center for the current text."
        )

    site_specific = any(
        term in lowered
        for term in ("our company", "our site", "yesterday's accident", "today's site")
    ) and any(
        term in lowered
        for term in ("legally determine", "specifically responsible", "legally permissible")
    )
    if site_specific:
        return (
            "The provided statutes alone cannot determine legal responsibility "
            "or operating permission for an individual site. The specific facts "
            "must be reviewed by the responsible authority or a qualified expert."
        )

    outside_dataset = any(
        term in lowered
        for term in (
            "minimum wage",
            "workers' compensation benefit",
            "workers compensation benefit",
            "insurance benefit amount",
        )
    )
    if outside_dataset:
        return REFUSAL_MESSAGE

    return None


class MistralNF4:
    """Mistral-7B-Instruct-v0.3을 CUDA NF4 4비트로 한 번만 로드합니다."""

    def __init__(self):
        import torch
        import transformers
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA를 사용할 수 없습니다. CUDA PyTorch가 설치된 "
                "SAFETY_RAG_PY311 환경인지 확인하세요."
            )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 사용 중인 Transformers 4.46.3과의 호환성을 위해 dtype가 아니라
        # torch_dtype를 전달합니다.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )
        self.last_generated_tokens = 0
        self.last_hit_limit = False

        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        print(
            f"[LLM] {MODEL_NAME} | Transformers={transformers.__version__} | "
            f"GPU={torch.cuda.get_device_name(0)} | NF4 4bit | VRAM={allocated:.0f}MB"
        )

    def generate(self, system: str, user: str, max_new_tokens: int) -> str:
        """Mistral 채팅 템플릿을 적용해 결정적으로 답변을 생성합니다."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        output = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"].strip()

        self.last_generated_tokens = len(
            self.tokenizer.encode(output, add_special_tokens=False)
        )
        self.last_hit_limit = self.last_generated_tokens >= max_new_tokens - 3
        return output


_CITATION_RE = re.compile(
    r"\[\s*C(\d+)(?:\s*[,/]\s*C?(\d+))*\s*\]",
    flags=re.IGNORECASE,
)


def _extract_citation_ids(answer: str, document_count: int) -> list[int]:
    """모델 답변의 [C1], [C1, C3]에서 유효한 근거 번호만 추출합니다."""
    found: list[int] = []
    for block in re.findall(r"\[\s*C[\dC,\s/]+\]", answer, flags=re.IGNORECASE):
        for number in re.findall(r"\d+", block):
            index = int(number)
            if 1 <= index <= document_count and index not in found:
                found.append(index)
    return found


def _clean_citation_ids(answer: str) -> str:
    """최종 화면에서는 내부 C-ID를 제거하고 검증 가능한 출처명으로 바꿉니다."""
    cleaned = re.sub(
        r"\[\s*C[\dC,\s/]+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _looks_truncated(answer: str, hit_limit: bool) -> bool:
    """토큰 한도 도달 또는 불완전한 영문 문장 종료를 탐지합니다."""
    if hit_limit or not answer.strip():
        return True
    tail = _clean_citation_ids(answer).rstrip()
    if not tail:
        return True
    return tail.endswith((",", ":", ";", "-", "(", "["))


class EnglishRagChain:
    """FAISS dense 검색과 BM25를 결합해 Mistral에 영문 근거를 전달합니다."""

    def __init__(self, store):
        self.store = store
        self.top_k = TOP_K
        self.score_threshold = SCORE_THRESHOLD
        self.llm = MistralNF4()

        # FAISS docstore의 모든 청크로 BM25 인덱스를 한 번만 만듭니다.
        self.all_documents = list(getattr(store.docstore, "_dict", {}).values())
        if not self.all_documents:
            raise RuntimeError("FAISS에서 BM25용 문서를 읽지 못했습니다.")

        from rank_bm25 import BM25Okapi

        corpus = []
        for document in self.all_documents:
            metadata = document.metadata
            title = (
                f"{metadata.get('law_name_en', '')} "
                f"{metadata.get('article_title_en', '')} "
                f"{metadata.get('article_title_en', '')}"
            )
            corpus.append(_tokenize(f"{title} {document.page_content}"))
        self.bm25 = BM25Okapi(corpus)
        print(f"[hybrid] BM25 준비 완료: {len(self.all_documents)}개 청크")

    def _dense_pairs(self, query: str, count: int):
        """FAISS raw 거리와 0~1 relevance 점수를 함께 반환합니다."""
        raw_pairs = self.store.similarity_search_with_score(query, k=count)
        score_function = self.store._select_relevance_score_fn()
        return [
            (
                document,
                max(0.0, min(1.0, float(score_function(float(raw_score))))),
            )
            for document, raw_score in raw_pairs
        ]

    def _hybrid_retrieve(self, question: str):
        """
        원 질문과 일반적인 하위 절을 각각 dense/BM25로 검색하고 순위를 결합합니다.

        기존 비교 실험과 동일하게 dense 0.45, BM25 0.55를 사용합니다.
        """
        fetch_k = max(20, self.top_k * 4)
        documents_by_key = {}
        dense_rank = defaultdict(float)
        lexical_rank = defaultdict(float)
        priority_keys = []

        for query_index, query in enumerate(_expand_queries(question)):
            dense = self._dense_pairs(query, fetch_k)
            for rank, (document, _) in enumerate(dense, 1):
                key = _document_key(document)
                documents_by_key[key] = document
                dense_rank[key] = max(dense_rank[key], 1.0 / rank)
                if query_index > 0 and rank == 1 and key not in priority_keys:
                    priority_keys.append(key)

            bm25_scores = self.bm25.get_scores(_tokenize(query))
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda index: bm25_scores[index],
                reverse=True,
            )[:fetch_k]
            for rank, index in enumerate(top_indices, 1):
                if bm25_scores[index] <= 0:
                    continue
                document = self.all_documents[index]
                key = _document_key(document)
                documents_by_key[key] = document
                lexical_rank[key] = max(lexical_rank[key], 1.0 / rank)
                if query_index > 0 and rank == 1 and key not in priority_keys:
                    priority_keys.append(key)

        max_dense = max(dense_rank.values(), default=1.0)
        max_lexical = max(lexical_rank.values(), default=1.0)
        priority = set(priority_keys)
        ranked = []
        for key, document in documents_by_key.items():
            dense = dense_rank.get(key, 0.0) / max_dense
            lexical = lexical_rank.get(key, 0.0) / max_lexical
            bonus = 0.10 if key in priority else 0.0
            score = 0.45 * dense + 0.55 * lexical + bonus
            ranked.append((score, document))
        ranked.sort(key=lambda item: item[0], reverse=True)

        # 같은 조문의 여러 조각이 top_k를 독점하지 않도록 출처 단위로 줄입니다.
        unique = []
        seen_sources = set()
        for score, document in ranked:
            source = _source_key(document)
            if source in seen_sources:
                continue
            seen_sources.add(source)
            unique.append((score, document))

        selected = unique[: self.top_k]
        selected_keys = {_document_key(document) for _, document in selected}

        # 복합질문 하위 절의 1위 근거가 밀린 경우 마지막 일반 후보와 교체합니다.
        score_by_key = {_document_key(document): score for score, document in unique}
        doc_by_key = {_document_key(document): document for score, document in unique}
        for key in priority_keys:
            if key in selected_keys or key not in doc_by_key:
                continue
            replacement = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if _document_key(selected[index][1]) not in priority
                ),
                None,
            )
            if replacement is None:
                continue
            selected_keys.discard(_document_key(selected[replacement][1]))
            selected[replacement] = (score_by_key[key], doc_by_key[key])
            selected_keys.add(key)

        selected.sort(key=lambda item: item[0], reverse=True)
        top_score = min(ranked[0][0], 1.0) if ranked else 0.0
        return [document for _, document in selected], round(float(top_score), 4)

    def _merge_annex_chunks(self, question: str, documents: list):
        """선택된 별표에 질문과 가장 가까운 형제 청크를 최대 2개 연결합니다."""
        query_tokens = set(_tokenize(question))
        merged = []
        for selected in documents:
            if not _source_key(selected).split(":", 1)[1].startswith("annex"):
                merged.append(selected)
                continue

            siblings = [
                document
                for document in self.all_documents
                if _source_key(document) == _source_key(selected)
                and _document_key(document) != _document_key(selected)
            ]
            siblings.sort(
                key=lambda document: len(
                    query_tokens & set(_tokenize(document.page_content))
                ),
                reverse=True,
            )
            texts = [selected.page_content] + [
                document.page_content for document in siblings[:2]
            ]
            combined = "\n[Related annex rows]\n".join(texts)[:2200]
            merged.append(
                type(selected)(
                    page_content=combined,
                    metadata=dict(selected.metadata),
                )
            )
        return merged

    def search(self, question: str) -> dict:
        """LLM 생성 없이 검색 결과만 점검합니다."""
        documents, top_score = self._hybrid_retrieve(question)
        documents = self._merge_annex_chunks(question, documents)
        return {
            "question": question,
            "contexts": [document.page_content for document in documents],
            "sources": [_source_name(document) for document in documents],
            "source_keys": [_source_key(document) for document in documents],
            "top_score": top_score,
        }

    def _refusal_result(
        self,
        question: str,
        answer: str,
        started_at: float,
        top_score=None,
        searched_sources=None,
    ) -> dict:
        """정적 범위 검사나 임계값에서 거부된 결과를 공통 형식으로 만듭니다."""
        return {
            "question": question,
            "answer": answer,
            "raw_answer": answer,
            "contexts": [],
            "sources": searched_sources or [],
            "source_keys": [],
            "prompt_sources": [],
            "model_used_sources": [],
            "used_sources": [],
            "used_source_keys": [],
            "status": "REFUSE",
            "citation_status": "NOT_APPLICABLE",
            "question_mode": "refusal",
            "max_new_tokens_used": 0,
            "generation_retried": False,
            "truncation_detected": False,
            "top_score": top_score,
            "latency": round(time.time() - started_at, 3),
        }

    def ask(self, question: str) -> dict:
        """질문 하나를 검색하고 근거 기반 영문 답변을 생성합니다."""
        started_at = time.time()
        question = str(question).strip()
        if not question:
            raise ValueError("질문이 비어 있습니다.")

        scope_answer = _scope_refusal(question)
        if scope_answer:
            return self._refusal_result(question, scope_answer, started_at)

        documents, top_score = self._hybrid_retrieve(question)
        if top_score < self.score_threshold:
            return self._refusal_result(
                question,
                REFUSAL_MESSAGE,
                started_at,
                top_score=top_score,
                searched_sources=[_source_name(document) for document in documents],
            )

        documents = self._merge_annex_chunks(question, documents)
        plan = _question_plan(question)

        context_blocks = [
            f"[C{index}] {_source_name(document)}\n{document.page_content}"
            for index, document in enumerate(documents, 1)
        ]
        user_prompt = (
            "[Evidence]\n"
            + "\n\n".join(context_blocks)
            + f"\n\n[Question]\n{question}"
            + f"\n\n[Output instruction]\n{plan['instruction']}"
        )

        raw_answer = self.llm.generate(
            STRICT_SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=plan["max_new_tokens"],
        )
        citation_ids = _extract_citation_ids(raw_answer, len(documents))
        truncated = _looks_truncated(raw_answer, self.llm.last_hit_limit)
        retried = False

        # 잘림 또는 전체 인용 누락이 있으면 동일 근거로 한 번만 재작성합니다.
        if truncated or (REFUSAL_MESSAGE not in raw_answer and not citation_ids):
            retried = True
            retry_limit = min(plan["max_new_tokens"] + 120, 460)
            retry_prompt = (
                user_prompt
                + "\n\n[Rewrite]\n"
                + "Rewrite the complete answer from the beginning. "
                + "Finish every sentence, keep it concise, and put the correct "
                + "C-ID after every supported claim."
            )
            candidate = self.llm.generate(
                STRICT_SYSTEM_PROMPT,
                retry_prompt,
                max_new_tokens=retry_limit,
            )
            candidate_ids = _extract_citation_ids(candidate, len(documents))
            candidate_truncated = _looks_truncated(
                candidate,
                self.llm.last_hit_limit,
            )
            if (candidate_ids and not citation_ids) or (
                truncated and not candidate_truncated
            ):
                raw_answer = candidate
                citation_ids = candidate_ids
                truncated = candidate_truncated

        if REFUSAL_MESSAGE in raw_answer:
            return self._refusal_result(
                question,
                REFUSAL_MESSAGE,
                started_at,
                top_score=top_score,
                searched_sources=[_source_name(document) for document in documents],
            )

        model_used_sources = [
            _source_name(documents[index - 1]) for index in citation_ids
        ]
        used_source_keys = [
            _source_key(documents[index - 1]) for index in citation_ids
        ]
        answer = _clean_citation_ids(raw_answer)
        if model_used_sources:
            answer += "\n\n(Sources: " + "; ".join(model_used_sources) + ")"

        if truncated:
            status = "ANSWER_TRUNCATED"
        elif model_used_sources:
            status = "ANSWER"
        else:
            status = "ANSWER_UNCITED"

        return {
            "question": question,
            "answer": answer,
            "raw_answer": raw_answer,
            "contexts": [document.page_content for document in documents],
            "sources": [_source_name(document) for document in documents],
            "source_keys": [_source_key(document) for document in documents],
            "prompt_sources": [_source_name(document) for document in documents],
            "model_used_sources": model_used_sources,
            "used_sources": model_used_sources,
            "used_source_keys": used_source_keys,
            "status": status,
            "citation_status": "CITED" if model_used_sources else "MISSING",
            "question_mode": plan["mode"],
            "max_new_tokens_used": plan["max_new_tokens"],
            "generation_retried": retried,
            "truncation_detected": truncated,
            "top_score": top_score,
            "latency": round(time.time() - started_at, 3),
        }


def build_rag_chain(store) -> EnglishRagChain:
    """고정 설정의 영문 RAG 체인을 생성합니다."""
    print(
        f"[chain] model={MODEL_NAME}, search=hybrid, top_k={TOP_K}, "
        f"threshold={SCORE_THRESHOLD}, prompt=strict, CUDA=forced"
    )
    return EnglishRagChain(store)
