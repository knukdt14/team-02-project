"""
evaluate.py
---------------------------------------------------
RAG 파이프라인 4단계: 평가.

계산하는 지표 (전부 산출 → 나중에 취사선택)
  1) BERTScore(P/R/F1): 모델 답변 ↔ 정답(GT)의 의미 유사도. (필수 지표)
  2) 응답시간(latency) : chain.ask() 가 측정. 초 단위.
  3) 키워드 포함률     : 정답의 핵심 수치·조문이 답변에 들어갔는지.
  4) RAGAS            : faithfulness / answer_relevancy / context_precision /
                        context_recall (LLM 심판).
  5) 검색 지표 (references.csv 기반) ← 정답 조문과 검색 결과를 대조
     - hit_at_k        : 정답 조문이 검색 결과에 하나라도 있는지 (0/1)
     - recall_at_k     : 필요한 정답 조문 중 몇 %를 찾았는지
     - precision_at_k  : 검색 결과 중 정답 조문 비율
     - mrr             : 첫 정답 조문이 몇 번째로 검색됐는지 (1/순위)
  6) citation_precision/recall:
     답변이 실제 사용했다고 표기한 출처(used_sources)가 정답 근거와 맞는지 평가
  7) refusal_acc      : 답변불가형(unanswerable) 질문을 제대로 거절했는지 (0/1)

입력
  - eval/questions.csv  : id, 대상법령, question, ground_truth_answer, 유형, 난이도
  - eval/references.csv : id, 근거_법령, 근거_조문, 근거_원문  (검색 지표의 정답)

출력
  - eval/results.csv    : 문항별 지표 + 실험조건(모델·프롬프트·top_k·검색방식 등)
  - 콘솔에 요약(평균, 유형별) 출력

설치(쓰는 것만)
  pip install bert-score pandas
  pip install ragas datasets            # RAGAS 사용 시 (기본 OPENAI_API_KEY 필요)
"""

import re
import csv
import time
from pathlib import Path


# =====================================================================
# 유틸: 정답에서 핵심 키워드(수치·조문) 자동 추출
# =====================================================================
# 숫자+단위, 제○조, 퍼센트 등
_KW_PATTERNS = [
    r"\d+(?:\.\d+)?\s*(?:미터|퍼센트|%|센티미터|도|년|개월|명|억원|만원|원|피피엠|ppm|럭스|초당|m/s)",
    r"제\s*\d+\s*조(?:의\s*\d+)?",
    r"\d+(?:\.\d+)?",  # 순수 숫자(위에서 안 걸린 것)
]


def extract_keywords(text: str) -> list[str]:
    """정답 문장에서 채점용 핵심 키워드를 뽑는다(중복 제거)."""
    kws = []
    for pat in _KW_PATTERNS:
        for m in re.findall(pat, text):
            k = m.strip().replace(" ", "")
            if k and k not in kws:
                kws.append(k)
    return kws


def keyword_rate(answer: str, keywords: list[str]) -> float:
    """정답 키워드 중 몇 %가 답변에 포함됐는지 (0~1)."""
    if not keywords:
        return None  # 채점할 키워드가 없으면 제외
    ans = answer.replace(" ", "")
    hit = sum(1 for k in keywords if k in ans)
    return round(hit / len(keywords), 3)


# =====================================================================
# 검색 지표 (references.csv 기반)
#   Hit@k / Recall@k / Precision@k / MRR / Citation Accuracy / Refusal Accuracy
# =====================================================================
def _law_short(name: str) -> str:
    """법령명을 법/시행령/규칙 3종으로 축약."""
    if "규칙" in name:
        return "규칙"
    if "시행령" in name:
        return "시행령"
    if "산업안전보건법" in name or name == "법":
        return "법"
    return ""


_ART_RE = re.compile(r"제\s*(\d+)\s*조(?:의\s*(\d+))?|별표\s*(\d*(?:의\s*\d+)?)")


def _normalize_article(key: str) -> str:
    """
    조문/별표 키를 비교 가능한 형태로 정규화합니다.

    법령 JSON에는 별표3이 '별표0003의00'처럼 저장되는 경우가 있습니다.
    이를 그대로 비교하면 실제 같은 별표를 틀렸다고 채점하므로,
    앞자리 0과 의미 없는 '의00'을 제거합니다.
    """
    compact = str(key).replace(" ", "")
    if compact.startswith("별표"):
        m = re.fullmatch(r"별표(\d*)(?:의(\d+))?", compact)
        if not m or not m.group(1):
            return "별표"
        main = str(int(m.group(1)))
        sub = m.group(2)
        if sub and int(sub) != 0:
            return f"별표{main}의{int(sub)}"
        return f"별표{main}"

    m = re.fullmatch(r"0*(\d+)(?:의0*(\d+))?", compact)
    if m:
        main = str(int(m.group(1)))
        sub = m.group(2)
        return f"{main}의{int(sub)}" if sub is not None else main
    return compact


def parse_gold_articles(ref_law: str, ref_article: str) -> list[tuple[str, str]]:
    """
    references.csv 의 근거_법령/근거_조문에서 정답 조문 목록을 추출.
    반환: [(법령축약, 조문키), ...]  예: [("규칙","37"), ("법","167"), ("","별표2")]
    법령축약이 ""면 법령 무관 매칭.
    '규칙 제37조 + 법 제38조·제167조' 같은 복합 표기도 처리:
    조문 앞에 나온 가장 가까운 법령 단서(법/시행령/규칙)를 따라간다.
    """
    # '제38·39조'처럼 두 번째 숫자에 '제/조'가 생략된 표기를 먼저 확장합니다.
    # 두 번 이상 연결된 경우를 위해 변화가 없을 때까지 반복합니다.
    expanded = ref_article
    while True:
        new = re.sub(
            r"제\s*(\d+(?:\s*의\s*\d+)?)\s*[·ㆍ]\s*"
            r"(\d+(?:\s*의\s*\d+)?)\s*조",
            r"제\1조·제\2조",
            expanded,
        )
        if new == expanded:
            break
        expanded = new

    default_law = _law_short(ref_law)
    gold = []
    current_law = default_law
    # 텍스트를 순회하며 법령 단서를 갱신하고, 조문 패턴을 수집
    tokens = re.split(r"(\s+|\+|·|ㆍ|,|/)", expanded)
    for tok in tokens:
        ls = _law_short(tok) if tok.strip() else ""
        if ls:
            current_law = ls
        for m in _ART_RE.finditer(tok):
            if m.group(1):  # 제N조(의M)
                key = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
                gold.append((current_law, _normalize_article(key)))
            elif m.group(0).startswith("별표"):
                key = "별표" + (m.group(3) or "")
                gold.append((current_law, _normalize_article(key)))
    # 중복 제거(순서 유지)
    seen, out = set(), []
    for g in gold:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _parse_source(source: str) -> tuple[str, str]:
    """
    검색 결과 source 문자열 → (법령축약, 조문키).
    지원 표기: '... 제37조', '... 제619조의2', '... 별표2', '... 37' (구형)
    """
    s = source.strip()
    # 신형: '제N조(의M)' 또는 '별표N'
    m = re.search(r"제\s*(\d+)\s*조(?:의\s*(\d+))?\s*$", s)
    if m:
        key = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
        return (_law_short(s[:m.start()]), _normalize_article(key))
    m = re.search(r"(별표\s*\d*(?:의\d+)?)\s*$", s)
    if m:
        return (_law_short(s[:m.start()]), _normalize_article(m.group(1)))
    # 구형: 끝이 숫자('37', '619의2')
    m = re.match(r"^(.*?)\s*(\d+(?:의\d+)?)$", s)
    if m:
        return (_law_short(m.group(1)), _normalize_article(m.group(2)))
    return (_law_short(s), "")


def _match(gold: tuple[str, str], src: tuple[str, str]) -> bool:
    """정답 조문과 검색 조문을 법령명과 정확한 조문/별표 번호로 비교합니다."""
    g_law, g_art = gold
    s_law, s_art = src
    g_art = _normalize_article(g_art)
    s_art = _normalize_article(s_art)
    if g_art != s_art:
        return False
    return g_law == "" or s_law == "" or g_law == s_law


def retrieval_metrics(sources: list[str], gold: list[tuple[str, str]]) -> dict:
    """Hit@k / Recall@k / Precision@k / MRR 계산. gold 없으면 전부 None."""
    if not gold:
        return {"hit_at_k": None, "recall_at_k": None,
                "precision_at_k": None, "mrr": None}
    parsed = [_parse_source(s) for s in sources]

    found = set()
    first_rank = None
    n_relevant_docs = 0
    for rank, src in enumerate(parsed, 1):
        matched = False
        for g in gold:
            if _match(g, src):
                found.add(g)
                matched = True
        if matched:
            n_relevant_docs += 1
            if first_rank is None:
                first_rank = rank

    return {
        "hit_at_k": 1 if found else 0,
        "recall_at_k": round(len(found) / len(gold), 3),
        "precision_at_k": round(n_relevant_docs / len(parsed), 3) if parsed else 0.0,
        "mrr": round(1 / first_rank, 3) if first_rank else 0.0,
    }


def citation_metrics(
    used_sources: list[str],
    gold: list[tuple[str, str]],
) -> dict:
    """
    실제 사용 출처(used_sources)를 정답 조문과 비교합니다.

    sources는 검색 상위 k개 전체라서 답변이 실제 사용하지 않은 문서도 포함합니다.
    따라서 인용 정확도는 반드시 모델이 [C1] 등으로 선택한 used_sources로 계산합니다.
    """
    if not gold:
        return {
            "citation_acc": None,
            "citation_precision": None,
            "citation_recall": None,
            "citation_f1": None,
        }

    cited = []
    for source in used_sources:
        parsed = _parse_source(source)
        if parsed not in cited:
            cited.append(parsed)

    if not cited:
        return {
            "citation_acc": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "citation_f1": 0.0,
        }

    matched_cited = {
        src for src in cited if any(_match(g, src) for g in gold)
    }
    matched_gold = {
        g for g in gold if any(_match(g, src) for src in cited)
    }
    precision = len(matched_cited) / len(cited)
    recall = len(matched_gold) / len(gold)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        # 과거 결과 파일과의 호환성을 위해 citation_acc는 precision과 같은 값입니다.
        "citation_acc": round(precision, 3),
        "citation_precision": round(precision, 3),
        "citation_recall": round(recall, 3),
        "citation_f1": round(f1, 3),
    }


_REFUSAL_PATTERNS = ["찾을 수 없습니다", "답변드릴 수 없", "답변하기 어렵",
                     "제공하기 어렵", "확인할 수 없", "확인이 어렵", "알 수 없습니다"]


def refusal_accuracy(answer: str, qtype: str, status: str = "") -> float | None:
    """답변불가형(unanswerable) 질문에서 거절했으면 1, 아니면 0. 그 외 유형은 None."""
    if qtype != "unanswerable":
        return None
    refused = status == "REFUSE" or any(p in answer for p in _REFUSAL_PATTERNS)
    return 1.0 if refused else 0.0


def answerability_accuracy(answer: str, qtype: str, status: str = "") -> float:
    """
    답변 가능 질문은 답하고, 답변 불가 질문은 거절했는지 모두 평가합니다.
    기존 refusal_acc는 unanswerable 2문항만 봐서 과도한 거절을 발견하지 못했습니다.
    """
    refused = status == "REFUSE" or any(p in answer for p in _REFUSAL_PATTERNS)
    should_refuse = qtype == "unanswerable"
    return 1.0 if refused == should_refuse else 0.0


def load_references(path) -> dict:
    """references.csv → {id: [(법령축약, 조문키), ...]}"""
    refs = {}
    p = Path(path)
    if not p.exists():
        print(f"  [경고] references.csv 없음({p}) — 검색 지표 생략")
        return refs
    with open(p, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            refs[row["id"]] = parse_gold_articles(
                row.get("근거_법령", ""), row.get("근거_조문", ""))
    return refs


# =====================================================================
# 1) BERTScore
# =====================================================================
def compute_bertscore(cands: list[str], refs: list[str]) -> list[dict]:
    """한국어 BERTScore Precision/Recall/F1 리스트 반환.
    반환: [{"bertscore_p":.., "bertscore_r":.., "bertscore_f1":..}, ...]"""
    from bert_score import score
    P, R, F1 = score(cands, refs, lang="ko", verbose=False)
    return [
        {
            "bertscore_p": round(float(p), 4),
            "bertscore_r": round(float(r), 4),
            "bertscore_f1": round(float(f), 4),
        }
        for p, r, f in zip(P, R, F1)
    ]


# =====================================================================
# 2) RAGAS (선택)
# =====================================================================
def compute_ragas(questions, answers, contexts_list, ground_truths,
                  judge: str = "upstage"):
    """
    RAGAS 지표를 문항별 DataFrame으로 반환.

    judge: 채점에 쓸 LLM 심판.
      - "upstage" (기본): Solar-pro + solar 임베딩. UPSTAGE_API_KEY 필요.
      - "openai"        : RAGAS 기본. OPENAI_API_KEY 필요.
    한국어는 심판 LLM에 따라 점수 변동이 있으니 참고 지표로 사용할 것.
    """
    # .env 로드 (단독 실행 대비)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )

    kwargs = {}
    if judge == "upstage":
        from langchain_upstage import ChatUpstage, UpstageEmbeddings
        judge_llm = ChatUpstage(model="solar-pro", temperature=0)
        judge_emb = UpstageEmbeddings(model="solar-embedding-1-large")
        # ragas 버전에 따라 래퍼가 필요한 경우와 아닌 경우가 있어 둘 다 대응
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            kwargs["llm"] = LangchainLLMWrapper(judge_llm)
            kwargs["embeddings"] = LangchainEmbeddingsWrapper(judge_emb)
        except ImportError:
            kwargs["llm"] = judge_llm
            kwargs["embeddings"] = judge_emb
    # judge == "openai" 면 kwargs 비움 → RAGAS 기본(OpenAI) 사용

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,       # 각 항목이 문자열 리스트
        "ground_truth": ground_truths,
    })
    result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        **kwargs,
    )
    return result.to_pandas()


# =====================================================================
# 3) 메인 평가 루프
# =====================================================================
def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def evaluate(
    chain,
    questions_csv: str,
    out_csv: str = "../eval/results.csv",
    exp_config: dict | None = None,
    references_csv: str | None = None,   # 미지정 시 questions.csv 옆의 references.csv
    use_bertscore: bool = True,
    use_keyword: bool = True,
    use_ragas: bool = False,
    ragas_judge: str = "upstage",   # "upstage" | "openai"
    save: bool = True,
):
    """
    Args:
        chain         : build_rag_chain() 결과
        questions_csv : GT 질문셋 경로
        out_csv       : 결과 저장 경로
        exp_config    : 실험조건 기록용 dict
                        (예: {"model":"Qwen2.5-7B","prompt":"cite","top_k":3,
                              "search":"similarity","store":"faiss","embed":"hf"})
        use_*         : 각 지표 on/off (전부 True면 모두 계산)
    """
    exp_config = exp_config or {}
    rows = load_csv(questions_csv)

    # references.csv 로드 (검색 지표의 정답 조문)
    if references_csv is None:
        references_csv = str(Path(questions_csv).parent / "references.csv")
    references = load_references(references_csv)

    # --- 각 문항에 대해 답변 생성 ---
    records = []
    for r in rows:
        q = r["question"]
        gt = r["ground_truth_answer"]
        qid = r.get("id", "")
        qtype = r.get("유형", "")
        res = chain.ask(q)   # {answer, contexts, sources, latency}

        rec = {
            "id": qid,
            "유형": qtype,
            "난이도": r.get("난이도", ""),
            "question": q,
            "ground_truth": gt,
            "answer": res["answer"],
            # 후처리 전 모델 출력도 저장해 모델 오류와 프로그램 오류를 구분합니다.
            "raw_answer": res.get("raw_answer", res["answer"]),
            "latency": res["latency"],
            "contexts": res["contexts"],
            "sources": "; ".join(res["sources"]),
            # 모델이 처음 고른 인용과 프로그램 검증 후 인용을 따로 저장합니다.
            "model_used_sources": "; ".join(
                res.get("model_used_sources", [])
            ),
            "used_sources": "; ".join(res.get("used_sources", [])),
            "citation_repaired": res.get("citation_repaired", False),
            "status": res.get("status", ""),
            "citation_status": res.get("citation_status", ""),
            "question_mode": res.get("question_mode", ""),
            "max_new_tokens_used": res.get("max_new_tokens_used", ""),
            "top_score": res.get("top_score", ""),
        }
        if use_keyword:
            kws = extract_keywords(gt)
            rec["keyword_rate"] = keyword_rate(res["answer"], kws)
            rec["keywords"] = ", ".join(kws)

        # --- 검색 지표 (references.csv 기반) ---
        gold = references.get(qid, [])
        rec.update(retrieval_metrics(res["sources"], gold))
        rec.update(citation_metrics(res.get("used_sources", []), gold))
        # 자동 교정 전 모델 자체 인용 점수도 남겨 후처리로 점수를 숨기지 않습니다.
        model_citation = citation_metrics(
            res.get("model_used_sources", []),
            gold,
        )
        for key, value in model_citation.items():
            rec[f"model_{key}"] = value
        rec["citation_repair_rate"] = (
            1.0 if res.get("citation_repaired", False) else 0.0
        )
        rec["refusal_acc"] = refusal_accuracy(
            res["answer"], qtype, res.get("status", "")
        )
        rec["answerability_acc"] = answerability_accuracy(
            res["answer"], qtype, res.get("status", "")
        )

        records.append(rec)

    # --- BERTScore (배치로 한 번에, P/R/F1 모두) ---
    if use_bertscore:
        scores = compute_bertscore(
            [x["answer"] for x in records],
            [x["ground_truth"] for x in records],
        )
        for x, s in zip(records, scores):
            x.update(s)  # bertscore_p / bertscore_r / bertscore_f1

    # --- RAGAS ---
    if use_ragas:
        try:
            df = compute_ragas(
                [x["question"] for x in records],
                [x["answer"] for x in records],
                [x["contexts"] for x in records],
                [x["ground_truth"] for x in records],
                judge=ragas_judge,
            )
            for i, x in enumerate(records):
                for col in ["faithfulness", "answer_relevancy",
                            "context_precision", "context_recall"]:
                    if col in df.columns:
                        x[col] = round(float(df.iloc[i][col]), 4)
        except Exception as e:
            print(f"  [RAGAS 건너뜀] {e}")

    # --- 저장 (save=False면 main이 통합 저장) ---
    if save:
        _save_results(records, out_csv, exp_config)
    _print_summary(records)
    return records


# =====================================================================
# 저장 & 요약
# =====================================================================
_METRIC_COLS = ["bertscore_p", "bertscore_r", "bertscore_f1",
                "keyword_rate", "latency",
                "hit_at_k", "recall_at_k", "precision_at_k", "mrr",
                "citation_acc", "citation_precision", "citation_recall",
                "citation_f1", "refusal_acc", "answerability_acc",
                "model_citation_acc", "model_citation_precision",
                "model_citation_recall", "model_citation_f1",
                "citation_repair_rate",
                "faithfulness", "answer_relevancy",
                "context_precision", "context_recall"]


def _save_results(records, out_csv, exp_config):
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    base_cols = ["id", "유형", "난이도", "question", "ground_truth",
                 "answer", "raw_answer", "keywords", "sources",
                 "model_used_sources", "used_sources", "citation_repaired",
                 "contexts", "status", "citation_status", "question_mode",
                 "max_new_tokens_used", "top_score"]
    exp_cols = list(exp_config.keys())
    cols = exp_cols + base_cols + _METRIC_COLS

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for x in records:
            row = [exp_config.get(c, "") for c in exp_cols]
            row += [x.get(c, "") for c in base_cols]
            row += [x.get(c, "") for c in _METRIC_COLS]
            w.writerow(row)
    print(f"\n[결과 저장] {out}  ({len(records)}문항)")


def _avg(records, key):
    vals = [x[key] for x in records if isinstance(x.get(key), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def _print_summary(records):
    print("\n===== 평가 요약 (전체 평균) =====")
    for k in _METRIC_COLS:
        v = _avg(records, k)
        if v is not None:
            print(f"  {k:20s}: {v}")

    # 유형별 BERTScore
    print("\n----- 유형별 BERTScore F1 -----")
    types = {}
    for x in records:
        types.setdefault(x.get("유형", ""), []).append(x)
    for t, xs in types.items():
        v = _avg(xs, "bertscore_f1")
        print(f"  {t:14s}: {v}  (n={len(xs)})")

    # 검색 vs 생성 진단 (Hit@k와 BERTScore 조합)
    hit = _avg(records, "hit_at_k")
    f1 = _avg(records, "bertscore_f1")
    if hit is not None:
        print("\n----- 검색/생성 진단 -----")
        print(f"  Hit@k={hit}, BERTScore F1={f1}")
        if hit < 0.7:
            print("  → 검색이 정답 조문을 자주 놓침: top_k/검색방식/임베딩 개선 우선")
        elif f1 is not None and f1 < 0.7:
            print("  → 검색은 되는데 답변 품질이 낮음: 프롬프트/모델/파인튜닝 개선 우선")
        else:
            print("  → 검색·생성 모두 양호")


# =====================================================================
# 단독 실행
# =====================================================================
if __name__ == "__main__":
    import argparse
    from load_data import load_data
    from build_vectorstore import build_vectorstore
    from rag_chain import build_rag_chain

    parser = argparse.ArgumentParser(description="RAG 평가 (BERTScore/키워드/응답시간/RAGAS)")
    parser.add_argument("data_path")
    parser.add_argument("--file_type", required=True, choices=["json", "pdf"])
    parser.add_argument("--questions_csv", default="../eval/questions.csv")
    parser.add_argument("--out_csv", default="../eval/results.csv")
    parser.add_argument("--store_type", default="faiss")
    parser.add_argument("--embedding_name", default="hf")
    parser.add_argument("--llm_type", default="openai")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--prompt_name", default="cite")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--search_type", default="similarity")
    parser.add_argument("--no_ragas", action="store_true", help="RAGAS 제외")
    args = parser.parse_args()

    docs = load_data(args.data_path, args.file_type)
    store = build_vectorstore(docs, store_type=args.store_type,
                              embedding_name=args.embedding_name, persist_dir=None)
    chain = build_rag_chain(
        store, llm_type=args.llm_type, model_name=args.model_name,
        prompt_name=args.prompt_name, top_k=args.top_k, search_type=args.search_type,
    )

    exp_config = {
        "model": args.model_name or args.llm_type,
        "prompt": args.prompt_name,
        "top_k": args.top_k,
        "search": args.search_type,
        "store": args.store_type,
        "embed": args.embedding_name,
    }

    evaluate(
        chain,
        questions_csv=args.questions_csv,
        out_csv=args.out_csv,
        exp_config=exp_config,
        use_ragas=not args.no_ragas,
    )
