"""
evaluate.py
---------------------------------------------------
RAG 파이프라인 4단계: 평가.

계산하는 지표 (전부 산출 → 나중에 취사선택)
  1) BERTScore(F1)   : 모델 답변 ↔ 정답(GT)의 의미 유사도. (필수 지표)
  2) 응답시간(latency): chain.ask() 가 측정. 초 단위.
  3) 키워드 포함률    : 정답의 핵심 수치·조문(예: 15미터, 제37조)이 답변에 들어갔는지.
                       BERTScore가 못 잡는 '수치 오류'를 보완.
  4) RAGAS           : faithfulness / answer_relevancy / context_precision /
                       context_recall (LLM 심판, hallucination·검색품질 자동화).

입력
  - eval/questions.csv  : id, 대상법령, question, ground_truth_answer, 유형, 난이도
  - eval/references.csv : id, 근거_법령, 근거_조문, 근거_원문  (RAGAS ground truth 근거)

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
def compute_ragas(questions, answers, contexts_list, ground_truths):
    """
    RAGAS 지표를 문항별 DataFrame으로 반환.
    기본적으로 OPENAI_API_KEY 를 사용(LLM 심판). 한국어는 변동성 있으니 참고용.
    """
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,       # 각 항목이 문자열 리스트
        "ground_truth": ground_truths,
    })
    result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
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
    use_bertscore: bool = True,
    use_keyword: bool = True,
    use_ragas: bool = True,
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

    # --- 각 문항에 대해 답변 생성 ---
    records = []
    for r in rows:
        q = r["question"]
        gt = r["ground_truth_answer"]
        res = chain.ask(q)   # {answer, contexts, sources, latency}

        rec = {
            "id": r.get("id", ""),
            "유형": r.get("유형", ""),
            "난이도": r.get("난이도", ""),
            "question": q,
            "ground_truth": gt,
            "answer": res["answer"],
            "latency": res["latency"],
            "contexts": res["contexts"],
            "sources": "; ".join(res["sources"]),
        }
        if use_keyword:
            kws = extract_keywords(gt)
            rec["keyword_rate"] = keyword_rate(res["answer"], kws)
            rec["keywords"] = ", ".join(kws)
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
                "faithfulness", "answer_relevancy",
                "context_precision", "context_recall"]


def _save_results(records, out_csv, exp_config):
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    base_cols = ["id", "유형", "난이도", "question", "ground_truth",
                 "answer", "keywords", "sources"]
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
