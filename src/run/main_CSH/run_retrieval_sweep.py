"""
run_retrieval_sweep.py
---------------------------------------------------
검색(retrieval) 성능만 측정하는 실험 도구.

LLM은 검색이 끝난 다음 단계에서 동작하므로, 검색 지표(Hit@k / Recall@k /
Precision@k / MRR)는 어떤 LLM을 쓰든 완전히 동일하다.
(실험 1에서 프롬프트 4종의 검색 지표가 소수점까지 같았던 것이 그 증거)

따라서 라마를 아예 띄우지 않고 검색만 돌리면, 조합 수십 개를 몇 분 만에 훑을 수 있다.

채점 로직은 evaluate.py 의 함수를 그대로 가져다 쓰므로 코드 중복이 없다.
main.py 의 벡터DB 저장 규칙도 그대로 따라가므로, 이미 만들어 둔 벡터DB는 재사용된다.

사용법:
    cd src
    python run_retrieval_sweep.py

결과:
    ../eval/results/retrieval_sweep_summary.csv   조합별 평균 지표
    ../eval/results/retrieval_sweep_detail.csv    문항별 상세 (실험 3 임계값 계산용)
"""

import csv
from pathlib import Path

# --- 팀 공용 코드 재사용 (LLM을 부르지 않는 모듈만 import 하므로 torch 안 뜸) ---
from main import _safe
from load_data import load_data
from build_vectorstore import build_vectorstore, load_vectorstore
from rag_chain import _source_name
from evaluate import load_csv, load_references, retrieval_metrics


# =====================================================================
# 1) 훑어볼 조합 - 여기만 고치면 실험이 바뀐다
# =====================================================================
CHUNK_SIZES = [500, 300, 200]          # 임베딩 128토큰 제한에 맞춰 줄여본다
TOP_KS = [3, 5, 10]                    # 더 많이 가져오면 정답이 들어올까
SEARCH_TYPES = ["similarity", "mmr"]   # mmr = 다양성 반영 → 중복 청크 완화 기대


# =====================================================================
# 2) 고정 설정 (main.py 기본값과 동일하게 맞춤)
# =====================================================================
DATA_PATH = "../data/laws_all.json"
FILE_TYPE = "json"
OVERLAP_SIZE = 50
STORE_TYPE = "faiss"
EMBEDDING_NAME = "hf"
STORE_DIR = "../stores"

QUESTIONS_CSV = "../eval/questions.csv"
RESULTS_DIR = "../eval/results"
SUMMARY_CSV = f"{RESULTS_DIR}/retrieval_sweep_summary.csv"
DETAIL_CSV = f"{RESULTS_DIR}/retrieval_sweep_detail.csv"

METRICS = ["hit_at_k", "recall_at_k", "precision_at_k", "mrr"]


# =====================================================================
# 3) 벡터DB 준비 - main.py 와 같은 규칙으로 저장/재사용
# =====================================================================
def get_store(chunk_size: int):
    """청크 크기별 벡터DB를 만들거나 불러온다. 이미 있으면 재임베딩하지 않는다."""
    key = (f"{_safe(Path(DATA_PATH).stem)}_c{chunk_size}"
           f"_o{OVERLAP_SIZE}_{STORE_TYPE}_{EMBEDDING_NAME}")
    persist_dir = Path(STORE_DIR) / key

    if persist_dir.exists() and any(persist_dir.iterdir()):
        print(f"[store] 재사용: {persist_dir}")
        return load_vectorstore(store_type=STORE_TYPE,
                                embedding_name=EMBEDDING_NAME,
                                persist_dir=str(persist_dir))

    print(f"[store] 새로 생성: {persist_dir}  (임베딩 중, 몇 분 걸릴 수 있음)")
    docs = load_data(DATA_PATH, FILE_TYPE,
                     chunk_size=chunk_size, overlap_size=OVERLAP_SIZE)
    return build_vectorstore(docs, store_type=STORE_TYPE,
                             embedding_name=EMBEDDING_NAME,
                             persist_dir=str(persist_dir))


# =====================================================================
# 4) 검색 한 번 (rag_chain.RagChain 과 같은 방식)
# =====================================================================
def retrieve(store, question: str, top_k: int, search_type: str):
    """
    반환: (출처 문자열 리스트, 최고 관련도 점수)
    mmr 은 점수를 제공하지 않으므로 점수는 None.
    """
    if search_type == "mmr":
        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": max(top_k * 4, 20)},
        )
        docs = retriever.invoke(question)
        return [_source_name(d) for d in docs], None

    pairs = store.similarity_search_with_relevance_scores(question, k=top_k)
    docs = [d for d, _ in pairs]
    top_score = max((s for _, s in pairs), default=0.0)
    return [_source_name(d) for d in docs], round(float(top_score), 4)


# =====================================================================
# 5) 조합 하나 평가
# =====================================================================
def run_one(store, questions, references, chunk_size, top_k, search_type):
    detail = []
    for q in questions:
        qid = q.get("id", "")
        sources, top_score = retrieve(store, q["question"], top_k, search_type)
        gold = references.get(qid, [])
        m = retrieval_metrics(sources, gold)

        detail.append({
            "chunk": chunk_size,
            "top_k": top_k,
            "search": search_type,
            "id": qid,
            "유형": q.get("유형", ""),
            "난이도": q.get("난이도", ""),
            "question": q["question"],
            "top_score": top_score,
            "sources": "; ".join(sources),
            "중복수": len(sources) - len(set(sources)),
            **m,
        })
    return detail


def avg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


# =====================================================================
# 6) 실행
# =====================================================================
def main():
    questions = load_csv(QUESTIONS_CSV)
    references = load_references(str(Path(QUESTIONS_CSV).parent / "references.csv"))
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    all_detail = []
    summary = []
    total = len(CHUNK_SIZES) * len(TOP_KS) * len(SEARCH_TYPES)
    n = 0

    for chunk_size in CHUNK_SIZES:
        print("\n" + "=" * 70)
        print(f"청크 크기 {chunk_size}자")
        print("=" * 70)
        store = get_store(chunk_size)

        for search_type in SEARCH_TYPES:
            for top_k in TOP_KS:
                n += 1
                print(f"  [{n}/{total}] chunk={chunk_size}, "
                      f"search={search_type}, top_k={top_k} ...", end=" ")

                detail = run_one(store, questions, references,
                                 chunk_size, top_k, search_type)
                all_detail += detail

                row = {"chunk": chunk_size, "search": search_type, "top_k": top_k}
                for m in METRICS:
                    row[m] = avg(detail, m)
                row["중복평균"] = avg(detail, "중복수")
                summary.append(row)
                print(f"Hit@k={row['hit_at_k']}")

    # ---------------- 저장 ----------------
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    with open(DETAIL_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_detail[0].keys()))
        w.writeheader()
        w.writerows(all_detail)

    # ---------------- 표 출력 ----------------
    print("\n" + "=" * 78)
    print("검색 성능 비교 (Hit@k 높은 순)")
    print("=" * 78)
    print(f"{'청크':<7}{'검색방식':<14}{'top_k':<8}"
          f"{'Hit@k':<9}{'Recall':<9}{'Precision':<11}{'MRR':<9}{'중복':<6}")
    print("-" * 78)
    for r in sorted(summary, key=lambda x: (-(x["hit_at_k"] or 0), -(x["mrr"] or 0))):
        print(f"{r['chunk']:<7}{r['search']:<14}{r['top_k']:<8}"
              f"{r['hit_at_k']:<9}{r['recall_at_k']:<9}"
              f"{r['precision_at_k']:<11}{r['mrr']:<9}{r['중복평균']:<6}")

    best = max(summary, key=lambda x: (x["hit_at_k"] or 0, x["mrr"] or 0))
    print("-" * 78)
    print(f"최고: chunk={best['chunk']}, search={best['search']}, "
          f"top_k={best['top_k']}  →  Hit@k={best['hit_at_k']}, MRR={best['mrr']}")
    print(f"\n기준선(실험1): chunk=500, similarity, top_k=3 → Hit@k=0.5, MRR=0.4167")
    print("=" * 78)
    print(f"\n[저장] {SUMMARY_CSV}")
    print(f"[저장] {DETAIL_CSV}")
    print("\n※ top_k 를 늘리면 Hit@k 는 오르지만 Precision 은 떨어지는 게 정상이다.")
    print("  LLM 에 넘기는 근거가 많아질수록 노이즈도 같이 늘어나므로,")
    print("  Hit@k 만 보고 top_k 를 무작정 키우면 답변 품질이 오히려 나빠질 수 있다.")


if __name__ == "__main__":
    main()
