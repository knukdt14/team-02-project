"""
run_embedding_sweep.py
---------------------------------------------------
임베딩 모델 × 청크 크기 대조 실험.

[배경]
실험 2에서 청크 크기를 500 → 300 → 200으로 줄여봤지만 Hit@k 개선이 거의 없었다.
하지만 그것은 "임베딩의 128토큰 제한을 피해가려는" 시도였을 뿐,
제한 자체를 없앤 것이 아니었다.

[이번 가설]
현재 임베딩 jhgan/ko-sbert-nli 는
  (1) 최대 입력이 128토큰이라 긴 청크의 뒷부분을 잘라먹고
  (2) 애초에 '검색용'이 아니라 '문장 유사도(NLI)'용으로 학습된 모델이다.
긴 입력을 받는 검색 전용 모델(BAAI/bge-m3, 8192토큰)로 바꾸면 개선될 것이다.

[실험 설계 — 2x2 대조]
                 청크 500      청크 1000
  ko-sbert(128)   기준선        더 나빠져야 정상  ← 잘림이 원인이라면
  bge-m3(8192)    ?             여기가 최선일 것

ko-sbert 가 청크 1000에서 나빠지고 bge-m3 는 안 나빠진다면,
'잘림이 원인'이라는 가설이 대조군과 함께 증명된다.

[중요]
팀 공용 코드(main.py, rag_chain.py)는 건드리지 않는다.
build_vectorstore() 에 embedding_model 인자가 이미 있어서, 이 파일에서 직접 넘기면 된다.
결과가 좋으면 그때 main.py 에 옵션을 뚫는 PR을 올린다.

사용법:
    cd src
    python run_embedding_sweep.py
"""

import csv
from pathlib import Path

from main import _safe
from load_data import load_data
from build_vectorstore import build_vectorstore, load_vectorstore
from rag_chain import _source_name
from evaluate import load_csv, load_references, retrieval_metrics


# =====================================================================
# 1) 훑어볼 조합
# =====================================================================
# (표시이름, 실제 모델명)  모델명 None = build_vectorstore 기본값(jhgan/ko-sbert-nli)
EMBEDDINGS = [
    ("ko-sbert", None),
    ("bge-m3", "BAAI/bge-m3"),
]

CHUNK_SIZES = [500, 1000]              # 긴 청크에서 차이가 드러나는지 확인
TOP_KS = [3, 5, 10]
SEARCH_TYPES = ["similarity", "mmr"]


# =====================================================================
# 2) 고정 설정
# =====================================================================
DATA_PATH = "../data/laws_all.json"
FILE_TYPE = "json"
OVERLAP_SIZE = 50
STORE_TYPE = "faiss"
EMBEDDING_NAME = "hf"                  # HuggingFace 계열 (모델명만 바꿔가며 씀)
STORE_DIR = "../stores"

QUESTIONS_CSV = "../eval/questions.csv"
RESULTS_DIR = "../eval/results"
SUMMARY_CSV = f"{RESULTS_DIR}/embedding_sweep_summary.csv"
DETAIL_CSV = f"{RESULTS_DIR}/embedding_sweep_detail.csv"

METRICS = ["hit_at_k", "recall_at_k", "precision_at_k", "mrr"]

# 실험 1 기준선 (chunk=500, ko-sbert, similarity, top_k=3)
BASELINE = {"hit_at_k": 0.5, "mrr": 0.4167, "precision_at_k": 0.2222}


# =====================================================================
# 3) 벡터DB 준비
#    저장 폴더 이름에 '임베딩 모델'까지 넣는다.
#    (main.py 는 embedding_name 만 넣어서, 모델을 바꿔도 옛 벡터DB를 재사용해버린다.
#     그 버그를 피하려고 여기서는 모델명을 키에 포함시킨다.)
# =====================================================================
def store_tag(emb_model):
    return EMBEDDING_NAME if emb_model is None else f"{EMBEDDING_NAME}-{_safe(emb_model)}"


def get_store(chunk_size: int, emb_model):
    key = (f"{_safe(Path(DATA_PATH).stem)}_c{chunk_size}"
           f"_o{OVERLAP_SIZE}_{STORE_TYPE}_{store_tag(emb_model)}")
    persist_dir = Path(STORE_DIR) / key

    if persist_dir.exists() and any(persist_dir.iterdir()):
        print(f"[store] 재사용: {persist_dir}")
        return load_vectorstore(store_type=STORE_TYPE,
                                embedding_name=EMBEDDING_NAME,
                                embedding_model=emb_model,
                                persist_dir=str(persist_dir))

    print(f"[store] 새로 생성: {persist_dir}")
    print("        (임베딩 중... bge-m3 는 모델이 2.2GB라 처음엔 다운로드도 필요)")
    docs = load_data(DATA_PATH, FILE_TYPE,
                     chunk_size=chunk_size, overlap_size=OVERLAP_SIZE)
    return build_vectorstore(docs, store_type=STORE_TYPE,
                             embedding_name=EMBEDDING_NAME,
                             embedding_model=emb_model,
                             persist_dir=str(persist_dir))


# =====================================================================
# 4) 검색 (rag_chain.RagChain 과 동일한 방식)
# =====================================================================
def retrieve(store, question, top_k, search_type):
    if search_type == "mmr":
        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": max(top_k * 4, 20)},
        )
        docs = retriever.invoke(question)
        return [_source_name(d) for d in docs], None

    pairs = store.similarity_search_with_relevance_scores(question, k=top_k)
    return ([_source_name(d) for d, _ in pairs],
            round(float(max((s for _, s in pairs), default=0.0)), 4))


def avg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


# =====================================================================
# 5) 실행
# =====================================================================
def main():
    questions = load_csv(QUESTIONS_CSV)
    references = load_references(str(Path(QUESTIONS_CSV).parent / "references.csv"))
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    all_detail, summary = [], []
    total = len(EMBEDDINGS) * len(CHUNK_SIZES) * len(SEARCH_TYPES) * len(TOP_KS)
    n = 0

    for emb_label, emb_model in EMBEDDINGS:
        for chunk_size in CHUNK_SIZES:
            print("\n" + "=" * 72)
            print(f"임베딩 {emb_label}  /  청크 {chunk_size}자")
            print("=" * 72)
            store = get_store(chunk_size, emb_model)

            for search_type in SEARCH_TYPES:
                for top_k in TOP_KS:
                    n += 1
                    print(f"  [{n}/{total}] {emb_label}, chunk={chunk_size}, "
                          f"{search_type}, top_k={top_k} ...", end=" ")

                    detail = []
                    for q in questions:
                        qid = q.get("id", "")
                        sources, top_score = retrieve(
                            store, q["question"], top_k, search_type)
                        detail.append({
                            "embed": emb_label, "chunk": chunk_size,
                            "search": search_type, "top_k": top_k,
                            "id": qid, "유형": q.get("유형", ""),
                            "난이도": q.get("난이도", ""),
                            "question": q["question"],
                            "top_score": top_score,
                            "sources": "; ".join(sources),
                            "중복수": len(sources) - len(set(sources)),
                            **retrieval_metrics(sources, references.get(qid, [])),
                        })
                    all_detail += detail

                    row = {"embed": emb_label, "chunk": chunk_size,
                           "search": search_type, "top_k": top_k}
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

    # ---------------- 전체 표 ----------------
    print("\n" + "=" * 86)
    print("전체 결과 (Hit@k 높은 순)")
    print("=" * 86)
    print(f"{'임베딩':<12}{'청크':<8}{'검색방식':<14}{'top_k':<8}"
          f"{'Hit@k':<9}{'Recall':<9}{'Prec':<9}{'MRR':<9}{'중복':<6}")
    print("-" * 86)
    for r in sorted(summary, key=lambda x: (-(x["hit_at_k"] or 0), -(x["mrr"] or 0))):
        print(f"{r['embed']:<12}{r['chunk']:<8}{r['search']:<14}{r['top_k']:<8}"
              f"{r['hit_at_k']:<9}{r['recall_at_k']:<9}"
              f"{r['precision_at_k']:<9}{r['mrr']:<9}{r['중복평균']:<6}")

    # ---------------- 핵심 2x2 대조표 ----------------
    print("\n" + "=" * 86)
    print("★ 가설 검증용 2x2 대조표  (검색방식=similarity 고정)")
    print("=" * 86)
    for top_k in TOP_KS:
        print(f"\n--- top_k = {top_k} ---")
        print(f"{'임베딩':<12}" + "".join(f"{'청크 '+str(c):<16}" for c in CHUNK_SIZES))
        for emb_label, _ in EMBEDDINGS:
            cells = []
            for c in CHUNK_SIZES:
                hit = next((r["hit_at_k"] for r in summary
                            if r["embed"] == emb_label and r["chunk"] == c
                            and r["search"] == "similarity" and r["top_k"] == top_k), None)
                cells.append(f"{hit}")
            print(f"{emb_label:<12}" + "".join(f"{v:<16}" for v in cells))

    print("\n" + "=" * 86)
    best = max(summary, key=lambda x: (x["hit_at_k"] or 0, x["mrr"] or 0))
    print(f"최고: {best['embed']} / 청크 {best['chunk']} / {best['search']} / "
          f"top_k {best['top_k']}  →  Hit@k={best['hit_at_k']}, MRR={best['mrr']}")
    print(f"실험1 기준선: ko-sbert / 청크 500 / similarity / top_k 3  →  "
          f"Hit@k={BASELINE['hit_at_k']}, MRR={BASELINE['mrr']}")
    print("=" * 86)
    print(f"\n[저장] {SUMMARY_CSV}")
    print(f"[저장] {DETAIL_CSV}")
    print("\n※ 검색 대상은 12문항이므로 Hit@k 는 0.0833 단위로만 움직인다.")
    print("  즉 0.0833 차이 = 문항 1개 차이. 작은 차이는 노이즈로 봐야 한다.")
    print("  ko-sbert 가 청크 1000에서 떨어지고 bge-m3 는 유지/상승한다면")
    print("  '128토큰 잘림이 원인'이라는 가설이 대조군과 함께 증명된 것이다.")


if __name__ == "__main__":
    main()
