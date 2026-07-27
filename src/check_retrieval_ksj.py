"""
check_retrieval_ksj.py
---------------------------------------------------
7B LLM을 로드하지 않고 저장된 벡터DB의 검색 성능만 빠르게 확인합니다.

실행:
    python src\\check_retrieval_ksj.py

출력:
    - 각 질문의 검색 상위 5개 출처
    - references.csv 기준 Hit@5 / Recall@5 / Precision@5 / MRR

이 점검이 통과한 뒤 전체 모델 평가를 실행하면, 검색 문제와 생성 문제를
분리해서 확인할 수 있습니다.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

from evaluate import load_references, retrieval_metrics
from main import CONFIG, get_or_build_store
from rag_chain import RagChain, get_prompt


# 어느 폴더에서 실행해도 프로젝트의 eval 파일을 찾도록 절대 경로를 사용합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _average(rows: list[dict], key: str) -> float:
    """None을 제외한 검색 지표의 평균을 계산합니다."""
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> int:
    # main.py와 같은 설정으로 저장된 FAISS를 로드합니다.
    # 저장소가 없을 때만 기존 코드가 한 번 생성합니다.
    args = SimpleNamespace(**CONFIG)
    args.search_type = "hybrid"
    args.top_k = 5
    store = get_or_build_store(args)

    # 검색만 사용할 것이므로 LLM은 None으로 두어 7B 모델을 전혀 로드하지 않습니다.
    chain = RagChain(
        store=store,
        llm=None,
        prompt=get_prompt("strict"),
        top_k=args.top_k,
        search_type=args.search_type,
        score_threshold=0.0,
    )

    questions_path = PROJECT_ROOT / "eval" / "questions.csv"
    references_path = PROJECT_ROOT / "eval" / "references.csv"
    references = load_references(references_path)
    metric_rows: list[dict] = []

    with questions_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = row.get("id", "")
            qtype = row.get("유형", "")

            # 답변불가형은 검색 정답 조문이 없으므로 검색 평균에서 제외합니다.
            if qtype == "unanswerable":
                continue

            result = chain.search(row["question"])
            metrics = retrieval_metrics(
                result["sources"],
                references.get(qid, []),
            )
            metric_rows.append(metrics)

            print(f"\n[{qid}] {row['question']}")
            for rank, source in enumerate(result["sources"], 1):
                print(f"  {rank}. {source}")
            print(
                "  검색지표:"
                f" Hit@5={metrics['hit_at_k']},"
                f" Recall@5={metrics['recall_at_k']},"
                f" MRR={metrics['mrr']}"
            )

    print("\n===== 검색 회귀 점검 평균 =====")
    for key in ("hit_at_k", "recall_at_k", "precision_at_k", "mrr"):
        print(f"{key:15s}: {_average(metric_rows, key):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
