"""
compare_embedding_results_ksj.py
---------------------------------------------------
기존 한국어 SBERT와 BGE-M3 결과 CSV를 같은 지표로 비교합니다.

이 스크립트는 모델을 실행하지 않고 이미 생성된 두 CSV만 읽습니다.
검색 성능을 우선 비교하고, 전체 답변 품질과 응답시간도 함께 표시합니다.

실행:
    python src/compare_embedding_results_ksj.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 임베딩 비교의 핵심은 검색 지표입니다.
# 답변 지표도 표시하지만 생성 모델의 미세한 출력 차이가 섞일 수 있습니다.
METRICS = [
    "hit_at_k",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "bertscore_f1",
    "keyword_rate",
    "citation_f1",
    "model_citation_f1",
    "latency",
]


def _load_csv(path: Path) -> list[dict[str, str]]:
    """UTF-8 BOM이 포함된 평가 CSV도 안전하게 읽습니다."""
    if not path.exists():
        raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"결과 파일이 비어 있습니다: {path}")
    return rows


def _number(value: str | None) -> float | None:
    """빈칸과 NaN을 평균 계산에서 제외할 수 있게 변환합니다."""
    try:
        number = float(value or "")
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _mean(rows: list[dict[str, str]], column: str) -> float | None:
    """숫자가 기록된 행만 사용해 평균을 계산합니다."""
    values = [_number(row.get(column)) for row in rows]
    filtered = [value for value in values if value is not None]
    return statistics.mean(filtered) if filtered else None


def _fmt(value: float | None) -> str:
    """표 출력용 소수점 형식입니다."""
    return "-" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="임베딩 모델 평가 결과 비교")
    parser.add_argument(
        "--baseline",
        default=str(
            PROJECT_ROOT
            / "eval"
            / "results"
            / "mistral7b_base_hybrid_v3_ksj.csv"
        ),
    )
    parser.add_argument(
        "--candidate",
        default=str(
            PROJECT_ROOT
            / "eval"
            / "results"
            / "mistral7b_base_hybrid_v3_bge_m3_ksj.csv"
        ),
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = _load_csv(baseline_path)
    candidate = _load_csv(candidate_path)

    # 실험 조건이 바뀌지 않았는지 핵심 열을 함께 출력합니다.
    condition_columns = [
        "model",
        "prompt",
        "top_k",
        "search",
        "store",
        "chunk",
    ]
    print("=" * 78)
    print("[임베딩 비교 조건]")
    for column in condition_columns:
        left = baseline[0].get(column, "")
        right = candidate[0].get(column, "")
        marker = "같음" if left == right else "다름"
        print(f"- {column:10s}: {left} | {right} ({marker})")
    print(
        "- embedding : "
        f"{baseline[0].get('embedding_model') or 'jhgan/ko-sbert-nli'}"
        " | "
        f"{candidate[0].get('embedding_model') or 'BAAI/bge-m3'}"
    )

    print("\n[전체 평균]")
    print(f"{'metric':24s} {'baseline':>10s} {'bge-m3':>10s} {'delta':>10s}")
    print("-" * 58)
    for metric in METRICS:
        left = _mean(baseline, metric)
        right = _mean(candidate, metric)
        delta = None if left is None or right is None else right - left
        delta_text = "-" if delta is None else f"{delta:+.4f}"
        print(
            f"{metric:24s} {_fmt(left):>10s} {_fmt(right):>10s} "
            f"{delta_text:>10s}"
        )

    # 검색 지표가 개선됐는지 단순 판정하되, 작은 평가셋의 한계를 알립니다.
    base_recall = _mean(baseline, "recall_at_k")
    bge_recall = _mean(candidate, "recall_at_k")
    base_mrr = _mean(baseline, "mrr")
    bge_mrr = _mean(candidate, "mrr")
    print("\n[판정]")
    if (
        base_recall is not None
        and bge_recall is not None
        and base_mrr is not None
        and bge_mrr is not None
        and (bge_recall > base_recall or bge_mrr > base_mrr)
    ):
        print("- BGE-M3가 Recall@k 또는 MRR을 개선했습니다.")
    elif bge_recall == base_recall and bge_mrr == base_mrr:
        print("- 핵심 검색 지표가 동일합니다. 속도와 운영 복잡도를 함께 비교하세요.")
    else:
        print("- 현재 평가셋에서는 기존 임베딩의 검색 지표가 더 높습니다.")
    print("- 질문 수가 적으므로 최종 교체 전 평가 질문을 늘려 재확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
