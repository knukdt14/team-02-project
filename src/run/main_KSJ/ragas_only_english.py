"""
기존 checkpoint_before_ragas.csv를 읽어서
Mistral 답변 생성 없이 RAGAS만 다시 계산합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from evaluate_english import compute_ragas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("UPSTAGE_API_KEY"):
    raise ValueError(
        f"UPSTAGE_API_KEY를 찾지 못했습니다. "
        f"{PROJECT_ROOT / '.env'} 파일을 확인하세요."
    )


LIST_COLUMNS = (
    "keywords",
    "sources",
    "source_keys",
    "used_sources",
    "used_source_keys",
)

FLOAT_COLUMNS = (
    "top_score",
    "keyword_rate",
    "hit_at_k",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "refusal_acc",
    "answerability_acc",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "latency",
)

BOOLEAN_COLUMNS = (
    "generation_retried",
    "truncation_detected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="기존 체크포인트에서 RAGAS만 다시 계산"
    )

    parser.add_argument(
        "checkpoint",
        help="checkpoint_before_ragas.csv 경로",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="최종 결과 CSV 경로. 생략하면 checkpoint 이름에서 자동 생성",
    )

    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    if not value:
        return []

    return [
        item.strip()
        for item in value.split(";")
        if item.strip()
    ]


def parse_float(value: str):
    if value is None or str(value).strip() == "":
        return None

    try:
        return float(value)
    except ValueError:
        return value


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def load_checkpoint(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV 헤더가 없습니다.")

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    for row in rows:
        contexts_value = row.get("contexts", "")

        if contexts_value:
            try:
                row["contexts"] = json.loads(contexts_value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{row.get('id')}의 contexts JSON 파싱 실패"
                ) from error
        else:
            row["contexts"] = []

        for column in LIST_COLUMNS:
            row[column] = parse_list(row.get(column, ""))

        for column in FLOAT_COLUMNS:
            row[column] = parse_float(row.get(column, ""))

        for column in BOOLEAN_COLUMNS:
            row[column] = parse_bool(row.get(column, ""))

        # 이전 실패 과정에서 값이 일부 존재해도 새로 계산하도록 제거
        for column in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ):
            row[column] = None

    return rows, fieldnames


def serialize_row(row: dict, fieldnames: list[str]) -> dict:
    output = dict(row)

    output["contexts"] = json.dumps(
        row.get("contexts", []),
        ensure_ascii=False,
    )

    for column in LIST_COLUMNS:
        value = row.get(column, [])

        if isinstance(value, list):
            output[column] = "; ".join(str(item) for item in value)

    return {
        column: output.get(column, "")
        for column in fieldnames
    }


def save_results(
    path: Path,
    records: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                serialize_row(record, fieldnames)
            )


def mean_metric(records: list[dict], key: str):
    values = [
        float(row[key])
        for row in records
        if isinstance(row.get(key), (int, float))
    ]

    if not values:
        return None

    return sum(values) / len(values)


def print_ragas_summary(records: list[dict]) -> None:
    print("\n===== RAGAS 결과 =====")

    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        value = mean_metric(records, metric)

        if value is not None:
            print(f"{metric:20s}: {value:.4f}")


def make_output_path(
    checkpoint_path: Path,
    output_argument: str | None,
) -> Path:
    if output_argument:
        return Path(output_argument)

    checkpoint_suffix = ".checkpoint_before_ragas"

    if checkpoint_path.stem.endswith(checkpoint_suffix):
        final_stem = checkpoint_path.stem[
            : -len(checkpoint_suffix)
        ]
    else:
        final_stem = (
            checkpoint_path.stem
            + ".ragas_completed"
        )

    return checkpoint_path.with_name(
        final_stem + checkpoint_path.suffix
    )


def main() -> int:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {checkpoint_path}"
        )

    output_path = make_output_path(
        checkpoint_path,
        args.output,
    )

    records, fieldnames = load_checkpoint(
        checkpoint_path
    )

    print(
        f"[체크포인트 로드] {checkpoint_path}"
    )
    print(f"[문항 수] {len(records)}")
    print("[RAGAS] Upstage 요청을 순차 실행합니다.")

    compute_ragas(records)

    save_results(
        output_path,
        records,
        fieldnames,
    )

    print_ragas_summary(records)
    print(f"\n[최종 결과 저장] {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())