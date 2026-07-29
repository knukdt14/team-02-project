"""
한국어/영어 1:1 청크의 정렬과 번역 무결성을 검사합니다.

검사 항목:
- 양쪽 청크 수가 모두 1,953개인지
- ``chunk_id``와 원본 메타데이터가 순서까지 같은지
- 한국어 원문이 번역 파일 안에서도 동일한지
- 영어 번역이 비어 있지 않은지
- 법령의 숫자가 누락되거나 추가되지 않았는지
- 영문 본문에 번역되지 않은 한글이 남았는지

실행:
    python src/validate_parallel_chunks.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from parallel_validation_utils import (
    classify_hangul,
    missing_number_values,
    number_values,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "laws_chunks_ko_1to1.jsonl"
TARGET_PATH = PROJECT_ROOT / "data" / "laws_chunks_en_1to1.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "validation_report_parallel.json"
EXPECTED_CHUNKS = 1953

METADATA_FIELDS = (
    "record_index",
    "chunk_index",
    "법령명",
    "법령종류",
    "장",
    "조문번호",
    "조문가지번호",
    "조문표시",
    "조문제목",
    "시행일자",
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 파싱 실패: {path} {line_number}번째 줄"
                ) from error
    return rows


def _error(errors: list[dict], chunk_id: str, kind: str, detail: str) -> None:
    errors.append(
        {
            "chunk_id": chunk_id,
            "kind": kind,
            "detail": detail,
        }
    )


def validate() -> dict:
    source_rows = _read_jsonl(SOURCE_PATH)
    target_rows = _read_jsonl(TARGET_PATH)
    errors: list[dict] = []
    warnings: list[dict] = []

    if len(source_rows) != EXPECTED_CHUNKS:
        _error(
            errors,
            "__file__",
            "source_count",
            f"{len(source_rows)} != {EXPECTED_CHUNKS}",
        )
    if len(target_rows) != EXPECTED_CHUNKS:
        _error(
            errors,
            "__file__",
            "target_count",
            f"{len(target_rows)} != {EXPECTED_CHUNKS}",
        )

    if len({row.get("chunk_id") for row in source_rows}) != len(source_rows):
        _error(errors, "__file__", "duplicate_source_id", "중복 ID 존재")
    if len({row.get("chunk_id") for row in target_rows}) != len(target_rows):
        _error(errors, "__file__", "duplicate_target_id", "중복 ID 존재")

    for index, (source, target) in enumerate(zip(source_rows, target_rows)):
        source_id = str(source.get("chunk_id", ""))
        target_id = str(target.get("chunk_id", ""))
        if source_id != target_id:
            _error(
                errors,
                target_id or f"row:{index}",
                "id_or_order",
                f"{source_id} != {target_id}",
            )
            continue

        for field in METADATA_FIELDS:
            if source.get(field, "") != target.get(field, ""):
                _error(
                    errors,
                    source_id,
                    "metadata",
                    f"{field}: {source.get(field)!r} != {target.get(field)!r}",
                )

        text_ko = str(source.get("text_ko", ""))
        if text_ko != str(target.get("text_ko", "")):
            _error(errors, source_id, "korean_text_changed", "text_ko 불일치")

        text_en = str(target.get("text_en", "")).strip()
        if not text_en:
            _error(errors, source_id, "empty_translation", "text_en이 비어 있음")
            continue

        source_numbers = number_values(text_ko)
        target_numbers = number_values(text_en)
        missing = missing_number_values(text_ko, text_en)
        added = sorted(target_numbers - source_numbers)
        if missing:
            _error(errors, source_id, "missing_numbers", str(missing))
        if added:
            # ①을 1.로 번역하는 등 목록 표기 변환으로 숫자가 추가될 수 있으므로
            # 자동 실패가 아니라 사람이 확인할 경고로 분류합니다.
            warnings.append(
                {
                    "chunk_id": source_id,
                    "kind": "added_numbers_check",
                    "detail": str(added),
                }
            )

        hard_hangul, table_labels = classify_hangul(text_en)
        if hard_hangul:
            _error(
                errors,
                source_id,
                "untranslated_hangul",
                ", ".join(hard_hangul[:20]),
            )
        if table_labels:
            warnings.append(
                {
                    "chunk_id": source_id,
                    "kind": "table_label_hangul",
                    "detail": ", ".join(table_labels[:20]),
                }
            )

        # 지나치게 짧은 번역은 숫자는 맞아도 내용이 누락됐을 가능성이 큽니다.
        if len(text_en) < max(20, int(len(text_ko) * 0.25)):
            warnings.append(
                {
                    "chunk_id": source_id,
                    "kind": "suspiciously_short",
                    "detail": f"ko={len(text_ko)}, en={len(text_en)}",
                }
            )

    report = {
        "expected_chunks": EXPECTED_CHUNKS,
        "source_chunks": len(source_rows),
        "target_chunks": len(target_rows),
        "aligned": not errors,
        "error_count": len(errors),
        "error_chunk_count": len(
            {
                item["chunk_id"]
                for item in errors
                if item["chunk_id"] != "__file__"
            }
        ),
        "error_summary": dict(Counter(item["kind"] for item in errors)),
        "warning_count": len(warnings),
        "warning_chunk_count": len(
            {
                item["chunk_id"]
                for item in warnings
                if item["chunk_id"] != "__file__"
            }
        ),
        "warning_summary": dict(Counter(item["kind"] for item in warnings)),
        "errors": errors,
        "warnings": warnings,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> int:
    report = validate()
    print("===== 1:1 영문 청크 검증 =====")
    print(f"- 예상 청크: {report['expected_chunks']:,}개")
    print(f"- 한국어 청크: {report['source_chunks']:,}개")
    print(f"- 영어 청크: {report['target_chunks']:,}개")
    print(f"- 오류: {report['error_count']:,}개")
    print(f"- 오류 청크: {report['error_chunk_count']:,}개")
    print(f"- 오류 종류: {report['error_summary']}")
    print(f"- 경고: {report['warning_count']:,}개")
    print(f"- 경고 종류: {report['warning_summary']}")
    print(f"- 보고서: {REPORT_PATH}")

    if report["errors"]:
        print("\n오류 청크만 재번역:")
        print("  python src/translate_parallel_chunks.py --repair-errors")
        print("  python src/validate_parallel_chunks.py")
        return 1

    print("\n검증 통과. 다음 단계:")
    print("  python src/main_english_upstage.py --build-store")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
