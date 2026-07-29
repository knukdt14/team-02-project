"""
한국어 고정 청크를 Upstage Solar로 1:1 영어 번역합니다.

핵심 원칙
---------
- 입력 ``chunk_id``와 출력 ``chunk_id``는 정확히 같아야 합니다.
- 영어 번역문을 다시 청킹하지 않습니다.
- 완료된 청크는 캐시에 즉시 저장하므로 중단 후 재실행할 수 있습니다.
- 배치 응답이 깨지면 자동으로 더 작은 배치로 나눠 재시도합니다.

실행:
    python src/translate_parallel_chunks.py

검증 오류만 다시 번역:
    python src/translate_parallel_chunks.py --repair-errors
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from parallel_validation_utils import translation_issues


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "laws_chunks_ko_1to1.jsonl"
CACHE_PATH = PROJECT_ROOT / "data" / "translation_cache_1to1.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "laws_chunks_en_1to1.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "validation_report_parallel.json"
EXPECTED_CHUNKS = 1953


SYSTEM_PROMPT = """
You are a professional legal translator.
Translate each Korean occupational safety and health statute chunk into clear,
faithful English.

Mandatory rules:
1. Return one JSON object for every input object, in the same order.
2. Copy each id exactly. Never add, omit, split, or merge an id.
3. Put only the English translation in text_en.
4. Preserve every Arabic numeral, article number, paragraph number, unit,
   formula, symbol, list structure, and legal condition.
5. Do not summarize, explain, answer, or add facts.
6. Keep legal terminology consistent.
7. Return only a valid JSON array. Do not use Markdown fences.
""".strip()

# 배치 JSON 응답이 계속 깨지는 단일 청크를 복구할 때 사용하는 프롬프트입니다.
# 이 경우 ID는 프로그램이 원본에서 직접 붙이므로 모델에는 번역문만 요청합니다.
SINGLE_TEXT_PROMPT = """
You are a professional legal translator.
Translate the supplied Korean occupational safety and health statute into
clear, faithful English.

Mandatory rules:
1. Return only the English translation as plain text.
2. Do not return JSON, Markdown fences, labels, explanations, or notes.
3. Preserve every Arabic numeral, article number, paragraph number, unit,
   formula, symbol, list structure, and legal condition.
4. Do not summarize or add facts.
""".strip()

REPAIR_PROMPT = """
You are repairing an English translation of a Korean occupational safety and
health statute chunk.

Mandatory rules:
1. Return only the corrected English translation as plain text.
2. Preserve the complete legal meaning. Never summarize or omit table rows.
3. Correct every issue listed by the validator.
4. Preserve every legal number semantically. Standard English scale conversion
   is allowed (for example, 10억원 = 1 billion won), but the value must be exact.
5. Translate Korean prose into English. A Korean table item marker may be
   rendered as a consistent Latin marker.
6. Preserve article, paragraph, annex, footnote, unit, formula, and condition
   references exactly.
7. Do not add explanations, labels, notes, JSON, or Markdown.
""".strip()


def _read_jsonl(path: Path) -> list[dict]:
    """빈 줄을 건너뛰며 UTF-8 JSONL을 읽습니다."""
    if not path.exists():
        return []
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """임시 파일로 먼저 쓴 후 교체하여 캐시 손상을 방지합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _append_cache(rows: list[dict]) -> None:
    """번역 성공분을 즉시 디스크에 기록해 작업 중단에 대비합니다."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def _cache_map() -> dict[str, dict]:
    """
    완성 출력과 캐시를 합쳐 현재 번역을 읽습니다.

    캐시 파일이 일부 유실돼도 완성된 1:1 출력이 있으면 복구 입력으로 재사용하고,
    같은 ID가 여러 번 있으면 캐시의 가장 마지막 결과를 우선합니다.
    """
    result = {}
    for path in (OUTPUT_PATH, CACHE_PATH):
        for row in _read_jsonl(path):
            chunk_id = str(row.get("chunk_id", "")).strip()
            text_en = str(row.get("text_en", "")).strip()
            if chunk_id and text_en:
                result[chunk_id] = {"chunk_id": chunk_id, "text_en": text_en}
    return result


def _extract_json_array(content: str) -> list[dict]:
    """Solar 응답에서 JSON 배열만 엄격하게 추출합니다."""
    cleaned = str(content).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # 앞뒤 설명이 붙어도 첫 배열 범위만 한 번 복구를 시도합니다.
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, list):
        raise ValueError("번역 응답이 JSON 배열이 아닙니다.")
    return parsed


def _validate_batch(source_rows: list[dict], translated: list[dict]) -> list[dict]:
    """ID·건수·빈 번역·숫자 보존을 검사하고 정규화된 캐시 행을 반환합니다."""
    expected_ids = [row["chunk_id"] for row in source_rows]
    actual_ids = [str(row.get("id", row.get("chunk_id", ""))) for row in translated]
    if actual_ids != expected_ids:
        raise ValueError(
            f"번역 ID 불일치: 예상={expected_ids}, 실제={actual_ids}"
        )

    normalized = []
    for source, target in zip(source_rows, translated):
        text_en = str(target.get("text_en", "")).strip()
        if not text_en:
            raise ValueError(f"빈 번역: {source['chunk_id']}")

        # 숫자 표기 차이는 전체 번역을 중단시키지 않고 경고만 남깁니다.
        # 실제 번역 누락 여부는 전체 완료 후 validate 단계에서 판정합니다.
        issues = translation_issues(source["text_ko"], text_en)
        missing_numbers = issues["missing_numbers"]
        if missing_numbers:
            print(
                f"[숫자 확인 경고] {source['chunk_id']}: "
                f"번역문에서 바로 확인되지 않은 값={missing_numbers}"
            )

        normalized.append(
            {
                "chunk_id": source["chunk_id"],
                "text_en": text_en,
            }
        )
    return normalized


def _translate_repair_as_plain_text(
    llm,
    row: dict,
    previous_text_en: str,
    issue_details: list[dict],
) -> list[dict]:
    """이전 번역과 정확한 실패 사유를 함께 주어 같은 오류의 반복을 막습니다."""
    from langchain_core.messages import HumanMessage, SystemMessage

    payload = {
        "chunk_id": row["chunk_id"],
        "validator_issues": issue_details,
        "korean_source": row["text_ko"],
        "previous_english_translation": previous_text_en,
    }
    response = llm.invoke(
        [
            SystemMessage(content=REPAIR_PROMPT),
            HumanMessage(
                content=(
                    "Correct the previous translation using this JSON input:\n"
                    + json.dumps(payload, ensure_ascii=False)
                )
            ),
        ]
    )
    translated = [
        {
            "id": row["chunk_id"],
            "text_en": _extract_plain_translation(response.content),
        }
    ]
    return _validate_batch([row], translated)


def _translate_once(llm, rows: list[dict]) -> list[dict]:
    """한 API 요청으로 작은 청크 묶음을 번역합니다."""
    from langchain_core.messages import HumanMessage, SystemMessage

    payload = [
        {"id": row["chunk_id"], "text_ko": row["text_ko"]}
        for row in rows
    ]
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Translate this JSON array:\n"
                    + json.dumps(payload, ensure_ascii=False)
                )
            ),
        ]
    )
    return _validate_batch(rows, _extract_json_array(response.content))


def _extract_plain_translation(content: str) -> str:
    """
    단일 청크 복구 응답에서 영어 번역문만 꺼냅니다.

    모델이 지시를 어기고 JSON을 반환한 경우도 한 번 처리하고, 그 외에는
    코드 펜스와 ``Translation:`` 같은 머리말만 제거합니다.
    """
    cleaned = str(content).strip()
    cleaned = re.sub(r"^```(?:json|text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 단일 복구에서도 정상 JSON을 반환했다면 text_en만 사용합니다.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            cleaned = str(parsed[0].get("text_en", "")).strip()
        elif isinstance(parsed, dict):
            cleaned = str(parsed.get("text_en", "")).strip()
        elif isinstance(parsed, str):
            cleaned = parsed.strip()
    except json.JSONDecodeError:
        pass

    # 흔히 붙는 불필요한 라벨만 제거하고 법령 본문은 그대로 보존합니다.
    cleaned = re.sub(
        r"^\s*(?:English\s+translation|Translation)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if not cleaned:
        raise ValueError("단일 청크 일반 텍스트 번역이 비어 있습니다.")
    return cleaned


def _translate_single_as_plain_text(llm, row: dict) -> list[dict]:
    """
    JSON 파싱이 반복 실패한 청크 하나를 일반 텍스트 형식으로 번역합니다.

    ``chunk_id``는 모델 응답을 신뢰하지 않고 원본 행에서 직접 사용하므로
    1:1 청크 대응 관계는 그대로 유지됩니다.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke(
        [
            SystemMessage(content=SINGLE_TEXT_PROMPT),
            HumanMessage(content=row["text_ko"]),
        ]
    )
    translated = [
        {
            "id": row["chunk_id"],
            "text_en": _extract_plain_translation(response.content),
        }
    ]
    return _validate_batch([row], translated)


def _translate_with_recovery(llm, rows: list[dict], retries: int) -> list[dict]:
    """재시도 후에도 실패하면 배치를 반으로 나눠 문제 청크를 격리합니다."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _translate_once(llm, rows)
        except Exception as error:  # API·형식·숫자 검증 오류를 모두 재시도합니다.
            last_error = error
            wait_seconds = min(2**attempt, 12)
            print(
                f"[재시도] {attempt}/{retries}, 청크={len(rows)}개: "
                f"{error} ({wait_seconds}초 후)"
            )
            time.sleep(wait_seconds)

    if len(rows) > 1:
        midpoint = len(rows) // 2
        print(f"[배치 분리] {len(rows)}개 → {midpoint}개/{len(rows) - midpoint}개")
        return (
            _translate_with_recovery(llm, rows[:midpoint], retries)
            + _translate_with_recovery(llm, rows[midpoint:], retries)
        )

    # 청크 하나에서도 JSON 문법 오류가 반복되면 JSON 형식을 포기합니다.
    # 모델에는 영어 본문만 요청하고 ID는 코드가 직접 붙여 작업을 이어갑니다.
    print(f"[단일 청크 복구] {rows[0]['chunk_id']}: 일반 텍스트 번역으로 전환")
    plain_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _translate_single_as_plain_text(llm, rows[0])
        except Exception as error:
            plain_error = error
            wait_seconds = min(2**attempt, 12)
            print(
                f"[단일 복구 재시도] {attempt}/{retries}: "
                f"{error} ({wait_seconds}초 후)"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"단일 청크 번역 실패: {rows[0]['chunk_id']} | "
        f"JSON 오류={last_error} | 일반 텍스트 오류={plain_error}"
    )


def _report_error_map() -> dict[str, list[dict]]:
    """검증 보고서의 실제 오류만 청크별로 묶습니다."""
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"검증 보고서가 없습니다: {REPORT_PATH}\n"
            "먼저 `python src/validate_parallel_chunks.py`를 실행하세요."
        )

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for item in report.get("errors", []):
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id == "__file__":
            continue
        grouped.setdefault(chunk_id, []).append(item)
    return grouped


def _repair_report_errors(
    llm,
    source_rows: list[dict],
    cache: dict[str, dict],
    retries: int,
) -> dict[str, dict]:
    """
    오류 청크만 이전 번역과 실패 사유를 포함해 교정하고 캐시를 원자적으로 교체합니다.

    기존 방식처럼 캐시를 먼저 지우지 않으므로 API 실패가 나도 정상 번역을 잃지
    않습니다. 최종 성공분만 새 캐시에 반영합니다.
    """
    grouped = _report_error_map()
    if not grouped:
        print("[복구] 교정할 실제 오류 청크가 없습니다.")
        return cache

    if CACHE_PATH.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = CACHE_PATH.with_name(
            f"{CACHE_PATH.stem}.backup_{timestamp}{CACHE_PATH.suffix}"
        )
        shutil.copy2(CACHE_PATH, backup)
        print(f"[복구] 캐시 백업: {backup}")

    source_map = {row["chunk_id"]: row for row in source_rows}
    repaired = 0
    failed: list[str] = []
    for number, (chunk_id, issue_details) in enumerate(grouped.items(), 1):
        source = source_map.get(chunk_id)
        previous = cache.get(chunk_id, {}).get("text_en", "")
        if source is None or not previous:
            failed.append(chunk_id)
            continue

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                fixed = _translate_repair_as_plain_text(
                    llm,
                    source,
                    previous,
                    issue_details,
                )[0]
                # 교정 후에도 같은 실제 오류가 남으면 다음 시도에 그 결과를 넣습니다.
                remaining = translation_issues(source["text_ko"], fixed["text_en"])
                hard_remaining = {
                    key: value
                    for key, value in remaining.items()
                    if key != "table_label_hangul" and value
                }
                if hard_remaining:
                    previous = fixed["text_en"]
                    issue_details = [
                        {
                            "chunk_id": chunk_id,
                            "kind": key,
                            "detail": str(value),
                        }
                        for key, value in hard_remaining.items()
                    ]
                    raise ValueError(f"교정 후 남은 오류={hard_remaining}")

                cache[chunk_id] = fixed
                repaired += 1
                break
            except Exception as error:
                last_error = error
                if attempt < retries:
                    wait_seconds = min(2**attempt, 12)
                    print(
                        f"[교정 재시도] {chunk_id} {attempt}/{retries}: "
                        f"{error} ({wait_seconds}초 후)"
                    )
                    time.sleep(wait_seconds)
        else:
            failed.append(chunk_id)
            print(f"[교정 실패] {chunk_id}: {last_error}")

        if number == 1 or number % 10 == 0 or number == len(grouped):
            print(f"  - 교정 {number}/{len(grouped)} 확인")

    _write_jsonl(CACHE_PATH, list(cache.values()))
    print(f"[복구] 교정 성공: {repaired}개 / 실패: {len(failed)}개")
    if failed:
        print(f"[복구] 실패 ID: {failed[:20]}")
    return cache


def _build_final(source_rows: list[dict], cache: dict[str, dict]) -> bool:
    """원본 순서를 유지해 한국어·영어가 같은 행에 있는 최종 JSONL을 만듭니다."""
    missing = [row["chunk_id"] for row in source_rows if row["chunk_id"] not in cache]
    if missing:
        print(f"[미완료] 번역되지 않은 청크: {len(missing):,}개")
        return False

    final_rows = []
    for source in source_rows:
        final_rows.append(
            {
                **source,
                "text_en": cache[source["chunk_id"]]["text_en"],
            }
        )
    _write_jsonl(OUTPUT_PATH, final_rows)
    print(f"[완료] 1:1 영문 청크: {len(final_rows):,}개")
    print(f"[저장] {OUTPUT_PATH}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="한국어 고정 청크 1:1 영문 번역")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="한 API 요청의 청크 수. 법령은 4개를 권장",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="배치 분리 전 재시도 횟수",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="시험 번역 개수. 0이면 남은 청크 전체",
    )
    parser.add_argument(
        "--repair-errors",
        action="store_true",
        help="검증 보고서의 실제 오류만 이전 번역과 실패 사유를 넣어 교정",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size는 1 이상이어야 합니다.")

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError as error:
        raise RuntimeError(
            "python-dotenv가 없습니다. requirements.txt를 설치하세요."
        ) from error

    source_rows = _read_jsonl(SOURCE_PATH)
    if len(source_rows) != EXPECTED_CHUNKS:
        raise ValueError(
            f"한국어 청크가 {len(source_rows)}개입니다. "
            f"먼저 전처리하여 {EXPECTED_CHUNKS}개를 만드세요."
        )

    cache = _cache_map()

    if args.repair_errors:
        from langchain_upstage import ChatUpstage

        llm = ChatUpstage(model="solar-pro", temperature=0)
        cache = _repair_report_errors(llm, source_rows, cache, args.retries)

    pending = [row for row in source_rows if row["chunk_id"] not in cache]
    if args.max_items:
        pending = pending[: args.max_items]

    print(
        f"[번역] 전체 {len(source_rows):,}개 / "
        f"완료 {len(cache):,}개 / 이번 실행 {len(pending):,}개"
    )
    if pending:
        from langchain_upstage import ChatUpstage

        # Upstage는 n=1만 지원하므로 후보 개수를 별도로 요청하지 않습니다.
        llm = ChatUpstage(model="solar-pro", temperature=0)
        batches = [
            pending[index : index + args.batch_size]
            for index in range(0, len(pending), args.batch_size)
        ]
        for batch_number, batch in enumerate(batches, 1):
            translated = _translate_with_recovery(llm, batch, args.retries)
            _append_cache(translated)
            for row in translated:
                cache[row["chunk_id"]] = row
            if batch_number == 1 or batch_number % 10 == 0 or batch_number == len(batches):
                print(f"  - {batch_number}/{len(batches)} 배치 완료")

    completed = _build_final(source_rows, cache)
    if completed:
        print("\n다음 단계:")
        print("  python src/validate_parallel_chunks.py")
        return 0

    print("\n중단 후 같은 명령을 다시 실행하면 캐시 다음부터 이어집니다.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
