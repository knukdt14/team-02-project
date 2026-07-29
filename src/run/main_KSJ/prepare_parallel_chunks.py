"""
한국어 법령을 먼저 고정 청크로 분할하는 전처리 스크립트.

언어 비교에서 청크 수가 달라지는 문제를 막기 위해 다음 순서를 강제합니다.

1. 한국어 원문을 기존 조건(500자, overlap 50)으로 한 번만 분할합니다.
2. 각 청크에 변하지 않는 ``chunk_id``를 부여합니다.
3. 이후 영어판은 이 청크를 1:1로 번역하며 절대로 다시 분할하지 않습니다.

실행:
    python src/prepare_parallel_chunks.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    # 현재 공용 프로젝트가 사용하는 신버전 패키지를 우선 사용합니다.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # 구버전 LangChain 환경에서도 같은 분할기를 사용할 수 있게 합니다.
    from langchain.text_splitter import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "laws_all_ko.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "laws_chunks_ko_1to1.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "parallel_chunks_manifest.json"

# 현재 한국어 공용 벡터DB와 동일한 청크 조건입니다.
CHUNK_SIZE = 500
OVERLAP_SIZE = 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# 현재 제공된 3개 법령 JSON에서 위 조건으로 생성되어야 하는 고정 개수입니다.
EXPECTED_SOURCE_RECORDS = 1311
EXPECTED_CHUNKS = 1953


def _source_sha256(path: Path) -> str:
    """원본 파일이 바뀌었는지 추적할 SHA-256 값을 계산합니다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _splitter() -> RecursiveCharacterTextSplitter:
    """한국어 공용 실험과 동일한 재귀 문자 분할기를 생성합니다."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE,
        separators=SEPARATORS,
        length_function=len,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """중간 실패로 완성 파일이 손상되지 않도록 임시 파일을 거쳐 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_korean_chunks() -> list[dict]:
    """법령 레코드를 순서대로 분할하고 안정적인 청크 ID를 붙입니다."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"한국어 원본을 찾을 수 없습니다: {SOURCE_PATH}")

    records = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    if len(records) != EXPECTED_SOURCE_RECORDS:
        raise ValueError(
            "한국어 원본 레코드 수가 예상과 다릅니다: "
            f"{len(records)} != {EXPECTED_SOURCE_RECORDS}. "
            "법령 데이터가 바뀌었다면 비교 실험 전에 기준을 다시 확정하세요."
        )

    splitter = _splitter()
    rows: list[dict] = []

    for record_index, record in enumerate(records):
        source_text = str(record.get("본문") or record.get("내용") or "").strip()
        if not source_text:
            continue

        # 분할은 여기에서 딱 한 번만 수행합니다.
        chunks = splitter.split_text(source_text)
        for chunk_index, text_ko in enumerate(chunks):
            # 레코드 순서와 내부 청크 순서로 결정되므로 재실행해도 ID가 같습니다.
            chunk_id = f"law:{record_index:04d}:{chunk_index:03d}"
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "record_index": record_index,
                    "chunk_index": chunk_index,
                    "text_ko": text_ko,
                    # 검색·인용·평가에 필요한 원본 메타데이터는 그대로 보존합니다.
                    "법령명": record.get("법령명", ""),
                    "법령종류": record.get("법령종류", ""),
                    "장": record.get("장", ""),
                    "조문번호": record.get("조문번호", ""),
                    "조문가지번호": record.get("조문가지번호", ""),
                    "조문표시": record.get("조문표시", ""),
                    "조문제목": record.get("조문제목", ""),
                    "시행일자": record.get("시행일자", ""),
                }
            )

    if len(rows) != EXPECTED_CHUNKS:
        raise ValueError(
            "청크 개수가 기준과 다릅니다: "
            f"{len(rows)} != {EXPECTED_CHUNKS}. "
            "LangChain 버전 또는 원본 데이터가 달라졌을 수 있습니다."
        )
    if len({row["chunk_id"] for row in rows}) != len(rows):
        raise ValueError("중복 chunk_id가 생성되었습니다.")

    _write_jsonl(OUTPUT_PATH, rows)
    manifest = {
        "source_file": SOURCE_PATH.name,
        "source_sha256": _source_sha256(SOURCE_PATH),
        "source_records": len(records),
        "source_language": "ko",
        "chunk_size": CHUNK_SIZE,
        "overlap_size": OVERLAP_SIZE,
        "separators": SEPARATORS,
        "chunk_count": len(rows),
        "translation_alignment": "ko_chunk_to_en_chunk_1to1",
        "english_rechunk": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    rows = build_korean_chunks()
    print(f"[완료] 한국어 고정 청크: {len(rows):,}개")
    print(f"[저장] {OUTPUT_PATH}")
    print(f"[설정] {MANIFEST_PATH}")
    print("\n다음 단계:")
    print("  python src/translate_parallel_chunks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
