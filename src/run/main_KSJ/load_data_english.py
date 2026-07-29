"""
1:1 영문 법령 청크 로더.

``laws_chunks_en_1to1.jsonl``은 한국어에서 먼저 만든 1,953개 청크를
그대로 번역한 파일입니다. 이 로더는 영어를 다시 분할하지 않고
JSONL 한 줄을 LangChain Document 한 개로 변환합니다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


EXPECTED_CHUNKS = 1953

LAW_NAME_EN = {
    "산업안전보건법": "Occupational Safety and Health Act",
    "산업안전보건법 시행령": (
        "Enforcement Decree of the Occupational Safety and Health Act"
    ),
    "산업안전보건기준에 관한 규칙": (
        "Rules on Occupational Safety and Health Standards"
    ),
}


def make_article_label_en(article_label: str) -> str:
    """한국어 조문·별표 식별자를 영문 화면 표기로 바꿉니다."""
    compact = re.sub(r"\s+", "", str(article_label or ""))

    annex = re.fullmatch(r"별표0*(\d+)(?:의0*(\d+))?", compact)
    if annex:
        main = str(int(annex.group(1)))
        sub = annex.group(2)
        return f"Annex {main}-{int(sub)}" if sub and int(sub) else f"Annex {main}"

    article = re.fullmatch(r"제?0*(\d+)(?:조)?(?:의0*(\d+))?", compact)
    if article:
        main = str(int(article.group(1)))
        sub = article.group(2)
        return f"Article {main}-{int(sub)}" if sub else f"Article {main}"

    return compact


def _read_jsonl(path: Path) -> list[dict]:
    """오류 위치를 알 수 있게 줄 번호를 포함하여 JSONL을 읽습니다."""
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


def load_parallel_english_chunks(path: str) -> list[Document]:
    """
    검증된 1:1 영문 청크를 재분할 없이 Document로 변환합니다.

    입력 개수가 1,953개가 아니거나 ID가 중복되면 잘못된 언어 비교를
    막기 위해 즉시 중단합니다.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"1:1 영문 청크가 없습니다: {source}\n"
            "`python src/prepare_parallel_chunks.py`와 "
            "`python src/translate_parallel_chunks.py`를 먼저 실행하세요."
        )

    rows = _read_jsonl(source)
    if len(rows) != EXPECTED_CHUNKS:
        raise ValueError(
            f"영문 청크 수가 {len(rows)}개입니다. "
            f"정확히 {EXPECTED_CHUNKS}개여야 합니다."
        )

    chunk_ids = [str(row.get("chunk_id", "")) for row in rows]
    if not all(chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk_id가 비어 있거나 중복되었습니다.")

    documents = []
    for row in rows:
        text_en = str(row.get("text_en", "")).strip()
        if not text_en:
            raise ValueError(f"빈 영어 번역: {row['chunk_id']}")

        law_name_ko = row.get("법령명", "")
        documents.append(
            Document(
                # 핵심: split_text를 호출하지 않고 번역 청크 하나를 그대로 씁니다.
                page_content=text_en,
                metadata={
                    "source": source.name,
                    "language": "en",
                    "chunk_id": row["chunk_id"],
                    "record_index": row.get("record_index", ""),
                    "chunk_index": row.get("chunk_index", ""),
                    "법령명": law_name_ko,
                    "법령종류": row.get("법령종류", ""),
                    "장": row.get("장", ""),
                    "조문번호": row.get("조문번호", ""),
                    "조문가지번호": row.get("조문가지번호", ""),
                    "조문표시": row.get("조문표시", ""),
                    "조문제목": row.get("조문제목", ""),
                    "시행일자": row.get("시행일자", ""),
                    "law_name_en": LAW_NAME_EN.get(law_name_ko, law_name_ko),
                    "article_label_en": make_article_label_en(
                        row.get("조문표시", "")
                    ),
                    # 제목은 메타데이터에 한국어로 보존되므로 검색 본문에는 쓰지 않습니다.
                    "article_title_en": "",
                    "alignment": "ko_chunk_to_en_chunk_1to1",
                },
            )
        )

    print(
        f"[load] {source.name} → {len(documents):,}개 "
        "(한국어 청크 1:1 번역, 영어 재청킹 없음)"
    )
    return documents


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="1:1 영문 청크 로드 검사")
    parser.add_argument("path")
    args = parser.parse_args()
    docs = load_parallel_english_chunks(args.path)
    print(docs[0].metadata)
    print(docs[0].page_content[:300])
