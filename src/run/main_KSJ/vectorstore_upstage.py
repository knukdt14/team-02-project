"""
Upstage Solar 임베딩 + FAISS 전용 벡터스토어 모듈.

비교 조건을 실수로 바꾸지 않도록 다른 임베딩 모델이나 저장소 선택 기능은
의도적으로 넣지 않았습니다. 문서 임베딩은 Upstage API에서 수행하고,
완성된 FAISS 인덱스는 로컬 ``stores/`` 폴더에 저장합니다.

Upstage API의 요청 토큰 제한을 넘지 않도록 문서를 작은 배치로 나누어
임베딩합니다.
"""

import json
from pathlib import Path
from typing import Sequence


EMBEDDING_MODEL = "solar-embedding-1-large"
EXPECTED_CHUNKS = 1953
ALIGNMENT = "ko_chunk_to_en_chunk_1to1"

# 한 번의 API 요청에 보낼 문서 개수
EMBEDDING_BATCH_SIZE = 10

# 비정상적으로 긴 청크를 잡기 위한 글자 수 기준
MAX_CHUNK_CHARACTERS = 12000


def get_embeddings():
    """
    Upstage 임베딩 객체를 생성합니다.

    API 키는 코드에 직접 적지 않고 프로젝트 루트의 .env 또는
    환경변수 UPSTAGE_API_KEY에서 읽습니다.
    """
    from langchain_upstage import UpstageEmbeddings

    return UpstageEmbeddings(
        model=EMBEDDING_MODEL,
    )


def _validate_documents(documents: Sequence) -> None:
    """
    벡터DB에 입력되는 문서가 1:1 영문 청크 조건을 만족하는지 확인합니다.
    """
    if len(documents) != EXPECTED_CHUNKS:
        raise ValueError(
            f"벡터DB 입력이 {len(documents)}개입니다. "
            f"1:1 비교 기준인 {EXPECTED_CHUNKS}개여야 합니다."
        )

    invalid_documents = [
        document
        for document in documents
        if document.metadata.get("alignment") != ALIGNMENT
    ]

    if invalid_documents:
        raise ValueError(
            "1:1 정렬 청크가 아닌 문서가 포함되어 있습니다. "
            f"잘못된 문서 수: {len(invalid_documents)}개"
        )


def _validate_chunk_lengths(documents: Sequence) -> None:
    """
    Upstage 임베딩 제한을 넘길 가능성이 있는 비정상적으로 긴 청크를 찾습니다.
    """
    long_documents = []

    for index, document in enumerate(documents, start=1):
        text_length = len(document.page_content)

        if text_length > MAX_CHUNK_CHARACTERS:
            long_documents.append(
                (
                    index,
                    document.metadata.get("chunk_id"),
                    text_length,
                )
            )

    if not long_documents:
        print(
            f"[검증] {MAX_CHUNK_CHARACTERS:,}자를 초과한 "
            "비정상 청크가 없습니다."
        )
        return

    print("[오류] 비정상적으로 긴 청크가 발견되었습니다.")

    for index, chunk_id, text_length in long_documents:
        print(
            f"  - 순번={index}, "
            f"chunk_id={chunk_id}, "
            f"글자 수={text_length:,}"
        )

    raise ValueError(
        f"{MAX_CHUNK_CHARACTERS:,}자를 초과한 청크가 있습니다. "
        "임베딩 전에 해당 번역 청크를 확인하세요."
    )


def _remove_incomplete_store(destination: Path) -> None:
    """
    이전 실행이 중단되면서 남은 불완전한 FAISS 파일을 삭제합니다.

    새 인덱스를 만들기 전에 index.faiss, index.pkl,
    store_manifest.json만 제거합니다.
    """
    filenames = (
        "index.faiss",
        "index.pkl",
        "store_manifest.json",
    )

    for filename in filenames:
        file_path = destination / filename

        if file_path.exists():
            file_path.unlink()
            print(f"[vectorstore] 기존 파일 삭제: {file_path}")


def build_vectorstore(
    documents,
    persist_dir: str,
    batch_size: int = EMBEDDING_BATCH_SIZE,
):
    """
    영문 법령 청크를 배치 단위로 임베딩하고 FAISS 인덱스를 저장합니다.

    전체 1,953개 문서를 한 번에 Upstage API로 보내면 요청 토큰 제한을
    초과할 수 있으므로, 작은 배치로 나누어 순차적으로 추가합니다.
    """
    from langchain_community.vectorstores import FAISS

    # 1:1 청크 개수와 정렬 정보 확인
    _validate_documents(documents)

    # 비정상적으로 긴 청크가 있는지 임베딩 전에 확인
    _validate_chunk_lengths(documents)

    if batch_size <= 0:
        raise ValueError(
            f"batch_size는 1 이상이어야 합니다. 현재 값: {batch_size}"
        )

    destination = Path(persist_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # 이전 실패 실행에서 생성된 불완전한 파일이 있으면 제거
    _remove_incomplete_store(destination)

    embeddings = get_embeddings()

    total_documents = len(documents)
    first_end = min(batch_size, total_documents)
    first_batch = documents[:first_end]

    print(
        f"[vectorstore] FAISS 생성을 시작합니다. "
        f"(전체={total_documents:,}, 배치={batch_size})"
    )

    # 첫 번째 배치로 FAISS 인덱스 생성
    try:
        store = FAISS.from_documents(
            first_batch,
            embeddings,
        )
    except Exception as error:
        raise RuntimeError(
            "첫 번째 배치 임베딩 중 오류가 발생했습니다.\n"
            f"문서 범위: 1~{first_end}\n"
            "Upstage API 키, 네트워크 또는 입력 길이를 확인하세요."
        ) from error

    print(
        f"[embedding] {first_end:,}/{total_documents:,} "
        f"({first_end / total_documents * 100:.1f}%)"
    )

    # 두 번째 배치부터 기존 FAISS 인덱스에 추가
    for start in range(first_end, total_documents, batch_size):
        end = min(start + batch_size, total_documents)
        batch = documents[start:end]

        try:
            store.add_documents(batch)

        except Exception as error:
            batch_info = []

            for offset, document in enumerate(batch, start=start + 1):
                batch_info.append(
                    {
                        "index": offset,
                        "chunk_id": document.metadata.get("chunk_id"),
                        "characters": len(document.page_content),
                    }
                )

            print("[오류] 실패한 배치의 청크 정보:")

            for item in batch_info:
                print(
                    f"  - 순번={item['index']}, "
                    f"chunk_id={item['chunk_id']}, "
                    f"글자 수={item['characters']:,}"
                )

            raise RuntimeError(
                "임베딩 중 오류가 발생했습니다.\n"
                f"문서 범위: {start + 1:,}~{end:,}\n"
                "배치 크기 또는 개별 청크 길이를 확인하세요."
            ) from error

        print(
            f"[embedding] {end:,}/{total_documents:,} "
            f"({end / total_documents * 100:.1f}%)"
        )

    # 완성된 FAISS 인덱스 저장
    store.save_local(str(destination))

    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "document_count": total_documents,
        "alignment": ALIGNMENT,
        "english_rechunk": False,
        "embedding_batch_size": batch_size,
        "max_chunk_characters": MAX_CHUNK_CHARACTERS,
    }

    manifest_path = destination / "store_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[vectorstore] FAISS 생성 완료: {destination} "
        f"(청크={total_documents:,}, "
        f"임베딩={EMBEDDING_MODEL}, "
        f"배치={batch_size})"
    )

    return store


def load_vectorstore(persist_dir: str):
    """
    이미 생성된 FAISS 인덱스를 재임베딩 없이 불러옵니다.
    """
    from langchain_community.vectorstores import FAISS

    source = Path(persist_dir)

    index_path = source / "index.faiss"
    pickle_path = source / "index.pkl"

    if not index_path.exists() or not pickle_path.exists():
        raise FileNotFoundError(
            f"FAISS 인덱스가 없습니다: {source}\n"
            "먼저 아래 명령을 실행하세요.\n"
            "python src/main_english_upstage.py --build-store"
        )

    manifest_path = source / "store_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"1:1 벡터DB 설정 파일이 없습니다: {manifest_path}\n"
            "이전 3,747개 DB를 재사용하지 말고 "
            "--build-store로 새로 만드세요."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    expected = {
        "embedding_model": EMBEDDING_MODEL,
        "document_count": EXPECTED_CHUNKS,
        "alignment": ALIGNMENT,
        "english_rechunk": False,
    }

    for key, expected_value in expected.items():
        actual_value = manifest.get(key)

        if actual_value != expected_value:
            raise ValueError(
                f"벡터DB 설정 불일치: "
                f"{key}={actual_value!r}, "
                f"예상={expected_value!r}.\n"
                "현재 저장소를 삭제하거나 새 경로에 다시 생성하세요."
            )

    store = FAISS.load_local(
        str(source),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

    print(
        f"[vectorstore] 저장된 FAISS 재사용: {source} "
        f"(청크={manifest['document_count']:,}, "
        f"임베딩={manifest['embedding_model']})"
    )

    return store