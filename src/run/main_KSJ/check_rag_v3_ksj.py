"""
check_rag_v3_ksj.py
---------------------------------------------------
v3의 핵심 후처리가 외부 모델 다운로드 없이 동작하는지 빠르게 검사합니다.

검사 대상:
  - 단순/복합 질문의 동적 생성 길이
  - 15m/s 답변에 잘못 붙은 30m/s 조문의 인용 교정
  - 별표 제목 청크에 관련 표 행을 병합하는 기능
"""

from collections import Counter

from rag_chain import (
    RagChain,
    _question_plan,
    validate_and_resolve_citations,
)


class FakeDocument:
    """LangChain 설치 여부와 무관하게 후처리만 검사하는 작은 문서 객체입니다."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def _doc(text: str, article: str, title: str) -> FakeDocument:
    """반복되는 테스트용 문서 메타데이터를 한 곳에서 만듭니다."""
    return FakeDocument(
        page_content=text,
        metadata={
            "법령명": "산업안전보건기준에 관한 규칙",
            "조문표시": article,
            "조문제목": title,
        },
    )


def check_dynamic_length() -> None:
    """단순 질문보다 복합질문에 더 긴 생성 예산이 배정되는지 확인합니다."""
    single = _question_plan("타워크레인 운전 중지 풍속은 얼마인가요?")
    multi = _question_plan(
        "타워크레인을 강풍에 계속 가동하다 사망하면 기준과 처벌은 무엇인가요?"
    )
    assert single["mode"] == "single"
    assert single["max_new_tokens"] == 160
    assert multi["mode"] == "duty_and_penalty"
    assert multi["max_new_tokens"] == 280


def check_citation_repair() -> None:
    """수치가 다른 잘못된 인용 C2/C3이 실제 15m/s 근거 C1로 바뀌는지 확인합니다."""
    docs = [
        _doc(
            "순간풍속이 초당 15미터를 초과하면 타워크레인의 운전작업을 중지한다.",
            "제37조",
            "악천후 및 강풍 시 작업 중지",
        ),
        _doc(
            "순간풍속이 초당 30미터를 초과할 우려가 있으면 이탈방지 조치를 한다.",
            "제140조",
            "폭풍에 의한 이탈 방지",
        ),
        _doc(
            "순간풍속이 초당 30미터를 초과한 후 이상 유무를 점검한다.",
            "제143조",
            "폭풍 등으로 인한 이상 유무 점검",
        ),
    ]
    raw_answer = (
        "타워크레인은 순간풍속이 초당 15미터를 초과하면 운전작업을 "
        "중지해야 합니다. [C2, C3]"
    )
    _, model_sources, verified_sources, repaired = (
        validate_and_resolve_citations(raw_answer, docs)
    )
    assert model_sources != verified_sources
    assert verified_sources == ["산업안전보건기준에 관한 규칙 제37조"]
    assert repaired is True


def check_annex_merge() -> None:
    """별표 제목만 검색돼도 질문과 가까운 '1명 이상' 표 행을 보강하는지 확인합니다."""
    def annex(text: str) -> FakeDocument:
        return FakeDocument(
            page_content=text,
            metadata={
                "법령명": "산업안전보건법 시행령",
                "조문표시": "별표0003의00",
                "조문제목": "안전관리자를 두어야 하는 사업과 선임방법",
            },
        )

    chain = object.__new__(RagChain)
    chain._all_docs = [
        annex("[별표 3] 안전관리자를 두어야 하는 사업"),
        annex("상시근로자 50명 이상 500명 미만 안전관리자 수 1명 이상"),
        annex("상시근로자 500명 이상 안전관리자 수 2명 이상"),
    ]
    chain._bm25_doc_tokens = []
    chain._bm25_term_freqs = []
    chain._bm25_doc_freq = Counter()
    chain._bm25_avg_len = 0.0
    chain._prepare_bm25()

    merged = chain._merge_annex_chunks(
        "상시근로자 60명인 사업장은 안전관리자를 몇 명 두어야 하나요?",
        [chain._all_docs[0]],
    )
    assert "1명 이상" in merged[0].page_content


def main() -> int:
    check_dynamic_length()
    check_citation_repair()
    check_annex_merge()
    print("[통과] 동적 출력 길이, 인용 검증, 별표 청크 병합")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
