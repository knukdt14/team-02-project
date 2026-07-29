"""
validate_ft_dataset.py
---------------------------------------------------
파인튜닝 전에 data/ft_dataset_v3_ksj.jsonl의 구조와 품질 위험을 검사합니다.

중요:
  - 한 줄이 하나의 {"messages": [...]} 학습 샘플입니다.
  - 원본 파일을 자동 수정하지 않고 문제를 보고만 합니다.

실행:
    python src/validate_ft_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from evaluate import _match, _parse_source, load_references


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize(text: str) -> str:
    """공백·문장부호 차이를 줄여 질문 중복과 평가셋 누수를 비교합니다."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()


def _question_from_user(content: str) -> str:
    """사용자 메시지에서 [질문] 뒤의 실제 질문만 추출합니다."""
    marker = "[질문]"
    if marker in content:
        return content.split(marker, 1)[1].strip()
    return content.strip()


def load_json_or_jsonl(path: Path) -> list[dict]:
    """JSON 배열과 JSONL을 모두 읽어 학습 샘플 리스트로 반환합니다."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{line_no}번째 줄 JSON 오류: {exc}") from exc
    return rows


def load_eval_questions(path: Path) -> set[str]:
    """평가 질문을 읽어 정규화된 질문 집합으로 반환합니다."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {
            _normalize(row.get("question", ""))
            for row in csv.DictReader(file)
            if row.get("question")
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="파인튜닝 JSON/JSONL 품질 검사")
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_ROOT / "data" / "ft_dataset_v3_ksj.jsonl"),
    )
    parser.add_argument(
        "--questions_csv",
        default=str(PROJECT_ROOT / "eval" / "questions.csv"),
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=300,
        help="학습 데이터 최소 개수",
    )
    parser.add_argument(
        "--min_multi_ratio",
        type=float,
        default=0.40,
        help="복합질문 최소 비율(0~1)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    rows = load_json_or_jsonl(data_path)
    eval_questions = load_eval_questions(Path(args.questions_csv))

    structural_errors = []
    questions = []
    answers = []
    citation_errors = []
    holdout_article_overlap = []
    exact_eval_overlap = []
    type_counts = Counter()
    refusal_count = 0
    multi_count = 0
    incomplete_answers = []
    multi_citation_errors = []
    user_lengths = []
    answer_lengths = []

    # 평가 질문뿐 아니라 정답 조문 자체도 학습 문맥에 들어가지 않았는지 검사합니다.
    references_path = Path(args.questions_csv).parent / "references.csv"
    holdout_by_id = load_references(references_path)
    holdout_articles = {
        article for articles in holdout_by_id.values() for article in articles
    }

    for index, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            structural_errors.append(
                f"{index}행: messages는 system/user/assistant 3개여야 합니다."
            )
            continue

        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"]:
            structural_errors.append(
                f"{index}행: role 순서가 {roles}입니다."
            )
            continue

        user_text = str(messages[1].get("content", ""))
        answer = str(messages[2].get("content", ""))
        question = _question_from_user(user_text)
        questions.append(question)
        answers.append(answer)
        sample_type = str(row.get("sample_type", "")).strip() or "unknown"
        type_counts[sample_type] += 1
        is_multi_sample = bool(row.get("is_multi", False)) or sample_type.startswith(
            "multi"
        )
        if is_multi_sample:
            multi_count += 1
        user_lengths.append(len(user_text))
        answer_lengths.append(len(answer))

        # 답변의 C-ID가 실제 사용자 문맥에 존재하는지 검사합니다.
        is_refusal = "찾을 수 없습니다" in answer or "답변" in answer and "없" in answer
        context_ids = set(re.findall(r"(?m)^\[(C\d+)\]\s+", user_text))
        answer_ids = set(re.findall(r"\[(C\d+)\]", answer))
        if is_refusal:
            refusal_count += 1
            if answer_ids:
                citation_errors.append(f"{index}행: 거절 답변에 인용 ID가 있습니다.")
        elif not answer_ids:
            citation_errors.append(f"{index}행: 일반 답변에 인용 ID가 없습니다.")
        elif not answer_ids.issubset(context_ids):
            citation_errors.append(
                f"{index}행: 답변 인용 {sorted(answer_ids)} 중 문맥에 없는 ID가 있습니다."
            )

        # 복합질문은 제공된 모든 근거를 빠짐없이 한 번 이상 사용해야 합니다.
        if is_multi_sample and answer_ids != context_ids:
            multi_citation_errors.append(
                f"{index}행: 복합질문 문맥 ID={sorted(context_ids)}, "
                f"답변 ID={sorted(answer_ids)}"
            )

        # 각 일반 답변 줄이 문장 중간에서 잘리지 않고 '다.'로 끝나는지 검사합니다.
        if not is_refusal:
            for line in answer.splitlines():
                clean_line = line.strip()
                if not clean_line:
                    continue
                clean_line = re.sub(r"\s*\[C\d+\]\s*$", "", clean_line).strip()
                if not clean_line.endswith("다."):
                    incomplete_answers.append(
                        f"{index}행: {clean_line[-80:]}"
                    )

        # [C1] 뒤 출처 표기를 읽어 평가 정답 조문과 겹치는지 확인합니다.
        for source in re.findall(r"(?m)^\[C\d+\]\s+([^\n]+)", user_text):
            parsed = _parse_source(source.strip())
            if any(_match(gold, parsed) for gold in holdout_articles):
                holdout_article_overlap.append((index, source.strip()))

        if _normalize(question) in eval_questions:
            exact_eval_overlap.append(index)

    question_counts = Counter(_normalize(q) for q in questions)
    answer_counts = Counter(a.strip() for a in answers)
    duplicate_questions = sum(count - 1 for count in question_counts.values() if count > 1)
    repeated_answers = [(answer, count) for answer, count in answer_counts.items() if count > 1]
    repeated_answers.sort(key=lambda item: item[1], reverse=True)
    multi_ratio = multi_count / len(rows) if rows else 0.0

    print("=" * 68)
    print("[파인튜닝 데이터 점검 결과]")
    print("=" * 68)
    print(f"파일: {data_path}")
    print(f"전체 샘플: {len(rows)}")
    print(f"고유 질문: {len(question_counts)}")
    print(f"고유 답변: {len(answer_counts)}")
    print(f"유형 분포: {dict(type_counts)}")
    print(f"복합질문: {multi_count}개 ({multi_ratio:.1%})")
    print(f"거절 샘플: {refusal_count}")
    print(f"완전 중복 질문 수: {duplicate_questions}")
    print(f"구조 오류 수: {len(structural_errors)}")
    print(f"인용 ID 오류 수: {len(citation_errors)}")
    print(f"복합질문 근거 누락 수: {len(multi_citation_errors)}")
    print(f"불완전한 답변 문장 수: {len(incomplete_answers)}")
    print(f"평가 질문과 완전히 겹치는 샘플: {len(exact_eval_overlap)}")
    print(f"평가 정답 조문이 문맥에 포함된 샘플: {len(holdout_article_overlap)}")
    if user_lengths and answer_lengths:
        print(
            "최대 글자 수: "
            f"user={max(user_lengths)}, assistant={max(answer_lengths)}"
        )

    if structural_errors:
        print("\n[구조 오류 예시]")
        for error in structural_errors[:10]:
            print("-", error)

    if citation_errors:
        print("\n[인용 형식 오류]")
        for error in citation_errors[:20]:
            print("-", error)

    if multi_citation_errors:
        print("\n[복합질문 근거 누락]")
        for error in multi_citation_errors[:20]:
            print("-", error)

    if incomplete_answers:
        print("\n[불완전한 답변 문장]")
        for error in incomplete_answers[:20]:
            print("-", error)

    if exact_eval_overlap:
        print("\n[평가셋 누수 위험 행]")
        print(", ".join(map(str, exact_eval_overlap)))

    if holdout_article_overlap:
        print("\n[평가 정답 조문 누수 위험]")
        for index, source in holdout_article_overlap[:20]:
            print(f"- {index}행: {source}")

    if repeated_answers:
        print("\n[반복 답변 상위 5개]")
        for answer, count in repeated_answers[:5]:
            preview = answer.replace("\n", " ")[:90]
            print(f"- {count}회: {preview}")

    print("\n판정:")
    fatal = (
        structural_errors
        or citation_errors
        or multi_citation_errors
        or incomplete_answers
        or exact_eval_overlap
        or holdout_article_overlap
        or duplicate_questions
        or len(rows) < args.min_samples
        or multi_ratio < args.min_multi_ratio
    )
    if fatal:
        print(
            "- 구조·인용·중복·평가 누수·데이터 개수·복합질문 비율을 "
            "고친 뒤 학습하세요."
        )
        return 1
    print(
        "- 구조, 인용 ID, 복합질문 비율, 문장 완결성, 질문 중복, "
        "평가셋 누수 검사를 통과했습니다."
    )
    print("- 자동 검사를 통과해도 학습 전 일부 샘플의 문장 자연스러움은 사람이 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
