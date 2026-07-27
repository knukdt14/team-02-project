"""
build_ft_dataset_v3_ksj.py
---------------------------------------------------
Mistral 7B QLoRA 재실험용 학습 데이터 v3를 생성합니다.

v2 결과에서 확인된 문제:
  - 94개 중 복합질문이 12개뿐이어서 복합 답변을 충분히 학습하지 못함
  - 긴 법령 문장을 그대로 답하게 해 출력이 반복되거나 토큰 한도에서 잘림
  - 검색 문서 중 실제 필요한 근거만 선택하는 훈련이 부족함

v3 설계:
  - 전체 360개
  - 복합질문 190개(52.8%)
  - 모든 일반 답변은 짧고 완결된 한 문장 단위
  - 관련 없는 근거를 섞은 샘플로 필요한 C-ID만 선택하도록 학습
  - 2개 및 3개 근거를 모두 답하는 복합질문을 별도로 구성
  - 평가 정답 조문과 평가 질문은 학습에서 제외

실행:
    python src/build_ft_dataset_v3_ksj.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from evaluate import (
    _law_short,
    _match,
    _normalize_article,
    parse_gold_articles,
)
from rag_chain import REFUSAL_MSG


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 긴 시스템 프롬프트는 한국어 토큰을 많이 사용해 실제 법령 문맥을 잘라낼 수 있습니다.
# 학습에서는 실행 프롬프트의 핵심 행동만 짧게 가르칩니다.
TRAIN_SYSTEM = (
    "제공된 법령 근거만 사용해 간결하게 답하세요. "
    "단순질문은 한 문장, 복합질문은 요구된 모든 항목을 한 문장씩 답하고 "
    "각 문장 끝에 실제 사용한 [C번호]를 붙이세요. "
    f"근거가 없으면 '{REFUSAL_MSG}'라고만 답하세요."
)


@dataclass(frozen=True)
class Candidate:
    """학습 샘플 조합에 사용할 짧고 완결된 법령 조문 한 건입니다."""

    law_name: str
    article: str
    title: str
    chapter: str
    sentence: str

    @property
    def source(self) -> str:
        """사람이 읽을 수 있는 출처 표기를 반환합니다."""
        return f"{self.law_name} {self.article}".strip()


def _normalize(text: str) -> str:
    """질문 중복을 검사하기 위해 공백과 문장부호를 제거합니다."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()


def _load_eval_questions(path: Path) -> set[str]:
    """평가 질문을 정규화해 완전 중복을 막습니다."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {
            _normalize(row.get("question", ""))
            for row in csv.DictReader(file)
            if row.get("question")
        }


def _load_holdout_articles(path: Path) -> set[tuple[str, str]]:
    """references.csv의 모든 정답 조문을 학습 제외 목록으로 만듭니다."""
    holdout: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            holdout.update(
                parse_gold_articles(
                    row.get("근거_법령", ""),
                    row.get("근거_조문", ""),
                )
            )
    return holdout


def _source_key(item: dict) -> tuple[str, str]:
    """같은 조문의 여러 항을 하나로 묶는 키입니다."""
    return (
        str(item.get("법령명", "")).strip(),
        str(item.get("조문표시") or item.get("조문번호", "")).strip(),
    )


def _is_holdout(item: dict, holdout: set[tuple[str, str]]) -> bool:
    """JSON 항목이 평가 정답 조문인지 확인합니다."""
    article = (
        str(item.get("조문표시") or item.get("조문번호", ""))
        .replace("제", "")
        .replace("조", "")
    )
    parsed = (
        _law_short(str(item.get("법령명", ""))),
        _normalize_article(article),
    )
    return any(_match(gold, parsed) for gold in holdout)


def _first_complete_sentence(items: list[dict], max_chars: int = 110) -> str:
    """
    조문에서 첫 번째 짧고 완결된 문장을 추출합니다.

    중간 절단된 답변을 학습하지 않도록 글자 수를 넘는 문장은 잘라 쓰지 않고
    후보에서 제외합니다. 괄호가 닫히지 않은 문장과 목록 예고문도 제외합니다.
    """
    parts = []
    for item in items:
        text = re.sub(
            r"\s+",
            " ",
            str(item.get("내용") or item.get("본문") or ""),
        ).strip()
        text = re.sub(r"^\[[^\]]+\]\s*", "", text)
        if text and text not in parts:
            parts.append(text)
    joined = " ".join(parts)
    joined = re.sub(
        r"^제\s*\d+\s*조(?:의\s*\d+)?\s*\([^)]*\)\s*",
        "",
        joined,
    )

    sentences = re.findall(r"[^.?!]*?다\.(?:\s|$)", joined)
    if not sentences:
        return ""
    sentence = sentences[0].strip()
    if not 30 <= len(sentence) <= max_chars:
        return ""
    if sentence.count("(") != sentence.count(")"):
        return ""
    if any(
        phrase in sentence
        for phrase in ("다음 각 호", "다음 각 목", "다음과 같다")
    ):
        return ""
    if any(symbol in sentence for symbol in ("┌", "├", "└", "│")):
        return ""
    return sentence


def _build_candidates(
    laws_path: Path,
    holdout: set[tuple[str, str]],
) -> list[Candidate]:
    """법령 JSON을 평가 누수 없는 짧은 조문 후보로 변환합니다."""
    items = json.loads(laws_path.read_text(encoding="utf-8-sig"))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        article = str(item.get("조문표시") or item.get("조문번호", ""))
        if article.startswith("별표") or _is_holdout(item, holdout):
            continue
        grouped[_source_key(item)].append(item)

    candidates = []
    for group in grouped.values():
        sentence = _first_complete_sentence(group)
        first = group[0]
        title = str(first.get("조문제목", "")).strip()
        if not sentence or not title:
            continue
        candidates.append(
            Candidate(
                law_name=str(first.get("법령명", "")).strip(),
                article=str(
                    first.get("조문표시") or first.get("조문번호", "")
                ).strip(),
                title=title,
                chapter=str(first.get("장", "")).strip(),
                sentence=sentence,
            )
        )
    return candidates


def _row(
    sample_type: str,
    question: str,
    contexts: list[tuple[str, Candidate]],
    answer: str,
    is_multi: bool,
) -> dict:
    """하나의 학습 샘플을 실제 RAG와 같은 messages 형식으로 만듭니다."""
    context_text = "\n\n".join(
        f"[{citation_id}] {candidate.source}\n{candidate.sentence}"
        for citation_id, candidate in contexts
    )
    return {
        "dataset_version": "v3",
        "sample_type": sample_type,
        "is_multi": is_multi,
        "messages": [
            {"role": "system", "content": TRAIN_SYSTEM},
            {
                "role": "user",
                "content": f"[근거]\n{context_text}\n\n[질문]\n{question}",
            },
            {"role": "assistant", "content": answer},
        ],
    }


def _balanced_single_candidates(
    candidates: list[Candidate],
    randomizer: random.Random,
) -> list[Candidate]:
    """
    단일질문이 한 법령 유형에만 치우치지 않도록 균형 있게 100개를 고릅니다.
    후보가 상대적으로 적은 유형을 먼저 확보한 뒤 나머지를 채웁니다.
    """
    by_law: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_law[candidate.law_name].append(candidate)
    for values in by_law.values():
        randomizer.shuffle(values)

    targets = {
        "산업안전보건법": 35,
        "산업안전보건법 시행령": 25,
        "산업안전보건기준에 관한 규칙": 40,
    }
    selected = []
    used = set()
    for law_name, target in targets.items():
        for candidate in by_law.get(law_name, [])[:target]:
            selected.append(candidate)
            used.add(candidate.source)

    # 특정 유형의 짧은 문장이 부족하면 다른 후보로 전체 100개를 맞춥니다.
    remainder = [c for c in candidates if c.source not in used]
    randomizer.shuffle(remainder)
    selected.extend(remainder[: max(0, 100 - len(selected))])
    return selected[:100]


def _single_rows(
    candidates: list[Candidate],
    eval_questions: set[str],
    used_questions: set[str],
) -> list[dict]:
    """짧은 한 문장 답변을 학습하는 단일질문 100개를 만듭니다."""
    templates = [
        "{title}에 관해 법령에서 정한 핵심은 무엇인가요?",
        "{source}의 {title}에서 정한 내용을 간단히 알려주세요.",
        "{title}에 관한 법령상 조치를 한 문장으로 설명해 주세요.",
        "{title} 조문의 핵심 내용을 알려주세요.",
    ]
    rows = []
    for index, candidate in enumerate(candidates):
        question = templates[index % len(templates)].format(
            source=candidate.source,
            title=candidate.title,
        )
        normalized = _normalize(question)
        if normalized in eval_questions or normalized in used_questions:
            continue
        used_questions.add(normalized)
        rows.append(
            _row(
                "single",
                question,
                [("C1", candidate)],
                f"{candidate.sentence} [C1]",
                is_multi=False,
            )
        )
    return rows


def _distractor_rows(
    candidates: list[Candidate],
    randomizer: random.Random,
    eval_questions: set[str],
    used_questions: set[str],
) -> list[dict]:
    """
    관련 근거 하나와 비관련 근거 하나를 함께 주고 필요한 C-ID만 고르게 합니다.
    모든 검색 결과를 출처로 쓰는 습관을 줄이기 위한 40개 샘플입니다.
    """
    rows = []
    shuffled = list(candidates)
    randomizer.shuffle(shuffled)
    for index, relevant in enumerate(shuffled):
        distractor = next(
            (
                candidate
                for candidate in shuffled[index + 1 :] + shuffled[:index]
                if candidate.law_name != relevant.law_name
                and candidate.title != relevant.title
            ),
            None,
        )
        if distractor is None:
            continue
        question = (
            f"{relevant.title}에 관해 직접 관련된 근거만 사용해 답해 주세요."
        )
        normalized = _normalize(question)
        if normalized in eval_questions or normalized in used_questions:
            continue
        used_questions.add(normalized)

        # 관련 근거가 항상 C1에 있지 않게 번갈아 배치합니다.
        if len(rows) % 2 == 0:
            contexts = [("C1", relevant), ("C2", distractor)]
            cite = "C1"
        else:
            contexts = [("C1", distractor), ("C2", relevant)]
            cite = "C2"
        rows.append(
            _row(
                "single_with_distractor",
                question,
                contexts,
                f"{relevant.sentence} [{cite}]",
                is_multi=False,
            )
        )
        if len(rows) >= 40:
            break
    return rows


def _related_combinations(
    candidates: list[Candidate],
    size: int,
    max_sentence_chars: int,
    randomizer: random.Random,
) -> list[tuple[Candidate, ...]]:
    """같은 법령·같은 장 안에서 의미상 가까운 조문 조합을 만듭니다."""
    by_section: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if len(candidate.sentence) <= max_sentence_chars:
            by_section[(candidate.law_name, candidate.chapter)].append(candidate)

    combinations = []
    for section_candidates in by_section.values():
        if len(section_candidates) < size:
            continue
        # 같은 장 안의 조합을 만들되, 한 장이 전체 데이터를 독점하지 않도록
        # 장별 최대 200개까지만 후보로 사용합니다.
        section_candidates = sorted(
            section_candidates,
            key=lambda candidate: candidate.article,
        )
        section_count = 0
        for group in itertools.combinations(section_candidates, size):
            if len({candidate.source for candidate in group}) != size:
                continue
            combinations.append(tuple(group))
            section_count += 1
            if section_count >= 200:
                break
    randomizer.shuffle(combinations)
    return combinations


def _multi_pair_rows(
    candidates: list[Candidate],
    randomizer: random.Random,
    eval_questions: set[str],
    used_questions: set[str],
) -> list[dict]:
    """두 근거를 빠짐없이 답하는 복합질문 130개를 만듭니다."""
    combinations = _related_combinations(
        candidates,
        size=2,
        max_sentence_chars=80,
        randomizer=randomizer,
    )
    rows = []
    for left, right in combinations:
        question = (
            f"{left.title} 및 {right.title}에서 정한 내용을 두 항목으로 각각 알려주세요."
        )
        normalized = _normalize(question)
        if normalized in eval_questions or normalized in used_questions:
            continue
        used_questions.add(normalized)
        answer = (
            f"- [{left.title}] {left.sentence} [C1]\n"
            f"- [{right.title}] {right.sentence} [C2]"
        )
        rows.append(
            _row(
                "multi_pair",
                question,
                [("C1", left), ("C2", right)],
                answer,
                is_multi=True,
            )
        )
        if len(rows) >= 130:
            break
    return rows


def _multi_triple_rows(
    candidates: list[Candidate],
    randomizer: random.Random,
    eval_questions: set[str],
    used_questions: set[str],
) -> list[dict]:
    """세 요구사항을 끝까지 답하는 형식을 학습하는 복합질문 60개를 만듭니다."""
    combinations = _related_combinations(
        candidates,
        size=3,
        max_sentence_chars=60,
        randomizer=randomizer,
    )
    rows = []
    for first, second, third in combinations:
        question = (
            f"{first.title}, {second.title}, {third.title}의 핵심을 "
            "세 항목으로 빠짐없이 설명해 주세요."
        )
        normalized = _normalize(question)
        if normalized in eval_questions or normalized in used_questions:
            continue
        used_questions.add(normalized)
        answer = (
            f"- [{first.title}] {first.sentence} [C1]\n"
            f"- [{second.title}] {second.sentence} [C2]\n"
            f"- [{third.title}] {third.sentence} [C3]"
        )
        rows.append(
            _row(
                "multi_triple",
                question,
                [("C1", first), ("C2", second), ("C3", third)],
                answer,
                is_multi=True,
            )
        )
        if len(rows) >= 60:
            break
    return rows


def _refusal_rows(
    candidates: list[Candidate],
    randomizer: random.Random,
    used_questions: set[str],
) -> list[dict]:
    """정적 법령 근거만으로 판단할 수 없는 질문 30개를 만듭니다."""
    questions = [
        "오늘 우리 공장을 바로 가동해도 되는지 최종 판단해 주세요.",
        "지난달 이후 법령이 실시간으로 개정됐는지 알려주세요.",
        "내일 이 사업장에서 사고가 발생할 확률을 계산해 주세요.",
        "현장 확인 없이 이 설비가 안전한지 인증해 주세요.",
        "근로자의 증상만 듣고 직업병 여부를 진단해 주세요.",
        "특정 회사가 실제로 법을 위반했는지 판결해 주세요.",
        "내일 시간대별 풍속을 예보해 주세요.",
        "사고 당사자의 민사상 책임 비율을 확정해 주세요.",
        "근거에 없는 벌금 액수를 추측해 주세요.",
        "회사의 비공개 내부 규정을 찾아 알려주세요.",
        "현재 감독기관의 조사 결과를 실시간으로 알려주세요.",
        "이 사진 없이 보호구 착용 상태가 적절한지 판정해 주세요.",
        "사고가 발생하지 않을 것이라고 보장해 주세요.",
        "담당자의 형사책임이 확정됐는지 알려주세요.",
        "우리 회사 보험에서 보상받을 금액을 계산해 주세요.",
        "다음 주 현장 점검에서 적발될 항목을 예측해 주세요.",
        "근로자의 치료 방법과 약을 처방해 주세요.",
        "법원에서 선고할 정확한 형량을 미리 알려주세요.",
        "지금 현장 기계의 고장 여부를 원격으로 확정해 주세요.",
        "제공되지 않은 회사 취업규칙 내용을 알려주세요.",
        "이번 사고의 과실 비율을 숫자로 확정해 주세요.",
        "실시간 산소농도를 측정해 알려주세요.",
        "오늘 신고된 산업재해 건수를 실시간으로 알려주세요.",
        "근로계약서 없이 임금 체불 여부를 확정해 주세요.",
        "현장 조건 없이 작업 재개 시간을 결정해 주세요.",
        "특정 근로자의 건강검진 결과를 조회해 주세요.",
        "다음 달 법 개정 내용을 미리 알려주세요.",
        "회사 관계자의 진술이 거짓인지 판단해 주세요.",
        "사고 원인을 현장 조사 없이 하나로 확정해 주세요.",
        "법령 근거 없이 회사에 유리한 답을 만들어 주세요.",
    ]
    context_candidates = list(candidates)
    randomizer.shuffle(context_candidates)
    rows = []
    for index, question in enumerate(questions):
        normalized = _normalize(question)
        if normalized in used_questions:
            continue
        used_questions.add(normalized)
        context = context_candidates[index % len(context_candidates)]
        rows.append(
            _row(
                "unanswerable",
                question,
                [("C1", context)],
                REFUSAL_MSG,
                is_multi=False,
            )
        )
    return rows


def build_dataset(
    laws_path: Path,
    questions_csv: Path,
    references_csv: Path,
    seed: int,
) -> list[dict]:
    """모든 유형을 합쳐 360개 학습 데이터를 생성합니다."""
    randomizer = random.Random(seed)
    holdout = _load_holdout_articles(references_csv)
    eval_questions = _load_eval_questions(questions_csv)
    candidates = _build_candidates(laws_path, holdout)
    if len(candidates) < 250:
        raise RuntimeError(
            f"짧고 완결된 조문 후보가 부족합니다: {len(candidates)}개"
        )

    used_questions: set[str] = set()
    single_candidates = _balanced_single_candidates(candidates, randomizer)
    rows = []
    rows.extend(
        _single_rows(single_candidates, eval_questions, used_questions)
    )
    rows.extend(
        _distractor_rows(
            candidates,
            randomizer,
            eval_questions,
            used_questions,
        )
    )
    rows.extend(
        _multi_pair_rows(
            candidates,
            randomizer,
            eval_questions,
            used_questions,
        )
    )
    rows.extend(
        _multi_triple_rows(
            candidates,
            randomizer,
            eval_questions,
            used_questions,
        )
    )
    rows.extend(_refusal_rows(candidates, randomizer, used_questions))

    expected = {
        "single": 100,
        "single_with_distractor": 40,
        "multi_pair": 130,
        "multi_triple": 60,
        "unanswerable": 30,
    }
    counts = Counter(row["sample_type"] for row in rows)
    if counts != expected:
        raise RuntimeError(
            f"목표 유형 수를 만들지 못했습니다. 실제={dict(counts)}, 목표={expected}"
        )

    randomizer.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="파인튜닝 데이터 v3 생성")
    parser.add_argument(
        "--laws_path",
        default=str(PROJECT_ROOT / "data" / "laws_all.json"),
    )
    parser.add_argument(
        "--questions_csv",
        default=str(PROJECT_ROOT / "eval" / "questions.csv"),
    )
    parser.add_argument(
        "--references_csv",
        default=str(PROJECT_ROOT / "eval" / "references.csv"),
    )
    parser.add_argument(
        "--output_path",
        default=str(PROJECT_ROOT / "data" / "ft_dataset_v3_ksj.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_dataset(
        Path(args.laws_path),
        Path(args.questions_csv),
        Path(args.references_csv),
        seed=args.seed,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(row["sample_type"] for row in rows)
    multi_count = sum(row["is_multi"] for row in rows)
    multi_ratio = multi_count / len(rows)
    print(f"[완료] {output}")
    print(f"[전체] {len(rows)}개")
    print(f"[유형] {dict(counts)}")
    print(f"[복합질문] {multi_count}개 ({multi_ratio:.1%})")
    print("[보호] 평가 질문과 평가 정답 조문은 학습에서 제외했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
