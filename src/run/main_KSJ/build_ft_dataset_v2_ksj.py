"""
build_ft_dataset_v2_ksj.py
---------------------------------------------------
현재 법령 JSON에서 Mistral QLoRA용 학습 데이터 v2를 만듭니다.

설계 원칙:
  1) 평가셋의 정답 조문은 학습에서 제외해 성능 비교의 누수를 막습니다.
  2) 같은 답변을 질문만 바꿔 반복하지 않고 조문당 한 샘플만 만듭니다.
  3) 실제 RAG와 같은 [C1], [C2] 근거 ID 형식을 사용합니다.
  4) 단일 근거, 복합 근거, 답변 불가 샘플을 섞어 답변 형식을 학습합니다.
  5) 정답은 JSON 원문 안의 문장만 잘라 사용하며 법적 내용을 새로 만들지 않습니다.

실행:
    python src/build_ft_dataset_v2_ksj.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from evaluate import (
    _law_short,
    _match,
    _normalize_article,
    parse_gold_articles,
)
from rag_chain import REFUSAL_MSG


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 실제 실행 프롬프트의 핵심 규칙만 짧게 유지합니다.
# 전체 strict 프롬프트를 그대로 넣으면 한국어 토큰 수가 커져 8GB GPU 학습에서
# 정작 법령 근거가 잘리는 문제가 있으므로, 학습 목적에 필요한 규칙만 남깁니다.
TRAIN_SYSTEM = (
    "당신은 산업안전보건 법령 상담 챗봇입니다. 제공된 근거만 사용해 한국어로 "
    "간결하게 답하세요. 각 주장 끝에 실제 사용한 [C번호]를 붙이고, 근거가 없으면 "
    f"'{REFUSAL_MSG}'라고만 답하세요."
)


def _source_name(item: dict) -> str:
    """법령 JSON 항목을 사람이 읽는 '법령명 조문' 표기로 바꿉니다."""
    article = item.get("조문표시") or str(item.get("조문번호", ""))
    return f"{item.get('법령명', '')} {article}".strip()


def _item_key(item: dict) -> tuple[str, str]:
    """같은 조문의 여러 항을 한 항목으로 묶는 키입니다."""
    return (
        str(item.get("법령명", "")).strip(),
        str(item.get("조문표시") or item.get("조문번호", "")).strip(),
    )


def _compact_body(items: list[dict], max_chars: int = 300) -> str:
    """
    같은 조문의 여러 항에서 학습에 쓸 완결된 짧은 원문을 고릅니다.

    표·별표와 OCR 테두리는 자동 데이터에서 제외하며, 문장이 길면 '다.' 단위로
    자릅니다. 의미를 바꾸는 요약문을 새로 생성하지 않습니다.
    """
    parts = []
    for item in items:
        text = str(item.get("내용") or item.get("본문") or "")
        text = re.sub(r"^\[[^\]]+\]\s*", "", text)
        text = re.sub(
            r"^제\s*\d+\s*조(?:의\s*\d+)?\s*\([^)]*\)\s*",
            "",
            text,
        )
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in parts:
            parts.append(text)

    joined = " ".join(parts)
    if not joined or any(symbol in joined for symbol in ("┌", "├", "└", "│")):
        return ""

    # 첫 문장이 목록 예고로 끝나면 다음 문장까지 포함합니다.
    sentences = re.findall(r".+?다\.(?:\s|$)", joined)
    if sentences:
        excerpt = sentences[0].strip()
        if (
            any(word in excerpt for word in ("다음 각 호", "다음과 같다", "다음 각 목"))
            and len(sentences) > 1
        ):
            expanded = f"{excerpt} {sentences[1].strip()}"
            if len(expanded) <= max_chars:
                excerpt = expanded
    else:
        excerpt = joined

    # 불완전한 문장을 억지로 자르면 모델이 중간에서 답을 끊는 형식을 학습합니다.
    # 한 문장 자체가 너무 긴 조문은 자동 학습 후보에서 제외합니다.
    if len(excerpt) > max_chars:
        return ""
    return excerpt


def _load_holdout_articles(references_csv: Path) -> list[tuple[str, str]]:
    """references.csv의 모든 평가 조문을 학습 제외 목록으로 만듭니다."""
    holdout: list[tuple[str, str]] = []
    with references_csv.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            for key in parse_gold_articles(
                row.get("근거_법령", ""),
                row.get("근거_조문", ""),
            ):
                if key not in holdout:
                    holdout.append(key)
    return holdout


def _is_holdout(item: dict, holdout: list[tuple[str, str]]) -> bool:
    """한 JSON 조문이 평가 정답 조문과 일치하는지 확인합니다."""
    law = _law_short(str(item.get("법령명", "")))
    article = _normalize_article(
        str(item.get("조문표시") or item.get("조문번호", ""))
        .replace("제", "")
        .replace("조", "")
    )
    return any(_match(gold, (law, article)) for gold in holdout)


def _make_row(context: str, question: str, answer: str, sample_type: str) -> dict:
    """한 샘플을 표준 messages JSONL 구조로 만듭니다."""
    return {
        "sample_type": sample_type,
        "messages": [
            {"role": "system", "content": TRAIN_SYSTEM},
            {
                "role": "user",
                "content": f"[근거]\n{context}\n\n[질문]\n{question}",
            },
            {"role": "assistant", "content": answer},
        ],
    }


def build_dataset(
    laws_path: Path,
    references_csv: Path,
    seed: int = 42,
) -> list[dict]:
    """법령 원문과 평가 제외 목록으로 재현 가능한 학습 샘플을 만듭니다."""
    randomizer = random.Random(seed)
    items = json.loads(laws_path.read_text(encoding="utf-8-sig"))
    holdout = _load_holdout_articles(references_csv)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        article = str(item.get("조문표시") or item.get("조문번호", ""))
        if article.startswith("별표") or _is_holdout(item, holdout):
            continue
        grouped[_item_key(item)].append(item)

    # 법령별 균형을 맞춰 한 종류의 문장 스타일만 과대표집되지 않게 합니다.
    candidates_by_law: dict[str, list[tuple[list[dict], str]]] = defaultdict(list)
    for group in grouped.values():
        excerpt = _compact_body(group)
        title = str(group[0].get("조문제목", "")).strip()
        if not title or len(excerpt) < 45:
            continue
        candidates_by_law[str(group[0].get("법령명", ""))].append((group, excerpt))

    singles = []
    single_pool = []
    question_templates = [
        "{source}의 '{title}'에서 정한 핵심 내용을 알려주세요.",
        "'{title}'에 관해 해당 조문이 정한 내용을 간단히 설명해 주세요.",
        "{title}에 관한 법령상 기준은 무엇인가요?",
    ]
    for law_name in sorted(candidates_by_law):
        candidates = candidates_by_law[law_name]
        randomizer.shuffle(candidates)
        for index, (group, excerpt) in enumerate(candidates[:24]):
            item = group[0]
            source = _source_name(item)
            title = str(item.get("조문제목", "")).strip()
            question = question_templates[index % len(question_templates)].format(
                source=source,
                title=title,
            )
            context = f"[C1] {source}\n{excerpt}"
            singles.append(
                _make_row(context, question, f"{excerpt} [C1]", "single")
            )
            single_pool.append((item, excerpt))

    # 같은 법령·같은 장의 서로 다른 조문을 짝지어 복합질문 형식을 학습합니다.
    pairs = []
    by_law_chapter: dict[tuple[str, str], list[tuple[dict, str]]] = defaultdict(list)
    for item, excerpt in single_pool:
        # 두 문맥을 넣는 복합 샘플은 토큰 예산을 위해 짧은 조문만 사용합니다.
        if len(excerpt) > 180:
            continue
        by_law_chapter[
            (str(item.get("법령명", "")), str(item.get("장", "")))
        ].append((item, excerpt))
    for group in by_law_chapter.values():
        for offset in range(0, len(group) - 1, 2):
            left, right = group[offset], group[offset + 1]
            pairs.append((left, right))
    randomizer.shuffle(pairs)

    multis = []
    for (left_item, left_text), (right_item, right_text) in pairs[:12]:
        left_source = _source_name(left_item)
        right_source = _source_name(right_item)
        left_title = str(left_item.get("조문제목", "")).strip()
        right_title = str(right_item.get("조문제목", "")).strip()
        context = (
            f"[C1] {left_source}\n{left_text}\n\n"
            f"[C2] {right_source}\n{right_text}"
        )
        question = (
            f"'{left_title}' 및 '{right_title}'에서 정한 내용을 각각 간단히 알려주세요."
        )
        answer = (
            f"- [{left_title}] {left_text} [C1]\n"
            f"- [{right_title}] {right_text} [C2]"
        )
        multis.append(_make_row(context, question, answer, "multi"))

    # 정적 법령 근거로 답할 수 없는 질문을 별도 유형으로 가르칩니다.
    refusal_questions = [
        "오늘 우리 공장을 바로 가동해도 문제가 없나요?",
        "지난달 이후 법령이 또 개정됐는지 실시간으로 알려주세요.",
        "우리 회사가 내일 사고를 낼 가능성을 예측해 주세요.",
        "현장 사진 없이 이 설비가 안전한지 최종 판정해 주세요.",
        "담당 근로자의 건강 상태를 진단해 주세요.",
        "특정 회사가 실제로 법을 위반했는지 판결해 주세요.",
        "내일 지역별 바람 속도를 예보해 주세요.",
        "이 사고에서 누구에게 민사상 책임이 있는지 확정해 주세요.",
        "제공된 근거와 관계없이 벌금 액수를 추측해 주세요.",
        "법령에 없는 회사 내부 규정의 최신 내용을 알려주세요.",
    ]
    refusals = []
    for index, question in enumerate(refusal_questions):
        fallback_item, fallback_text = single_pool[index % len(single_pool)]
        fallback_context = f"[C1] {_source_name(fallback_item)}\n{fallback_text}"
        refusals.append(
            _make_row(
                fallback_context,
                question,
                REFUSAL_MSG,
                "unanswerable",
            )
        )

    rows = singles + multis + refusals
    randomizer.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="파인튜닝 데이터 v2 생성")
    parser.add_argument(
        "--laws_path",
        default=str(PROJECT_ROOT / "data" / "laws_all.json"),
    )
    parser.add_argument(
        "--references_csv",
        default=str(PROJECT_ROOT / "eval" / "references.csv"),
    )
    parser.add_argument(
        "--output_path",
        default=str(PROJECT_ROOT / "data" / "ft_dataset_v2_ksj.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_dataset(
        Path(args.laws_path),
        Path(args.references_csv),
        seed=args.seed,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = defaultdict(int)
    for row in rows:
        counts[row["sample_type"]] += 1
    print(f"[완료] {output}")
    print(f"[샘플 수] 전체={len(rows)}, 유형별={dict(counts)}")
    print("[보호] eval/references.csv의 정답 조문은 학습에서 제외했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
