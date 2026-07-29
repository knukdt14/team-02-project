"""
영문 RAG 평가 모듈.

발표 지표인 Keyword, Hit@k, Faithfulness, BERTScore F1, MRR, Latency를
문항별로 저장하고 전체 평균을 출력합니다. 검색 평가는 references.csv의
``gold_keys``와 실제 검색된 ``source_keys``를 정확히 비교합니다.
"""

from __future__ import annotations

import csv
import copy
import json
import re
from pathlib import Path


PRESENTATION_METRICS = (
    "keyword_rate",
    "hit_at_k",
    "faithfulness",
    "bertscore_f1",
    "mrr",
    "latency",
)


def _read_csv(path: str) -> list[dict]:
    """Excel에서 저장한 UTF-8 BOM CSV도 안전하게 읽습니다."""
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _answer_body(answer: str) -> str:
    """프로그램이 붙인 출처 꼬리표를 제거한 순수 생성 답변을 반환합니다."""
    return re.sub(
        r"\n*\(Sources:\s*.*?\)\s*$",
        "",
        str(answer),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


_KEYWORD_PATTERNS = (
    r"\bArticle\s+\d+(?:-\d+)?\b",
    r"\bAnnex\s+\d+(?:-\d+)?\b",
    r"\b\d+(?:\.\d+)?\s*(?:%|ppm|meters?\s+per\s+second|m/s|"
    r"centimeters?|cm|degrees?|years?|months?|workers?|people|"
    r"million\s+won|billion\s+(?:won|KRW)|won|lux)\b",
    r"\b\d+(?:\.\d+)?\b",
)


def _normalize_keyword(text: str) -> str:
    return re.sub(r"\s+", "", str(text).lower())


def extract_keywords(ground_truth: str) -> list[str]:
    """정답의 조문·숫자·단위를 자동 키워드로 추출합니다."""
    keywords = []
    for pattern in _KEYWORD_PATTERNS:
        for match in re.findall(pattern, str(ground_truth), flags=re.IGNORECASE):
            normalized = _normalize_keyword(match)
            if normalized and normalized not in keywords:
                keywords.append(normalized)
    return keywords


def keyword_rate(answer: str, keywords: list[str]) -> float | None:
    """자동 추출된 정답 키워드 중 답변에 포함된 비율입니다."""
    if not keywords:
        return None
    normalized_answer = _normalize_keyword(answer)
    hits = sum(keyword in normalized_answer for keyword in keywords)
    return round(hits / len(keywords), 4)


def load_gold_sources(path: str) -> dict[str, list[str]]:
    """references.csv의 세미콜론 구분 gold_keys를 ID별로 읽습니다."""
    result = {}
    for row in _read_csv(path):
        result[row["id"]] = [
            value.strip().lower()
            for value in row.get("gold_keys", "").split(";")
            if value.strip()
        ]
    return result


def retrieval_metrics(source_keys: list[str], gold_keys: list[str]) -> dict:
    """Hit@k와 MRR을 포함한 검색 지표를 계산합니다."""
    if not gold_keys:
        return {
            "hit_at_k": None,
            "recall_at_k": None,
            "precision_at_k": None,
            "mrr": None,
        }

    retrieved = [value.lower() for value in source_keys]
    gold = list(dict.fromkeys(value.lower() for value in gold_keys))
    found = [value for value in gold if value in retrieved]
    relevant_positions = [
        index + 1
        for index, value in enumerate(retrieved)
        if value in set(gold)
    ]
    return {
        "hit_at_k": 1.0 if found else 0.0,
        "recall_at_k": round(len(found) / len(gold), 4),
        "precision_at_k": round(
            len(relevant_positions) / len(retrieved),
            4,
        ) if retrieved else 0.0,
        "mrr": round(1 / min(relevant_positions), 4)
        if relevant_positions else 0.0,
    }


def citation_metrics(used_keys: list[str], gold_keys: list[str]) -> dict:
    """모델이 실제로 선택한 C-ID 출처가 정답 조문과 맞는지 계산합니다."""
    if not gold_keys:
        return {
            "citation_precision": None,
            "citation_recall": None,
            "citation_f1": None,
        }

    cited = list(dict.fromkeys(value.lower() for value in used_keys))
    gold = list(dict.fromkeys(value.lower() for value in gold_keys))
    if not cited:
        return {
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "citation_f1": 0.0,
        }

    matched_cited = [value for value in cited if value in set(gold)]
    matched_gold = [value for value in gold if value in set(cited)]
    precision = len(matched_cited) / len(cited)
    recall = len(matched_gold) / len(gold)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "citation_f1": round(f1, 4),
    }


def compute_bertscore(records: list[dict]) -> None:
    """영문 답변과 영문 정답의 BERTScore P/R/F1을 배치 계산합니다."""
    from bert_score import score

    precision, recall, f1 = score(
        [row["answer_body"] for row in records],
        [row["ground_truth"] for row in records],
        lang="en",
        verbose=False,
    )
    for row, p_value, r_value, f_value in zip(
        records,
        precision,
        recall,
        f1,
    ):
        row["bertscore_p"] = round(float(p_value), 4)
        row["bertscore_r"] = round(float(r_value), 4)
        row["bertscore_f1"] = round(float(f_value), 4)


def compute_ragas(records: list[dict]) -> None:
    """
    Upstage Solar를 심판으로 RAGAS 네 지표를 계산합니다.

    정상적인 답변불가 문항은 근거 컨텍스트가 없으므로 RAGAS 대상에서 제외합니다.
    반대로 답변 가능한 문항을 잘못 거부했다면 Faithfulness를 0으로 기록합니다.
    """
    eligible_indices = [
        index
        for index, row in enumerate(records)
        if row["type"] != "unanswerable" and row["contexts"]
    ]
    for row in records:
        if row["type"] != "unanswerable" and not row["contexts"]:
            row["faithfulness"] = 0.0

    if not eligible_indices:
        return

    from datasets import Dataset
    from langchain_upstage import ChatUpstage, UpstageEmbeddings
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    judge_llm = ChatUpstage(model="solar-pro", temperature=0)
    judge_embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    ragas_kwargs = {}
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        ragas_kwargs["llm"] = LangchainLLMWrapper(judge_llm)
        ragas_kwargs["embeddings"] = LangchainEmbeddingsWrapper(judge_embeddings)
    except ImportError:
        # 구버전 RAGAS는 LangChain 객체를 직접 받습니다.
        ragas_kwargs["llm"] = judge_llm
        ragas_kwargs["embeddings"] = judge_embeddings

    # RAGAS answer_relevancy의 기본 strictness=3은 한 요청에 n=3 후보를
    # 요구합니다. Upstage Chat API는 n=1만 허용하므로 별도 복사본만 1로
    # 낮춥니다. Faithfulness 등 다른 지표의 계산 방식은 바꾸지 않습니다.
    answer_relevancy_metric = copy.deepcopy(answer_relevancy)
    if not hasattr(answer_relevancy_metric, "strictness"):
        raise RuntimeError(
            "설치된 RAGAS의 answer_relevancy API가 예상과 다릅니다. "
            "requirements.txt의 RAGAS 범위를 다시 설치하세요."
        )
    answer_relevancy_metric.strictness = 1

    # API 일시 오류에 대비해 재시도하고, 동시 요청 수는 과도하지 않게 제한합니다.
    try:
        from ragas.run_config import RunConfig

        ragas_kwargs["run_config"] = RunConfig(
            timeout=180,
            max_retries=10,
            max_wait=60,
            max_workers=1,
        )
    except ImportError:
        # 구버전에서는 RAGAS 자체 기본 재시도 설정을 사용합니다.
        pass

    selected = [records[index] for index in eligible_indices]
    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in selected],
            "answer": [row["answer_body"] for row in selected],
            "contexts": [row["contexts"] for row in selected],
            "ground_truth": [row["ground_truth"] for row in selected],
        }
    )
    result = ragas_evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy_metric,
            context_precision,
            context_recall,
        ],
        # 지표 계산 실패를 NaN으로 숨기지 않고 즉시 알려 줍니다.
        # 그래야 "RAGAS를 실행했다"고 오해한 결과 CSV가 만들어지지 않습니다.
        raise_exceptions=True,
        **ragas_kwargs,
    ).to_pandas()

    metric_names = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
    for result_index, record_index in enumerate(eligible_indices):
        for metric_name in metric_names:
            if metric_name in result.columns:
                value = result.iloc[result_index][metric_name]
                records[record_index][metric_name] = round(float(value), 4)


def _is_refusal(record: dict) -> bool:
    """체인이 명시적으로 거부했는지 판별합니다."""
    return record.get("status") == "REFUSE"


def _mean(records: list[dict], key: str) -> tuple[float | None, int]:
    values = [
        float(row[key])
        for row in records
        if isinstance(row.get(key), (int, float))
    ]
    if not values:
        return None, 0
    return round(sum(values) / len(values), 4), len(values)


def _print_summary(records: list[dict]) -> None:
    """발표 지표와 보조 진단 지표의 평균·분모를 출력합니다."""
    print("\n===== 발표용 성능평가 지표 =====")
    for metric in PRESENTATION_METRICS:
        value, count = _mean(records, metric)
        if value is None:
            continue
        suffix = "초" if metric == "latency" else ""
        print(f"{metric:18s}: {value:.4f}{suffix} (n={count}/{len(records)})")

    print("\n===== 보조 진단 =====")
    for metric in (
        "recall_at_k",
        "precision_at_k",
        "citation_f1",
        "answerability_acc",
        "truncation_rate",
    ):
        value, count = _mean(records, metric)
        if value is not None:
            print(f"{metric:18s}: {value:.4f} (n={count}/{len(records)})")


def _save(records: list[dict], output_csv: str, experiment: dict) -> None:
    """실험 설정과 문항별 결과를 UTF-8 BOM CSV로 저장합니다."""
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    result_columns = [
        "id",
        "type",
        "difficulty",
        "question",
        "ground_truth",
        "answer",
        "answer_body",
        "raw_answer",
        "keywords",
        "contexts",
        "sources",
        "source_keys",
        "used_sources",
        "used_source_keys",
        "status",
        "question_mode",
        "max_new_tokens_used",
        "generation_retried",
        "truncation_detected",
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
    ]
    columns = list(experiment) + result_columns

    with open(output, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {**experiment, **record}
            row["contexts"] = json.dumps(
                record.get("contexts", []),
                ensure_ascii=False,
            )
            for list_column in (
                "sources",
                "source_keys",
                "used_sources",
                "used_source_keys",
                "keywords",
            ):
                value = record.get(list_column, [])
                row[list_column] = "; ".join(value) if isinstance(value, list) else value
            writer.writerow({column: row.get(column, "") for column in columns})

    print(f"\n[결과 저장] {output} ({len(records)}문항)")


def evaluate(
    chain,
    questions_csv: str,
    references_csv: str,
    output_csv: str,
    experiment: dict,
    use_ragas: bool = True,
) -> list[dict]:
    """평가셋 전체를 실행하고 모든 지표를 계산·저장합니다."""
    questions = _read_csv(questions_csv)
    gold_by_id = load_gold_sources(references_csv)
    records = []

    for number, row in enumerate(questions, 1):
        print(f"[평가] {number}/{len(questions)} {row['id']}")
        result = chain.ask(row["question"])
        question_type = row.get("type", "")
        gold_keys = gold_by_id.get(row["id"], [])
        answer_body = _answer_body(result["answer"])
        keywords = extract_keywords(row["ground_truth_answer"])

        record = {
            "id": row["id"],
            "type": question_type,
            "difficulty": row.get("difficulty", ""),
            "question": row["question"],
            "ground_truth": row["ground_truth_answer"],
            "answer": result["answer"],
            "answer_body": answer_body,
            "raw_answer": result.get("raw_answer", result["answer"]),
            "keywords": keywords,
            "contexts": result.get("contexts", []),
            "sources": result.get("sources", []),
            "source_keys": result.get("source_keys", []),
            "used_sources": result.get("used_sources", []),
            "used_source_keys": result.get("used_source_keys", []),
            "status": result.get("status", ""),
            "question_mode": result.get("question_mode", ""),
            "max_new_tokens_used": result.get("max_new_tokens_used", ""),
            "generation_retried": result.get("generation_retried", False),
            "truncation_detected": result.get("truncation_detected", False),
            "top_score": result.get("top_score", ""),
            "keyword_rate": keyword_rate(answer_body, keywords),
            "latency": result["latency"],
        }
        record.update(
            retrieval_metrics(record["source_keys"], gold_keys)
        )
        record.update(
            citation_metrics(record["used_source_keys"], gold_keys)
        )

        should_refuse = question_type == "unanswerable"
        refused = _is_refusal(record)
        record["refusal_acc"] = (
            1.0 if refused else 0.0
        ) if should_refuse else None
        record["answerability_acc"] = 1.0 if refused == should_refuse else 0.0
        record["truncation_rate"] = (
            1.0 if record["truncation_detected"] else 0.0
        )
        records.append(record)

    compute_bertscore(records)

    if use_ragas:
        # RAGAS API가 중간에 실패해도 14/28문항의 로컬 생성·검색 결과를
        # 잃지 않도록 먼저 체크포인트를 저장합니다.
        checkpoint_csv = str(
            Path(output_csv).with_name(
                f"{Path(output_csv).stem}.checkpoint_before_ragas.csv"
            )
        )
        _save(records, checkpoint_csv, experiment)
        try:
            # 최종 비교에서는 RAGAS 실패를 빈 값으로 숨기지 않습니다.
            compute_ragas(records)
        except Exception as error:
            raise RuntimeError(
                "RAGAS 평가가 실패했습니다. 기본 평가는 아래 체크포인트에 "
                f"보존되었습니다: {checkpoint_csv}\n원인: {error}"
            ) from error

    _save(records, output_csv, experiment)
    if use_ragas:
        # 최종 CSV 저장에 성공한 뒤에만 임시 체크포인트를 정리합니다.
        Path(checkpoint_csv).unlink(missing_ok=True)
    _print_summary(records)
    return records
