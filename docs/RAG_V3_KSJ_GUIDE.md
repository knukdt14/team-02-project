# Mistral 7B Hybrid RAG v3 실행·파인튜닝 가이드

## 1. 고정한 베이스라인

이번 버전도 비교 조건을 유지합니다.

| 항목 | 값 |
|---|---|
| LLM | `mistralai/Mistral-7B-Instruct-v0.3` |
| 양자화 | NF4 4비트 |
| 검색 | dense + BM25 hybrid |
| 벡터 저장소 | FAISS |
| `top_k` | 5 |
| 프롬프트 | strict |
| 실행 장치 | CUDA 필수 |

검색 조건은 바꾸지 않고, 검색된 근거를 답변으로 만드는 단계와 파인튜닝
데이터를 수정했습니다. 따라서 v2와 v3 차이는 검색 모델 교체 효과가 아닙니다.

## 2. v3에서 바꾼 네 가지

### 복합질문 템플릿

단순 질문은 결론부터 1~3문장으로 답합니다. 처벌까지 묻는 복합질문은 작업
기준·사업주의 의무·처벌로, 풍속 단계 질문은 작업 중지·사전 조치·사후
조치로 나눠 답합니다. 특정 정답 조문은 프롬프트에 미리 넣지 않습니다.

### 동적 출력 길이

- 단순 질문: 최대 160 토큰
- 복합질문: 최대 280 토큰

복합질문만 길이를 늘려 답변 중간 절단을 줄이고, 각 항목은 1~2문장으로
제한해 장황함을 막습니다.

### 인용 검증

모델이 쓴 `[C번호]`를 그대로 믿지 않습니다. 각 답변 문장과 검색 문서의
수치·조문·핵심 행위를 비교해 틀린 인용을 제거하고, 강하게 일치하는 근거로
교정합니다.

평가 CSV에는 다음 값이 따로 저장됩니다.

- `model_used_sources`: 모델이 원래 고른 출처
- `used_sources`: 내용 검증을 통과한 출처
- `citation_repaired`: 인용 교정 여부
- `model_citation_*`: 교정 전 모델 자체 인용 점수
- `citation_*`: 교정 후 시스템 최종 인용 점수

### 별표 청크 병합

FAISS 검색 결과에서 별표 제목 청크 하나만 남기지 않고, 같은 별표에서 질문과
가까운 표 행을 최대 2개 찾아 동일한 C-ID 문맥에 연결합니다. `top_k=5`는
그대로이며 벡터DB를 다시 만들 필요가 없습니다.

## 3. 파일 배치 후 최초 점검

프로젝트 루트에서 실행합니다.

```bat
conda activate SAFETY_RAG_PY311
python src\check_mistral_env.py
```

기존 `stores` 폴더가 있으면 그대로 재사용합니다. 없다면 최초 실행 때 한 번
생성합니다.

## 4. 파인튜닝 전 v3 베이스라인

검색 결과부터 확인합니다.

```bat
check_retrieval_ksj.bat
```

질문 한 개를 실행합니다.

```bat
run_mistral_ksj.bat 상시근로자 60명인 제조업 사업장은 안전관리자를 몇 명 두어야 하나요?
```

전체 평가를 실행합니다.

```bat
evaluate_mistral_ksj.bat
```

결과 파일:

```text
eval/results/mistral7b_base_hybrid_v3_ksj.csv
```

## 5. 새 파인튜닝 데이터

원본 `data/ft_dataset.json`은 비교를 위해 보존합니다. 새 파일은 다음 명령으로
다시 만들 수 있습니다.

```bat
prepare_finetune_data_ksj.bat
```

생성 파일:

```text
data/ft_dataset_v2_ksj.jsonl
```

구성:

- 단일 근거 72개
- 복합 근거 12개
- 답변 불가 10개
- 총 94개

생성기는 평가 정답 조문을 제외하고, 질문 중복·C-ID 오류·평가 누수를
`validate_ft_dataset.py`로 검사합니다. 정답은 원문 문장을 사용하며 불완전하게
잘린 긴 문장은 자동 후보에서 제외합니다.

## 6. QLoRA 학습

```bat
finetune_mistral_ksj.bat
```

기본 설정:

| 항목 | 값 |
|---|---|
| 학습 방식 | QLoRA |
| 최대 길이 | 448 |
| epoch | 2 |
| learning rate | `1e-4` |
| batch | 1 |
| gradient accumulation | 8 |
| 출력 | `models/mistral7b_ksj_adapter_v2` |

VRAM 부족이 발생하면 브라우저·다른 GPU 프로그램을 종료한 뒤 다시 실행합니다.
그래도 부족하면 아래처럼 길이만 줄입니다.

```bat
python src\finetune_mistral_qlora.py --max_length 384
```

## 7. 파인튜닝 후 같은 조건으로 평가

```bat
evaluate_mistral_qlora_ksj.bat
```

결과 파일:

```text
eval/results/mistral7b_qlora_hybrid_v3_ksj.csv
```

두 결과는 모델 경로 외 모든 조건이 동일합니다.

## 8. 반드시 비교할 항목

| 목적 | 열 |
|---|---|
| 검색 성공 | `hit_at_k`, `recall_at_k`, `mrr` |
| 모델 자체 인용 | `model_citation_precision`, `model_citation_recall` |
| 최종 인용 | `citation_precision`, `citation_recall` |
| 인용 교정 빈도 | `citation_repair_rate` |
| 답변 내용 | `answer`, `raw_answer`, `bertscore_f1`, `keyword_rate` |
| 답변 가능 판단 | `answerability_acc` |
| 속도 | `latency` |

파인튜닝 후 `BERTScore`만 오르고 `model_citation_*`가 떨어지면 좋은 개선이
아닙니다. 실제 답변을 읽고, 복합질문의 모든 항목이 있는지와 각 수치의 출처가
맞는지를 함께 확인해야 합니다.

## 9. 권장 실험 순서

1. v3 원본 Mistral 베이스라인 평가
2. 새 학습 데이터에서 임의의 단일·복합·거절 샘플 각 5개 사람 검수
3. QLoRA 학습
4. 같은 질문셋으로 QLoRA 평가
5. 두 CSV의 내용·인용·속도 비교
6. QLoRA가 실제로 나아진 경우에만 어댑터 채택

