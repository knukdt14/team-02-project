# 임베딩 모델 비교 실험

## 목적

최종 RAG 설정은 그대로 유지하고 임베딩 모델만 교체하여 검색 성능 차이를
확인합니다.

| 항목 | 기준 실험 | 비교 실험 |
|---|---|---|
| LLM | Mistral-7B-Instruct-v0.3 | 동일 |
| 프롬프트 | strict | 동일 |
| 검색 | hybrid | 동일 |
| 벡터DB | FAISS | 동일 |
| top_k | 5 | 동일 |
| chunk/overlap | 500/50 | 동일 |
| 임베딩 | jhgan/ko-sbert-nli | BAAI/bge-m3 |

## 실행 순서

프로젝트 최상위 폴더에서 실행합니다.

```bat
conda activate SAFETY_RAG_PY311
evaluate_embedding_bge_m3_ksj.bat
compare_embedding_ksj.bat
```

첫 실행에서는 BGE-M3를 다운로드하고 다음 전용 FAISS 폴더를 만듭니다.

```text
stores/laws_all_c500_o50_faiss_hf_bge-m3
```

기존 임베딩의 FAISS는 그대로 유지되며 다시 만들지 않습니다.

## 결과 파일

```text
eval/results/mistral7b_base_hybrid_v3_bge_m3_ksj.csv
```

기준 결과:

```text
eval/results/mistral7b_base_hybrid_v3_ksj.csv
```

## 판단 기준

임베딩 모델 비교에서는 생성 점수보다 아래 검색 지표를 먼저 봅니다.

- `hit_at_k`: 정답 조문을 하나 이상 찾았는지
- `recall_at_k`: 필요한 정답 조문을 얼마나 찾았는지
- `mrr`: 첫 정답 조문이 검색 순위의 얼마나 앞에 있는지
- `precision_at_k`: 검색 결과 중 정답 조문의 비율
- `latency`: 전체 응답시간

현재 평가 질문 수가 적기 때문에 점수가 같거나 차이가 작다면 바로 최종 모델을
교체하지 말고 평가 질문을 늘려 다시 확인해야 합니다.

## 주의

- 8GB GPU는 Mistral 7B에 사용하므로 임베딩 장치는 CPU로 유지합니다.
- 임베딩 모델이 달라지면 기존 FAISS를 재사용하면 안 됩니다.
- 코드가 모델명을 벡터DB 경로에 포함하므로 두 임베딩 결과는 서로 섞이지 않습니다.
