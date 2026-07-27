# Mistral 7B KSJ 실험 가이드

> 이 문서는 초기 v2 기록입니다. 현재 실행·파인튜닝 순서는
> `docs/RAG_V3_KSJ_GUIDE.md`를 우선 확인하세요.

## 1. 파일을 `_ksj.py`로 복사하지 않는 이유

`main_ksj.py`, `rag_chain_ksj.py`처럼 핵심 소스를 통째로 복사하면 버그 수정이 양쪽에 따로 들어가고, 어느 파일이 최신인지 곧 알기 어려워집니다.

이 작업본은 `feat/ksj` 브랜치에서 기존 정식 소스를 수정합니다. 원본과의 비교는 Git이 담당합니다.

```bat
git branch --show-current
git diff main...feat/ksj
git status
```

개인 이름은 서로 충돌하면 안 되는 산출물에만 사용합니다.

- 평가 결과: `eval/results/mistral7b_base_hybrid_v2_ksj.csv`
- QLoRA 결과: `eval/results/mistral7b_qlora_hybrid_v2_ksj.csv`
- 학습 어댑터: `models/mistral7b_ksj_adapter/`
- 실행 파일: `run_mistral_ksj.bat`

## 2. 이번 변경에서 유지한 파일

- `data/fetch_law.py`: 변경하지 않음
- 원본 법령 JSON: 변경하지 않음
- `data/ft_dataset.json`: 자동 변경하지 않음

파인튜닝 데이터는 먼저 검사하고, 사람이 법 조문과 답변을 검수한 뒤 수정해야 합니다.

## 3. 변경·추가한 파일과 이유

| 파일 | 변경 내용 | 이유 |
|---|---|---|
| `src/main.py` | Mistral 7B, 4비트, CUDA, strict, top-k 5 기본값 | KSJ 실험을 같은 명령으로 재현 |
| `src/rag_chain.py` | NF4, CUDA, 160토큰, `[C1]`, hybrid 검색, 복합질의 분해 | 속도·검색 누락·장황함·가짜 출처 개선 |
| `src/build_vectorstore.py` | 임베딩 장치 `cpu` 옵션 | 8GB VRAM을 7B LLM에 양보 |
| `src/evaluate.py` | 정확한 별표 비교, 실제 사용 출처 채점, 과도한 거절 채점 | 기존 평가 설계 오류 수정 |
| `src/check_mistral_env.py` | CUDA/패키지 점검 | CPU 폴백이나 설치 오류를 실행 전에 발견 |
| `src/validate_ft_dataset.py` | JSONL 구조·중복·누수·조문 누락 검사 | 낮은 품질 데이터를 바로 학습하는 실수 방지 |
| `src/finetune_mistral_qlora.py` | 8GB용 QLoRA 학습 | 전체 모델 대신 작은 어댑터만 학습 |
| `requirements-mistral.txt` | bitsandbytes/PEFT 등 추가 | 팀 공통 패키지와 개인 모델 의존성 분리 |
| `*_mistral_*.bat` | 질문·베이스라인·QLoRA 평가 명령 | 긴 명령을 매번 다시 입력하지 않음 |
| `evaluate_qwen_control_ksj.bat` | 같은 설정의 Qwen 대조군 | 모델 외 조건을 고정한 공정한 비교 |

## 4. RTX 5070 Laptop 8GB에서 가능한가?

가능하지만 4비트 양자화가 필요합니다.

- FP16 7B 가중치만 대략 14GB라서 8GB GPU에 들어가지 않습니다.
- NF4 4비트는 가중치가 대략 4GB 전후이며 실행 오버헤드를 포함해도 8GB에서 추론할 가능성이 높습니다.
- 임베딩 모델은 CPU에 두어 GPU 메모리를 Mistral에 양보합니다.
- QLoRA 학습은 추론보다 메모리가 빠듯합니다. 기본값 `batch=1`, `max_length=384`, gradient checkpointing을 유지하세요.

## 5. 최초 1회 환경 설치

프로젝트 루트에서 실행합니다.

```bat
conda activate SAFETY_RAG_PY311
python -m pip install -r requirements-mistral.txt
python src\check_mistral_env.py
```

점검 마지막 줄이 `실행 가능`이어야 합니다.

## 6. 파인튜닝 전에 베이스라인부터 측정

먼저 질문 한 개를 확인합니다.

```bat
check_retrieval_ksj.bat
run_mistral_ksj.bat
```

첫 명령은 7B 모델 없이 실제 저장된 벡터DB의 검색 순위만 확인합니다.

다른 질문은 뒤에 바로 적습니다.

```bat
run_mistral_ksj.bat 크레인 작업 중 강풍이 불면 어떻게 해야 하나요?
```

그다음 전체 평가를 실행합니다.

```bat
evaluate_mistral_ksj.bat
```

결과:

```text
eval/results/mistral7b_base_hybrid_v2_ksj.csv
```

기존 결과 CSV는 수정 전 평가 코드로 계산되어 새 점수와 직접 비교하면 안 됩니다. 같은 수정 코드와 같은 실험 조건에서 Qwen 대조군을 한 번 다시 실행하세요.

```bat
evaluate_qwen_control_ksj.bat
```

대조군 결과:

```text
eval/results/qwen15_control_hybrid_v2_ksj.csv
```

## 7. 파인튜닝 데이터 검사

```bat
python src\validate_ft_dataset.py
python src\finetune_mistral_qlora.py --dry_run
```

확인할 항목:

1. 평가 질문과 완전히 겹치는 학습 질문이 없는지
2. 일반 답변에 정확한 법령명·조문이 있는지
3. 같은 답변이 지나치게 반복되지 않는지
4. 근거에 없는 내용을 정답에 넣지 않았는지
5. 답변 불가 질문의 거절 문구가 일관적인지

`ft_dataset.json`은 구조상 학습할 수 있지만, 데이터 수가 작고 반복 답변이 있어 품질 검수 없이 바로 학습하면 모델 성능이 오히려 떨어질 수 있습니다.

원본 데이터의 `(근거: 법령 제○조)` 형식은 현재 RAG의 `[C1]` 형식과 다릅니다. 학습 스크립트가 메모리에서 자동 변환하므로 원본 파일은 보존되며, 파인튜닝 모델도 실제 실행과 같은 인용 ID를 학습합니다.

## 8. QLoRA 학습

베이스라인 CSV를 보존한 다음 실행합니다.

```bat
python src\finetune_mistral_qlora.py
```

GPU 메모리 부족 오류가 나면 먼저 다른 GPU 프로그램을 종료합니다. 그래도 부족하면 아래처럼 길이를 줄입니다.

```bat
python src\finetune_mistral_qlora.py --max_length 256
```

학습 결과는 Git에 넣지 않는 `models/mistral7b_ksj_adapter/`에 저장됩니다.

## 9. 파인튜닝 후 같은 조건으로 재평가

```bat
evaluate_mistral_qlora_ksj.bat
```

비교 대상:

| 항목 | 베이스라인 | QLoRA |
|---|---|---|
| 모델 | 원본 Mistral 7B Instruct | 원본 + KSJ LoRA 어댑터 |
| 데이터/청크 | 동일 | 동일 |
| 검색/Top-k | hybrid / 5 | hybrid / 5 |
| 프롬프트 | strict | strict |
| 생성 길이 | 160 | 160 |
| 결과 파일 | `mistral7b_base_hybrid_v2_ksj.csv` | `mistral7b_qlora_hybrid_v2_ksj.csv` |

모델 효과만 보려면 모델 외 조건을 모두 동일하게 유지해야 합니다.

## 10. 결과에서 우선 볼 열

- `hit_at_k`, `recall_at_k`: 검색이 정답 조문을 찾았는지
- `citation_precision`, `citation_recall`: 실제 인용한 출처가 맞고 충분한지
- `answerability_acc`: 답할 질문과 거절할 질문을 올바르게 구분했는지
- `bertscore_f1`: 정답 문장과 의미가 비슷한지
- `latency`: 질문당 응답시간
- `answer`, `used_sources`: 점수뿐 아니라 실제 답변과 실제 사용 출처
- `raw_answer`, `citation_status`: 모델 원문과 인용 형식 실패 여부

검색 지표가 낮으면 임베딩·청크·검색을 먼저 고치고, 검색은 좋은데 답변이 나쁘면 프롬프트·모델·파인튜닝을 고칩니다.

`ANSWER_UNCITED`는 답변을 못 찾았다는 뜻이 아닙니다. 모델이 답은 생성했지만
`[C1]` 형식의 인용을 빠뜨렸다는 뜻입니다. 이전 버전처럼 이 답변을 강제로
“찾을 수 없음”으로 바꾸지 않고, 실제 답변과 인용 오류를 따로 평가합니다.

## 11. 커밋 예시

```bat
git add src requirements-mistral.txt *.bat docs\MISTRAL_KSJ_GUIDE.md .gitignore
git commit -m "feat: add Mistral 7B 4-bit baseline and QLoRA workflow"
git push -u origin feat/ksj
```

팀의 `main` 브랜치에는 테스트가 끝난 뒤 Pull Request로 합칩니다.
