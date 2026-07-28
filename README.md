# 산업안전보건 법령 RAG 질의응답 시스템

산업안전보건 관련 법령을 근거로 현장 안전 질문에 답변하는 RAG 기반 챗봇입니다.
질문에 관련된 법령 조문을 검색해 근거로 제시하며, 근거가 없는 질문은 답변을 거부합니다.

**예시**

```
Q. 타워크레인은 순간풍속 얼마를 초과하면 운전을 멈춰야 하나요?
A. 순간풍속이 초당 15미터를 초과하는 경우 운전작업을 중지해야 합니다.
   (근거: 산업안전보건기준에 관한 규칙 제37조)
```

## 팀 구성 및 역할

| 이름 | 담당 모델 |
|---|---|
| 류인환(팀장) | EXAONE-3.5-7.8B |
| 김도경 | Qwen2.5-7B |
| 김상준 | Mistral-7B |
| 최성호 | Llama-3.1-3B |

## 시스템 구조

```
① 법령 수집
       ↓
② 전처리 · 분할
       ↓
③ 임베딩 · 저장
       ↓
④ 검색
       ↓
⑤ 답변 생성
```

## 폴더 구조

```
team-02-project/
├── data/                        # 법령 데이터
│   ├── fetch_law.py             # Open API 수집 + 조문 단위 전처리
│   └── laws_all.json            # 수집 결과
│   └── ft_dataset.json          # 파인튜닝 학습 데이터셋
├── src/
│   ├── load_data.py             # 데이터 로드 + 청킹
│   ├── build_vectorstore.py     # 임베딩 + 벡터스토어 구축
│   ├── rag_chain.py             # 검색 + LLM 연결
│   ├── evaluate.py              # 평가 지표 계산
│   └── main.py                  # 실행 진입점
├── eval/
│   ├── questions.csv            # 평가셋 14문항 (질문 + 정답)
│   ├── references.csv           # 정답 근거 조문
│   └── results/                 # 실험 결과 CSV
├── report/                      # 보고서
├── slides/                      # 발표 자료
└── requirements.txt
```

## 설치 및 실행

### 환경 요구사항

- Python 3.11+
- NVIDIA GPU (VRAM 8GB 이상 권장)
- CUDA 12.8 이상

### 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# PyTorch는 CUDA 버전으로 별도 설치 (일반 설치 시 CPU 버전이 설치됨)
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### API 키 설정

프로젝트 루트에 `.env` 파일 생성:

```
UPSTAGE_API_KEY=up_xxxxxxxxxxxx
```

### 데이터 수집

`fetch_law.py` 상단의 `OC` 변수에 [국가법령정보 공동활용](https://open.law.go.kr)에서
발급받은 인증키를 입력한 뒤 실행합니다.

```bash
python data/fetch_law.py
```

### 실행

```bash
cd src

# 평가셋 전체 실행
python main.py

# 단일 질문 테스트
python main.py --ask "타워크레인 풍속 기준은?"

# 설정 변경 예시
python main.py --embedding_name upstage --top_k 3 --score_threshold 0.24 --model_name Qwen/Qwen2.5-7B-Instruct --run_name qwen_baseline
```

**주요 인자**

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--model_name` | 사용할 LLM | Qwen/Qwen2.5-7B-Instruct |
| `--embedding_name` | 임베딩 모델 | hf |
| `--top_k` | 검색해 LLM에 전달할 chunk 수 | 3 |
| `--chunk_size` / `--overlap_size` | 문서 분할 크기 | 500 / 50 |
| `--search_type` | 검색 방식  | similarity |
| `--prompt_name` | 프롬프트 | basic |
| `--score_threshold` | 관련도 임계값 | 0.0 |
| `--load_in_4bit` | 4bit 양자화 | off |
| `--trust_remote_code` | 커스텀 코드 모델 허용 | off |
| `--run_name` | 결과 CSV 파일명 | 자동 생성 |

## 데이터

### 대상 법령 3종

| 법령 | 비고 |
|---|---|
| 산업안전보건법 | 의무·벌칙 등 상위 규범 |
| 산업안전보건법 시행령 | 선임 기준·대상 사업장 |
| 산업안전보건기준에 관한 규칙 | 현장 기술 기준 (수치 규정) |

조문 1,260개를 조문 단위로 구조화하여 1,940개 chunk 생성.
조문 가지번호(제619조의2), 장(章) 정보, 별표 구조를 메타데이터로 보존합니다.

### 평가셋 (14문항)

- 법 · 시행령 · 규칙 각 난이도 상/중/하 — 9문항
- 다중조문 종합형 — 3문항
- 답변불가형(거절이 정답) — 2문항

### 학습셋 (100문항)

- 개념/의무형 30 · 대상/조건형 30 · 현장/기술기준형 30 · 거절형 10


## 평가 방법

성능 저하의 원인이 **검색인지 생성인지 분리 진단**할 수 있도록 지표를 계층화했습니다.

| 계층 | 지표 |
|---|---|
| 검색 품질 | Hit@k · Recall@k · Precision@k · MRR |
| 답변 품질 | BERTScore(P/R/F1) · 키워드 포함률 |
| 환각 억제 | RAGAS Faithfulness · Context Precision/Recall |
| 안전성 | Citation Accuracy · Refusal Accuracy |
| 효율 | Latency |

## 참고

- [국가법령정보 공동활용](https://open.law.go.kr) — 법령 데이터 출처
