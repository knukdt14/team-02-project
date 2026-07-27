"""
train_llama_lora.py
---------------------------------------------------
실험 5 — Llama-3.2-3B QLoRA 파인튜닝.

[배경]
전처리·검색·시스템 설계만으로 여기까지 왔다.
  Hit@k 0.5 → 0.8333 / keyword_rate 0.2881 → 0.4880 / refusal_acc 0.0 → 1.0
이제 남은 것은 '모델 자체'를 이 도메인에 맞추는 것이다.

[가설]
RAG에서 파인튜닝은 '지식을 넣는 것'이 아니라 '형식과 태도를 가르치는 것'이다.
지식은 검색이 담당한다. 따라서 다음과 같이 나뉠 것으로 예상한다.

  Hit@k        거의 변화 없음   (검색을 건드리지 않으므로)
  citation_acc 상승            (인용 형식을 학습)
  keyword_rate 상승            (답변 스타일 정렬)
  한자 혼입     감소            (한국어 출력 교정)
  multi 유형    개선?           (실험 4에서 유일하게 하락한 항목)

이 '무엇이 오르고 무엇이 안 오르는가'를 데이터로 보여주는 것이 목표다.

[구성]  8GB VRAM 대응
  - 베이스 모델을 4bit 로 올리고(QLoRA), LoRA 어댑터만 학습한다
  - gradient checkpointing 으로 활성값 메모리를 줄인다
  - batch 1 + 누적 8 (실질 batch 8)
  - trl 없이 peft + transformers.Trainer 만 사용
    (trl 은 최신 transformers 를 요구해 환경을 깨뜨렸던 이력이 있음)

[사용법]
    cd src

    # 1) 스모크 테스트 - 1스텝만 돌려 OOM 여부 확인 (2~3분)
    python train_llama_lora.py --smoke

    # 2) 본 학습
    python train_llama_lora.py

산출물: ../models/llama3.2-3b-lora   (.gitignore 대상이라 커밋되지 않음)

[학습 후 평가]
    rag_chain.py 가 LoRA 어댑터 폴더를 자동 감지하도록 되어 있으므로
    (팀원이 추가한 기능) model_name 에 어댑터 경로를 주면 그대로 평가된다.

    python main.py --model_name ../models/llama3.2-3b-lora \
        --embedding_model BAAI/bge-m3 --chunk_size 1000 --top_k 3 \
        --prompt_name strict --score_threshold 0.45 \
        --load_in_4bit --temperature 0 --run_name llama3.2-3b_ft_v1
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    from dotenv import load_dotenv
    load_dotenv()          # HF_TOKEN
except ImportError:
    pass


# =====================================================================
# 1) 설정
# =====================================================================
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH = "../data/ft_dataset.json"        # chat 형식 JSONL
OUTPUT_DIR = "../models/llama3.2-3b-lora"
EVAL_QUESTIONS = "../eval/questions.csv"     # 데이터 누수 검사용

MAX_LEN = 1024          # 근거+질문+답변이 들어갈 길이. 8GB 기준 무난한 값
EPOCHS = 5              # 100건이라 적게 돌면 학습이 거의 안 됨
LR = 2e-4
BATCH = 1               # VRAM 때문에 1
ACCUM = 8               # 누적해서 실질 batch 8

LORA = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # attention + MLP 전체에 붙인다. 소규모 데이터에서는 이쪽이 안정적이다.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)


# =====================================================================
# 2) 데이터
# =====================================================================
def load_jsonl(path):
    """한 줄에 하나씩 들어있는 JSON 을 읽는다(ft_dataset.json 은 JSONL 형식)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_leakage(samples, questions_csv):
    """
    데이터 누수 검사.
    평가셋 질문이 학습 데이터에 그대로 들어 있으면 점수는 오르지만
    '시험 문제를 미리 외운' 것이 되어 발표에서 지적당한다.
    """
    p = Path(questions_csv)
    if not p.exists():
        print(f"  [건너뜀] 평가셋을 찾을 수 없음: {p}")
        return

    with open(p, encoding="utf-8-sig") as f:
        eval_qs = [r["question"].strip() for r in csv.DictReader(f)]

    train_text = "\n".join(
        m["content"] for s in samples for m in s["messages"] if m["role"] == "user"
    )

    hits = [q for q in eval_qs if q and q in train_text]
    if hits:
        print(f"  [경고] 평가셋 질문 {len(hits)}개가 학습 데이터에 그대로 있습니다:")
        for q in hits:
            print(f"         - {q[:60]}")
        print("         → 데이터 누수입니다. 해당 항목을 학습셋에서 제외하세요.")
    else:
        print(f"  [확인] 평가셋 {len(eval_qs)}문항 중 학습 데이터와 겹치는 것 없음")


class ChatSFTDataset(Dataset):
    """
    chat 형식(messages)을 학습용 토큰으로 변환한다.

    핵심: '답변 부분만' 학습한다.
      질문과 근거(프롬프트)는 label 을 -100 으로 막아 손실 계산에서 제외한다.
      이렇게 해야 모델이 '질문을 따라 쓰는 법'이 아니라
      '답변하는 법'을 배운다.
    """

    def __init__(self, samples, tokenizer, max_len):
        self.items = []
        skipped = 0

        for s in samples:
            msgs = s["messages"]
            # 프롬프트(마지막 assistant 를 뺀 부분) / 전체
            prompt_text = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                msgs, tokenize=False)

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]
                skipped += 1
            n_prompt = min(len(prompt_ids), len(full_ids))

            labels = list(full_ids)
            labels[:n_prompt] = [-100] * n_prompt      # 프롬프트 구간 손실 제외

            if all(x == -100 for x in labels):
                continue                                # 답변이 통째로 잘린 항목은 버림

            self.items.append({"input_ids": full_ids, "labels": labels})

        print(f"  [데이터] 학습 샘플 {len(self.items)}개"
              + (f" (길이 초과로 잘린 것 {skipped}개)" if skipped else ""))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id):
    """배치 안에서 가장 긴 것에 맞춰 오른쪽을 채운다."""
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }
    return collate


# =====================================================================
# 3) 학습
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Llama-3.2-3B QLoRA 파인튜닝")
    ap.add_argument("--smoke", action="store_true",
                    help="1스텝만 돌려 OOM 여부만 확인")
    ap.add_argument("--data_path", default=DATA_PATH)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    ap.add_argument("--epochs", type=float, default=EPOCHS)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    print("=" * 66)
    print("실험 5 — QLoRA 파인튜닝")
    print(f"  베이스 모델 : {BASE_MODEL}")
    print(f"  연산 dtype  : {'bfloat16' if bf16 else 'float16'}")
    print(f"  출력        : {args.output_dir}")
    print("=" * 66)

    # --- 데이터 ---
    samples = load_jsonl(args.data_path)
    print(f"\n[1/4] 데이터 {len(samples)}건 로드: {args.data_path}")
    check_leakage(samples, EVAL_QUESTIONS)

    # --- 토크나이저 ---
    print("\n[2/4] 토크나이저")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        # Llama 계열은 pad_token 이 없어 배치 학습 시 에러가 난다
        tokenizer.pad_token = tokenizer.eos_token
        print("  [보정] pad_token 이 없어 eos_token 으로 설정")

    dataset = ChatSFTDataset(samples, tokenizer, args.max_len)

    # --- 모델 (4bit) ---
    print("\n[3/4] 베이스 모델 4bit 로드")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_cfg,
        torch_dtype=dtype,
        device_map={"": 0},          # 전부 GPU 0 에. auto 는 CPU 로 새는 수가 있다
    )
    model.config.use_cache = False   # gradient checkpointing 과 함께 쓰면 필수
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(**LORA))
    model.print_trainable_parameters()

    # --- 학습 ---
    print("\n[4/4] 학습 시작")
    targs = TrainingArguments(
        output_dir=args.output_dir + "_ckpt",
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        num_train_epochs=args.epochs,
        max_steps=1 if args.smoke else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="no",          # 어댑터는 마지막에 한 번만 저장
        optim="paged_adamw_8bit",    # VRAM 절약
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        report_to=[],                # wandb 등 외부 로깅 끔
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    trainer.train()

    if args.smoke:
        print("\n" + "=" * 66)
        print("스모크 테스트 통과 — OOM 없이 1스텝 완료.")
        print("이제 본 학습을 돌리세요:  python train_llama_lora.py")
        print("=" * 66)
        return

    # --- 저장 ---
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    print("\n" + "=" * 66)
    print(f"학습 완료 → {out}")
    print("=" * 66)
    print("\n평가 명령어:")
    print("  python main.py --model_name ../models/llama3.2-3b-lora \\")
    print("      --embedding_model BAAI/bge-m3 --chunk_size 1000 --top_k 3 \\")
    print("      --prompt_name strict --score_threshold 0.45 \\")
    print("      --load_in_4bit --temperature 0 --run_name llama3.2-3b_ft_v1 \\")
    print("      --results_dir ../eval/results")
    print("\n※ 비교 기준 (파인튜닝 전, 같은 설정)")
    print("   bertscore_f1 0.7574 / keyword_rate 0.4880 / citation_acc 0.4167")
    print("   hit_at_k 0.8333 / refusal_acc 1.0")
    print("   → hit_at_k 는 거의 그대로여야 정상이다(검색을 건드리지 않았으므로).")
    print("     citation_acc 와 keyword_rate 가 오르면 가설대로다.")


if __name__ == "__main__":
    main()
