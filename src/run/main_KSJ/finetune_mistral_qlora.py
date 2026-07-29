"""
finetune_mistral_qlora.py
---------------------------------------------------
Mistral-7B-Instruct-v0.3을 8GB GPU에서 QLoRA 방식으로 미세조정합니다.

주의:
  1) 먼저 원본 Mistral 베이스라인 평가를 끝낸 뒤 학습하세요.
  2) 8GB VRAM에서는 batch_size=1, max_length=512로 시작합니다.
     메모리가 부족하면 --max_length 448로 낮추세요.
  3) 이 파일은 전체 모델이 아니라 작은 LoRA 어댑터만 저장합니다.
  4) ft_dataset_v3_ksj.jsonl은 학습 전 validate_ft_dataset.py로 검사합니다.
  5) v3 기본 학습값은 epoch=1, learning_rate=5e-5입니다.

환경 검사만:
    python src/finetune_mistral_qlora.py --dry_run

실제 학습:
    python src/finetune_mistral_qlora.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from rag_chain import REFUSAL_MSG, get_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def load_jsonl(path: Path) -> list[dict]:
    """한 줄에 한 샘플이 저장된 JSONL 파일을 읽습니다."""
    rows = []
    with path.open(encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{line_no}번째 줄 JSON 오류: {exc}") from exc
            messages = row.get("messages")
            roles = [message.get("role") for message in messages or []]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(
                    f"{line_no}번째 줄 role 순서가 system/user/assistant가 아닙니다."
                )
            rows.append(row)
    return rows


def adapt_to_citation_id_format(row: dict) -> dict:
    """
    원본 학습 샘플을 현재 RAG의 [C1] 인용 형식으로 메모리에서 변환합니다.

    원본 파일:
        [산업안전보건법 제38조] ...
        답변 ... (근거: 산업안전보건법 제38조)

    학습에 넣는 형식:
        [C1] 산업안전보건법 제38조
        ...
        답변 ... [C1]

    v2/v3 데이터는 이미 실제 RAG와 같은 [C1]/[C2] 형식이므로 그대로 사용합니다.
    과거 ft_dataset.json도 실험 재현을 위해 읽을 수 있게 구형 형식만 변환합니다.
    """
    messages = row["messages"]
    user_content = str(messages[1]["content"])
    assistant_content = str(messages[2]["content"]).strip()

    already_uses_c_ids = bool(re.search(r"(?m)^\[C\d+\]\s+", user_content))
    if not already_uses_c_ids:
        # [근거], [질문] 표지는 유지하고 구형 법령 출처 대괄호만 [C1]로 바꿉니다.
        user_content = re.sub(
            r"(?m)^\[(?!근거\]|질문\])([^\]]+)\]\s*",
            lambda match: f"[C1] {match.group(1).strip()}\n",
            user_content,
            count=1,
        )

    if REFUSAL_MSG in assistant_content:
        assistant_content = REFUSAL_MSG
    elif not re.search(r"\[C\d+(?:\s*,\s*C\d+)*\]", assistant_content):
        # 기존 사람이 읽는 출처 문구를 제거하고 런타임 검증용 ID를 붙입니다.
        assistant_content = re.sub(
            r"\s*\(\s*근거\s*:\s*[^)]*\)\s*$",
            "",
            assistant_content,
        ).strip()
        assistant_content = f"{assistant_content} [C1]"

    return {
        "messages": [
            # v2/v3는 토큰 절약을 위해 만든 짧은 학습 시스템 메시지를 보존합니다.
            # 구형 데이터만 현재 strict 프롬프트로 보완합니다.
            {
                "role": "system",
                "content": (
                    str(messages[0].get("content", ""))
                    if already_uses_c_ids
                    else get_prompt("strict")["system"]
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


class ChatDataset:
    """
    채팅 샘플을 causal language model 학습 입력으로 변환합니다.

    system/user 프롬프트 토큰은 labels=-100으로 마스킹하고,
    assistant 답변 부분만 손실(loss)을 계산해 답변 형식을 학습시킵니다.
    """

    def __init__(self, rows, tokenizer, max_length: int):
        self.examples = []
        self.truncated_examples = 0

        for row in rows:
            # 디스크 원본은 유지하고 학습 직전에만 [C1] 형식으로 변환합니다.
            messages = adapt_to_citation_id_format(row)["messages"]

            # 전체 대화와 assistant 답변 직전 프롬프트를 같은 chat template로 만듭니다.
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )

            # 여기서는 먼저 자르지 않고 토큰화합니다.
            # Mistral 토크나이저는 한국어를 비교적 많은 토큰으로 나누므로,
            # 앞에서 max_length로 잘라버리면 뒤쪽 assistant 답변이 통째로 사라집니다.
            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
            )["input_ids"]
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]

            # 전체 대화와 답변 직전 프롬프트가 공통으로 가지는 토큰 길이를 찾습니다.
            # 공통 부분 뒤가 모델이 학습해야 할 assistant 답변 토큰입니다.
            common_length = 0
            for full_id, prompt_id in zip(full_ids, prompt_ids):
                if full_id != prompt_id:
                    break
                common_length += 1

            prompt_prefix = full_ids[:common_length]
            assistant_ids = full_ids[common_length:]

            # 템플릿 호환 문제 등으로 답변 토큰을 구분하지 못한 샘플만 제외합니다.
            if not assistant_ids:
                continue

            # 답변이 비정상적으로 길어도 전체 길이의 절반은 질문/근거에 남겨둡니다.
            max_answer_tokens = max(64, max_length // 2)
            if len(assistant_ids) > max_answer_tokens:
                assistant_ids = assistant_ids[:max_answer_tokens]
                if tokenizer.eos_token_id is not None:
                    assistant_ids[-1] = tokenizer.eos_token_id

            # 남은 토큰 예산 안에서 프롬프트를 보존합니다.
            # 길면 시작 부분(system 지시)과 끝 부분(질문)을 남기고,
            # 가운데의 긴 법령 근거만 줄입니다.
            prompt_budget = max_length - len(assistant_ids)
            if len(prompt_prefix) > prompt_budget:
                self.truncated_examples += 1
                head_budget = min(64, max(1, prompt_budget // 3))
                tail_budget = prompt_budget - head_budget
                if tail_budget > 0:
                    prompt_prefix = (
                        prompt_prefix[:head_budget]
                        + prompt_prefix[-tail_budget:]
                    )
                else:
                    prompt_prefix = prompt_prefix[:head_budget]

            input_ids = prompt_prefix + assistant_ids

            # system/user/근거 부분은 -100으로 마스킹하고 assistant만 학습합니다.
            labels = [-100] * len(prompt_prefix) + list(assistant_ids)
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mistral 7B QLoRA 미세조정")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_ROOT / "data" / "ft_dataset_v3_ksj.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "models" / "mistral7b_ksj_adapter_v3"),
    )
    # v3 데이터는 근거와 답변을 짧게 만들어 512 토큰 안에 최대한 보존합니다.
    # 8GB GPU에서 OOM이 발생하면 실행 시 --max_length 448을 추가하면 됩니다.
    parser.add_argument("--max_length", type=int, default=512)
    # 360개 데이터의 과적합을 줄이기 위해 한 번만 학습합니다.
    parser.add_argument("--epochs", type=float, default=1.0)
    # 기존 1e-4보다 학습률을 낮춰 베이스 모델의 능력 손상을 줄입니다.
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="데이터만 읽고 모델 다운로드·학습은 하지 않음",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.data_path))
    print(f"[데이터] 학습 샘플 {len(rows)}개")
    if not rows:
        raise RuntimeError("학습 데이터가 비어 있습니다.")

    if args.dry_run:
        print("[DRY RUN] 데이터 구조 확인 완료. 모델은 다운로드하지 않았습니다.")
        return 0

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA 학습에는 CUDA가 필요합니다.")

    set_seed(args.seed)
    random.seed(args.seed)
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[연산 dtype] {compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # NF4 + double quantization은 8GB GPU에서 7B 가중치 메모리를 줄이는 핵심 설정입니다.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map={"": 0},
        # 구버전 Transformers와도 호환되도록 dtype 대신 torch_dtype를 사용합니다.
        # 최신 버전에서 보이는 deprecation 경고는 학습 동작에는 영향을 주지 않습니다.
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Mistral attention/MLP의 선형층에만 작은 LoRA 가중치를 추가합니다.
    # r=8은 8GB VRAM에서 안정성을 우선한 보수적인 값입니다.
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatDataset(rows, tokenizer, args.max_length)
    print(
        f"[토큰화] 실제 학습 샘플 {len(train_dataset)}개 "
        f"(긴 프롬프트 축약 {train_dataset.truncated_examples}개)"
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "학습 가능한 샘플이 0개입니다. chat template과 토큰화 결과를 확인하세요."
        )

    # DataCollatorForSeq2Seq는 길이가 다른 input/label을 배치 단위로 패딩합니다.
    # label의 패딩 값 -100은 손실 계산에서 제외됩니다.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()

    # 전체 7B 모델이 아니라 수십~수백 MB 수준의 LoRA 어댑터만 저장합니다.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[완료] LoRA 어댑터 저장: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
