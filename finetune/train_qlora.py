"""
train_qlora.py
---------------------------------------------------
EXAONE-3.5-7.8B QLoRA 파인튜닝 (8GB VRAM 대응)

방식: 4bit 양자화된 베이스 모델 + LoRA 어댑터 학습
  - 베이스 모델 가중치는 얼려두고(frozen), 작은 어댑터만 학습
  - 8GB GPU에서 7.8B 모델 학습이 가능한 사실상 유일한 방법
  - 결과물은 수십 MB 어댑터 (베이스 모델 전체를 저장하지 않음)

학습 목표 (RAGAS 진단 기반):
  - faithfulness 0.622 개선 → 근거에 없는 내용을 덧붙이지 않도록
  - 답변 형식 통일 (근거 인용 → 결론)
  - 근거 없으면 거절하는 행동 학습

입력: finetune_dataset.jsonl (100문항, 누수 검증 PASS)
출력: ../models/exaone-ft/  (LoRA 어댑터)

설치:
  pip install peft trl datasets bitsandbytes accelerate

실행:
  python train_qlora.py
  python train_qlora.py --epochs 5 --lr 1e-4      # 하이퍼파라미터 조정
"""

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "../data/ft_dataset.json"
OUT_DIR = BASE.parent / "models" / "exaone-ft"


# =====================================================================
# 설정 (실험 변수)
# =====================================================================
CONFIG = dict(
    model_name="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    output_dir=str(OUT_DIR),
    # 학습 하이퍼파라미터
    epochs=3,                    # 100문항이면 3~5 권장
    lr=2e-4,                     # QLoRA 표준값
    batch_size=1,                # 8GB VRAM → 1 고정
    grad_accum=8,                # 실질 배치 = 1 x 8 = 8
    max_seq_len=1024,            # 근거+질문+답변 길이
    # LoRA 설정
    lora_r=16,                   # 랭크 (클수록 표현력↑, 메모리↑)
    lora_alpha=32,               # 보통 r의 2배
    lora_dropout=0.05,
    # 기타
    seed=42,
    save_steps=50,
    logging_steps=5,
)


def parse_args():
    p = argparse.ArgumentParser(description="EXAONE QLoRA 파인튜닝")
    for k, v in CONFIG.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif isinstance(v, int):
            p.add_argument(f"--{k}", type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(f"--{k}", type=float, default=v)
        else:
            p.add_argument(f"--{k}", default=v)
    return p.parse_args()


# =====================================================================
# 데이터 로드 — messages 형식 → chat template 적용된 텍스트
# =====================================================================
def load_dataset_from_jsonl(path, tokenizer):
    """
    finetune_dataset.jsonl의 messages를 모델 고유 대화 형식으로 변환.
    추론(rag_chain.py)에서도 apply_chat_template을 쓰므로 형식이 일치한다.
    """
    from datasets import Dataset

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            records.append({"text": text})

    print(f"[data] {len(records)}개 학습 샘플 로드")
    print(f"[data] 샘플 미리보기:\n{records[0]['text'][:300]}...\n")
    return Dataset.from_list(records)


# =====================================================================
# EXAONE 호환 패치
#   EXAONE의 커스텀 modeling 코드는 get_input_embeddings()를 구현하지 않아
#   PEFT(LoRA 주입)와 gradient checkpointing이 실패한다.
#   → 임베딩 층을 찾아 클래스에 주입한다. (HF 캐시 파일은 건드리지 않음)
# =====================================================================
def patch_input_embeddings(model) -> str | None:
    import torch.nn as nn

    def _inject(owner, attr):
        cls = type(owner)
        cls._input_embed_layer = attr
        cls.get_input_embeddings = lambda self, _a=attr: getattr(self, _a)
        cls.set_input_embeddings = lambda self, v, _a=attr: setattr(self, _a, v)
        return f"{cls.__name__}.{attr}"

    # ① 흔한 구조: ExaoneForCausalLM.transformer(ExaoneModel).wte
    owners = [model]
    for a in ("transformer", "model", "base_model"):
        sub = getattr(model, a, None)
        if sub is not None:
            owners.append(sub)

    for owner in owners:
        for attr in ("wte", "embed_tokens", "word_embeddings", "tok_embeddings"):
            emb = getattr(owner, attr, None)
            if isinstance(emb, nn.Embedding):
                path = _inject(owner, attr)
                # 최상위에서도 접근 가능하도록 보강
                if owner is not model:
                    type(model).get_input_embeddings = lambda self, _e=emb: _e
                    type(model).set_input_embeddings = lambda self, v: None
                return path

    # ② 폴백: 전체 모듈에서 가장 큰 Embedding(=토큰 임베딩)을 찾아 연결
    best_name, best = None, None
    for name, sub in model.named_modules():
        if isinstance(sub, nn.Embedding):
            if best is None or sub.num_embeddings > best.num_embeddings:
                best_name, best = name, sub
    if best is not None:
        type(model).get_input_embeddings = lambda self, _e=best: _e
        type(model).set_input_embeddings = lambda self, v: None
        return f"(fallback) {best_name}"
    return None


# =====================================================================
# 메인
# =====================================================================
def main():
    args = parse_args()

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    if not Path(DATA).exists():
        raise SystemExit(f"학습 데이터 없음: {DATA}\n"
                         f"먼저 build_finetune100.py 를 실행하세요.")

    # --- 토크나이저 ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 데이터셋 ---
    dataset = load_dataset_from_jsonl(DATA, tokenizer)

    # --- 연산 dtype 결정 ---
    #   bf16 지원 GPU면 bf16 사용 (GradScaler 불필요 → dtype 충돌 없음, 학습 안정)
    #   미지원 구형 GPU면 fp16 폴백
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[dtype] {'bfloat16' if use_bf16 else 'float16'} 사용")

    # --- 4bit 양자화 베이스 모델 ---
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[model] 로딩 중: {args.model_name} (4bit)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False          # 학습 시 필수

    # EXAONE 호환 패치 (PEFT 주입 전에 반드시 수행)
    patched = patch_input_embeddings(model)
    if patched:
        print(f"[patch] 입력 임베딩 연결: {patched}")
    else:
        print("[patch] 경고: 임베딩 층을 찾지 못했습니다. LoRA 주입이 실패할 수 있습니다.")

    model = prepare_model_for_kbit_training(model)

    # --- LoRA 어댑터 ---
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # 주의: 모델 구조마다 모듈명이 다름. 자동 탐색 실패 시 아래를 조정.
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] 학습 파라미터: {trainable:,} / 전체 {total:,} "
          f"({trainable/total*100:.3f}%)")

    # --- 학습 설정 ---
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="paged_adamw_8bit",           # 메모리 절약 옵티마이저
        bf16=use_bf16,                      # bf16이면 GradScaler 미사용
        fp16=not use_bf16,
        gradient_checkpointing=True,        # 메모리 절약 (속도 ↓)
        max_length=args.max_seq_len,
        dataset_text_field="text",
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n[train] 학습 시작 — epochs={args.epochs}, lr={args.lr}, "
          f"실질 배치={args.batch_size * args.grad_accum}")
    trainer.train()

    # --- 저장 (어댑터만) ---
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[done] LoRA 어댑터 저장 완료: {args.output_dir}")
    print("\n다음 단계: 파인튜닝 모델로 평가 실행")
    print(f'  python main_copy.py --model_name {args.output_dir} \\')
    print( '    --embedding_name upstage --top_k 3 --score_threshold 0.24 \\')
    print( '    --load_in_4bit --trust_remote_code --run_name RIH_finetuned')


if __name__ == "__main__":
    main()
