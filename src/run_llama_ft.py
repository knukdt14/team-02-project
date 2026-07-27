"""
run_llama_ft.py
---------------------------------------------------
실험 5 평가 — QLoRA 파인튜닝 모델을 최종 설정에서 평가한다.

[비교 대상]  파인튜닝 전, 완전히 동일한 설정
    eval/results/llama3.2-3b_bge-m3_c1000_strict_k3_th045.csv
      bertscore_f1 0.7574 / keyword_rate 0.4880 / citation_acc 0.4167
      hit_at_k 0.8333 / refusal_acc 1.0 / latency 2.34초

검색·프롬프트·게이트를 전부 그대로 두고 모델만 바꾸므로,
차이는 오로지 파인튜닝 효과다.

[가설]
    hit_at_k       거의 변화 없음   (검색을 건드리지 않았으므로)
    citation_acc   상승            (인용 형식을 학습)
    keyword_rate   상승            (답변 스타일 정렬)
    한자 혼입       감소
    multi 유형      개선?           (실험 4에서 유일하게 하락한 항목)

[주의]  학습 loss가 0.014까지 떨어져 과적합이 의심된다.
        모델이 검색 근거를 쓰지 않고 외운 답을 뱉으면
        keyword_rate 는 오르는데 hit_at_k 대비 답변이 겉도는 형태로 나타난다.
        그런 조짐이 보이면 에폭을 줄여 재학습한다(3분이면 된다).

    python train_llama_lora.py --epochs 2

사용법:
    cd src
    python run_llama_ft.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


ADAPTER = "../models/llama3.2-3b-lora"

COMMON = {
    "llm_type": "hf",
    "model_name": ADAPTER,
    "embedding_model": "BAAI/bge-m3",
    "chunk_size": 1000,
    "prompt_name": "strict",
    "top_k": 3,
    "score_threshold": 0.45,
    "temperature": 0,
    "results_dir": "../eval/results",
}

FLAGS = ["load_in_4bit"]

# 결과 파일 이름 꼬리표. 재학습할 때마다 --tag 로 바꿔서 덮어쓰기를 막는다.
#   예) python run_llama_ft.py --tag v2_ep2
DEFAULT_TAG = "v1"


def build_command(exp):
    cfg = {**COMMON, **exp}
    cmd = [sys.executable, "main.py"]
    for key, value in cfg.items():
        cmd += [f"--{key}", str(value)]
    for flag in FLAGS:
        cmd.append(f"--{flag}")
    return cmd


def main():
    ap = argparse.ArgumentParser(description="파인튜닝 모델 평가")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="결과 파일 이름 꼬리표 (예: v2_ep2)")
    args = ap.parse_args()

    experiments = [{"run_name": f"llama3.2-3b_ft_{args.tag}"}]

    if not Path(ADAPTER, "adapter_config.json").exists():
        print(f"[중단] 어댑터를 찾을 수 없습니다: {ADAPTER}")
        print("       먼저 학습을 돌리세요:  python train_llama_lora.py")
        return

    print("=" * 70)
    print("실험 5 평가 — QLoRA 파인튜닝 모델")
    print(f"  어댑터 : {ADAPTER}")
    print("=" * 70)
    print("[파인튜닝 전 기준 — 동일 설정]")
    print("  bertscore_f1 0.7574 / keyword_rate 0.4880 / citation_acc 0.4167")
    print("  hit_at_k 0.8333 / refusal_acc 1.0 / latency 2.34초")

    failed = []
    for i, exp in enumerate(experiments, 1):
        cmd = build_command(exp)
        print("\n" + "=" * 70)
        print(f"[{i}/{len(experiments)}] {exp['run_name']}")
        print("=" * 70)
        if subprocess.run(cmd).returncode != 0:
            failed.append(exp)

    print("\n" + "=" * 70)
    print(f"완료: {len(experiments) - len(failed)}/{len(experiments)} 성공")
    print("=" * 70)
    print("\n※ 읽는 법")
    print("  로그에 '[HFChatLLM] LoRA 어댑터 감지' 가 떠야 파인튜닝 모델이 쓰인 것이다.")
    print("  안 뜨면 베이스 모델로 평가한 것이므로 경로를 확인할 것.")
    print()
    print("  hit_at_k 0.8333 유지     → 정상 (검색은 안 건드렸으므로)")
    print("  citation_acc 상승        → 가설대로 (인용 형식 학습)")
    print("  keyword_rate 상승        → 가설대로 (답변 스타일 정렬)")
    print("  전부 하락               → 과적합 의심, 에폭 줄여 재학습")


if __name__ == "__main__":
    main()
