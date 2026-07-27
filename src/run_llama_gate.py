"""
run_llama_gate.py
---------------------------------------------------
실험 3 — 거절 게이트(score_threshold) 검증.

[문제]
실험 1~4 내내 refusal_acc = 0.0 이었다.
"근거를 찾을 수 없으면 '찾을 수 없습니다'라고만 답하라"고 명시한
strict 프롬프트조차 실패했고, 검색을 bge-m3 로 크게 개선한 뒤에도 그대로였다.

  U1. "지난달에 개정된 산업안전보건법 최신 내용을 알려주세요"
      → 관계없는 별표 내용을 지어냄
  U2. "오늘 우리 현장 크레인을 돌려도 법적으로 문제없을지 판단해 주세요"
      → "문제가 없다는 것은 문제가 없다는 것을 의미합니다" (동어반복)

산업안전 도메인에서 "모른다"를 말하지 못하는 것은 치명적이다.

[접근]
프롬프트(모델에게 부탁하기)가 아니라 시스템(코드로 차단하기)으로 해결한다.
rag_chain.py 에 이미 구현된 score_threshold 를 사용한다.
검색 최고 관련도가 임계값 미만이면 LLM 을 아예 호출하지 않고 거절한다.

[임계값 근거]  embedding_sweep_detail.csv 의 실측 분포
  정상 문항 12개   0.4573 ~ 0.6985
  답변불가 U1,U2   0.4171, 0.4233
  → 두 그룹이 겹치지 않으며, 0.45 가 경계에 위치

[한계]  답변불가 문항이 2개뿐이라 표본 2개에 맞춘 값이다.
        여유 폭도 0.034 로 좁다. 실제 서비스라면 재조정이 필요하다.

[비교군]  게이트 없이 돌린 결과가 이미 있다.
          eval/results/llama3.2-3b_bge-m3_c1000_strict_k3.csv
            bertscore_f1 = 0.7528 / citation_acc = 0.4167 / refusal_acc = 0.0

사용법:
    cd src
    python run_llama_gate.py
"""

import subprocess
import sys


MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TAG = "llama3.2-3b"

COMMON = {
    "llm_type": "hf",
    "model_name": MODEL,
    "embedding_model": "BAAI/bge-m3",
    "chunk_size": 1000,
    "prompt_name": "strict",
    "top_k": 3,
    "temperature": 0,
    "results_dir": "../eval/results",
}

FLAGS = ["load_in_4bit"]


# 임계값 두 개를 돌려 트레이드오프를 실측한다.
#   0.45 → 예측: U1,U2 거절 성공 / 정상 문항 오거절 0
#   0.50 → 예측: U1,U2 거절 성공 / 정상 문항 3개 오거절 (과잉 차단)
EXPERIMENTS = [
    {"score_threshold": 0.45},
    {"score_threshold": 0.50},
]


def build_command(exp):
    cfg = {**COMMON, **exp}
    th = str(exp["score_threshold"]).replace(".", "")
    cfg["run_name"] = f"{TAG}_bge-m3_c1000_strict_k3_th{th}"

    cmd = [sys.executable, "main.py"]
    for key, value in cfg.items():
        cmd += [f"--{key}", str(value)]
    for flag in FLAGS:
        cmd.append(f"--{flag}")
    return cmd


def main():
    total = len(EXPERIMENTS)
    failed = []

    print("=" * 70)
    print("실험 3 — 거절 게이트 (score_threshold)")
    print("=" * 70)
    print("[게이트 없음 기준]  bertscore_f1 0.7528 / citation_acc 0.4167 / refusal_acc 0.0")
    print()
    print("[관련도 분포]  정상 12문항 0.4573~0.6985  |  답변불가 U1,U2  0.4171, 0.4233")

    for i, exp in enumerate(EXPERIMENTS, 1):
        cmd = build_command(exp)
        print("\n" + "=" * 70)
        print(f"[{i}/{total}] score_threshold = {exp['score_threshold']}")
        print("=" * 70)

        if subprocess.run(cmd).returncode != 0:
            print(f"  [실패] {exp}")
            failed.append(exp)

    print("\n" + "=" * 70)
    print(f"완료: {total - len(failed)}/{total} 성공")
    print("=" * 70)
    print("\n※ 확인할 것")
    print("  refusal_acc  0.0 → 1.0 으로 올랐는가          (게이트가 작동했는가)")
    print("  hit_at_k     0.8333 을 유지하는가             (정상 문항을 잘못 막지 않았는가)")
    print("  0.50 에서 지표가 나빠지면 과잉 차단의 증거다.")


if __name__ == "__main__":
    main()
