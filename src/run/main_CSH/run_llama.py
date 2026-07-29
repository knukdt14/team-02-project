"""
run_llama.py
---------------------------------------------------
라마 실험 실행기. main.py 를 설정만 바꿔가며 여러 번 실행한다.

main.py 를 복사하지 않고 '실행'만 시키므로,
팀에서 main.py 를 개선하면 그 내용이 그대로 반영된다.

사용법:
    cd src
    python run_llama.py

실험을 추가/변경하려면 아래 EXPERIMENTS 목록만 고치면 된다.
"""

import subprocess
import sys


# =====================================================================
# 1) 공통 설정 - 모든 실험에 똑같이 적용
# =====================================================================
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TAG = "llama3.2-3b"          # 결과 파일 이름 앞에 붙는 꼬리표

COMMON = {
    "llm_type": "hf",
    "model_name": MODEL,
    "temperature": 0,                  # 0 = 항상 같은 답 (실험 재현성)
    "results_dir": "../eval/results",  # git 에 커밋되는 위치
}

# 값 없이 켜기만 하는 옵션 (8GB VRAM 대응)
FLAGS = ["load_in_4bit"]


# =====================================================================
# 2) 실험 목록 - 여기에 줄을 추가하면 실험이 늘어난다
# =====================================================================
EXPERIMENTS = [
    {"prompt_name": "basic",  "top_k": 3},
    {"prompt_name": "cite",   "top_k": 3},
    {"prompt_name": "cot",    "top_k": 3},
    {"prompt_name": "strict", "top_k": 3},
]


# =====================================================================
# 3) 실행 - main.py 를 하나씩 별도 프로세스로 돌린다
#    프로세스를 분리해야 실험이 끝날 때마다 GPU 메모리가 완전히 비워진다.
# =====================================================================
def build_command(exp):
    """실험 설정 하나를 main.py 실행 명령어로 변환."""
    cfg = {**COMMON, **exp}
    cfg["run_name"] = f"{TAG}_{exp['prompt_name']}_k{exp['top_k']}"

    cmd = [sys.executable, "main.py"]
    for key, value in cfg.items():
        cmd += [f"--{key}", str(value)]
    for flag in FLAGS:
        cmd.append(f"--{flag}")
    return cmd


def main():
    total = len(EXPERIMENTS)
    failed = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        cmd = build_command(exp)
        print("\n" + "=" * 66)
        print(f"[{i}/{total}] {exp}")
        print(" ".join(cmd))
        print("=" * 66)

        if subprocess.run(cmd).returncode != 0:
            print(f"  [실패] {exp}")
            failed.append(exp)

    print("\n" + "=" * 66)
    print(f"완료: {total - len(failed)}/{total} 성공")
    for exp in failed:
        print(f"  실패 -> {exp}")
    print("=" * 66)


if __name__ == "__main__":
    main()
