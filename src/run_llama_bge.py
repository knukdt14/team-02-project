"""
run_llama_bge.py
---------------------------------------------------
실험 4 — 검색 개선이 실제 답변 품질로 이어지는지 확인.

[배경]
실험 2-2에서 임베딩을 bge-m3 로 바꾸자 검색 지표가 크게 올랐다.
  Hit@k 0.5 → 0.9167 / MRR 0.4167 → 0.7292
하지만 이것은 '정답 조문을 찾아왔다'는 뜻일 뿐,
라마가 그 근거를 실제로 잘 활용하는지는 별개 문제다.

[확인할 것]
  bertscore_f1  : 0.7159 (실험1 strict) 에서 오르는가
  citation_acc  : 0.375  (실험1 strict) 에서 오르는가
  refusal_acc   : 0.0    에서 변화가 있는가 (아마 없을 것 — 실험 3에서 다룸)

오르면  → 검색 개선이 최종 성능으로 이어졌다는 증거
안 오르면 → "검색은 고쳤는데 3B 모델이 못 쓴다" 는 발견이 되고,
            이것이 파인튜닝(실험 5)의 명분이 된다.
어느 쪽이 나와도 발표 재료가 된다.

[대조군 설계]
실험 1은 top_k=3 이었고 이번 최적 설정은 top_k=5 다.
그냥 비교하면 개선이 '임베딩 덕분'인지 'top_k 를 키운 덕분'인지 알 수 없으므로,
top_k=3 조건을 하나 넣어 변수를 분리한다.

[주의]
run_llama.py 는 실험 1 기록용으로 그대로 둔다(재현성).
이 파일은 실험 4 전용이며, 결과 파일 이름에 임베딩/청크 꼬리표가 붙어
실험 1 결과를 덮어쓰지 않는다.

사용법:
    cd src
    python run_llama_bge.py
"""

import subprocess
import sys


# =====================================================================
# 1) 공통 설정 — 실험 2-2에서 찾은 최적 검색 설정
# =====================================================================
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TAG = "llama3.2-3b"

EMBEDDING_MODEL = "BAAI/bge-m3"        # 검색 전용 모델, 최대 8192토큰
CHUNK_SIZE = 1000                      # bge-m3 는 길어도 잘리지 않음

COMMON = {
    "llm_type": "hf",
    "model_name": MODEL,
    "embedding_model": EMBEDDING_MODEL,
    "chunk_size": CHUNK_SIZE,
    "temperature": 0,                  # 0 = 항상 같은 답 (재현성)
    "results_dir": "../eval/results",
}

FLAGS = ["load_in_4bit"]               # 8GB VRAM 대응


# =====================================================================
# 2) 실험 목록
# =====================================================================
EXPERIMENTS = [
    # 실험 2-2에서 가장 좋았던 설정
    {"prompt_name": "strict", "top_k": 5},
    {"prompt_name": "cite", "top_k": 5},

    # 대조군: 실험 1과 top_k 를 맞춤
    # → 이것과 실험1(strict/k3/ko-sbert)의 차이가 '순수 임베딩 효과'
    {"prompt_name": "strict", "top_k": 3},
]


# =====================================================================
# 3) 실행
# =====================================================================
def build_command(exp):
    cfg = {**COMMON, **exp}
    # 실험 1 결과 파일과 이름이 겹치지 않도록 임베딩/청크 꼬리표를 붙인다
    cfg["run_name"] = (f"{TAG}_bge-m3_c{CHUNK_SIZE}"
                       f"_{exp['prompt_name']}_k{exp['top_k']}")

    cmd = [sys.executable, "main.py"]
    for key, value in cfg.items():
        cmd += [f"--{key}", str(value)]
    for flag in FLAGS:
        cmd.append(f"--{flag}")
    return cmd


def main():
    total = len(EXPERIMENTS)
    failed = []

    print("=" * 66)
    print("실험 4 — bge-m3 검색으로 라마 답변 품질 재평가")
    print(f"  임베딩 : {EMBEDDING_MODEL}")
    print(f"  청크   : {CHUNK_SIZE}자")
    print("=" * 66)
    print("\n[비교 기준] 실험 1 (ko-sbert / 청크500 / strict / top_k 3)")
    print("  bertscore_f1 = 0.7159")
    print("  citation_acc = 0.375")
    print("  refusal_acc  = 0.0")

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
    print("\n※ 검색 지표(hit_at_k 등)는 실험 2-2와 같은 값이 나와야 정상이다.")
    print("  다르게 나오면 벡터DB를 잘못 잡은 것이니 설정을 다시 확인할 것.")


if __name__ == "__main__":
    main()
