"""
check_mistral_ksj_env.py
---------------------------------------------------
Mistral 7B를 실행하기 전에 현재 가상환경이 준비되었는지 빠르게 확인합니다.

이 파일은 모델을 다운로드하지 않습니다.
Python/PyTorch/CUDA/GPU 메모리/필수 패키지만 검사하므로 먼저 실행해도 안전합니다.

실행:
    python src/check_mistral_ksj_env.py
"""

from __future__ import annotations

import sys
from importlib import metadata


# 현재 프로젝트에서 Mistral 실행에 꼭 필요한 패키지 이름입니다.
REQUIRED_PACKAGES = [
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sentencepiece",
]


def _version(package_name: str) -> str | None:
    """설치된 패키지 버전을 반환하고, 미설치이면 None을 반환합니다."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    """환경을 점검하고 실행 가능하면 0, 문제가 있으면 1을 반환합니다."""
    print("=" * 68)
    print("[Mistral 7B 실행 환경 점검]")
    print("=" * 68)
    print(f"Python: {sys.version.split()[0]}")

    # 프로젝트 표준 가상환경은 Python 3.11입니다.
    python_ok = sys.version_info[:2] == (3, 11)
    print(f"Python 3.11: {'OK' if python_ok else '확인 필요'}")

    try:
        import torch
    except ImportError:
        print("PyTorch: 설치되지 않음")
        print("CUDA용 PyTorch를 먼저 설치해야 합니다.")
        return 1

    print(f"PyTorch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA 사용 가능: {cuda_ok}")
    print(f"PyTorch CUDA 버전: {torch.version.cuda}")

    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name}")
        print(f"GPU 메모리: {total_gib:.1f} GiB")
        if total_gib < 7.5:
            print("[주의] 7B 4비트 추론도 메모리가 빠듯할 수 있습니다.")
    else:
        print("[오류] Mistral 7B는 이 프로젝트에서 CPU로 실행하지 않습니다.")

    packages_ok = True
    print("\n[필수 패키지]")
    for package in REQUIRED_PACKAGES:
        version = _version(package)
        if version is None:
            packages_ok = False
            print(f"- {package}: 미설치")
        else:
            print(f"- {package}: {version}")

    if not packages_ok:
        print("\n다음 명령으로 Mistral 추가 패키지를 설치하세요.")
        print("python -m pip install -r requirements-ksj.txt")

    ok = python_ok and cuda_ok and packages_ok
    print("\n점검 결과:", "실행 가능" if ok else "수정 필요")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
