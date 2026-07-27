@echo off
REM Run fast v3 regression checks without downloading an LLM.
chcp 65001 > nul
cd /d "%~dp0"

python src\check_rag_v3_ksj.py
