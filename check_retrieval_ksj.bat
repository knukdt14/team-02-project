@echo off
REM Check retrieval metrics without loading the 7B language model.
chcp 65001 > nul
cd /d "%~dp0"

python src\check_retrieval_ksj.py
