@echo off
REM Print a side-by-side comparison of the baseline and BGE-M3 CSV files.
chcp 65001 > nul
cd /d "%~dp0"

python src\compare_embedding_results_ksj.py
