@echo off
REM Train the v3 QLoRA adapter after the dataset validation succeeds.
chcp 65001 > nul
cd /d "%~dp0"

python src\check_mistral_env.py
if errorlevel 1 exit /b 1

python src\validate_ft_dataset.py --min_samples 300 --min_multi_ratio 0.40
if errorlevel 1 exit /b 1

python src\finetune_mistral_qlora.py
