@echo off
REM Build and validate the v3 QLoRA dataset.
chcp 65001 > nul
cd /d "%~dp0"

REM Build 360 samples from the current law JSON source.
python src\build_ft_dataset_v3_ksj.py
if errorlevel 1 exit /b 1

REM Check counts, multi-question ratio, citations, duplicates, and leakage.
python src\validate_ft_dataset.py --min_samples 300 --min_multi_ratio 0.40
if errorlevel 1 exit /b 1

REM Verify the training input without downloading the model.
python src\finetune_mistral_qlora.py --dry_run
if errorlevel 1 exit /b 1

echo v3 fine-tuning data is ready.
