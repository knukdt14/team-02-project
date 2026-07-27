@echo off
REM Evaluate the original Mistral 7B baseline with fixed KSJ settings.
REM The result filename is unique, so the team baseline CSV is not overwritten.
chcp 65001 > nul
cd /d "%~dp0"

python src\check_mistral_env.py
if errorlevel 1 exit /b 1

python src\main.py ^
  --llm_type hf ^
  --model_name mistralai/Mistral-7B-Instruct-v0.3 ^
  --load_in_4bit ^
  --force_cuda ^
  --embedding_device cpu ^
  --prompt_name strict ^
  --search_type hybrid ^
  --top_k 5 ^
  --temperature 0 ^
  --max_new_tokens 160 ^
  --multi_max_new_tokens 280 ^
  --run_name mistral7b_base_hybrid_v3_ksj ^
  --no-use_ragas
