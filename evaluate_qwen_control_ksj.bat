@echo off
REM Fair control run: only the LLM differs from the Mistral baseline.
REM Retrieval, prompt, top-k and generation settings are kept identical.
chcp 65001 > nul
cd /d "%~dp0"

python src\check_mistral_env.py
if errorlevel 1 exit /b 1

python src\main.py ^
  --llm_type hf ^
  --model_name Qwen/Qwen2.5-1.5B-Instruct ^
  --no-load_in_4bit ^
  --force_cuda ^
  --embedding_device cpu ^
  --prompt_name strict ^
  --search_type hybrid ^
  --top_k 5 ^
  --temperature 0 ^
  --max_new_tokens 160 ^
  --multi_max_new_tokens 280 ^
  --run_name qwen15_control_hybrid_v2_ksj ^
  --no-use_ragas
