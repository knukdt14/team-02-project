@echo off
REM Evaluate the v3 QLoRA adapter under the same RAG settings as the baseline.
REM Run finetune_mistral_qlora.py before this file.
chcp 65001 > nul
cd /d "%~dp0"

python src\check_mistral_env.py
if errorlevel 1 exit /b 1

python src\main.py ^
  --llm_type hf ^
  --model_name models\mistral7b_ksj_adapter_v3 ^
  --load_in_4bit ^
  --force_cuda ^
  --embedding_device cpu ^
  --prompt_name strict ^
  --search_type hybrid ^
  --top_k 5 ^
  --temperature 0 ^
  --max_new_tokens 160 ^
  --multi_max_new_tokens 280 ^
  --run_name mistral7b_qlora_ftv3_hybrid_ksj ^
  --no-use_ragas
