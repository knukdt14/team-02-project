@echo off
REM Run one Mistral RAG question from the project root.
REM English comments avoid Windows cmd encoding problems.
chcp 65001 > nul
cd /d "%~dp0"

python src\check_mistral_env.py
if errorlevel 1 exit /b 1

if "%~1"=="" (
    set "QUESTION=순간풍속이 초당 몇 미터를 초과하면 타워크레인의 운전작업을 중지해야 하나요?"
) else (
    set "QUESTION=%*"
)

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
  --ask "%QUESTION%"
