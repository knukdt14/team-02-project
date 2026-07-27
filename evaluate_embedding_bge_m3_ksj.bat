@echo off
REM Compare BGE-M3 while keeping every other final RAG setting unchanged.
REM The first run downloads BGE-M3 and builds a separate FAISS index.
chcp 65001 > nul
cd /d "%~dp0"

python src\check_mistral_env.py
if errorlevel 1 exit /b 1

python src\main.py ^
  --llm_type hf ^
  --model_name mistralai/Mistral-7B-Instruct-v0.3 ^
  --load_in_4bit ^
  --force_cuda ^
  --embedding_name hf ^
  --embedding_model BAAI/bge-m3 ^
  --embedding_device cpu ^
  --prompt_name strict ^
  --search_type hybrid ^
  --store_type faiss ^
  --top_k 5 ^
  --chunk_size 500 ^
  --overlap_size 50 ^
  --temperature 0 ^
  --max_new_tokens 160 ^
  --multi_max_new_tokens 280 ^
  --run_name mistral7b_base_hybrid_v3_bge_m3_ksj ^
  --no-use_ragas
