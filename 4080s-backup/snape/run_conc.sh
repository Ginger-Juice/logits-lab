#!/bin/bash
# usage: run_conc.sh <num_ctx> <out_prefix>   (runs in background via setsid nohup)
cd /root/autodl-tmp/snape
export MODEL=modelscope.cn/unsloth/Qwen3.8-27B-GGUF:Q4_K_M
export NUM_CTX=${1:-32768}
export LENGTHS=512,4096 REPEATS=3 GEN=256 CONCURRENCY=1,2,4 WARMUP=1
export RAW=${RAW:-0} IGNORE_EOS=${IGNORE_EOS:-0}
export OUT=${2:-bench_q4km_conc_ctx32k}.csv
/root/miniconda3/bin/python3 ollama_bench_min.py
