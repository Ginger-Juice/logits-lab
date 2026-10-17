#!/bin/bash
# waits for Stage 1 to finish, stops our SGLang server, then runs Stage 2 capture and Stage 3 analysis
cd /root/autodl-tmp/layer-probe/pipeline_v1
until grep -q RUN_DONE logs/gen.log; do sleep 15; done
echo "gen finished $(date)" > logs/after_gen.log
bash /root/autodl-tmp/snape-ctx/stop.sh >> logs/after_gen.log 2>&1
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/snape-ctx/hf PYTHONPATH=/root/autodl-tmp/snape-ctx/pylib PYTHONUNBUFFERED=1
( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> logs/capture_nvsmi.log; sleep 5; done ) &
MON=$!
echo "capture start $(date)" >> logs/after_gen.log
/root/miniconda3/envs/qwen38/bin/python scripts/capture.py > logs/capture.log 2>&1
echo "capture end rc=$? $(date)" >> logs/after_gen.log
kill $MON
nvidia-smi --query-gpu=memory.used --format=csv >> logs/after_gen.log
PYTHONPATH=/root/autodl-tmp/snape-ctx/pylib_plot MPLBACKEND=Agg NJ=48 /root/miniconda3/envs/qwen38/bin/python scripts/analyze.py > logs/analyze.log 2>&1
echo "analysis end rc=$? $(date)" >> logs/after_gen.log
du -sh outputs outputs/* >> logs/after_gen.log
echo ALL_DONE >> logs/after_gen.log
