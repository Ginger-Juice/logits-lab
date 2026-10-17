#!/bin/bash
cd /root/autodl-tmp/layer-probe/pipeline_v1
export HF_HOME=/root/autodl-tmp/snape-ctx/hf HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1
/root/miniconda3/envs/qwen38/bin/python scripts/gen.py > logs/gen.log 2>&1
echo RUN_DONE >> logs/gen.log
