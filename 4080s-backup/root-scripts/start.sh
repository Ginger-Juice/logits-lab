#!/bin/bash

echo "(自动执行忽略)手动执行，需要先切换环境: "
echo "命令: conda activate qwen38"

pkill -f "sglang serve"
pkill -f "webui.py"

# sudo apt update
# sudo apt install ffmpeg -y

export LD_LIBRARY_PATH=/root/miniconda3/envs/qwen38/lib:$LD_LIBRARY_PATH

# 1. 强制切换到 /root 目录，防止相对路径找不到 webui.py 或 models
cd /root

# 2. 核心修复：强制 Python 禁用输出缓冲，让日志实时写入文件！
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export TORCH_NUM_THREADS=16

# 建议将 webui 和 sglang 的日志分开，避免两个进程同时写一个文件导致日志错乱
# 如果你非要合在一起，可以把下面的 > /root/webui.log 2>&1 删掉
python webui.py > /root/webui.log 2>&1 & 

sglang serve \
  --trust-remote-code \
  --model-path /root/autodl-tmp/qwen38-27b-fp8 \
  --served-model-name qwen3.8-27b-fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.82 \
  --attention-backend flashinfer \
  --chunked-prefill-size 4096 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 6008 \
  --cuda-graph-max-bs-decode 8 \
  --max-mamba-cache-size 10 \
  --mamba-full-memory-ratio 3.0 \
  --allow-auto-truncate \
  --disable-prefill-cuda-graph \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /root/autodl-tmp/model_dspark \
  --speculative-draft-attention-backend flashinfer \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype bfloat16 
  
