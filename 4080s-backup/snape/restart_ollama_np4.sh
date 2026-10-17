#!/bin/bash
# Restart ollama serve with OLLAMA_NUM_PARALLEL=4 (keeps OLLAMA_MODELS, network_turbo via start_ollama.sh)
# Uses exact process-name match (pkill -x) so the calling ssh/bash shell is never matched.
LOG=/root/autodl-tmp/snape/serve_np4.log
pkill -x ollama
sleep 3
pkill -x llama-server
sleep 2
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=60m
cd /root/autodl-tmp/snape
setsid nohup /root/autodl-tmp/snape/start_ollama.sh > "$LOG" 2>&1 < /dev/null &
sleep 5
pgrep -a ollama
