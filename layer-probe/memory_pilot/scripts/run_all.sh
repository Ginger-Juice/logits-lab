#!/bin/bash
# CPU only. Uses env python unchanged; matplotlib from existing pylib_plot appended to sys.path inside plot.py
cd /root/autodl-tmp/layer-probe/pipeline_v1/memory_pilot/scripts
PY=/root/miniconda3/envs/qwen38/bin/python
export CUDA_VISIBLE_DEVICES=
L=../logs/mem_pilot.log
$PY mem_pilot.py > $L 2>&1 && $PY summarize.py > ../results/summary_print.txt 2>>$L && $PY plot.py >> $L 2>&1
echo EXIT_DONE $? >> $L
