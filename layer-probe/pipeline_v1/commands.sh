#!/bin/bash
# 在用户的 Windows PowerShell 上执行（box 不能直接 SSH）。脚本以 base64(tar.gz) 方式上传到 /root/autodl-tmp/layer-probe/pipeline_v1/scripts/
SSH="ssh -p 40351 root@connect.westd.seetacloud.com"
P=/root/autodl-tmp/layer-probe/pipeline_v1
# 0) 数据集（ModelScope）
$SSH "mkdir -p $P/data && curl -sSL -o $P/data/gsm8k_test.parquet 'https://modelscope.cn/api/v1/datasets/AI-ModelScope/gsm8k/repo?Revision=master&FilePath=main/test-00000-of-00001.parquet' && sha256sum $P/data/gsm8k_test.parquet"
# 1) SGLang（最终配置；第一次尝试用的 C3 extra_buffer 配置并发只有 1，已放弃）
$SSH "cd /root/autodl-tmp/snape-ctx && bash launch.sh pv1d_novis_strict_mf096 0.96 --disable-fast-image-processor --chunked-prefill-size 2048 --max-mamba-cache-size 9 --mamba-radix-cache-strategy no_buffer --disable-overlap-schedule"
# 2) Stage 1 生成（等 /health_generate 就绪，可续跑；结果写入 outputs/generations.jsonl 和 gen_meta.json）
$SSH "cd $P/scripts && (CONC=3 setsid nohup bash run_gen.sh >/dev/null 2>&1 </dev/null &)"
# 3) 串联脚本：等 gen.log 出现 RUN_DONE -> stop.sh 停掉 SGLang -> Stage 2 capture.py -> Stage 3 analyze.py
$SSH "cd $P/scripts && (setsid nohup bash run_after_gen.sh >/dev/null 2>&1 </dev/null &)"
# 4) 生成统计
$SSH "cd $P/scripts && /root/miniconda3/envs/qwen38/bin/python gen_stats.py > ../outputs/gen_stats.txt"
# 5) 取回结果（不含 npz）：先 tar 到 /root/autodl-tmp/snape-ctx/pv1_bundle.tgz，scp 到 Windows %TEMP%，再 CopyToBox
