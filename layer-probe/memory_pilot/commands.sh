#!/bin/bash
# 在 Windows PowerShell 上执行
SSH="ssh -p 40351 root@connect.westd.seetacloud.com"
M=/root/autodl-tmp/layer-probe/pipeline_v1/memory_pilot
# 1) 上传脚本（box: scripts/ 打包成 tgz -> CopyFromBox 到 C:\Users\ASUS\mem_pilot_stage\mp4.tgz）
scp -P 40351 C:\Users\ASUS\mem_pilot_stage\mp4.tgz root@connect.westd.seetacloud.com:$M/scripts/mp4.tgz
$SSH "cd $M/scripts && md5sum mp4.tgz && tar xzf mp4.tgz"   # md5 9ebefb8fe3cbe29340e0f9a38e0d43da
# 2) 运行（纯 CPU：mem_pilot.py -> summarize.py -> plot.py）
$SSH "cd $M/scripts && (setsid nohup bash run_all.sh >/dev/null 2>&1 </dev/null &)"
# 3) 取回结果
$SSH "cd $M && tar czf mp_results.tgz results scripts/*.py scripts/run_all.sh logs"
scp -P 40351 root@connect.westd.seetacloud.com:$M/mp_results.tgz C:\Users\ASUS\mem_pilot_stage\mp_results.tgz   # 然后 CopyToBox
