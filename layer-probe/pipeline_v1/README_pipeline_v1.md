# layer-probe / pipeline_v1（从 AutoDL 5090 服务器同步过来的副本）
服务器路径：/root/autodl-tmp/layer-probe/pipeline_v1/scripts/

- `gen.py`：Stage 1，用 SGLang 的 `/generate` 生成，prompt ids 在本地渲染
  - 环境变量：OUT_DIR、N=40、SEED=1234、CONDS=off,low,medium,xhigh、MAXTOT=7000、CONC
- `run_gen.sh`：后台跑 gen.py 的包装脚本。`stop_gen.sh`：按 PID 精确结束 gen.py
- `capture.py`：Stage 2，transformers 5.12.1 加 #47990 补丁，teacher forcing 前向，用 hook 采集 embedding 和 64 层
  - 输出：features_{cond}.npz、capture_meta.json
- `analyze.py`：Stage 3，内容是 probe（按 qid 分组 5 折交叉验证，加打乱标签基线）、CKA、差均值方向余弦，输出 CSV 和 PNG
- `gen_stats.py`：统计每个条件的准确率、thinking 长度、截断数
- `run_after_gen.sh`：等生成结束，停掉 SGLang，再依次跑 capture 和 analyze
- 服务器上的启动和停止脚本在 /root/autodl-tmp/snape-ctx/launch.sh 和 stop.sh
- 本次用的 SGLang 参数见 commands.sh
