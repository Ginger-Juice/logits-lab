# local-lab：本地 RTX 3080 上的 Qwen3.5 隐状态最小实验（冒烟测试）

> 目的：在本地 Windows PC（RTX 3080 10GB）上跑通「thinking 开/关 → 生成 → teacher-forced 前向抓取隐状态 → CKA / 线性探针」的完整流水线。
> **n = 20 道题，仅为流水线冒烟测试，任何数值都不具备统计意义。**

## 目录结构

```
D:\knowleen\local-lab\
├─ .venv\                      # 独立虚拟环境（Python 3.12，不改动系统 Python / 驱动）
├─ models\Qwen3.5-2B\          # 从 ModelScope 下载的模型（Qwen/Qwen3.5-2B, master）
│   └─ model_revision.json     # 下载时记录的 ModelScope 文件 revision / sha256 前缀
├─ setup_env.ps1               # 建 venv + 安装依赖的脚本
├─ download_model.ps1          # 用 curl 从 ModelScope 下载模型文件
├─ logs\                       # 安装、下载、实验运行日志
└─ exp_min\
    ├─ run_min.py              # 实验主脚本（固定 seed=1234，贪心解码）
    ├─ questions.json          # 20 道题（10 道算术应用题 + 10 道常识题，带标准答案）
    └─ outputs\                # 实验输出（见下）
```

## 环境搭建步骤（已执行，可复现）

1. 检查：`nvidia-smi`（RTX 3080 10GB，驱动 528.49 / CUDA 12.0）、`py -0p`（使用 3.12）、`git --version`、D 盘剩余空间（需 ≥ 15 GB）。
2. 建 venv 并安装依赖：
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File D:\knowleen\local-lab\setup_env.ps1
   ```
   - **torch 最终版本：2.7.1+cu118**（CUDA 11.8 运行时，驱动 528.49 原生支持）。
     - 先装的 torch 2.14.1+cu126（官方源）能 import、`is_available()=True`，但任何显存分配都报
       `cudaErrorDevicesUnavailable`（CUDA-capable device(s) is/are busy or unavailable）；用 ctypes 直接调驱动 API（cuCtxCreate/cuMemAlloc）正常，
       判断是 CUDA 12.6 运行时与 12.0 驱动不兼容。没有升级驱动，而是在同一 venv 内换成 cu118 轮子：
       ```powershell
       powershell -NoProfile -ExecutionPolicy Bypass -File D:\knowleen\local-lab\fix_torch_and_run.ps1
       ```
       该脚本从上海交大镜像 `https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/` 下载 whl（实测约 6.5 MB/s；官方源约 1.5 MB/s，阿里云约 0.2–0.4 MB/s），本地 pip 安装后直接跑实验。
       whl 保存在 `D:\knowleen\local-lab\wheels\`。
   - 其他包走清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`：transformers==5.12.1（与服务器一致）、accelerate 1.15.0、modelscope 1.40.2、safetensors 0.8.0、numpy 2.5.4、scikit-learn 1.9.1、matplotlib 3.11.2、pandas 3.0.6。完整列表见 `logs\pip_freeze.txt`。
   - flash-linear-attention 0.5.2 能 pip 装上，但它依赖 triton（Windows 上没有），导致 `Qwen3_5ForCausalLM` 导入失败，**已卸载**；causal-conv1d 未安装。
     transformers 使用纯 torch 实现的 GatedDeltaNet（`is_fast_path_available=False`）。
3. 下载模型（ModelScope，无需 HF）：
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File D:\knowleen\local-lab\download_model.ps1
   # 或： .venv\Scripts\modelscope download --model Qwen/Qwen3.5-2B --local_dir D:\knowleen\local-lab\models\Qwen3.5-2B
   ```

## 重新运行实验

```powershell
cd D:\knowleen\local-lab\exp_min
..\.venv\Scripts\python.exe run_min.py                       # 全流程：sanity,gen,feat,analysis
..\.venv\Scripts\python.exe run_min.py --stages analysis     # 只基于已保存特征重跑分析
..\.venv\Scripts\python.exe run_min.py --stages feat,analysis --hash_weights 0
```
可选参数：`--model`、`--out`、`--questions`、`--max_new_on 1024`、`--max_new_off 256`。

## 实验流程

1. **sanity**：加载模型（`AutoModelForCausalLM` → `Qwen3_5ForCausalLM`，只加载文本部分，bf16，cuda；视觉塔与 MTP 权重被忽略），记录加载后显存，用一题分别测试 thinking 开 / 关的生成；计算权重 sha256。
2. **gen**：20 题 × {thinking ON（`enable_thinking=True`，max_new_tokens=1024），thinking OFF（256）}，贪心解码；保存 prompt/生成 token id、文本、是否答对。
   - 答对判定（简单匹配）：数字答案 = 答案文本中抽出的任一数字等于标准答案；文字答案 = 大小写不敏感子串匹配。ON 条件只看 `</think>` 之后的文本；若在 1024 token 内没有输出 `</think>`，记为截断且判错。
3. **feat**：对「prompt + 生成内容（去掉末尾 EOS）」做一次 teacher-forced 前向，用 forward hook 抓 embedding 输出 + 每个 decoder layer 输出（最终 RMSNorm 之前），在以下位置拷到 CPU fp32：
   - `last_prompt`：prompt 最后一个 token
   - `think_mean`：thinking 段（`<think>` 之后到 `</think>` 之前）的平均（仅 ON）
   - `answer_start`：答案第一个非空白 token（ON 为 `</think>` 之后）
   - `last_token`：最后一个生成内容 token
4. **analysis**：
   - 每层 ON vs OFF 的 linear CKA 与 debiased CKA（位置：last_prompt / answer_start / last_token）；
   - ON 条件下 layer × layer linear CKA 热图（last_prompt、answer_start）；
   - 每层逻辑回归探针（StandardScaler + LogisticRegression），区分 ON/OFF，按题目分组的 5 折 GroupKFold，并给出 20 次标签打乱的基线。

## 输出文件（`exp_min\outputs\`）

| 文件 | 内容 |
|---|---|
| `metadata.json` | 版本、GPU/驱动、模型路径与 revision/sha256、layer_types、加载信息、各阶段耗时与峰值显存、生成统计 |
| `generations.jsonl` | 每题每条件一行：prompt、prompt_ids、gen_ids、文本、thinking 段范围、答案起点、是否答对、耗时 |
| `features_on.npz` / `features_off.npz` | `[20, 层数+1, hidden]` fp32 数组（各位置），以及 valid_* 掩码、index_* 位置下标、seq_len |
| `analysis\cka_on_vs_off_per_layer.csv/.png` | 每层 ON vs OFF CKA |
| `analysis\cka_layer_x_layer_on_*.csv`、`cka_layer_x_layer_on.png` | ON 条件 layer×layer CKA |
| `analysis\probe_on_vs_off_per_layer.csv/.png` | 探针准确率与打乱基线 |
| `analysis\analysis_summary.json` | 以上结果汇总 |
| `D:\knowleen\local-lab\logs\run_min.log` | 运行日志（另有 setup_env.log、fix_torch_and_run.log、download_model.log、pip_freeze.txt） |

## 注意事项 / 局限

- **n=20，只是流水线冒烟测试**，CKA 在 n 这么小时偏差很大（debiased CKA 方差也大），探针结果不能解读为科学结论。
- `last_prompt` 位置 ON/OFF 的输入本身不同：ON 的 prompt 以 `<think>\n` 结尾，OFF 的 prompt 以 `<think>\n\n</think>\n\n` 结尾（Qwen3.5 chat template 的行为），所以该位置探针接近 100% 是平凡的（输入 token 不同），不代表"隐状态编码了思考模式"。
- answer_start 位置的上下文也不同（ON 前面有整段思考），差异混杂了长度/内容效应。
- 贪心解码下 thinking 模式可能出现重复、在 1024 token 内未结束；这些题 answer_start 无效，会从对应分析中剔除（metadata 中有计数）。官方推荐 thinking 模式用采样（temperature=1.0, top_p=0.95 等），这里为了可复现用贪心。
- 未安装 flash-linear-attention 时 GatedDeltaNet 走 torch 回退实现，速度较慢；数值与 fast path 可能有微小差异。
- 驱动 528.49（CUDA 12.0）：cu126 版 torch 在这台机器上无法分配显存，所以用 cu118。若以后升级驱动（≥ 560），可换回与服务器一致的新版 torch（本实验未改动驱动）。
- 隐状态取自 decoder layer 输出（未经过最终 norm）；embedding 层为第 0 层。
- last_prompt 位置：同一条件下 20 道题的最后一个 token 相同（ON 都是 `<think>\n` 里的 `\n`，OFF 都是 `</think>\n\n`），所以 embedding 层（第 0 层）在该位置各题完全一样，方差为 0，那一层的 CKA 没有意义（linear 只是数值噪声，debiased 为 NaN）。

## 本次运行结果（2026-10-17 23:04–23:15，UTC+8；n=20，仅冒烟测试）

- 模型：ModelScope `Qwen/Qwen3.5-2B`（master；权重文件 revision 5bc562c7…，sha256 aa33250c4fc6… 与 API 一致），model_type `qwen3_5`（文本部分 `qwen3_5_text`），24 层，hidden 2048，
  layer_types = 每 4 层一组 [linear, linear, linear, full] × 6（第 4/8/…/24 层为 full_attention）；以 `Qwen3_5ForCausalLM` 加载，missing/unexpected keys = 0，参数量 1,881,825,088（仅文本部分）。
- 显存：加载后 3.505 GB；各阶段峰值（max_memory_allocated）：load 3.506 / sanity 3.533 / gen 3.574 / feat 3.653 GB。
- 耗时：load 5.7 s，sanity 9.2 s，gen 501.4 s（ON 426.6 s，OFF 74.7 s），feat 11.7 s，analysis 107.8 s。
- sanity（"What is 17 + 25?"）：ON 生成 88 token（思考 83 token），答 42；OFF 生成 3 token，答 42。
- 生成（贪心）：OFF 20/20 答对；ON 13/20 答对，其余 7 题（a01, a04, a08, a09, f03, f05, f10）在 1024 token 内未输出 `</think>`（截断，判错）；完成的 13 题全部答对。ON 平均生成 627 token。
- CKA（ON vs OFF）：last_prompt 第 1–24 层 linear CKA 0.94–0.99；answer_start（n=13）0.54–0.83，越深越高；last_token 0.23–0.65，随深度上升。
- 探针（GroupKFold 5 折，准确率 / 20 次打乱基线）：last_prompt 所有层 1.000（平凡，输入 token 不同），基线 0.388–0.471；
  answer_start（n=13 题）emb 0.650，第 1–2 层 0.933/0.967，第 3、5–15 层 1.000（第 4 层 0.967），第 16–24 层 0.883–0.967，基线 0.413–0.486；
  last_token emb 0.800，第 1–24 层 0.925–1.000，基线 0.457–0.499。
