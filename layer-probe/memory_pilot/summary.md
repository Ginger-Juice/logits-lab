# 先导实验：把 hidden states 当记忆（Qwen3.8-27B-FP8，pipeline_v1 的已有特征，纯 CPU）

## 跑了什么
- 数据：40 道 GSM8K 题 × 4 个条件（off/low/medium/xhigh），复用 pipeline_v1 的 features_*.npz（65 个 index × 5120 维；index 0 是 embedding）和 generations.jsonl。按 qid 排序对齐，4 个条件的 qid 集合完全相同
- 位置：last_prompt、think_start、think_end、answer_start、last_token、think_mean（think_samples 没用）
- 全注意力层：特征 index 4,8,…,64，共 16 层，已对照 config 的 layer_types 确认。其余 48 层是 GDN。**GDN 的递归状态（linear-attn recurrent state）当时没有采集，本实验只看了 residual stream 的 hidden states**
- 1）跨条件检索：用条件 A 下题目 q 的向量做 query，记忆库是条件 B 下的全部 40 题，用余弦相似度。归一化有三种：raw、每个库各自去均值（center）、每个库各自 z-score。统计 top-1/top-5/MRR，覆盖全部 12 个有向条件对；summary 里把两个方向取平均。**并列名次一律按最差算**：比如 embedding 层在模板 token 位置上所有题的向量都一样，这时 top-1 记为 0，不会被误算成 1
  - 额外做了跨位置检索：query 用 A 的 last_prompt，库用 B 的 last_token / think_mean / answer_start
  - 表层基线：prompt / output / prompt+output 的 token 计数向量（BoT），也用 raw/center/zscore/tfidf 四种归一化，计算方式和上面一样
- 2）内容保留：160 个样本混在一起（40 qid × 4 条件），用 GroupKFold(5) 按 qid 分组。模型是 StandardScaler + RidgeCV（alpha 取 1e0 到 1e6）。目标有三个：
  - log10(|gold|)，看 R²
  - 题目里数字的个数，看 R²
  - 题目字符长度三分位（3 类，各 13/13/14 题），看准确率
  - 对照组：在 qid 层面打乱标签，做 5 次置换；以及 BoT 特征上的 ridge
- 3）没有做文本 embedding 基线（要下载模型）。只做了 BoT 基线
- 结果都在 results/ 下：retrieval_all.csv、retrieval_summary.csv、retrieval_bot.csv、retrieval_crosspos*.csv、decoding_all.csv、decoding_summary.csv，以及 4 张检索图和 1 张解码图。图里灰色竖线和方块标的是全注意力层

## 关键数字（全部来自 results/*.csv）
### 检索（同一题跨条件，库大小 40，随机猜 top-1 = 0.025）
- **表层基线已经到顶**：BoT prompt 在 12 个有向条件对上、4 种归一化下 top-1 都是 1.000。BoT output（只看生成的 token）用 zscore 也全是 1.000；raw 下 off→xhigh 是 0.875，center 下是 0.700。prompt token 集合的 Jaccard 均值在 0.68 到 0.96 之间（off↔medium 0.965，因为 medium 不插 system 消息）
- last_prompt / think_start（这两个位置是同一个模板 token，所以 embedding 层 top-1=0）：去均值或 z-score 之后，从第 1 层起各条件对 top-1 基本都是 1.0。zscore 下 off↔xhigh 在 64 层里有 35 层 top-1=1；低效层集中在 40 到 60 附近，最低约 0.7 到 0.9。raw 余弦明显更差，例如 last_prompt off↔xhigh 的 raw 平均 top-1 只有 0.16（全注意力层）/0.13（GDN），最好的一层是 L52 的 0.425。也就是说，**模板带来的共同方向会淹没内容信号，必须做去均值或 z-score**
- think_end（zscore）：off↔xhigh 最好的层是 L52，top-1 1.000（n=38）；low↔xhigh 最好的是 L50，0.987；off↔low 最好的是 L52，0.912。这个位置在中间层明显变差（层均值只有 0.53 到 0.80），到 50 层以后才恢复
- answer_start（zscore）：off↔xhigh 最好的是 L37，0.776（n=38）；low↔xhigh 最好的是 L64，0.868；low↔medium 最好的是 L48，0.925
- last_token（答案结尾，zscore）：off↔xhigh 最好的是 L4，0.875（L4 是全注意力层）；low↔xhigh 最好的是 L3，0.675；medium↔xhigh 最好的是 L13，0.625。在这个位置，hidden states **低于** BoT output zscore 的 1.000
- think_mean（只有 low/medium/xhigh 有，zscore）：embedding 层就已经有 0.96 到 1.0（等于是 token embedding 的平均，本质上也是一种 bag-of-tokens），各层基本都是 1.0
- 跨位置（zscore）：query 用 last_prompt，库用其他条件的 think_mean，最好的层 top-1 在 0.95 到 1.00；库用 last_token，最好的层只有 0.70 到 0.85
- 全注意力层和 GDN 层：每个设置下两类层的平均 top-1 只差几个百分点。全注意力层略高的情况居多（center/zscore 共 66 个设置，51 个全注意力更高，平均高 0.015，最多高 0.083，answer_start 上差得最多，平均 +0.044），但 think_start 上反过来，GDN 略高（例如 zscore off↔xhigh 是 0.984 对 0.992）。没有看到稳定的层类型差异

### 内容保留（解码，n=160，GroupKFold 按 qid）
- log10(gold)：last_token 在 L21 的 R² 是 **0.780**。同一层打乱标签后 R² 均值 −0.536；所有层里打乱标签的最大值是 0.020；BoT prompt 是 0.027。不过 last_token 位于生成的答案（包括 \boxed{}）之后，它更像是在读出模型自己写出来的答案，不能说明它记住了题目本身。last_prompt 上 log10(gold) 最好只有 0.110（L46），而打乱标签在各层的最大值就有 0.357，所以**看不出比噪声好**
- 数字个数：last_prompt 最好的是 L37，R² 0.328；think_mean L20 是 0.370；think_start L7 是 0.308。对照组：打乱标签在各层的最大值约 0.03 到 0.14，BoT prompt 是 0.244。hidden states 只比 BoT 高一点
- 长度三分类：think_start L35 准确率 0.712，last_prompt L33 0.688；embedding 层约 0.15 到 0.18；打乱标签的均值约 0.33，各层最大值约 0.48 到 0.53；BoT prompt ridge 是 0.350（高维稀疏特征加上 n 很小，ridge 没学出东西，所以这是个弱基线）
- 全注意力层和 GDN 层：解码 R² 或准确率的层均值基本一样，例如 last_token log10_gold 是 0.616 对 0.620

## 结论（谨慎说）
1. 在这批数据上，“跨推理强度检索同一道题”是一个**饱和任务**：只看 prompt 的 token 计数就有 100% top-1，hidden states（去均值或 z-score 后）同样接近 100%，**没有超过表层基线**。所以还不能说明 hidden states 带来了文本表层以外的记忆价值
2. 有一个实用结论：存 hidden states 当记忆时，**每个库要做去均值或 z-score**。raw 余弦会被模板或位置带来的共同方向主导
3. 生成结束位置（last_token）的向量里能很好地线性读出答案的数量级（R² 0.78）。但 prompt 末尾读不出答案的数量级，说明在 prompt 阶段，“答案”还没有被线性编码进去（n=40，结论很弱）
4. 全注意力层和 GDN 层在这两项上没有可靠差异。GDN 的递归状态没采集，这部分没有测

## 局限
- n=40 题，只有 GSM8K；检索库只有 40 个，top-1 的分辨率是 1/40，置信区间很宽
- 同一道题在不同条件下的 prompt 只有模板不同，跨条件检索本质上很大一部分是表层重叠，BoT 已经 100%
- 没有做改写或语义检索，没有文本 embedding 基线
- 只用了线性 ridge 和单一的 GroupKFold 切分；打乱标签只做了 5 次，有的设置下打乱后 R² 波动很大
- 没有覆盖 linear-attention 的递归状态和 KV 缓存
- 跨条件检索的层选择是在测试结果上挑最好的一层（best layer），属于事后挑选，有乐观偏差

## 建议的下一步
- 做一个**真正的语义检索任务**：给同一道 GSM8K 题生成改写版，或者换数字、换实体的变体，再找与它解法结构相同的其他题当正例（例如 GSM8K 里同类题、MATH 的同一 subtype）。在这个任务上和 BoT/TF-IDF 以及一个小的文本 embedding 模型（<1GB）比较；检索前先在开发集上选好层和归一化方式，避免事后挑选
- “可复用”的下游验证：把检索到的 hidden state 或推理文本作为提示注入到新题里，看准确率或 think token 数有没有变化。这一步需要 GPU，需要你批准后才能做
- 补采 GDN 的递归状态（每层一个固定大小的矩阵），作为一种“压缩记忆”，与全注意力层的 hidden states 对比
