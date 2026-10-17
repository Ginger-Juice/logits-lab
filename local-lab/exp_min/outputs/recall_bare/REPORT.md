# Recall sweep

Question style: `bare`. Carrier, thinking off: `Reply with only the final answer.`

Each stored question is the original wording, with no extra instruction. The carrier is still only “Reply with only the final answer.”
Each trial adds `alpha * (h_question - h_carrier)` at one site during prefill, then greedy-decodes up to 120 new tokens. A number counts only when it is the whole reply. A short string such as Au counts only when it is the whole reply. A longer name may appear inside the reply. `intrusion` applies the same test to the next question's answer. `score = recall - intrusion`.

Layer 0 is the embedding output. Layers 1–24 are decoder blocks. Full-attention blocks are 4, 8, 12, 16, 20, 24.

Highest score in this run: **user_end / alpha 1.0 / layer 24** (recall 0.0, intrusion 0.0, score 0.0, n=20).

## Best-layer replies

- a01 miss: Please provide the question you would like me to answer.<|im_end|>
- a02 miss: Please provide the question you would like me to answer.<|im_end|>
- a03 miss: Please provide the question you would like me to answer.<|im_end|>
- a04 miss: Please provide the question you would like me to answer.<|im_end|>
- a05 miss: Please provide the question you would like me to answer.<|im_end|>
- a06 miss: Please provide the question you would like me to answer.<|im_end|>
- a07 miss: Please provide the question you would like me to answer.<|im_end|>
- a08 miss: Please provide the question you would like me to answer.<|im_end|>
- a09 miss: Please provide the question you would like me to answer.<|im_end|>
- a10 miss: Please provide the question you would like me to answer.<|im_end|>
- f01 miss: Please provide the question you would like me to answer.<|im_end|>
- f02 miss: Please provide the question you would like me to answer.<|im_end|>
- f03 miss: Please provide the question you would like me to answer.<|im_end|>
- f04 miss: Please provide the question you would like me to answer.<|im_end|>
- f05 miss: Please provide the question you would like me to answer.<|im_end|>
- f06 miss: Please provide the question you would like me to answer.<|im_end|>
- f07 miss: Please provide the question you would like me to answer.<|im_end|>
- f08 miss: Please provide the question you would like me to answer.<|im_end|>
- f09 miss: Please provide the question you would like me to answer.<|im_end|>
- f10 miss: Please provide the question you would like me to answer.<|im_end|>

## Ranking

| direction | alpha | layer | n | recall | intrusion | score |
|---|---|---|---|---|---|---|
| user_end | 1.0 | 24 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 24 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 23 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 23 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 22 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 22 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 21 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 21 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 20 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 20 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 19 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 19 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 18 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 18 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 17 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 17 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 16 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 16 | 20 | 0.0 | 0.0 | 0.0 |
| baseline | 0.0 | -1 | 20 | 0.0 | 0.0 | 0.0 |

## Logit lens diagnostic

Median rank of the gold answer's first token (1 = top of the vocabulary).
`last` is the position that predicts the first generated token. `user_end` predicts the token after the question, which is still inside the prompt.

| layer | user_end residual | last residual | last delta |
|---|---|---|---|
| 0 | 47 | 49 | 1 |
| 1 | 224192 | 245859 | 217335 |
| 2 | 245287 | 186294 | 215563 |
| 3 | 231325 | 218396 | 234833 |
| 4 | 242284 | 132244 | 113486 |
| 5 | 194570 | 141280 | 148296 |
| 6 | 233657 | 142483 | 107467 |
| 7 | 181632 | 194979 | 81046 |
| 8 | 212047 | 219744 | 225894 |
| 9 | 152193 | 244437 | 190167 |
| 10 | 194249 | 247566 | 208406 |
| 11 | 205638 | 240903 | 197543 |
| 12 | 178041 | 189909 | 123570 |
| 13 | 167553 | 181771 | 27764 |
| 14 | 107569 | 141685 | 11456 |
| 15 | 124660 | 48558 | 55326 |
| 16 | 172398 | 46247 | 66617 |
| 17 | 140828 | 13912 | 66630 |
| 18 | 128425 | 11920 | 45255 |
| 19 | 55568 | 9425 | 34518 |
| 20 | 64731 | 18555 | 49702 |
| 21 | 132621 | 8065 | 17281 |
| 22 | 134043 | 13052 | 21188 |
| 23 | 22706 | 3042 | 87221 |
| 24 | 700 | 186 | 185677 |

A low rank here means the answer token is already readable before injection. If that layer also wins the recall score, the memory is acting as an answer cache. A high rank together with a high recall score means the answer showed up only after the vector was injected into the carrier.

n = 20 questions. This ranks the pipeline; it is not a claim about the layer's function in general.
