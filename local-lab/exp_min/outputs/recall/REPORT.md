# Recall sweep

Question style: `wrapped`. Carrier, thinking off: `Reply with only the final answer.`

Each stored question ends with “Reply with only the final answer.” The carrier is only that sentence, so the question text is absent at recall time.
Each trial adds `alpha * (h_question - h_carrier)` at one site during prefill, then greedy-decodes up to 24 new tokens. A number counts only when it is the whole reply. A short string such as Au counts only when it is the whole reply. A longer name may appear inside the reply. `intrusion` applies the same test to the next question's answer. `score = recall - intrusion`.

Layer 0 is the embedding output. Layers 1–24 are decoder blocks. Full-attention blocks are 4, 8, 12, 16, 20, 24.

Highest score in this run: **last / alpha 2.0 / layer 20** (recall 0.4, intrusion 0.0, score 0.4, n=20).

## Best-layer replies

- a01 miss:  1<|im_end|>
- a02 miss: To provide a specific reply, please provide the question or topic you would like me to address.<|im_end|>
- a03 miss: To provide a specific reply, please provide the question or topic you would like me to address.<|im_end|>
- a04 miss: To provide a specific reply, please provide the question or topic you would like me to address.<|im_end|>
- a05 miss: If you provide the specific question or problem you need help with, I will be happy to give you the final answer.
- a06 miss: 1<|im_end|>
- a07 miss: To provide a specific reply, please provide the question or topic you would like me to address.<|im_end|>
- a08 miss: First, I need to know the specific question you are asking. Please provide the details of your query so I can give
- a09 miss: First, let's clarify the premise: the statement "The Earth is flat" is a factually incorrect claim. The
- a10 miss: To provide a specific reply, please provide the question or topic you would like me to address.<|im_end|>
- f01 hit:  Paris<|im_end|>
- f02 hit:  Au<|im_end|>
- f03 hit: 7<|im_end|>
- f04 hit:  Mars<|im_end|>
- f05 hit:  William Shakespeare<|im_end|>
- f06 miss:   None provided.<|im_end|>
- f07 hit: 100<|im_end|>
- f08 hit:  Tokyo<|im_end|>
- f09 miss:  Carbon-14 dating<|im_end|>
- f10 hit: 6<|im_end|>

## Ranking

| direction | alpha | layer | n | recall | intrusion | score |
|---|---|---|---|---|---|---|
| last | 2.0 | 20 | 20 | 0.4 | 0.0 | 0.4 |
| last | 2.0 | 21 | 20 | 0.4 | 0.05 | 0.35 |
| last | 2.0 | 22 | 20 | 0.35 | 0.0 | 0.35 |
| last | 2.0 | 19 | 20 | 0.35 | 0.0 | 0.35 |
| last | 2.0 | 18 | 20 | 0.35 | 0.05 | 0.3 |
| last | 2.0 | 23 | 20 | 0.3 | 0.0 | 0.3 |
| last | 1.0 | 22 | 20 | 0.3 | 0.0 | 0.3 |
| last | 1.0 | 21 | 20 | 0.3 | 0.0 | 0.3 |
| last | 1.0 | 20 | 20 | 0.3 | 0.0 | 0.3 |
| last | 1.0 | 19 | 20 | 0.3 | 0.0 | 0.3 |
| last | 1.0 | 18 | 20 | 0.3 | 0.0 | 0.3 |
| last | 1.0 | 24 | 20 | 0.25 | 0.0 | 0.25 |
| last | 2.0 | 24 | 20 | 0.25 | 0.0 | 0.25 |
| last | 1.0 | 23 | 20 | 0.25 | 0.0 | 0.25 |
| last | 1.0 | 17 | 20 | 0.25 | 0.0 | 0.25 |
| last | 2.0 | 16 | 20 | 0.2 | 0.0 | 0.2 |
| last | 2.0 | 17 | 20 | 0.2 | 0.05 | 0.15 |
| last | 0.5 | 24 | 20 | 0.15 | 0.0 | 0.15 |
| last | 0.5 | 23 | 20 | 0.15 | 0.0 | 0.15 |
| last | 0.5 | 20 | 20 | 0.1 | 0.0 | 0.1 |
| last | 0.5 | 21 | 20 | 0.05 | 0.0 | 0.05 |
| last | 0.5 | 19 | 20 | 0.05 | 0.0 | 0.05 |
| last | 0.5 | 18 | 20 | 0.05 | 0.0 | 0.05 |
| last | 0.5 | 17 | 20 | 0.05 | 0.0 | 0.05 |
| last | 1.0 | 16 | 20 | 0.05 | 0.0 | 0.05 |
| user_end | 0.5 | 24 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 24 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 24 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 24 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 24 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 24 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 23 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 23 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 23 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 23 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 23 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 23 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 22 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 22 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 22 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 22 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 22 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 22 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 22 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 21 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 21 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 21 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 21 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 21 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 21 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 20 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 20 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 20 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 20 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 20 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 20 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 19 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 19 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 19 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 19 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 19 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 19 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 18 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 18 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 18 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 18 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 18 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 18 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 17 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 17 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 17 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 17 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 17 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 17 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 16 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 16 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 16 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 16 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 16 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 16 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 16 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 15 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 15 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 15 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 15 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 15 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 15 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 15 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 15 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 15 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 14 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 14 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 14 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 14 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 14 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 14 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 14 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 14 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 14 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 13 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 13 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 13 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 13 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 13 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 13 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 13 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 13 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 13 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 12 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 12 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 12 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 12 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 12 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 12 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 12 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 12 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 12 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 11 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 11 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 11 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 11 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 11 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 11 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 11 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 11 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 11 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 10 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 10 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 10 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 10 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 10 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 10 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 10 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 10 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 10 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 9 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 9 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 9 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 9 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 9 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 9 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 9 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 9 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 9 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 8 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 8 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 8 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 8 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 8 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 8 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 8 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 8 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 8 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 7 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 7 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 7 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 7 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 7 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 7 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 7 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 7 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 7 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 6 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 6 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 6 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 6 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 6 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 6 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 6 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 6 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 6 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 5 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 5 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 5 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 5 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 5 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 5 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 5 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 5 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 5 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 4 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 4 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 4 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 4 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 4 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 4 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 4 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 4 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 4 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 3 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 3 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 3 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 3 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 3 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 3 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 3 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 3 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 3 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 2 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 2 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 2 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 2 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 2 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 2 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 2 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 2 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 2 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 1 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 1 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 1 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 1 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 1 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 1 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 1 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 1 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 1 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 0.5 | 0 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 1.0 | 0 | 20 | 0.0 | 0.0 | 0.0 |
| user_end | 2.0 | 0 | 20 | 0.0 | 0.0 | 0.0 |
| last | 1.0 | 0 | 20 | 0.0 | 0.0 | 0.0 |
| user_mean | 1.0 | 0 | 20 | 0.0 | 0.0 | 0.0 |
| all_pos | 1.0 | 0 | 20 | 0.0 | 0.0 | 0.0 |
| last | 0.5 | 0 | 20 | 0.0 | 0.0 | 0.0 |
| last | 2.0 | 0 | 20 | 0.0 | 0.0 | 0.0 |
| mean_at_last | 1.0 | 0 | 20 | 0.0 | 0.0 | 0.0 |
| baseline | 0.0 | -1 | 20 | 0.0 | 0.0 | 0.0 |

## Logit lens diagnostic

Median rank of the gold answer's first token (1 = top of the vocabulary).
`last` is the position that predicts the first generated token. `user_end` predicts the token after the question, which is still inside the prompt.

| layer | user_end residual | last residual | last delta |
|---|---|---|---|
| 0 | 51 | 49 | 1 |
| 1 | 68534 | 245121 | 136038 |
| 2 | 41106 | 181216 | 207941 |
| 3 | 24032 | 211671 | 231104 |
| 4 | 111354 | 149054 | 82951 |
| 5 | 134877 | 171271 | 213490 |
| 6 | 149516 | 201385 | 170286 |
| 7 | 183629 | 225041 | 197311 |
| 8 | 138824 | 222319 | 229899 |
| 9 | 83281 | 242258 | 193901 |
| 10 | 71712 | 247345 | 221948 |
| 11 | 123051 | 223227 | 145746 |
| 12 | 81948 | 194670 | 98994 |
| 13 | 102668 | 113864 | 2793 |
| 14 | 98662 | 103728 | 2032 |
| 15 | 134247 | 23419 | 6699 |
| 16 | 129046 | 38429 | 80338 |
| 17 | 31955 | 3037 | 33997 |
| 18 | 64509 | 1459 | 11233 |
| 19 | 22732 | 293 | 5410 |
| 20 | 89701 | 256 | 1641 |
| 21 | 156732 | 949 | 2781 |
| 22 | 137175 | 109 | 319 |
| 23 | 29384 | 4 | 45 |
| 24 | 682 | 1 | 5 |

A low rank here means the answer token is already readable before injection. If that layer also wins the recall score, the memory is acting as an answer cache. A high rank together with a high recall score means the answer showed up only after the vector was injected into the carrier.

n = 20 questions. This ranks the pipeline; it is not a claim about the layer's function in general.
