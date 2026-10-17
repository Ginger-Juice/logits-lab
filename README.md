# logits-lab

Experiments on whether a Qwen hidden state can be stored and later recalled as a memory of the current question.

- `local-lab/` — RTX 3080 smoke pipeline on Qwen3.5-2B (generate, capture layer states, CKA / probes).
- `local-lab/exp_min/recall_mem.py` — recall test: subtract a neutral prompt, inject the difference into a prompt that does not contain the question, and score whether the original answer comes back.
- `layer-probe/` — notes for the larger Qwen3.8-27B capture pipeline.
