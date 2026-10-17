"""Summaries of Stage 1 generations: per-condition accuracy, thinking token stats, truncation."""
import json, sys, os, numpy as np, pandas as pd
OUT = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/layer-probe/pipeline_v1/outputs'
R = [json.loads(l) for l in open(os.path.join(OUT, 'generations.jsonl'))]
df = pd.DataFrame([{k: r[k] for k in ['cond', 'qid', 'gold', 'parsed', 'correct', 'boxed', 'finish_reason', 'n_prompt', 'n_out', 'think_tokens', 'answer_tokens', 'truncated_in_think', 'seconds', 'max_new_tokens']} for r in R])
order = ['off', 'low', 'medium', 'xhigh']
g = df.groupby('cond')
s = pd.DataFrame(dict(n=g.size(), acc=g.correct.mean(), n_correct=g.correct.sum(), boxed_rate=g.boxed.mean(), n_finish_length=g.finish_reason.apply(lambda x: (x == 'length').sum()),
                      n_trunc_in_think=g.truncated_in_think.sum(), prompt_mean=g.n_prompt.mean(), out_mean=g.n_out.mean(), out_max=g.n_out.max(),
                      think_mean=g.think_tokens.mean(), think_median=g.think_tokens.median(), think_p90=g.think_tokens.quantile(.9), think_max=g.think_tokens.max(),
                      answer_mean=g.answer_tokens.mean(), sec_mean=g.seconds.mean())).reindex(order).round(3)
s.to_csv(os.path.join(OUT, 'gen_summary.csv')); df.to_csv(os.path.join(OUT, 'gen_per_sample.csv'), index=False)
print(s.to_string())
w = df.pivot(index='qid', columns='cond', values='correct')[order]
print('\nquestions wrong in any cond:\n', w[~w.all(1)].to_string())
