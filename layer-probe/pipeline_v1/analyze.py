"""Stage 3: smoke analysis (PIPELINE CHECK ONLY, tiny n). Reads features_{cond}.npz, writes CSV+PNG to OUT_DIR/analysis.
Needs numpy, sklearn, pandas, matplotlib."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from joblib import Parallel, delayed
NJ = int(os.environ.get('NJ', 32)); NSHUF = int(os.environ.get('NSHUF', 3))
OUT = os.environ.get('OUT_DIR', '/root/autodl-tmp/layer-probe/pipeline_v1/outputs'); AD = os.path.join(OUT, 'analysis'); os.makedirs(AD, exist_ok=True)
CONDS = ['off', 'low', 'medium', 'xhigh']; EFF = ['low', 'medium', 'xhigh']
F = {c: np.load(os.path.join(OUT, f'features_{c}.npz')) for c in CONDS if os.path.exists(os.path.join(OUT, f'features_{c}.npz'))}
NL = F['off']['last_prompt'].shape[1]
FULL = [i for i in range(NL - 1) if i % 4 == 3]          # decoder layer ids with full attention (3,7,...,63) -> feature idx i+1
rng = np.random.default_rng(1234)
def mark(ax):
    for i in FULL: ax.axvline(i + 1, color='grey', alpha=.18, lw=.8)
    ax.set_xlabel('feature index (0=embedding, i=decoder layer i-1); grey = full-attention layers')

def probe(X, y, g, seed=0, n_shuf=None):
    n_shuf = n_shuf or NSHUF; rng = np.random.default_rng(1234 + seed)
    """grouped 5-fold CV accuracy per layer + shuffled-label baseline (labels permuted within... globally)"""
    k = min(5, len(np.unique(g)))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=1234) if len(np.unique(y)) > 1 else None
    def score(yy):
        accs = []
        for tr, te in cv.split(X, yy, g):
            if len(np.unique(yy[tr])) < 2: continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)); m.fit(X[tr], yy[tr]); accs.append((m.predict(X[te]) == yy[te]).mean())
        return float(np.mean(accs)) if accs else np.nan
    a = score(y); s = np.nanmean([score(rng.permutation(y)) for _ in range(n_shuf)])
    return a, s

rows = []
# 1) thinking on vs off probe at last_prompt and answer_start. on = each effort cond vs off (paired by qid, group=qid)
for pos in ['last_prompt', 'answer_start']:
    for on_c in EFF + ['all_on']:
        cs = EFF if on_c == 'all_on' else [on_c]
        Xs, ys, gs = [], [], []
        for c, lab in [('off', 0)] + [(c, 1) for c in cs]:
            v = F[c][f'index_{pos}'] >= 0
            Xs.append(F[c][pos][v]); ys.append(np.full(v.sum(), lab)); gs.append(F[c]['qids'][v])
        X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
        res = Parallel(n_jobs=NJ)(delayed(probe)(X[:, l], y, g, l) for l in range(NL))
        for l, (a, s) in enumerate(res):
            rows.append(dict(task=f'on_vs_off[{on_c}]', pos=pos, layer=l, acc=a, shuffled=s, n=len(y), n_pos=int(y.sum())))
        print('probe on/off', pos, on_c, 'n', len(y), flush=True)
# 2) correctness probe (if balance allows): pooled over think conditions at answer_start and last_prompt
corr_note = {}
for pos in ['answer_start', 'last_prompt']:
    Xs, ys, gs = [], [], []
    for c in CONDS:
        v = F[c][f'index_{pos}'] >= 0
        Xs.append(F[c][pos][v]); ys.append(F[c]['correct'][v].astype(int)); gs.append(F[c]['qids'][v])
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    nmin = int(min(y.sum(), len(y) - y.sum())); corr_note[pos] = dict(n=len(y), n_correct=int(y.sum()), n_minority=nmin)
    if nmin >= 5:
        res = Parallel(n_jobs=NJ)(delayed(probe)(X[:, l], y, g, l) for l in range(NL))
        for l, (a, s) in enumerate(res):
            rows.append(dict(task='correctness[pooled]', pos=pos, layer=l, acc=a, shuffled=s, n=len(y), n_pos=int(y.sum())))
pr = pd.DataFrame(rows); pr.to_csv(os.path.join(AD, 'probe_per_layer.csv'), index=False)

# 3) linear CKA between conditions per layer at answer_start (paired by qid)
def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Kx = X @ X.T; Ky = Y @ Y.T
    return float((Kx * Ky).sum() / (np.linalg.norm(Kx) * np.linalg.norm(Ky)))
crow = []
pairs = [(a, b) for i, a in enumerate(CONDS) for b in CONDS[i + 1:]]
for pos in ['answer_start', 'last_prompt']:
    for a, b in pairs:
        qa = {q: i for i, q in enumerate(F[a]['qids']) if F[a][f'index_{pos}'][i] >= 0}
        qb = {q: i for i, q in enumerate(F[b]['qids']) if F[b][f'index_{pos}'][i] >= 0}
        common = sorted(set(qa) & set(qb))
        if len(common) < 5: continue
        ia = [qa[q] for q in common]; ib = [qb[q] for q in common]
        for l in range(NL):
            crow.append(dict(pos=pos, a=a, b=b, layer=l, cka=cka(F[a][pos][ia, l].astype(np.float64), F[b][pos][ib, l].astype(np.float64)), n=len(common)))
ck = pd.DataFrame(crow); ck.to_csv(os.path.join(AD, 'cka_between_conditions.csv'), index=False)

# 4) difference-of-means direction (effort - off) per layer at answer_start; cosine between efforts
drow = []
for pos in ['answer_start', 'last_prompt']:
    dirs = {}
    for c in EFF:
        v = F[c][f'index_{pos}'] >= 0 if pos != 'think_mean' else ~np.isnan(F[c]['think_mean'][:, 0, 0])
        if pos == 'think_mean':
            base = None
        vo = F['off'][f'index_{pos}'] >= 0 if pos != 'think_mean' else None
        if pos == 'think_mean':
            dirs[c] = F[c]['think_mean'][v].mean(0)        # raw mean (no off baseline for thinking span)
        else:
            dirs[c] = F[c][pos][v].mean(0) - F['off'][pos][vo].mean(0)
    if pos == 'think_mean':   # use deviation from the mean over efforts
        mu = np.mean([dirs[c] for c in EFF], 0); dirs = {c: dirs[c] - mu for c in EFF}
    for i, a in enumerate(EFF):
        for b in EFF[i + 1:]:
            for l in range(NL):
                x, y = dirs[a][l], dirs[b][l]
                drow.append(dict(pos=pos, a=a, b=b, layer=l, cos=float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)),
                                 norm_a=float(np.linalg.norm(x)), norm_b=float(np.linalg.norm(y))))
dm = pd.DataFrame(drow); dm.to_csv(os.path.join(AD, 'diffmeans_cosine.csv'), index=False)

# ---- plots
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, pos in zip(axs, ['last_prompt', 'answer_start']):
    for on_c in EFF + ['all_on']:
        d = pr[(pr.task == f'on_vs_off[{on_c}]') & (pr.pos == pos)]; ax.plot(d.layer, d.acc, label=on_c)
    d = pr[(pr.task == 'on_vs_off[all_on]') & (pr.pos == pos)]; ax.plot(d.layer, d.shuffled, 'k--', label='shuffled (all_on)')
    ax.set_title(f'thinking on vs off probe @ {pos} (grouped 5-fold CV)  PIPELINE CHECK, tiny n'); ax.set_ylabel('acc'); ax.legend(fontsize=8); mark(ax)
plt.tight_layout(); plt.savefig(os.path.join(AD, 'probe_on_vs_off.png'), dpi=110); plt.close()
if (pr.task == 'correctness[pooled]').any():
    fig, ax = plt.subplots(figsize=(12, 4))
    for pos in ['answer_start', 'last_prompt']:
        d = pr[(pr.task == 'correctness[pooled]') & (pr.pos == pos)]
        if len(d): ax.plot(d.layer, d.acc, label=f'{pos}'); ax.plot(d.layer, d.shuffled, '--', label=f'{pos} shuffled')
    ax.set_title('correctness probe (pooled conditions) PIPELINE CHECK, tiny n, imbalanced'); ax.legend(fontsize=8); mark(ax)
    plt.tight_layout(); plt.savefig(os.path.join(AD, 'probe_correctness.png'), dpi=110); plt.close()
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, pos in zip(axs, ['answer_start', 'last_prompt']):
    for (a, b), d in ck[ck.pos == pos].groupby(['a', 'b']): ax.plot(d.layer, d.cka, label=f'{a} vs {b}')
    ax.set_title(f'linear CKA between conditions @ {pos} (paired by qid)'); ax.set_ylabel('CKA'); ax.legend(fontsize=8); mark(ax)
plt.tight_layout(); plt.savefig(os.path.join(AD, 'cka_between_conditions.png'), dpi=110); plt.close()
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, pos in zip(axs, ['answer_start', 'last_prompt']):
    for (a, b), d in dm[dm.pos == pos].groupby(['a', 'b']): ax.plot(d.layer, d.cos, label=f'{a} vs {b}')
    ax.set_title(f'cosine of diff-of-means directions @ {pos}' + (' (effort - off)' if pos != 'think_mean' else ' (effort mean - mean over efforts)'))
    ax.legend(fontsize=8); mark(ax)
plt.tight_layout(); plt.savefig(os.path.join(AD, 'diffmeans_cosine.png'), dpi=110); plt.close()

# highlights
hl = {'correctness_balance': corr_note}
for t in pr.task.unique():
    for pos in pr.pos.unique():
        d = pr[(pr.task == t) & (pr.pos == pos)]
        if len(d): b = d.loc[d.acc.idxmax()]; hl[f'{t}@{pos}'] = dict(best_layer=int(b.layer), best_acc=round(b.acc, 3), mean_acc=round(d.acc.mean(), 3), mean_shuffled=round(d.shuffled.mean(), 3), acc_layer0=round(d.acc.iloc[0], 3), n=int(b.n))
for pos in ck.pos.unique():
    for (a, b), d in ck[ck.pos == pos].groupby(['a', 'b']):
        hl[f'cka {a}-{b}@{pos}'] = dict(min=round(d.cka.min(), 3), argmin=int(d.layer[d.cka.idxmin()]), mean=round(d.cka.mean(), 3), last=round(d.cka.iloc[-1], 3))
for pos in dm.pos.unique():
    for (a, b), d in dm[dm.pos == pos].groupby(['a', 'b']):
        hl[f'dm_cos {a}-{b}@{pos}'] = dict(mean=round(d.cos.mean(), 3), min=round(d.cos.min(), 3), at_layer64=round(d.cos.iloc[-1], 3))
json.dump(hl, open(os.path.join(AD, 'highlights.json'), 'w'), indent=1)
print(json.dumps(hl, indent=1)); print('ANALYSIS_DONE')
