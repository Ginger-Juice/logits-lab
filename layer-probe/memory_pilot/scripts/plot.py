import sys; sys.path.append('/root/autodl-tmp/snape-ctx/pylib_plot')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O='/root/autodl-tmp/layer-probe/pipeline_v1/memory_pilot/results/'
FULL=list(range(4,65,4))
R=pd.read_csv(O+'retrieval_all.csv'); D=pd.read_csv(O+'decoding_all.csv'); X=pd.read_csv(O+'retrieval_crosspos.csv')
POS=['last_prompt','think_start','think_end','answer_start','last_token','think_mean']
pairs=[('off','xhigh'),('low','xhigh'),('low','medium'),('off','low')]
def mark(ax):
    for L in FULL: ax.axvline(L,color='0.85',lw=0.8,zorder=0)
for mode in ['center','zscore','raw']:
    fig,axs=plt.subplots(2,3,figsize=(16,8),sharey=True)
    for ax,p in zip(axs.ravel(),POS):
        mark(ax)
        for a,b in pairs:
            h=R[(R.kind=='hidden')&(R.pos==p)&(R.norm==mode)&(R['query']==a)&(R.bank==b)].sort_values('layer')
            if len(h)==0: continue
            l,=ax.plot(h.layer,h.top1,label=f'{a}->{b} (n={int(h.n.iloc[0])})')
            ax.scatter(h.layer[h.full_attn],h.top1[h.full_attn],s=14,color=l.get_color(),marker='s')
        bo=R[(R.kind=='bot_output')&(R.norm=='center')&(R['query']=='off')&(R.bank=='xhigh')].top1.values
        bp=R[(R.kind=='bot_prompt')&(R.norm=='raw')&(R['query']=='off')&(R.bank=='xhigh')].top1.values
        ax.axhline(bp[0],ls='--',color='k',lw=1,label='BoT prompt off->xhigh (raw)')
        ax.axhline(bo[0],ls=':',color='r',lw=1,label='BoT output off->xhigh (center)')
        ax.axhline(1/40,ls='-.',color='0.5',lw=1,label='chance 1/40')
        ax.set_title(p); ax.set_xlabel('feature index (0=emb; squares/grey lines = full-attn)'); ax.set_ylim(-0.02,1.02)
    axs[0,0].set_ylabel('top-1'); axs[0,0].legend(fontsize=7)
    fig.suptitle(f'Cross-condition retrieval top-1 per layer (norm={mode}, N=40 bank)'); fig.tight_layout(); fig.savefig(O+f'retrieval_top1_{mode}.png',dpi=110); plt.close(fig)
# cross position
fig,axs=plt.subplots(1,3,figsize=(16,4.5),sharey=True)
for ax,bp in zip(axs,['last_token','think_mean','answer_start']):
    mark(ax)
    for a,b in [('off','off'),('xhigh','xhigh'),('off','xhigh'),('xhigh','off'),('low','medium')]:
        h=X[(X.qpos=='last_prompt')&(X.bpos==bp)&(X.norm=='center')&(X['query']==a)&(X.bank==b)].sort_values('layer')
        if len(h): ax.plot(h.layer,h.top1,label=f'{a}->{b} (n={int(h.n.iloc[0])})')
    ax.axhline(1/40,ls='-.',color='0.5'); ax.set_title(f'query last_prompt -> bank {bp} (center)'); ax.set_xlabel('feature index'); ax.set_ylim(-0.02,1.02)
axs[0].set_ylabel('top-1'); axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(O+'retrieval_crosspos_top1.png',dpi=110); plt.close(fig)
# decoding
T=[('log10_gold','R2'),('n_numbers','R2'),('len_bin','acc (3-class)')]
fig,axs=plt.subplots(1,3,figsize=(17,4.8))
for ax,(t,yl) in zip(axs,T):
    mark(ax)
    for p in POS:
        g=D[(D.feat==p)&(D.target==t)].sort_values('layer')
        l,=ax.plot(g.layer,g.score,label=f'{p} (n={int(g.n.iloc[0])})',lw=1.2)
    g=D[(D.feat=='last_prompt')&(D.target==t)].sort_values('layer')
    ax.fill_between(g.layer,g.shuf_mean,g.shuf_max,color='0.6',alpha=0.4,label='shuffled (last_prompt, mean..max)')
    for k,ls in [('bot_prompt','--'),('bot_prompt+output',':')]:
        v=D[(D.feat==k)&(D.target==t)].score.values[0]; ax.axhline(v,color='k',ls=ls,lw=1,label=k)
    ax.set_title(t); ax.set_ylabel(yl); ax.set_xlabel('feature index (grey lines = full-attn)')
    if t!='len_bin': ax.set_ylim(-1,1)
    else: ax.set_ylim(0,1)
axs[0].legend(fontsize=7); fig.suptitle('Ridge decoding, pooled 160 (40 qid x 4 cond), GroupKFold(5) by qid'); fig.tight_layout(); fig.savefig(O+'decoding_per_layer.png',dpi=110); plt.close(fig)
print('plots ok')
