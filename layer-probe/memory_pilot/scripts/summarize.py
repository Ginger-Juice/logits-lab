import numpy as np, pandas as pd
O='/root/autodl-tmp/layer-probe/pipeline_v1/memory_pilot/results/'
FULL=list(range(4,65,4)); POS=['last_prompt','think_start','think_end','answer_start','last_token','think_mean']
R=pd.read_csv(O+'retrieval_all.csv'); D=pd.read_csv(O+'decoding_all.csv'); X=pd.read_csv(O+'retrieval_crosspos.csv')
# ---------- summaries ----------
lines=[]
H=R[R.kind=='hidden']
sym=[('off','xhigh'),('low','xhigh'),('low','medium'),('medium','xhigh'),('off','low'),('off','medium')]
srows=[]
for p in POS:
    for mode in ['raw','center','zscore']:
        for a,b in sym:
            h=H[(H.pos==p)&(H.norm==mode)&(((H['query']==a)&(H.bank==b))|((H['query']==b)&(H.bank==a)))]
            if len(h)==0: continue
            g=h.groupby('layer')[['top1','top5','mrr']].mean()
            best=g.top1.idxmax()
            dec=g.drop(index=0)
            srows.append(dict(pos=p,norm=mode,pair=f'{a}<->{b}',n=int(h.n.min()),emb_top1=g.top1[0],best_layer=best,best_top1=g.top1[best],best_top5=g.top5[best],best_mrr=g.mrr[best],
                mean_top1_full=dec.top1[dec.index.isin(FULL)].mean(),mean_top1_gdn=dec.top1[~dec.index.isin(FULL)].mean(),
                n_layers_top1_eq1=int((dec.top1>=1-1e-9).sum())))
S=pd.DataFrame(srows); S.to_csv(O+'retrieval_summary.csv',index=False)
B=R[R.kind!='hidden'].copy(); B['pair']=B['query']+'->'+B['bank']; B.to_csv(O+'retrieval_bot.csv',index=False)
ds=[]
for (f,t),g in D[D.layer>=0].groupby(['feat','target']):
    g=g.set_index('layer'); best=g.score.idxmax(); dec=g.drop(index=0)
    ds.append(dict(feat=f,target=t,emb_score=g.score[0],best_layer=best,best_score=g.score[best],shuf_mean_at_best=g.shuf_mean[best],shuf_max_over_layers=g.shuf_max.max(),
                   mean_full=dec.score[dec.index.isin(FULL)].mean(),mean_gdn=dec.score[~dec.index.isin(FULL)].mean()))
DS=pd.DataFrame(ds); DS.to_csv(O+'decoding_summary.csv',index=False)
# cross-position summary
xs=[]
for (qp,bp,a,b,mode),g in X.groupby(['qpos','bpos','query','bank','norm']):
    g=g.set_index('layer'); best=g.top1.idxmax(); dec=g.drop(index=0)
    xs.append(dict(qpos=qp,bpos=bp,query=a,bank=b,norm=mode,n=int(g.n.min()),emb_top1=g.top1[0],best_layer=best,best_top1=g.top1[best],best_mrr=g.mrr[best],
       mean_top1_full=dec.top1[dec.index.isin(FULL)].mean(),mean_top1_gdn=dec.top1[~dec.index.isin(FULL)].mean()))
pd.DataFrame(xs).to_csv(O+'retrieval_crosspos_summary.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
print(S.round(3).to_string()); print(DS.round(3).to_string()); print(pd.DataFrame(xs).round(3).to_string())
bb=B
print(bb.pivot_table(index=['kind','norm'],columns='pair',values='top1').round(3).to_string())
print(D[D.layer<0].round(3).to_string())
