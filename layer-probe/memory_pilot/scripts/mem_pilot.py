# CPU-only "hidden states as memory" pilot on existing pipeline_v1 features. No GPU, no model load.
import os
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]: os.environ[v]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys, json, re, time, itertools
import numpy as np, pandas as pd
from joblib import Parallel, delayed
P='/root/autodl-tmp/layer-probe/pipeline_v1/outputs/'
O='/root/autodl-tmp/layer-probe/pipeline_v1/memory_pilot/results/'
os.makedirs(O, exist_ok=True)
CONDS=['off','low','medium','xhigh']
POS=['last_prompt','think_start','think_end','answer_start','last_token','think_mean']
cfg=json.load(open('/root/autodl-tmp/qwen38-27b-fp8/config.json')); tc=cfg.get('text_config',cfg)
FULL=[i+1 for i,t in enumerate(tc['layer_types']) if t=='full_attention']  # feature index (0=embedding)
assert FULL==list(range(4,65,4)), FULL
NL=65
t0=time.time()
# ---------- load ----------
gens={}
for l in open(P+'generations.jsonl'):
    r=json.loads(l); gens[(r['cond'],r['qid'])]=r
F={}; QIDS=None
for c in CONDS:
    z=np.load(P+f'features_{c}.npz')
    q=z['qids']; o=np.argsort(q)
    if QIDS is None: QIDS=q[o]
    assert (q[o]==QIDS).all()
    for p in POS: F[(c,p)]=z[p][o].astype(np.float32)
    print('loaded',c,round(time.time()-t0,1),flush=True)
N=len(QIDS)
VALID={k:~np.isnan(v).any(axis=(1,2)) for k,v in F.items()}
nanrep=[dict(cond=c,pos=p,n_valid=int(VALID[(c,p)].sum()),n_nan_rows=int((~VALID[(c,p)]).sum()),nan_qids=' '.join(map(str,QIDS[~VALID[(c,p)]]))) for c in CONDS for p in POS]
pd.DataFrame(nanrep).to_csv(O+'nan_report.csv',index=False); print(pd.DataFrame(nanrep).to_string(),flush=True)
# ---------- targets ----------
def num(s): return float(str(s).replace(',','').strip())
qtext=[gens[('off',int(q))]['question'] for q in QIDS]
for c in CONDS: assert all(gens[(c,int(q))]['question']==qtext[i] for i,q in enumerate(QIDS))
gold=np.array([num(gens[('off',int(q))]['gold']) for q in QIDS])
y_log=np.log10(np.maximum(np.abs(gold),1.0))
y_nnum=np.array([len(re.findall(r'\d+(?:[.,]\d+)*',t)) for t in qtext],float)
qlen=np.array([len(t) for t in qtext]); cuts=np.quantile(qlen,[1/3,2/3]); y_lbin=np.digitize(qlen,cuts)
pd.DataFrame(dict(qid=QIDS,gold=gold,log10_gold=y_log,n_numbers=y_nnum,q_chars=qlen,len_bin=y_lbin)).to_csv(O+'targets.csv',index=False)
# ---------- bag of tokens ----------
def bot(kind):
    seqs={}
    for c in CONDS:
        for q in QIDS:
            r=gens[(c,int(q))]
            seqs[(c,q)]=r['prompt_ids'] if kind=='prompt' else (r['output_ids'] if kind=='output' else r['prompt_ids']+r['output_ids'])
    vocab=sorted({t for s in seqs.values() for t in s}); vi={t:i for i,t in enumerate(vocab)}
    M={}
    for c in CONDS:
        X=np.zeros((N,len(vocab)),np.float32)
        for i,q in enumerate(QIDS):
            for t in seqs[(c,q)]: X[i,vi[t]]+=1
        M[c]=X
    return M
BOT={k:bot(k) for k in ['prompt','output','prompt+output']}
# surface overlap stats
ov=[]
for a,b in itertools.combinations(CONDS,2):
    j=[]; d=[]
    for q in QIDS:
        pa=gens[(a,int(q))]['prompt_ids']; pb=gens[(b,int(q))]['prompt_ids']
        sa,sb=set(pa),set(pb); j.append(len(sa&sb)/len(sa|sb)); d.append(abs(len(pa)-len(pb)))
    ov.append(dict(a=a,b=b,prompt_jaccard_mean=np.mean(j),prompt_len_absdiff_mean=np.mean(d)))
pd.DataFrame(ov).to_csv(O+'prompt_overlap.csv',index=False)
# ---------- retrieval ----------
def prep(X,mode):
    X=X.astype(np.float64)
    if mode in('center','zscore','tfidf_center'): X=X-X.mean(0)
    if mode=='zscore': X=X/(X.std(0)+1e-6)
    return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
def tfidf(A,B):
    df=((A>0).sum(0)+(B>0).sum(0)); idf=np.log((2*N+1)/(df+1))+1
    return A*idf, B*idf
def scores(Q,M,v=None):
    if v is not None: Q,M=Q[v],M[v]
    n=len(Q)
    if n<2: return dict(n=n,top1=np.nan,top5=np.nan,mrr=np.nan,margin=np.nan)
    S=Q@M.T; true=np.diag(S)
    rank=(S>=true[:,None]).sum(1)  # pessimistic tie-breaking: ties (incl. identical vectors) count against; self counted -> rank>=1
    return dict(n=n,top1=np.mean(rank==1),top5=np.mean(rank<=5),mrr=np.mean(1/rank),
                margin=np.mean(true-np.max(np.where(np.eye(n,dtype=bool),-np.inf,S),1)))
PAIRS=[(a,b) for a in CONDS for b in CONDS if a!=b]
rows=[]
for p in POS:
    for a,b in PAIRS:
        for mode in ['raw','center','zscore']:
            v=VALID[(a,p)]&VALID[(b,p)]
            if v.sum()<2: continue
            for L in range(NL):
                Q=prep(F[(a,p)][v,L],mode); M=prep(F[(b,p)][v,L],mode)
                rows.append(dict(kind='hidden',pos=p,query=a,bank=b,norm=mode,layer=L,full_attn=L in FULL,**scores(Q,M)))
for k,BM in BOT.items():
    for a,b in PAIRS:
        for mode in ['raw','center','zscore','tfidf']:
            A,B=BM[a],BM[b]
            if mode=='tfidf': A,B=tfidf(A,B); Q,M=prep(A,'raw'),prep(B,'raw')
            else: Q,M=prep(A,mode),prep(B,mode)
            rows.append(dict(kind='bot_'+k,pos='-',query=a,bank=b,norm=mode,layer=-1,full_attn=False,**scores(Q,M)))
R=pd.DataFrame(rows); R.to_csv(O+'retrieval_all.csv',index=False)
# cross-position retrieval (memory written at end-of-reasoning, queried with a fresh prompt): query=last_prompt of A, bank=last_token / think_mean of B
xr=[]
for qp,bp in [('last_prompt','last_token'),('last_prompt','think_mean'),('last_prompt','answer_start')]:
    for a,b in PAIRS+[(c,c) for c in CONDS]:
        for mode in ['center','zscore']:
            v=VALID[(a,qp)]&VALID[(b,bp)]
            if v.sum()<2: continue
            for L in range(NL):
                xr.append(dict(qpos=qp,bpos=bp,query=a,bank=b,norm=mode,layer=L,full_attn=L in FULL,**scores(prep(F[(a,qp)][v,L],mode),prep(F[(b,bp)][v,L],mode))))
pd.DataFrame(xr).to_csv(O+'retrieval_crosspos.csv',index=False)
print('retrieval done',round(time.time()-t0,1),flush=True)
# ---------- decoding (pooled 160 = 40 qid x 4 cond, GroupKFold(5) by qid) ----------
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
ALPH=np.logspace(0,6,13)
groups=np.tile(QIDS,len(CONDS)); qidx=np.tile(np.arange(N),len(CONDS))
TGT={'log10_gold':(y_log,'reg'),'n_numbers':(y_nnum,'reg'),'len_bin':(y_lbin,'cls')}
NPERM=5; rng=np.random.default_rng(1234); PERMS=[rng.permutation(N) for _ in range(NPERM)]
def cvscore(X,y,kind,g):
    folds=list(GroupKFold(5).split(X,groups=g))
    pred=np.zeros(len(y),float)
    for tr,te in folds:
        m=make_pipeline(StandardScaler(),RidgeCV(alphas=ALPH) if kind=='reg' else RidgeClassifierCV(alphas=ALPH))
        m.fit(X[tr],y[tr]); pred[te]=m.predict(X[te])
    if kind=='reg': return 1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
    return np.mean(pred==y)
def task(name,X,layer,full,keep):
    out=[]; X=X[keep]; qi=qidx[keep]; g=groups[keep]
    for t,(yq,kind) in TGT.items():
        y=yq[qi].astype(float)
        s=cvscore(X,y,kind,g)
        sh=[cvscore(X,yq[pm][qi].astype(float),kind,g) for pm in PERMS]
        out.append(dict(feat=name,layer=layer,full_attn=full,target=t,n=int(keep.sum()),score=s,shuf_mean=np.mean(sh),shuf_max=np.max(sh)))
    return out
jobs=[]
for p in POS:
    Xall=np.stack([F[(c,p)] for c in CONDS])  # 4,N,65,D
    keep=np.concatenate([VALID[(c,p)] for c in CONDS])
    for L in range(NL):
        jobs.append(delayed(task)(p,Xall[:,:,L].reshape(len(CONDS)*N,-1),L,L in FULL,keep))
for k in ['prompt','prompt+output']:
    jobs.append(delayed(task)('bot_'+k,np.concatenate([BOT[k][c] for c in CONDS]),-1,False,np.ones(len(groups),bool)))
res=Parallel(n_jobs=64,verbose=0)(jobs)
D=pd.DataFrame([r for rr in res for r in rr]); D.to_csv(O+'decoding_all.csv',index=False)
print('decoding done',round(time.time()-t0,1),flush=True)
json.dump(dict(full_attn_feature_idx=FULL,N=int(N),conds=CONDS,positions=POS,len_bin_cuts=cuts.tolist(),len_bin_counts=np.bincount(y_lbin).tolist(),
               alphas=ALPH.tolist(),nperm=NPERM,elapsed_s=round(time.time()-t0,1),numpy=np.__version__),open(O+'meta.json','w'),indent=1)
print('main done (run summarize.py next)',round(time.time()-t0,1),flush=True)
