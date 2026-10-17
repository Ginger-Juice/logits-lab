"""Stage 2: teacher-forced capture with transformers 5.12.1 + PR#47990 patch (pipeline_v1).
Index 0 = embedding output; index i (1..64) = output of decoder layer i-1 (pre final norm).
env: OUT_DIR, MP, CONDS, MAXN (limit samples per cond, for smoke)"""
import os, re, sys, json, time, hashlib, platform
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, torch, transformers
import transformers.integrations.finegrained_fp8 as fg
import transformers.quantizers.quantizers_utils as qu
from transformers import AutoModelForCausalLM, AutoTokenizer
MP = os.environ.get('MP', '/root/autodl-tmp/qwen38-27b-fp8')
OUT = os.environ.get('OUT_DIR', '/root/autodl-tmp/layer-probe/pipeline_v1/outputs')
CONDS = os.environ.get('CONDS', 'off,low,medium,xhigh').split(',')
MAXN = int(os.environ.get('MAXN', 10**9)); NS = 8; NCHK = 64
G = 2**30

def should_convert_module_anchored(full_name, patterns=None):   # = PR #47990
    if patterns is None: return True
    return not any(re.match(f"{k}\\.", full_name) or re.match(f"{k}$", full_name) or full_name.endswith(k) for k in patterns)
patched = []
for mod in (qu, fg):
    mod.should_convert_module = should_convert_module_anchored; patched.append(mod.__name__)
for name, mod in list(sys.modules.items()):
    if name.startswith('transformers') and getattr(mod, 'should_convert_module', None) is not None and mod.__dict__.get('should_convert_module') is not should_convert_module_anchored:
        mod.should_convert_module = should_convert_module_anchored; patched.append(name)
print('patched', patched, flush=True)

tok = AutoTokenizer.from_pretrained(MP)
THINK_S, THINK_E = tok.convert_tokens_to_ids('<think>'), tok.convert_tokens_to_ids('</think>')
t0 = time.time(); torch.cuda.reset_peak_memory_stats()
model = AutoModelForCausalLM.from_pretrained(MP, dtype='auto', device_map={'': 0}); model.eval()
load_s = time.time() - t0
bad = [n for n, m in model.named_modules() if type(m) is torch.nn.Linear and m.weight.dtype == torch.float8_e4m3fn]
nfp8 = sum(1 for m in model.modules() if type(m).__name__ == 'FP8Linear')
assert not bad, f'plain Linear with fp8 weights: {bad[:3]}'
print(f'loaded {load_s:.1f}s FP8Linear={nfp8} alloc={torch.cuda.memory_allocated()/G:.2f}GB', flush=True)
layers = model.model.layers; L = len(layers); D = model.config.hidden_size; NL = L + 1
layer_types = list(getattr(model.config, 'layer_types', []))

cur = {}
def mk(idx):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        h = h[0]
        cur['pos'][:, idx] = h[cur['pidx']].float().cpu()
        a, b = cur['span']
        if b > a: cur['mean'][idx] = h[a:b].float().mean(0).cpu()
    return hook
fin = {}
def norm_hook(mod, inp, out):
    fin['h'] = out[0, cur['chk'][0]:cur['chk'][1]].detach()
hooks = [model.model.embed_tokens.register_forward_hook(mk(0))] + [l.register_forward_hook(mk(i + 1)) for i, l in enumerate(layers)]
hooks.append(model.model.norm.register_forward_hook(norm_hook))

def positions(r, cond):
    P = r['n_prompt']; content = r['output_ids'][:r['content_len']]; ids = r['prompt_ids'] + content; T = len(ids)
    ws = lambda i: tok.decode([ids[i]]).strip() == ''
    pr = r['prompt_ids']; ts_prompt = max(i for i, t in enumerate(pr) if t == THINK_S)
    pos = dict(last_prompt=P - 1, think_start=ts_prompt, think_end=None, answer_start=None, last_token=T - 1)
    span = (0, 0); samples = None
    if cond == 'off':
        pos['think_end'] = max(i for i, t in enumerate(pr) if t == THINK_E)   # template-inserted empty think block
        a0 = P
    else:
        te = r['think_end_out']
        span = (P, P + te) if te is not None else (P, T)
        if span[1] > span[0]: samples = np.linspace(span[0], span[1] - 1, NS).round().astype(int).tolist()
        if te is not None: pos['think_end'] = P + te
        a0 = P + te + 1 if te is not None else None
    if a0 is not None:
        for i in range(a0, T):
            if not ws(i): pos['answer_start'] = i; break
    return ids, pos, span, samples

POS = ['last_prompt', 'think_start', 'think_end', 'answer_start', 'last_token']
recs = [json.loads(l) for l in open(os.path.join(OUT, 'generations.jsonl'))]
timing = {}; dev = 'cuda'
for cond in CONDS:
    rs = sorted([r for r in recs if r['cond'] == cond], key=lambda r: r['qid'])[:MAXN]
    n = len(rs)
    arr = {p: np.full((n, NL, D), np.nan, np.float32) for p in POS}
    tmean = np.full((n, NL, D), np.nan, np.float32); tsamp = np.full((n, NS, NL, D), np.nan, np.float32)
    idx = {p: np.full(n, -1, np.int64) for p in POS}; sidx = np.full((n, NS), -1, np.int64); spans = np.full((n, 2), -1, np.int64)
    seq_len = np.zeros(n, np.int64); secs = []; agree = []; peaks = []
    for k, r in enumerate(rs):
        ids, pos, span, samples = positions(r, cond)
        valid = [p for p in POS if pos[p] is not None]
        plist = [pos[p] for p in valid] + (samples or [])
        cur['pidx'] = torch.tensor(plist, device=dev); cur['pos'] = torch.zeros(len(plist), NL, D); cur['span'] = span; cur['mean'] = torch.full((NL, D), float('nan'))
        P = r['n_prompt']; cur['chk'] = (P - 1, min(P - 1 + NCHK, len(ids) - 1))
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        with torch.inference_mode():
            x = torch.tensor([ids], device=dev)
            model(input_ids=x, use_cache=False, logits_to_keep=1)
            pred = model.lm_head(fin['h']).argmax(-1).tolist()
        torch.cuda.synchronize(); el = time.time() - t; secs.append(el); peaks.append(torch.cuda.max_memory_allocated() / G)
        tgt = ids[cur['chk'][0] + 1: cur['chk'][1] + 1]
        agree.append(float(np.mean([a == b for a, b in zip(pred, tgt)])) if tgt else float('nan'))
        for j, p in enumerate(valid): arr[p][k] = cur['pos'][j].numpy(); idx[p][k] = pos[p]
        if samples:
            tsamp[k] = cur['pos'][len(valid):].numpy(); sidx[k] = samples
        if span[1] > span[0]: tmean[k] = cur['mean'].numpy(); spans[k] = span
        seq_len[k] = len(ids)
        print(f'{cond} qid={r["qid"]} T={len(ids)} {el:.2f}s peak={peaks[-1]:.2f}GB tf_agree64={agree[-1]:.3f}', flush=True)
    fn = os.path.join(OUT, f'features_{cond}.npz')
    np.savez(fn, qids=np.array([r['qid'] for r in rs]), correct=np.array([r['correct'] for r in rs]),
             truncated_in_think=np.array([r['truncated_in_think'] for r in rs]), think_tokens=np.array([r['think_tokens'] for r in rs]),
             **arr, think_mean=tmean, think_samples=tsamp, **{f'index_{p}': idx[p] for p in POS}, index_think_samples=sidx,
             think_span=spans, seq_len=seq_len, tf_agree64=np.array(agree))
    timing[cond] = dict(n=n, total_s=round(sum(secs), 1), s_per_sample=round(float(np.mean(secs)), 2), max_s=round(max(secs), 2),
                        peak_gb_max=round(max(peaks), 3), seq_len_max=int(seq_len.max()), seq_len_mean=round(float(seq_len.mean()), 1),
                        tf_agree64_mean=round(float(np.nanmean(agree)), 4), file_mb=round(os.path.getsize(fn) / 2**20, 1))
    print('COND_DONE', cond, timing[cond], flush=True)
for h in hooks: h.remove()

import importlib.metadata as md
def md5f(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
meta = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), stage='capture', seed=1234, python=sys.version, platform=platform.platform(),
            versions={k: md.version(k) for k in ['torch', 'transformers', 'accelerate', 'kernels', 'numpy', 'safetensors', 'sglang', 'scikit-learn']},
            gpu=torch.cuda.get_device_name(0), patch=dict(pr='huggingface/transformers#47990', namespaces=patched, n_fp8linear=nfp8, plain_linear_fp8=len(bad)),
            model=dict(path=MP, config_md5=md5f(os.path.join(MP, 'config.json')), safetensors_md5sum_txt_md5=md5f(os.path.join(MP, 'safetensors-md5sum.txt')),
                       chat_template_md5=md5f(os.path.join(MP, 'chat_template.jinja')), class_=type(model).__name__, num_layers=L, hidden=D,
                       full_attention_layers=[i for i, t in enumerate(layer_types) if t == 'full_attention']),
            tokens=dict(think_start=THINK_S, think_end=THINK_E, think_start_rt=tok.decode([THINK_S]), think_end_rt=tok.decode([THINK_E])),
            positions_note='idx0=embedding, idx i=decoder layer i-1 output (pre final norm). off: think_start/think_end are the template-inserted empty <think></think> in prompt. '
                           'think_mean/think_samples over [P, P+think_end_out) (whole output if truncated). answer_start=first non-whitespace token after </think> (off: first output token).',
            load_s=round(load_s, 1), timing=timing, script_md5=md5f(__file__),
            gen_script_md5=md5f(os.path.join(os.path.dirname(__file__), 'gen.py')))
json.dump(meta, open(os.path.join(OUT, 'capture_meta.json'), 'w'), indent=1, ensure_ascii=False)
print('CAPTURE_DONE', flush=True)
