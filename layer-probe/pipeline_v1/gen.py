"""Stage 1: generation via SGLang /generate with self-rendered prompt ids (pipeline_v1).
env: OUT_DIR, N (40), SEED (1234), CONDS (off,low,medium,xhigh), MAXTOT (7000), CONC (4), URL"""
import os, json, random, re, time, threading, urllib.request, hashlib, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from transformers import AutoTokenizer
MP = os.environ.get('MP', '/root/autodl-tmp/qwen38-27b-fp8')
OUT = os.environ.get('OUT_DIR', '/root/autodl-tmp/layer-probe/pipeline_v1/outputs')
DATA = os.environ.get('DATA', '/root/autodl-tmp/layer-probe/pipeline_v1/data/gsm8k_test.parquet')
N = int(os.environ.get('N', 40)); SEED = int(os.environ.get('SEED', 1234))
CONDS = os.environ.get('CONDS', 'off,low,medium,xhigh').split(',')
MAXTOT = int(os.environ.get('MAXTOT', 7000)); CONC = int(os.environ.get('CONC', 4))
URL = os.environ.get('URL', 'http://127.0.0.1:30010')
os.makedirs(OUT, exist_ok=True)
GEN = os.path.join(OUT, 'generations.jsonl'); META = os.path.join(OUT, 'gen_meta.json')
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
tok = AutoTokenizer.from_pretrained(MP)
THINK_S, THINK_E = tok.convert_tokens_to_ids('<think>'), tok.convert_tokens_to_ids('</think>')
IM_END = tok.convert_tokens_to_ids('<|im_end|>')
EOS = set([IM_END, tok.eos_token_id] + ([tok.pad_token_id] if tok.pad_token_id is not None else []))

def render(q, cond):
    msgs = [{'role': 'user', 'content': q + SUFFIX}]
    kw = dict(enable_thinking=False) if cond == 'off' else dict(enable_thinking=True, reasoning_effort=cond)
    txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    return txt, tok(txt, add_special_tokens=False)['input_ids']

def post(path, body, timeout=3600):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def gold_of(a): return a.split('####')[-1].strip().replace(',', '')
NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
def parse(ans):
    b = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', ans)
    src = b[-1] if b else ans
    nums = NUM.findall(src.replace('$', ''))
    return (nums[-1].replace(',', '').rstrip('.') if nums else None), bool(b)
def correct(p, g):
    try: return p is not None and abs(float(p) - float(g)) < 1e-6
    except ValueError: return False

def split(out_ids, cond):
    content = list(out_ids)
    while content and content[-1] in EOS: content.pop()
    if cond == 'off':
        return dict(think_end_out=None, think_tokens=0, truncated_in_think=False, think_ids=[], ans_ids=content, content_len=len(content))
    if THINK_E in content:
        te = content.index(THINK_E)
        return dict(think_end_out=te, think_tokens=te, truncated_in_think=False, think_ids=content[:te], ans_ids=content[te + 1:], content_len=len(content))
    return dict(think_end_out=None, think_tokens=len(content), truncated_in_think=True, think_ids=content, ans_ids=[], content_len=len(content))

if __name__ == '__main__':
    df = pd.read_parquet(DATA)
    idx = sorted(random.Random(SEED).sample(range(len(df)), N))
    qs = [(int(i), df.question[i], gold_of(df.answer[i])) for i in idx]
    # template verification
    ver = {}
    for c in CONDS:
        txt, ids = render(qs[0][1], c)
        ver[c] = dict(n_prompt=len(ids), head=txt[:400], tail=txt[-60:],
                      has_low='effort is set to low' in txt, has_xhigh='effort is set to xhigh' in txt, has_system='<|im_start|>system' in txt)
    meta = dict(model=MP, data=DATA, n=N, seed=SEED, qids=[q[0] for q in qs], conds=CONDS, maxtot=MAXTOT, conc=CONC, suffix=SUFFIX,
                think_start_id=THINK_S, think_end_id=THINK_E, im_end_id=IM_END, eos_ids=sorted(EOS), template_check=ver,
                script_md5=hashlib.md5(open(__file__, 'rb').read()).hexdigest())
    print(json.dumps(ver, indent=1)[:3000], flush=True)
    # wait for server
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(URL + '/health_generate', timeout=60); break
        except Exception:
            if time.time() - t0 > 1800: sys.exit('server not ready')
            time.sleep(10)
    meta['server_info'] = {k: v for k, v in json.loads(urllib.request.urlopen(URL + '/get_server_info').read()).items()
                           if k in ('max_total_num_tokens', 'kv_cache_dtype', 'mem_fraction_static', 'chunked_prefill_size', 'model_path', 'version', 'allow_auto_truncate', 'max_mamba_cache_size')}
    # cross-check: server-side chat template with chat_template_kwargs gives same prompt length
    xc = {}
    for c in CONDS:
        kw = dict(enable_thinking=False) if c == 'off' else dict(enable_thinking=True, reasoning_effort=c)
        r = post('/v1/chat/completions', dict(model='qwen3.8-27b-fp8', messages=[{'role': 'user', 'content': qs[0][1] + SUFFIX}],
                                              max_tokens=1, temperature=0, chat_template_kwargs=kw))
        xc[c] = dict(server_prompt_tokens=r['usage']['prompt_tokens'], local=ver[c]['n_prompt'])
    meta['chat_template_kwargs_crosscheck'] = xc; print('crosscheck', xc, flush=True)
    done = set()
    if os.path.exists(GEN):
        for l in open(GEN): r = json.loads(l); done.add((r['cond'], r['qid']))
    lock = threading.Lock(); fo = open(GEN, 'a')
    def job(c, qid, q, g):
        txt, p_ids = render(q, c)
        mx = MAXTOT - len(p_ids)
        t = time.time()
        r = post('/generate', dict(input_ids=p_ids, sampling_params=dict(temperature=0, max_new_tokens=mx, skip_special_tokens=False)))
        el = time.time() - t
        out_ids = r.get('output_ids'); mi = r['meta_info']
        if out_ids is None: out_ids = tok(r['text'], add_special_tokens=False)['input_ids']
        s = split(out_ids, c)
        ans = tok.decode(s['ans_ids'], skip_special_tokens=True); think = tok.decode(s['think_ids'], skip_special_tokens=False)
        p, boxed = parse(ans)
        fr = mi.get('finish_reason'); fr = fr.get('type') if isinstance(fr, dict) else fr
        rec = dict(cond=c, qid=qid, gold=g, question=q, prompt_ids=p_ids, n_prompt=len(p_ids), output_ids=out_ids, n_out=len(out_ids),
                   max_new_tokens=mx, finish_reason=fr, server_prompt_tokens=mi.get('prompt_tokens'), server_completion_tokens=mi.get('completion_tokens'),
                   think_tokens=s['think_tokens'], think_end_out=s['think_end_out'], truncated_in_think=s['truncated_in_think'],
                   content_len=s['content_len'], answer_tokens=len(s['ans_ids']), reasoning_text=think, answer_text=ans,
                   parsed=p, boxed=boxed, correct=correct(p, g), seconds=round(el, 2))
        with lock:
            fo.write(json.dumps(rec, ensure_ascii=False) + '\n'); fo.flush()
            print(c, qid, fr, len(out_ids), 'think', s['think_tokens'], 'trunc', s['truncated_in_think'], 'ok', rec['correct'], f'{el:.0f}s', flush=True)
        return len(out_ids)
    meta['timing'] = {}
    for c in CONDS:
        todo = [q for q in qs if (c, q[0]) not in done]
        t = time.time()
        with ThreadPoolExecutor(CONC) as ex: ntoks = list(ex.map(lambda q: job(c, *q), todo))
        el = time.time() - t
        meta['timing'][c] = dict(n=len(todo), seconds=round(el, 1), out_tokens=sum(ntoks), agg_tok_s=round(sum(ntoks) / max(el, 1e-9), 1))
        print('COND_DONE', c, meta['timing'][c], flush=True)
        json.dump(meta, open(META, 'w'), indent=1, ensure_ascii=False)
    print('GEN_DONE', flush=True)
