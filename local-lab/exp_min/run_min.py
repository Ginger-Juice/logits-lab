# -*- coding: utf-8 -*-
"""Minimal hidden-state experiment: Qwen3.5 thinking ON vs OFF (pipeline smoke test).

Stages: sanity -> gen -> feat -> analysis  (run all by default)
  python run_min.py                      # all stages
  python run_min.py --stages analysis    # re-run analysis only from saved features
"""
import argparse, json, os, re, sys, time, random, platform, hashlib, subprocess, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
DEFAULT_MODEL = os.path.join(LAB, "models", "Qwen3.5-2B")
SEED = 1234

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
ap.add_argument("--questions", default=os.path.join(HERE, "questions.json"))
ap.add_argument("--stages", default="sanity,gen,feat,analysis")
ap.add_argument("--max_new_on", type=int, default=1024)
ap.add_argument("--max_new_off", type=int, default=256)
ap.add_argument("--hash_weights", type=int, default=1)
args = ap.parse_args()
STAGES = [s.strip() for s in args.stages.split(",")]
OUT = args.out
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
META_PATH = os.path.join(OUT, "metadata.json")
GEN_PATH = os.path.join(OUT, "generations.jsonl")
CONDS = {"on": True, "off": False}
POS_NAMES = ["last_prompt", "think_mean", "answer_start", "last_token"]

def log(*a):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)

def load_meta():
    if os.path.exists(META_PATH):
        return json.load(open(META_PATH, encoding="utf-8"))
    return {}

def save_meta(m):
    json.dump(m, open(META_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    import torch
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

meta = load_meta()
meta.setdefault("timings_s", {}); meta.setdefault("peak_vram_gb", {})
questions = json.load(open(args.questions, encoding="utf-8"))
need_model = any(s in STAGES for s in ("sanity", "gen", "feat"))

# ---------------------------------------------------------------- model
model = tok = None
if need_model:
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
    set_seed()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    cfg_full = AutoConfig.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    model, load_info = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", output_loading_info=True)
    model.eval()
    torch.cuda.synchronize()
    t_load = time.time() - t0
    tcfg = model.config
    layers = model.model.layers
    L = len(layers)
    # fast-path availability for gated delta net
    fast = {}
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
        fast["is_fast_path_available"] = bool(getattr(mq, "is_fast_path_available", None))
    except Exception as e:
        fast["err"] = repr(e)
    for mod in ("fla", "causal_conv1d"):
        try:
            __import__(mod); fast[mod] = "importable"
        except Exception as e:
            fast[mod] = f"not available ({type(e).__name__})"
    gpu = torch.cuda.get_device_properties(0)
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        drv = "?"
    import accelerate, sklearn, matplotlib, pandas, safetensors
    try:
        import modelscope; ms_ver = modelscope.__version__
    except Exception:
        ms_ver = "?"
    rev = {}
    rp = os.path.join(args.model, "model_revision.json")
    if os.path.exists(rp):
        rev = json.load(open(rp, encoding="utf-8"))
    meta.update({
        "created": datetime.datetime.now().isoformat(),
        "seed": SEED,
        "python": sys.version, "platform": platform.platform(),
        "versions": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "transformers": transformers.__version__, "accelerate": accelerate.__version__,
                     "modelscope": ms_ver, "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, "pandas": pandas.__version__,
                     "safetensors": safetensors.__version__},
        "gpu": {"name": gpu.name, "total_mem_gb": round(gpu.total_memory / 1024**3, 3), "driver": drv},
        "linear_attn_fast_path": fast,
        "model": {"path": args.model, "repo": "Qwen/Qwen3.5-2B (ModelScope)", "revision_info": rev,
                  "class": type(model).__name__, "model_type_full": cfg_full.model_type,
                  "model_type_text": tcfg.model_type, "num_hidden_layers": L,
                  "layer_types": list(getattr(tcfg, "layer_types", [])),
                  "hidden_size": tcfg.hidden_size,
                  "n_params": int(sum(p.numel() for p in model.parameters())),
                  "missing_keys": list(load_info.get("missing_keys", [])),
                  "unexpected_keys_count": len(load_info.get("unexpected_keys", [])),
                  "unexpected_keys_sample": list(load_info.get("unexpected_keys", []))[:10],
                  "mismatched_keys": [str(x) for x in load_info.get("mismatched_keys", [])]},
        "vram_after_load_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
    })
    meta["timings_s"]["load"] = round(t_load, 2)
    meta["peak_vram_gb"]["load"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    save_meta(meta)
    log("loaded", type(model).__name__, "layers", L, "VRAM after load GB", meta["vram_after_load_gb"],
        "missing", len(meta["model"]["missing_keys"]), "unexpected", meta["model"]["unexpected_keys_count"])
    log("fast path:", fast)

    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    EOS_IDS = sorted({i for i in [IM_END, tok.eos_token_id, getattr(tcfg, "eos_token_id", None)]
                      if isinstance(i, int) and i >= 0})
    THINK_END = tok.convert_tokens_to_ids("</think>")
    meta["tokens"] = {"im_end": IM_END, "eos_ids": EOS_IDS, "think_end": THINK_END,
                      "think_end_roundtrip": tok.decode([THINK_END])}
    save_meta(meta)

    def build_prompt(q, thinking):
        msgs = [{"role": "user", "content": q}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking)

    @torch.no_grad()
    def generate(prompt, max_new):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        gc = GenerationConfig(do_sample=False, max_new_tokens=max_new, eos_token_id=EOS_IDS,
                              pad_token_id=EOS_IDS[0], temperature=None, top_p=None, top_k=None)
        out = model.generate(**enc, generation_config=gc)
        p_ids = enc["input_ids"][0].tolist()
        g_ids = out[0, len(p_ids):].tolist()
        return p_ids, g_ids

def split_generation(g_ids, thinking):
    """Return dict with think span, answer start index (in gen coords), content end, texts."""
    content = list(g_ids)
    while content and content[-1] in EOS_IDS:
        content.pop()
    finished = len(content) < len(g_ids)
    if thinking:
        if THINK_END in content:
            te = content.index(THINK_END)
            think_span = (0, te)
            a0 = te + 1
            truncated_in_think = False
        else:
            think_span = (0, len(content)); a0 = None; truncated_in_think = True
    else:
        think_span = None; a0 = 0; truncated_in_think = False
    ans_start = None
    if a0 is not None:
        for i in range(a0, len(content)):
            if tok.decode([content[i]]).strip() != "":
                ans_start = i; break
    think_text = tok.decode(content[think_span[0]:think_span[1]]) if think_span else ""
    ans_text = tok.decode(content[a0:]) if a0 is not None else ""
    return dict(content_len=len(content), finished=finished, think_span=think_span,
                answer_start=ans_start, truncated_in_think=truncated_in_think,
                think_text=think_text, answer_text=ans_text, think_tokens=(think_span[1] - think_span[0]) if think_span else 0)

NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
def is_correct(ans_text, answers):
    t = ans_text.lower()
    nums = set()
    for m in NUM_RE.findall(ans_text):
        try: nums.add(float(m.replace(",", "")))
        except ValueError: pass
    for a in answers:
        if re.fullmatch(r"-?\d+(\.\d+)?", a):
            if float(a) in nums: return True
        elif a.lower() in t:
            return True
    return False

# ---------------------------------------------------------------- sanity
if "sanity" in STAGES:
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    if args.hash_weights:
        h = {}
        for fn in sorted(os.listdir(args.model)):
            if fn.endswith(".safetensors") or fn == "config.json":
                sh = hashlib.sha256()
                with open(os.path.join(args.model, fn), "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 24), b""): sh.update(chunk)
                h[fn] = sh.hexdigest()
        meta["model"]["sha256"] = h
    q = "What is 17 + 25? Answer briefly."
    san = {}
    for c, th in CONDS.items():
        p = build_prompt(q, th)
        p_ids, g_ids = generate(p, 512 if th else 128)
        sp = split_generation(g_ids, th)
        san[c] = {"prompt_tail": p[-60:], "n_gen": len(g_ids), "finished": sp["finished"],
                  "think_tokens": sp["think_tokens"], "answer_text": sp["answer_text"][:400],
                  "think_text_head": sp["think_text"][:300], "correct_42": is_correct(sp["answer_text"], ["42"])}
        log(f"sanity thinking={c}: n_gen={len(g_ids)} think_tokens={sp['think_tokens']} answer={sp['answer_text'][:200]!r}")
    meta["sanity"] = san
    torch.cuda.synchronize()
    meta["timings_s"]["sanity"] = round(time.time() - t0, 2)
    meta["peak_vram_gb"]["sanity"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    save_meta(meta)

# ---------------------------------------------------------------- gen
if "gen" in STAGES:
    set_seed()
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    recs = []
    with open(GEN_PATH, "w", encoding="utf-8") as fo:
        for qi, q in enumerate(questions):
            for c, th in CONDS.items():
                t1 = time.time()
                p = build_prompt(q["question"], th)
                p_ids, g_ids = generate(p, args.max_new_on if th else args.max_new_off)
                sp = split_generation(g_ids, th)
                corr = is_correct(sp["answer_text"], q["answers"])
                r = {"qid": q["id"], "qtype": q["type"], "cond": c, "question": q["question"],
                     "answers": q["answers"], "prompt": p, "prompt_ids": p_ids, "gen_ids": g_ids,
                     "gen_text": tok.decode(g_ids), "n_prompt": len(p_ids), "n_gen": len(g_ids),
                     "correct": corr, "gen_time_s": round(time.time() - t1, 2),
                     **{k: sp[k] for k in ("content_len", "finished", "think_span", "answer_start",
                                           "truncated_in_think", "think_tokens", "answer_text")}}
                fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
                recs.append(r)
                log(f"{q['id']} {c}: n_gen={len(g_ids)} think={sp['think_tokens']} finished={sp['finished']} "
                    f"correct={corr} t={r['gen_time_s']}s ans={sp['answer_text'][-80:]!r}")
    torch.cuda.synchronize()
    meta["timings_s"]["gen"] = round(time.time() - t0, 2)
    meta["peak_vram_gb"]["gen"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    gs = {}
    for c in CONDS:
        rr = [r for r in recs if r["cond"] == c]
        gs[c] = {"n": len(rr), "acc": sum(r["correct"] for r in rr) / len(rr),
                 "acc_arith": float(np.mean([r["correct"] for r in rr if r["qtype"] == "arith"])),
                 "acc_fact": float(np.mean([r["correct"] for r in rr if r["qtype"] == "fact"])),
                 "n_finished": sum(r["finished"] for r in rr),
                 "n_truncated_in_think": sum(r["truncated_in_think"] for r in rr),
                 "mean_gen_tokens": float(np.mean([r["n_gen"] for r in rr])),
                 "mean_think_tokens": float(np.mean([r["think_tokens"] for r in rr])),
                 "total_gen_time_s": round(sum(r["gen_time_s"] for r in rr), 2)}
    meta["gen_summary"] = gs
    save_meta(meta)
    log("gen summary", json.dumps(gs))

# ---------------------------------------------------------------- feat
if "feat" in STAGES:
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    recs = [json.loads(l) for l in open(GEN_PATH, encoding="utf-8")]
    qids = [q["id"] for q in questions]
    D = model.config.hidden_size
    NL = L + 1  # embeddings + each decoder layer output (pre final norm)
    store = {}
    cur = {}
    def mk_hook(idx):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            h = h[0]  # batch=1 -> [T, D]
            sel = cur["sel"]
            for name, spec in sel.items():
                if spec is None: continue
                if spec[0] == "pos":
                    v = h[spec[1]]
                else:
                    v = h[spec[1]:spec[2]].float().mean(0)
                cur["buf"][name][idx] = v.float().cpu().numpy()
        return hook
    hooks = [model.model.embed_tokens.register_forward_hook(mk_hook(0))]
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(mk_hook(i + 1)))
    seq_lens = {}
    for c in CONDS:
        arr = {n: np.full((len(qids), NL, D), np.nan, dtype=np.float32) for n in POS_NAMES}
        valid = {n: np.zeros(len(qids), dtype=bool) for n in POS_NAMES}
        idxs = {n: np.full(len(qids), -1, dtype=np.int64) for n in POS_NAMES}
        lens = []
        for qi, qid in enumerate(qids):
            r = next(x for x in recs if x["qid"] == qid and x["cond"] == c)
            P = r["n_prompt"]
            content = r["gen_ids"][:r["content_len"]]
            ids = r["prompt_ids"] + content
            T = len(ids); lens.append(T)
            sel = {"last_prompt": ("pos", P - 1), "think_mean": None, "answer_start": None,
                   "last_token": ("pos", T - 1)}
            if r["think_span"] and r["think_span"][1] > r["think_span"][0]:
                a, b = r["think_span"]; sel["think_mean"] = ("mean", P + a, P + b)
            if r["answer_start"] is not None:
                sel["answer_start"] = ("pos", P + r["answer_start"])
            cur["sel"] = sel
            cur["buf"] = {n: np.zeros((NL, D), np.float32) for n in POS_NAMES}
            with torch.no_grad():
                x = torch.tensor([ids], device=dev)
                model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
            for n in POS_NAMES:
                if sel[n] is not None:
                    arr[n][qi] = cur["buf"][n]; valid[n][qi] = True
                    idxs[n][qi] = sel[n][1]
        np.savez_compressed(os.path.join(OUT, f"features_{c}.npz"), qids=np.array(qids),
                            **arr, **{f"valid_{n}": valid[n] for n in POS_NAMES},
                            **{f"index_{n}": idxs[n] for n in POS_NAMES}, seq_len=np.array(lens))
        seq_lens[c] = {"max": int(max(lens)), "mean": float(np.mean(lens))}
        log(f"features {c} saved; valid:", {n: int(valid[n].sum()) for n in POS_NAMES}, "seq_len", seq_lens[c])
    for h in hooks: h.remove()
    torch.cuda.synchronize()
    meta["timings_s"]["feat"] = round(time.time() - t0, 2)
    meta["peak_vram_gb"]["feat"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    meta["feat_seq_lens"] = seq_lens
    meta["feat_note"] = "index 0 = embedding output; index i = output of decoder layer i (before final RMSNorm). EOS tokens stripped before teacher forcing."
    save_meta(meta)

# ---------------------------------------------------------------- analysis
if "analysis" in STAGES:
    t0 = time.time()
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold, cross_val_score
    AD = os.path.join(OUT, "analysis")
    F = {c: np.load(os.path.join(OUT, f"features_{c}.npz")) for c in CONDS}
    lt = meta.get("model", {}).get("layer_types", [])
    NL = F["on"]["last_prompt"].shape[1]
    layer_labels = ["emb"] + [f"{i}{'F' if (lt and lt[i-1]=='full_attention') else 'L'}" for i in range(1, NL)]

    def gram(X):
        X = X.astype(np.float64); return X @ X.T
    def center(K):
        n = K.shape[0]; H = np.eye(n) - 1.0 / n; return H @ K @ H
    def linear_cka(X, Y):
        K, Lm = center(gram(X)), center(gram(Y))
        return float((K * Lm).sum() / (np.linalg.norm(K) * np.linalg.norm(Lm)))
    def hsic_u(K, Lm):
        n = K.shape[0]
        K = K.copy(); Lm = Lm.copy(); np.fill_diagonal(K, 0); np.fill_diagonal(Lm, 0)
        one = np.ones(n)
        t1 = (K * Lm).sum()
        t2 = (one @ K @ one) * (one @ Lm @ one) / ((n - 1) * (n - 2))
        t3 = 2.0 / (n - 2) * (one @ K @ Lm @ one)
        return (t1 + t2 - t3) / (n * (n - 3))
    def debiased_cka(X, Y):
        K, Lm = gram(X), gram(Y)
        return float(hsic_u(K, Lm) / np.sqrt(hsic_u(K, K) * hsic_u(Lm, Lm)))

    rows = []
    cross_pos = ["last_prompt", "answer_start", "last_token"]
    for pos in cross_pos:
        m = F["on"][f"valid_{pos}"] & F["off"][f"valid_{pos}"]
        for l in range(NL):
            X = F["on"][pos][m, l]; Y = F["off"][pos][m, l]
            rows.append({"position": pos, "layer": l, "layer_label": layer_labels[l], "n": int(m.sum()),
                         "linear_cka_on_vs_off": linear_cka(X, Y),
                         "debiased_cka_on_vs_off": debiased_cka(X, Y) if m.sum() > 3 else np.nan})
    cka_df = pd.DataFrame(rows)
    cka_df.to_csv(os.path.join(AD, "cka_on_vs_off_per_layer.csv"), index=False)

    fig, axes = plt.subplots(1, len(cross_pos), figsize=(5 * len(cross_pos), 4), sharey=True)
    for ax, pos in zip(axes, cross_pos):
        d = cka_df[cka_df.position == pos]
        ax.plot(d.layer, d.linear_cka_on_vs_off, "o-", label="linear CKA")
        ax.plot(d.layer, d.debiased_cka_on_vs_off, "s--", label="debiased CKA")
        for i, t in enumerate(lt):
            if t == "full_attention": ax.axvline(i + 1, color="gray", alpha=0.25, lw=3)
        ax.set_title(f"{pos} (n={int(d.n.iloc[0])})"); ax.set_xlabel("layer (0=emb; gray=full_attention)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("CKA thinking-ON vs OFF"); axes[0].legend()
    fig.suptitle("Qwen3.5-2B CKA ON vs OFF per layer (smoke test, n<=20)")
    fig.tight_layout(); fig.savefig(os.path.join(AD, "cka_on_vs_off_per_layer.png"), dpi=130); plt.close(fig)

    lxl = {}
    for pos in ["last_prompt", "answer_start"]:
        m = F["on"][f"valid_{pos}"]
        Xs = F["on"][pos][m]
        M = np.zeros((NL, NL))
        for i in range(NL):
            for j in range(i, NL):
                M[i, j] = M[j, i] = linear_cka(Xs[:, i], Xs[:, j])
        lxl[pos] = M
        pd.DataFrame(M, index=layer_labels, columns=layer_labels).to_csv(
            os.path.join(AD, f"cka_layer_x_layer_on_{pos}.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, pos in zip(axes, lxl):
        im = ax.imshow(lxl[pos], vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xticks(range(NL)); ax.set_xticklabels(layer_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(NL)); ax.set_yticklabels(layer_labels, fontsize=6)
        ax.set_title(f"thinking-ON linear CKA layer x layer @ {pos} (n={int(F['on'][f'valid_{pos}'].sum())})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(AD, "cka_layer_x_layer_on.png"), dpi=130); plt.close(fig)

    # probe ON vs OFF
    rng = np.random.RandomState(SEED)
    N_PERM = 20
    prow = []
    for pos in cross_pos:
        m = F["on"][f"valid_{pos}"] & F["off"][f"valid_{pos}"]
        nq = int(m.sum())
        groups = np.concatenate([np.arange(nq), np.arange(nq)])
        y = np.concatenate([np.ones(nq, int), np.zeros(nq, int)])
        n_splits = min(5, nq)
        cv = GroupKFold(n_splits=n_splits)
        perms = [rng.permutation(y) for _ in range(N_PERM)]
        for l in range(NL):
            X = np.concatenate([F["on"][pos][m, l], F["off"][pos][m, l]]).astype(np.float64)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
            acc = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="accuracy")
            sh = [cross_val_score(clf, X, yp, groups=groups, cv=cv, scoring="accuracy").mean() for yp in perms]
            prow.append({"position": pos, "layer": l, "layer_label": layer_labels[l], "n_questions": nq,
                         "n_samples": 2 * nq, "cv_folds": n_splits, "probe_acc_mean": float(acc.mean()),
                         "probe_acc_std": float(acc.std()), "shuffled_acc_mean": float(np.mean(sh)),
                         "shuffled_acc_std": float(np.std(sh)), "n_perm": N_PERM})
    probe_df = pd.DataFrame(prow)
    probe_df.to_csv(os.path.join(AD, "probe_on_vs_off_per_layer.csv"), index=False)
    fig, axes = plt.subplots(1, len(cross_pos), figsize=(5 * len(cross_pos), 4), sharey=True)
    for ax, pos in zip(axes, cross_pos):
        d = probe_df[probe_df.position == pos]
        ax.errorbar(d.layer, d.probe_acc_mean, yerr=d.probe_acc_std, fmt="o-", label="probe (GroupKFold)", capsize=2)
        ax.errorbar(d.layer, d.shuffled_acc_mean, yerr=d.shuffled_acc_std, fmt="s--", label="shuffled labels", capsize=2)
        ax.axhline(0.5, color="k", lw=0.8); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{pos} (n_q={int(d.n_questions.iloc[0])})"); ax.set_xlabel("layer (0=emb)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy ON vs OFF"); axes[0].legend()
    fig.suptitle("Logistic probe thinking ON vs OFF (smoke test, tiny n)")
    fig.tight_layout(); fig.savefig(os.path.join(AD, "probe_on_vs_off_per_layer.png"), dpi=130); plt.close(fig)

    meta["timings_s"]["analysis"] = round(time.time() - t0, 2)
    summ = {"cka": cka_df.round(4).to_dict(orient="records"), "probe": probe_df.round(4).to_dict(orient="records")}
    json.dump(summ, open(os.path.join(AD, "analysis_summary.json"), "w"), indent=1)
    save_meta(meta)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(cka_df.round(4).to_string())
        print(probe_df[["position", "layer_label", "n_questions", "probe_acc_mean", "shuffled_acc_mean"]].round(3).to_string())

log("DONE stages", STAGES, "timings", meta.get("timings_s"), "peak", meta.get("peak_vram_gb"))
