# -*- coding: utf-8 -*-
"""Recall a question from a layer residual, without the question text in the prompt.

Memory at layer l is the difference between the question prompt and a neutral carrier
at the same kind of position. Recall adds that difference back into the carrier and
generates. A layer counts as a memory if the injected question's answer comes back,
and a different question's answer does not.

Phases: capture, lens, sweep, report
"""
import argparse, json, os, re, time, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
DEFAULT_MODEL = os.path.join(LAB, "models", "Qwen3.5-2B")
SEED = 1234
CARRIER = "Answer the remembered question with only the final answer."
NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--questions", default=os.path.join(HERE, "questions.json"))
ap.add_argument("--out", default=os.path.join(HERE, "outputs", "recall"))
ap.add_argument("--phases", default="capture,lens,sweep,report")
ap.add_argument("--max_new", type=int, default=24)
ap.add_argument("--alphas", default="0.5,1,2")
args = ap.parse_args()
PHASES = [s.strip() for s in args.phases.split(",") if s.strip()]
ALPHAS = [float(s) for s in args.alphas.split(",") if s.strip()]
os.makedirs(args.out, exist_ok=True)

def log(*a):
    msg = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(args.out, "run.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def is_correct(ans_text, answers):
    t = ans_text.lower()
    nums = set()
    for m in NUM_RE.findall(ans_text):
        try:
            nums.add(float(m.replace(",", "")))
        except ValueError:
            pass
    for a in answers:
        if re.fullmatch(r"-?\d+(\.\d+)?", a):
            if float(a) in nums:
                return True
        elif a.lower() in t:
            return True
    return False

questions = json.load(open(args.questions, encoding="utf-8"))
H_PATH = os.path.join(args.out, "hiddens.npz")
LENS_PATH = os.path.join(args.out, "lens.json")
SWEEP_PATH = os.path.join(args.out, "sweep.jsonl")
REPORT_PATH = os.path.join(args.out, "REPORT.md")

need_model = any(s in PHASES for s in ("capture", "lens", "sweep"))
model = tok = None
dev = None
if need_model:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    log("loading model")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.layers
    L = len(layers)  # decoder layers; index 0 in our arrays is the embedding output
    NL = L + 1
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    EOS_IDS = sorted({i for i in [IM_END, tok.eos_token_id] if isinstance(i, int) and i >= 0})
    log("loaded", type(model).__name__, "layers", L, "vram_gb", round(torch.cuda.memory_allocated() / 1024**3, 3))

def build_prompt(text):
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def locate(ids):
    """Content span inside the user turn, and the last content token."""
    end = ids.index(IM_END)
    # <|im_start|> user \n  ...content... <|im_end|>
    start = 3
    if end <= start:
        start = 1
    return start, end, end - 1

@torch.no_grad()
def capture_prompt(text):
    prompt = build_prompt(text)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
    ids = enc["input_ids"][0].tolist()
    start, end, user_end = locate(ids)
    buf = [None] * NL
    hooks = []

    def mk(idx):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            buf[idx] = h[0].float().cpu()  # [T, D]
        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(mk(0)))
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(mk(i + 1)))
    model(input_ids=enc["input_ids"], attention_mask=torch.ones_like(enc["input_ids"]), use_cache=False)
    for h in hooks:
        h.remove()
    stacked = torch.stack(buf, 0)  # [NL, T, D]
    return {
        "ids": ids,
        "user_start": start,
        "user_end": user_end,
        "user_stop": end,
        "last": len(ids) - 1,
        "h": stacked.numpy(),
    }

def vec_at(pack, layer, site):
    h = pack["h"][layer]
    if site == "user_end":
        return h[pack["user_end"]]
    if site == "last":
        return h[pack["last"]]
    if site == "user_mean":
        return h[pack["user_start"]:pack["user_stop"]].mean(0)
    raise KeyError(site)

if "capture" in PHASES:
    t0 = time.time()
    log("capture carrier and", len(questions), "questions")
    carrier = capture_prompt(CARRIER)
    qpacks = [capture_prompt(q["question"]) for q in questions]
    # Store user_end / last / user_mean for every layer. Full sequences are large; keep vectors only.
    D = carrier["h"].shape[-1]
    sites = ["user_end", "last", "user_mean"]
    arrays = {}
    for site in sites:
        arrays[f"carrier_{site}"] = np.stack([vec_at(carrier, l, site) for l in range(NL)]).astype(np.float32)
        arrays[f"q_{site}"] = np.stack(
            [[vec_at(p, l, site) for l in range(NL)] for p in qpacks]
        ).astype(np.float32)  # [N, NL, D]
    meta = {
        "carrier_text": CARRIER,
        "carrier_user_end": carrier["user_end"],
        "carrier_last": carrier["last"],
        "carrier_user_start": carrier["user_start"],
        "carrier_user_stop": carrier["user_stop"],
        "q_user_end": [p["user_end"] for p in qpacks],
        "n_layers_with_emb": NL,
        "hidden": int(D),
        "qids": [q["id"] for q in questions],
    }
    np.savez_compressed(H_PATH, **arrays)
    json.dump(meta, open(os.path.join(args.out, "capture_meta.json"), "w", encoding="utf-8"), indent=2)
    # same-question on/off is not captured; check that question vectors differ from the carrier
    delta = arrays["q_user_end"] - arrays["carrier_user_end"][None, :, :]
    norms = np.linalg.norm(delta, axis=-1)  # [N, NL]
    log("capture done", round(time.time() - t0, 1), "s; user_end delta L2 median per layer",
        [round(float(np.median(norms[:, l])), 2) for l in range(NL)])

def load_hiddens():
    z = np.load(H_PATH)
    meta = json.load(open(os.path.join(args.out, "capture_meta.json"), encoding="utf-8"))
    return z, meta

if "lens" in PHASES:
    import torch
    z, meta = load_hiddens()
    gold_ids = []
    for q in questions:
        ids = tok.encode(q["answers"][0], add_special_tokens=False)
        gold_ids.append(ids[0] if ids else -1)
    rows = []
    log("logit lens over", z["q_user_end"].shape[1], "layers")
    with torch.no_grad():
        for site in ("user_end", "last", "user_mean"):
            Q = z[f"q_{site}"]  # [N, NL, D]
            C = z[f"carrier_{site}"]
            for li in range(Q.shape[1]):
                # question residual and the memory delta, batched over questions
                hq = torch.tensor(Q[:, li], device=dev, dtype=torch.bfloat16)
                hd = torch.tensor(Q[:, li] - C[li], device=dev, dtype=torch.bfloat16)
                for kind, h in (("h", hq), ("delta", hd)):
                    logits = model.lm_head(model.model.norm(h)).float()
                    logp = torch.log_softmax(logits, dim=-1)
                    for qi, q in enumerate(questions):
                        gid = gold_ids[qi]
                        if gid < 0:
                            continue
                        rank = int((logits[qi] > logits[qi, gid]).sum().item()) + 1
                        top = torch.topk(logits[qi], 5)
                        rows.append({
                            "site": site, "layer": li, "kind": kind, "qid": q["id"], "qtype": q["type"],
                            "gold": q["answers"][0], "gold_id": gid,
                            "rank": rank, "logprob": round(float(logp[qi, gid]), 4),
                            "top5": [tok.decode([int(i)]) for i in top.indices.tolist()],
                        })
    json.dump(rows, open(LENS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    # median rank of the gold token from the raw residual at user_end
    by = {}
    for r in rows:
        if r["site"] == "user_end" and r["kind"] == "h":
            by.setdefault(r["layer"], []).append(r["rank"])
    med = {str(k): int(np.median(v)) for k, v in sorted(by.items())}
    log("lens user_end median gold rank by layer", med)

def done_keys():
    keys = set()
    if os.path.exists(SWEEP_PATH):
        for line in open(SWEEP_PATH, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            keys.add((r["direction"], r["alpha"], r["layer"], r["qid"]))
    return keys

@torch.no_grad()
def generate_with_patch(prompt_len_ids, layer_idx, pos, delta, alpha, mode):
    """mode pos: add alpha*delta at one index. mode all: add it at every prefill position."""
    import torch
    device_delta = torch.tensor(delta, device=dev, dtype=torch.bfloat16)
    prompt_len = len(prompt_len_ids)
    module = model.model.embed_tokens if layer_idx == 0 else layers[layer_idx - 1]

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        if h.shape[1] != prompt_len:
            return out
        h = h.clone()
        if mode == "all":
            h[0] = h[0] + alpha * device_delta
        else:
            h[0, pos] = h[0, pos] + alpha * device_delta
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = module.register_forward_hook(hook)
    try:
        ids = torch.tensor([prompt_len_ids], device=dev)
        gc = GenerationConfig(
            do_sample=False, max_new_tokens=args.max_new, eos_token_id=EOS_IDS,
            pad_token_id=EOS_IDS[0], temperature=None, top_p=None, top_k=None,
        )
        out = model.generate(input_ids=ids, generation_config=gc)
        gen = out[0, prompt_len:].tolist()
    finally:
        handle.remove()
    text = tok.decode(gen, skip_special_tokens=False)
    return text

if "sweep" in PHASES:
    import torch
    z, meta = load_hiddens()
    carrier_prompt = build_prompt(CARRIER)
    carrier_ids = tok(carrier_prompt, add_special_tokens=False)["input_ids"]
    # re-locate on this tokenization; must match capture
    _, _, user_end = locate(carrier_ids)
    last = len(carrier_ids) - 1
    if user_end != meta["carrier_user_end"] or last != meta["carrier_last"]:
        raise SystemExit(f"carrier positions changed: now {user_end},{last} vs {meta['carrier_user_end']},{meta['carrier_last']}")
    directions = []
    for a in ALPHAS:
        directions.append(("user_end", a, "pos", user_end, "user_end"))
    directions.append(("last", 1.0, "pos", last, "last"))
    directions.append(("user_mean", 1.0, "pos", user_end, "user_mean"))
    directions.append(("all_pos", 1.0, "all", user_end, "user_end"))
    seen = done_keys()
    NL = z["q_user_end"].shape[1]
    t0 = time.time()
    n_new = 0
    for name, alpha, mode, pos, site in directions:
        Q = z[f"q_{site}"]
        C = z[f"carrier_{site}"]
        for li in range(NL):
            for qi, q in enumerate(questions):
                key = (name, alpha, li, q["id"])
                if key in seen:
                    continue
                delta = Q[qi, li] - C[li]
                other = questions[(qi + 1) % len(questions)]
                text = generate_with_patch(carrier_ids, li, pos, delta, alpha, mode)
                rec = {
                    "direction": name, "alpha": alpha, "layer": li, "qid": q["id"], "qtype": q["type"],
                    "gold": q["answers"], "other_id": other["id"], "other_gold": other["answers"],
                    "hit_own": is_correct(text, q["answers"]),
                    "hit_other": is_correct(text, other["answers"]),
                    "text": text.replace("\n", " ")[:400],
                    "gen_s": None,
                }
                with open(SWEEP_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_new += 1
                if n_new % 20 == 0:
                    log(f"sweep {n_new} last {name} a={alpha} layer={li} {q['id']} own={rec['hit_own']} other={rec['hit_other']}")
    log("sweep finished", n_new, "new gens in", round(time.time() - t0, 1), "s")

def summarize():
    rows = [json.loads(l) for l in open(SWEEP_PATH, encoding="utf-8") if l.strip()]
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["direction"], r["alpha"], r["layer"])].append(r)
    ranking = []
    for (direction, alpha, layer), rs in groups.items():
        n = len(rs)
        recall = sum(r["hit_own"] for r in rs) / n
        intrusion = sum(r["hit_other"] for r in rs) / n
        ranking.append({
            "direction": direction, "alpha": alpha, "layer": layer, "n": n,
            "recall": round(recall, 4), "intrusion": round(intrusion, 4),
            "score": round(recall - intrusion, 4),
        })
    ranking.sort(key=lambda r: (r["score"], -r["layer"]), reverse=True)
    # lens summary if present
    lens_lines = []
    if os.path.exists(LENS_PATH):
        lens = json.load(open(LENS_PATH, encoding="utf-8"))
        buck = defaultdict(list)
        for r in lens:
            if r["site"] == "user_end":
                buck[(r["kind"], r["layer"])].append(r["rank"])
        lens_lines.append("Median rank of the gold answer's first token at user_end (1 = top of the vocabulary).")
        lens_lines.append("")
        lens_lines.append("| layer | rank from residual | rank from delta |")
        lens_lines.append("|---|---|---|")
        layers_present = sorted({k[1] for k in buck})
        for li in layers_present:
            rh = int(np.median(buck.get(("h", li), [0])))
            rd = int(np.median(buck.get(("delta", li), [0])))
            lens_lines.append(f"| {li} | {rh} | {rd} |")
    best = ranking[0] if ranking else None
    lines = []
    lines.append("# Recall sweep")
    lines.append("")
    lines.append(f"Carrier prompt, thinking off: `{CARRIER}`")
    lines.append("")
    lines.append("The carrier does not contain the question. Each trial adds `alpha * (h_question - h_carrier)` at one layer during prefill, then greedy-decodes up to "
                 f"{args.max_new} new tokens. `recall` is the fraction of trials whose text contains that question's answer. "
                 "`intrusion` is the fraction that contains the next question's answer. `score = recall - intrusion`.")
    lines.append("")
    lines.append("Layer 0 is the embedding output. Layers 1–24 are decoder blocks. Full-attention blocks are 4, 8, 12, 16, 20, 24.")
    lines.append("")
    if best:
        lines.append(f"Highest score in this run: **{best['direction']} / alpha {best['alpha']} / layer {best['layer']}** "
                     f"(recall {best['recall']}, intrusion {best['intrusion']}, score {best['score']}, n={best['n']}).")
        lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| direction | alpha | layer | n | recall | intrusion | score |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in ranking:
        lines.append(f"| {r['direction']} | {r['alpha']} | {r['layer']} | {r['n']} | {r['recall']} | {r['intrusion']} | {r['score']} |")
    lines.append("")
    if lens_lines:
        lines.append("## Logit lens diagnostic")
        lines.append("")
        lines.extend(lens_lines)
        lines.append("")
        lines.append("A low rank here means the answer token is already readable before injection. "
                     "If that layer also wins the recall score, the memory is acting as an answer cache. "
                     "A high rank together with a high recall score means the answer showed up only after the vector was injected into the carrier.")
    lines.append("")
    lines.append("n = 20 questions. This ranks the pipeline; it is not a claim about the layer's function in general.")
    text = "\n".join(lines) + "\n"
    open(REPORT_PATH, "w", encoding="utf-8").write(text)
    json.dump(ranking, open(os.path.join(args.out, "ranking.json"), "w", encoding="utf-8"), indent=2)
    log("wrote", REPORT_PATH)
    if best:
        log("BEST", best)

if "report" in PHASES:
    if not os.path.exists(SWEEP_PATH):
        log("no sweep yet")
    else:
        summarize()
