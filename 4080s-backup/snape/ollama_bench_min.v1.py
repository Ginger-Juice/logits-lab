#!/usr/bin/env python3
"""Minimal Ollama streaming benchmark (stdlib only). TTFT / TPOT / prefill & decode tok/s."""
import json, time, urllib.request, statistics, sys, random, csv, os
URL = "http://127.0.0.1:11434/api/generate"
MODEL = os.environ.get("MODEL", "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:Q4_K_M")
NUM_CTX = int(os.environ.get("NUM_CTX", "32768"))
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "512,4096,16384").split(",")]
REPEATS = int(os.environ.get("REPEATS", "3"))
GEN = int(os.environ.get("GEN", "256"))
OUT = os.environ.get("OUT", "bench_results.csv")
FILLER = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens

def run(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": True, "raw": True,
                       "options": {"num_ctx": NUM_CTX, "num_predict": GEN, "temperature": 0}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; last = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            d = json.loads(line)
            if ttft is None and d.get("response"):
                ttft = time.perf_counter() - t0
            if d.get("done"):
                last = d
    wall = time.perf_counter() - t0
    pe, pd = last["prompt_eval_count"], last["prompt_eval_duration"] / 1e9
    ec, ed = last["eval_count"], last["eval_duration"] / 1e9
    return dict(prompt_tokens=pe, gen_tokens=ec, ttft_s=round(ttft, 4), wall_s=round(wall, 3),
                prefill_tok_s=round(pe / pd, 1) if pd else None,
                decode_tok_s=round(ec / ed, 2) if ed else None,
                tpot_ms=round(ed / ec * 1000, 2) if ec else None,
                tpot_wall_ms=round((wall - ttft) / max(ec - 1, 1) * 1000, 2))

def prompt_of(n_tokens):
    tag = f"[run {random.randint(0, 10**9)}] "  # defeat prompt cache
    return tag + FILLER * max(1, n_tokens // 10) + "\nSummarize the above in one paragraph:"

rows = []
print("warmup:", run(prompt_of(64)))
for L in LENGTHS:
    for i in range(REPEATS):
        r = run(prompt_of(L)); r.update(target_len=L, rep=i); rows.append(r); print(r, flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nsummary (median):")
for L in LENGTHS:
    g = [r for r in rows if r["target_len"] == L]
    med = lambda k: statistics.median(r[k] for r in g)
    print(f"len~{L}: prompt_tok={med('prompt_tokens')} TTFT={med('ttft_s')}s prefill={med('prefill_tok_s')} tok/s "
          f"decode={med('decode_tok_s')} tok/s TPOT={med('tpot_ms')} ms")
