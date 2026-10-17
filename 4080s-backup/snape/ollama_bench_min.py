#!/usr/bin/env python3
"""Minimal Ollama streaming benchmark (stdlib only).
TTFT / TPOT / prefill & decode tok/s, with optional concurrency.

env: MODEL, NUM_CTX, LENGTHS (comma), REPEATS, GEN, OUT (csv), CONCURRENCY (comma list, e.g. "1,2,4"),
     RAW (1=raw prompt, 0=templated, default 0), IGNORE_EOS (1 to send options.ignore_eos), WARMUP (1/0),
     THINK (unset=don't send; "false"/"true" sent as top-level think)
Each "round" launches CONCURRENCY simultaneous requests (threads, barrier-synchronised).
Aggregate output throughput = sum(eval_count of the round) / round wall time.
"""
import json, time, urllib.request, statistics, random, csv, os, threading
URL = os.environ.get("URL", "http://127.0.0.1:11434/api/generate")
MODEL = os.environ.get("MODEL", "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:Q4_K_M")
NUM_CTX = int(os.environ.get("NUM_CTX", "32768"))
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "512,4096").split(",")]
REPEATS = int(os.environ.get("REPEATS", "3"))
GEN = int(os.environ.get("GEN", "256"))
OUT = os.environ.get("OUT", "bench_results.csv")
CONCS = [int(x) for x in os.environ.get("CONCURRENCY", "1").split(",")]
RAW = os.environ.get("RAW", "0") == "1"
IGNORE_EOS = os.environ.get("IGNORE_EOS", "0") == "1"
WARMUP = os.environ.get("WARMUP", "1") == "1"
THINK = os.environ.get("THINK")
FILLER = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens
ASK = ("\n\nIgnore the repetition above. Now write a very long, detailed essay (at least 2000 words) "
       "about the history of computing, with many sections. Begin immediately.")

def run(prompt):
    opts = {"num_ctx": NUM_CTX, "num_predict": GEN, "temperature": 0}
    if IGNORE_EOS:
        opts["ignore_eos"] = True
    payload = {"model": MODEL, "prompt": prompt, "stream": True, "raw": RAW, "options": opts}
    if THINK is not None:
        payload["think"] = THINK.lower() == "true"
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; last = None; chunks = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            d = json.loads(line)
            if ttft is None and (d.get("response") or d.get("thinking")):
                ttft = time.perf_counter() - t0
            if d.get("response") or d.get("thinking"):
                chunks += 1
            if d.get("done"):
                last = d
    t1 = time.perf_counter(); wall = t1 - t0
    pe, pd = last.get("prompt_eval_count", 0), last.get("prompt_eval_duration", 0) / 1e9
    ec, ed = last.get("eval_count", 0), last.get("eval_duration", 0) / 1e9
    ttft = ttft if ttft is not None else wall
    return dict(t_start=t0, t_end=t1, prompt_tokens=pe, gen_tokens=ec, stream_chunks=chunks,
                done_reason=last.get("done_reason"), ttft_s=round(ttft, 4), wall_s=round(wall, 3),
                prefill_tok_s=round(pe / pd, 1) if pd else None,
                decode_tok_s=round(ec / ed, 2) if ed else None,
                tpot_ms=round(ed / ec * 1000, 2) if ec else None,
                tpot_wall_ms=round((wall - ttft) / max(ec - 1, 1) * 1000, 2))

def prompt_of(n_tokens):
    tag = f"[run {random.randint(0, 10**9)}] "  # defeat prompt cache
    return tag + FILLER * max(1, n_tokens // 10) + ASK

def round_of(n, L):
    res = [None] * n; err = [None] * n
    bar = threading.Barrier(n)
    prompts = [prompt_of(L) for _ in range(n)]
    def worker(i):
        bar.wait()
        try:
            res[i] = run(prompts[i])
        except Exception as e:
            err[i] = repr(e)
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    if any(err):
        raise RuntimeError(err)
    t0 = min(r["t_start"] for r in res); t1 = max(r["t_end"] for r in res)
    tot = sum(r["gen_tokens"] for r in res)
    return res, round(t1 - t0, 3), tot

rows = []; rounds = []
if WARMUP:
    w, _, _ = round_of(1, 64); print("warmup:", w[0], flush=True)
for C in CONCS:
    for L in LENGTHS:
        for i in range(REPEATS):
            res, rw, tot = round_of(C, L)
            agg = round(tot / rw, 2)
            rounds.append(dict(conc=C, target_len=L, rep=i, round_wall_s=rw, total_gen_tokens=tot, agg_tok_s=agg))
            print(f"conc={C} len={L} rep={i} round_wall={rw}s total_gen={tot} agg={agg} tok/s", flush=True)
            base = min(r["t_start"] for r in res)
            for j, r in enumerate(res):
                r = dict(r); r["start_offset_s"] = round(r.pop("t_start") - base, 4); r.pop("t_end")
                r.update(conc=C, target_len=L, rep=i, req=j, round_wall_s=rw, round_agg_tok_s=agg)
                rows.append(r); print("  ", r, flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(OUT.replace(".csv", "") + "_rounds.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rounds[0].keys())); w.writeheader(); w.writerows(rounds)
print("\nsummary (median):")
for C in CONCS:
    for L in LENGTHS:
        g = [r for r in rows if r["conc"] == C and r["target_len"] == L]
        rg = [r for r in rounds if r["conc"] == C and r["target_len"] == L]
        med = lambda k: statistics.median(r[k] for r in g if r[k] is not None)
        print(f"conc={C} len~{L}: prompt_tok={med('prompt_tokens')} gen_tok min/med/max="
              f"{min(r['gen_tokens'] for r in g)}/{med('gen_tokens')}/{max(r['gen_tokens'] for r in g)} "
              f"TTFT={med('ttft_s')}s prefill={med('prefill_tok_s')} decode={med('decode_tok_s')} tok/s "
              f"TPOT={med('tpot_ms')}ms TPOT_wall={med('tpot_wall_ms')}ms "
              f"agg={statistics.median(r['agg_tok_s'] for r in rg)} tok/s", flush=True)
