import torch, traceback
for step, fn in [("init", lambda: torch.cuda.init()), ("cap", lambda: torch.cuda.get_device_capability(0)),
                 ("arch_list", lambda: torch.cuda.get_arch_list()), ("mem_get_info", lambda: torch.cuda.mem_get_info()),
                 ("empty", lambda: torch.empty(4, device="cuda").shape), ("zeros", lambda: torch.zeros(4, device="cuda").sum().item())]:
    try: print(step, fn(), flush=True)
    except Exception as e: print(step, "FAILED", repr(e)[:200], flush=True)