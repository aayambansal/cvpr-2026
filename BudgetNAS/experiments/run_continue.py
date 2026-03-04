#!/usr/bin/env python3
"""Continue v2 experiments from where run_all_v2 left off."""
import os, sys, json, time, random, gc
import numpy as np
import torch

# Import everything from run_all_v2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all_v2 import (
    DEV, SEEDS, DOM, BS, EPOCHS, CHUNKS, LR, WD, N_TRAIN, N_TEST,
    get_data, run_fixed, run_growing, run_ewc, run_er, run_den, run_rnas, run_bnas,
    run_grad, save_json, RES
)

def main():
    # Load existing results
    rpath = os.path.join(RES, "v2_results.json")
    with open(rpath) as f:
        ALL = json.load(f)

    methods = [
        ("fixed", run_fixed),
        ("growing", run_growing),
        ("ewc", run_ewc),
        ("er", run_er),
        ("den", run_den),
        ("rnas", lambda d,s: run_rnas(d,s,3)),
        ("bnas", lambda d,s: run_bnas(d,s,3)),
    ]

    # ── finish remaining seeds ──
    if "seeds" not in ALL:
        ALL["seeds"] = {}
    for seed in SEEDS:
        sseed = str(seed)
        if sseed not in ALL["seeds"]:
            ALL["seeds"][sseed] = {}
        data = get_data(seed)
        for mk, fn in methods:
            if mk in ALL["seeds"][sseed]:
                continue  # already done
            print(f"  Seed {seed} / {mk}...", end=" ", flush=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            r = fn(data, seed)
            ALL["seeds"][sseed][mk] = r
            print(f"done ({r['over']['t']:.0f}s)")
            save_json(ALL, "v2_results.json")
            gc.collect()

    # ── budget sweep (seed 42 only) ──
    if "budget_sweep" not in ALL:
        ALL["budget_sweep"] = {}
    data42 = get_data(42)
    for B in [0, 1, 2, 3, 5]:
        sb = str(B)
        if sb in ALL["budget_sweep"]:
            continue
        print(f"  Budget B={B}...", end=" ", flush=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        r = run_bnas(data42, 42, B=B)
        ALL["budget_sweep"][sb] = r
        print(f"done ({r['over']['t']:.0f}s)")
        save_json(ALL, "v2_results.json")

    # ── gradual shift (seed 42) ──
    if "gradual" not in ALL:
        ALL["gradual"] = {}
    for meth in ["fixed", "bnas"]:
        if meth in ALL["gradual"]:
            continue
        print(f"  Gradual {meth}...", end=" ", flush=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        r = run_grad(data42, 42, meth)
        ALL["gradual"][meth] = r
        print(f"done ({r['over']['t']:.0f}s)")
        save_json(ALL, "v2_results.json")

    # ── AGGREGATE ──
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (mean +/- std over 3 seeds)")
    print(f"{'='*70}")
    mks = ["fixed","growing","ewc","er","den","rnas","bnas"]
    agg = {}
    for mk in mks:
        agg[mk] = {}
        for dk in DOM:
            fa=[]; av=[]
            for s in SEEDS:
                ss = str(s)
                es=[t for t in ALL["seeds"][ss][mk]["timeline"] if t["ds"]==dk]
                if es: fa.append(es[-1]["acc"]); av.append(np.mean([t["acc"] for t in es]))
            fp=ALL["seeds"][str(SEEDS[0])][mk]["timeline"]
            dse=[t for t in fp if t["ds"]==dk]
            agg[mk][dk]=dict(fm=float(np.mean(fa)*100), fs=float(np.std(fa)*100),
                             am=float(np.mean(av)*100), a_s=float(np.std(av)*100),
                             p=dse[-1]["p"] if dse else 0, bl=dse[-1]["bl"] if dse else 0)
        wt=[ALL["seeds"][str(s)][mk]["over"]["t"] for s in SEEDS]
        mt=[ALL["seeds"][str(s)][mk]["over"]["mut"] for s in SEEDS]
        agg[mk]["wt"]=float(np.mean(wt)); agg[mk]["mut"]=float(np.mean(mt))
        nm=ALL["seeds"][str(SEEDS[0])][mk]["method"]
        print(f"\n  {nm}")
        for dk in DOM:
            a=agg[mk][dk]
            print(f"    {dk:10s}: {a['fm']:.1f}+/-{a['fs']:.1f}%  avg={a['am']:.1f}+/-{a['a_s']:.1f}%  {a['p']/1e6:.2f}M  {a['bl']}blk")
        print(f"    time={agg[mk]['wt']:.0f}s  muts={agg[mk]['mut']:.0f}")

    ALL["aggregated"] = agg

    # Budget sweep summary
    print(f"\n{'='*50} BUDGET SWEEP {'='*50}")
    for B in [0,1,2,3,5]:
        sb = str(B)
        r=ALL["budget_sweep"][sb]["timeline"]
        line = f"  B={B}: "
        for dk in DOM:
            es=[t for t in r if t["ds"]==dk]
            if es: line+=f"{dk}={es[-1]['acc']*100:.1f}% "
        print(line)

    save_json(ALL, "v2_results.json")
    print(f"\nAll results saved to {os.path.join(RES,'v2_results.json')}")

if __name__ == "__main__":
    main()
