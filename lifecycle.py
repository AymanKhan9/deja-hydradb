"""D: a tool disappears from tools/list. (1) @webmcpx's scenario on real HydraDB: learn a task that uses tool A, remove A,
repeat the task -> A must never be loaded and the token budget must still be packed. (2) Stream experiment (twin): drop
10% of tools halfway through; compare reading the live catalog against trusting stale memory."""
import json, random, sys
import numpy as np
import retr
from retr import Learned, deja_rank, pack
from hydra_store import HydraLearned, hid, write_catalog, remove_tool, read_catalog

BUDGET = 2000
cat, tasks = retr.build(); names = list(cat); nix = {n: k for k, n in enumerate(names)}
TE, Q = retr.embed(None, "data/tool_emb.npy"), retr.embed(None, "data/q_emb.npy")
Learned.Q = Q; HydraLearned.tool_names = {hid("tool", n): n for n in cat}
maxtok = max(c["tokens"] for c in cat.values())

def loadout(L, t, active):
    sim = TE @ Q[t["i"]]; order = [names[k] for k in np.argsort(-sim, kind="stable") if names[k] in active][:400]
    return pack(deja_rank(L, t["q"], Q[t["i"]], sim, order, nix, active=active)[0], cat, BUDGET)

def scenario(n_cases=40, seed=0):
    rnd = random.Random(seed); order = list(range(len(tasks))); rnd.shuffle(order)
    H = HydraLearned(); H.bulk_load([tasks[i] for i in order[:2000]]); write_catalog(H.db, names)
    cases = []
    for i in order[2000:]:
        if len(cases) == n_cases: break
        t = tasks[i]; H.ingest(t)                                         # 1. learn a task that used tool A
        active = read_catalog(H.db); lo = loadout(H, t, active)
        A = next((g for g in sorted(t["gold"]) if g in lo), None)
        if A is None: continue                                            #    (need A to be loaded before removal)
        remove_tool(H.db, A)                                              # 2. A leaves tools/list
        active2 = read_catalog(H.db); lo2 = loadout(H, t, active2)         # 3. repeat the task
        tok = sum(cat[n]["tokens"] for n in lo2)
        cases.append({"tool": A, "catalog_dropped": len(active) - len(active2), "A_absent": A not in lo2, "A_in_catalog": A in active2,
                      "tokens": tok, "packed": BUDGET - maxtok <= tok <= BUDGET, "others_kept": len(set(lo) - {A} - set(lo2))})
    return cases

def stream(seed):
    order = list(range(len(tasks))); random.Random(seed).shuffle(order); half = len(order) // 2
    gone = set(random.Random(seed + 100).sample(names, len(names) // 10)); full = set(names); after = full - gone
    L = Learned(); res = {"filtered": [], "stale": [], "vector": []}
    for pos, i in enumerate(order):
        t = tasks[i]
        if pos >= half and not (t["gold"] & gone):                        # tasks still answerable after the removal
            for k, act in (("filtered", after), ("stale", None)):
                if act is None:                                           # stale: memory still proposes removed tools
                    sim = TE @ Q[i]; V = [names[j] for j in np.argsort(-sim, kind="stable") if names[j] in after][:400]
                    lo = pack(deja_rank(L, t["q"], Q[i], sim, V, nix)[0], cat, BUDGET)
                    res[k].append((t["gold"] <= set(lo), sum(cat[n]["tokens"] for n in lo if n in gone)))
                else:
                    lo = loadout(L, t, act); res[k].append((t["gold"] <= set(lo), 0))
            sim = TE @ Q[i]; V = [names[j] for j in np.argsort(-sim, kind="stable") if names[j] in after][:400]
            res["vector"].append((t["gold"] <= set(pack(V, cat, BUDGET)), 0))
        L.ingest(t)
    return {k: {"success": float(np.mean([a for a, _ in v])), "tokens_on_removed_tools": float(np.mean([b for _, b in v])), "n": len(v)} for k, v in res.items()}

if __name__ == "__main__":
    cases = scenario()
    S = {"cases": len(cases), "A_never_loaded": sum(c["A_absent"] for c in cases), "A_gone_from_catalog": sum(not c["A_in_catalog"] for c in cases),
         "budget_still_packed": sum(c["packed"] for c in cases), "max_tokens": max(c["tokens"] for c in cases)}
    print("scenario (HydraDB):", S); assert S["A_never_loaded"] == S["cases"] == S["A_gone_from_catalog"]
    ST = {s: stream(s) for s in (1, 2, 3)}
    for s, r in ST.items(): print(f"stream seed {s}:", {k: (round(v['success'], 3), round(v['tokens_on_removed_tools'], 1)) for k, v in r.items()})
    json.dump({"scenario": S, "scenario_cases": cases, "stream": ST}, open("data/lifecycle.json", "w"), indent=1)
