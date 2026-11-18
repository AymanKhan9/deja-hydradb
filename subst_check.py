"""A tool leaves the catalog but a near-identical tool stays (tool-tool cosine >= 0.80). Does the loadout carry the
substitute? Filter only (shipped: a removed tool's votes are dropped) vs passing them to its closest live substitute.
Memory-only gets the same treatment, so the gain is measured like for like. 10% of tools removed halfway through the
stream; memory keeps learning the recorded calls. Orderings 1-3.   uv run python subst_check.py -> data/subst_check.json"""
import collections, json, random
import numpy as np
import retr
from retr import deja_rank, memory_rank, search_rank, pack

B, COS = 2000, 0.80
cat, tasks = retr.build(); names = list(cat); nix = {n: k for k, n in enumerate(names)}
TE, Q = retr.embed(None, "data/tool_emb.npy"), retr.embed(None, "data/q_emb.npy"); retr.Learned.Q = Q

def one(seed):
    order = list(range(len(tasks))); random.Random(seed).shuffle(order); half = len(order) // 2
    gone = set(random.Random(seed + 100).sample(names, len(names) // 10)); live = [n for n in names if n not in gone]; after = set(live)
    S = TE[[nix[n] for n in sorted(gone)]] @ TE[[nix[n] for n in live]].T
    subs = {n: [live[k] for k in np.argsort(-row)[:5] if row[k] >= COS] for row, n in zip(S, sorted(gone))}
    sub = {n: v[0] for n, v in subs.items() if v}                # the closest live substitute
    L = retr.Learned(); res = collections.defaultdict(list); skipped = 0
    for pos, i in enumerate(order):
        t = tasks[i]; g = t["gold"]
        if pos >= half and g & gone:
            if all(subs[x] for x in g & gone):
                sim = TE @ Q[i]; V = [names[k] for k in np.argsort(-sim, kind="stable") if names[k] in after][:400]
                ok = lambda r: (lambda lo: all(x in lo if x not in gone else any(s in lo for s in subs[x]) for x in g))(set(pack(r, cat, B)))
                res["tool search"].append(ok(search_rank(t["q"], sim, V, nix)))
                res["memory-only, filter"].append(ok(memory_rank(L, t["q"], Q[i], sim, V, nix, active=after)))
                res["memory-only + substitute"].append(ok(memory_rank(L, t["q"], Q[i], sim, V, nix, active=after, sub=sub)))
                res["Deja, filter (shipped)"].append(ok(deja_rank(L, t["q"], Q[i], sim, V, nix, active=after)[0]))
                res["Deja + substitute"].append(ok(deja_rank(L, t["q"], Q[i], sim, V, nix, active=after, sub=sub)[0]))
            else: skipped += 1
        L.ingest(t)
    out = {"n": len(res["tool search"]), "no_substitute_skipped": skipped, **{k: float(np.mean(v)) for k, v in res.items()},
           "deja_sub_minus_memory_sub": [int(a) - int(b) for a, b in zip(res["Deja + substitute"], res["memory-only + substitute"])]}
    print(seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != "deja_sub_minus_memory_sub"}, flush=True)
    return out

if __name__ == "__main__":
    R = {s: one(s) for s in (1, 2, 3)}
    d = np.array([x for r in R.values() for x in r.pop("deja_sub_minus_memory_sub")], float)
    bs = [np.random.default_rng(k).choice(d, len(d)).mean() for k in range(2000)]
    out = {"orderings": R, "deja_minus_memory_with_substitute": {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}}
    json.dump(out, open("data/subst_check.json", "w"), indent=1); print(json.dumps(out["deja_minus_memory_with_substitute"]))
