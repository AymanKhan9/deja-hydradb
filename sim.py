"""Prequential replay (predict -> score -> ingest) of Toucan traces with an in-memory store.
Decides the headline (spike check 3) and later serves as the oracle the HydraDB store must match."""
import collections, json, random, sys
from toucan import load, mentions_for, norm

def grams(text, n=6):
    w = norm(text).split()
    return {" ".join(w[i:i + k]) for k in range(1, n + 1) for i in range(len(w) - k + 1)}

class Mem:
    def __init__(s):
        s.kv, s.stab = {}, collections.defaultdict(lambda: [0, 0])
        s.ment, s.eparams = collections.defaultdict(set), collections.defaultdict(set)
        s.eprod = {}
    def stable(s, tool):  # ponytail: unseen tools count as stable; flip to strict if precision suffers
        same, tot = s.stab[tool]
        return tot < 3 or same / tot >= 0.8
    def bindings(s, srv, question, tools):
        best = {}
        for g in grams(question):
            for v in s.ment.get((srv, g), ()):
                if not s.stable(s.eprod[(srv, v)]): continue
                for p in s.eparams[(srv, v)]:
                    if p.split("\x1f")[0] in tools and len(g) > best.get(p, (0, ""))[0]: best[p] = (len(g), v)
        return {p: v for p, (_, v) in best.items()}
    def ingest(s, t):
        srv, calls = t["servers"], t["calls"]
        for i, c in enumerate(calls):
            key = (srv, c["tool"], json.dumps(c["args"], sort_keys=True))
            old = s.kv.get(key)
            if old is not None:
                for a, j, k, v in t["flows"]:
                    if a == i: s.stab[c["tool"]][0] += v in old; s.stab[c["tool"]][1] += 1
            if c["out"]: s.kv[key] = c["out"]
        for i, j, k, v in t["flows"]:
            s.eprod[(srv, v)] = calls[i]["tool"]
            s.eparams[(srv, v)].add(calls[j]["tool"] + "\x1f" + k)
            for m in mentions_for(calls[i], v): s.ment[(srv, m)].add(v)

def score(store, t, prior):
    srv, calls = t["servers"], t["calls"]
    tools = {c["tool"] for c in calls}               # oracle loadout: isolates binding from tool retrieval
    pred = store.bindings(srv, t["question"], tools)
    used = {(c["tool"] + "\x1f" + k): v for c in calls for k, v in ((k, str(v).strip()) for k, v in c["args"].items() if not isinstance(v, (dict, list)))}
    hs_ok = sum(1 for p, v in pred.items() if used.get(p) == v)
    hs_bad = sum(1 for p, v in pred.items() if p in used and used[p] != v)
    prods = collections.defaultdict(list)
    for i, j, k, v in t["flows"]: prods[i].append((calls[j]["tool"] + "\x1f" + k, v))
    r = dict(prior=prior, producers=len(prods), kv=0, kv_bad=0, deja=0, hs_ok=hs_ok, hs_bad=hs_bad)
    for i, deps in prods.items():
        c = calls[i]; old = store.kv.get((srv, c["tool"], json.dumps(c["args"], sort_keys=True)))
        kv_ok = old is not None and all(v in old for _, v in deps)
        r["kv"] += kv_ok; r["kv_bad"] += old is not None and not kv_ok
        r["deja"] += (kv_ok and store.stable(c["tool"])) or all(pred.get(p) == v for p, v in deps)
    return r

def replay(traces, seed=None):
    if seed is not None: traces = random.Random(seed).sample(traces, len(traces))
    store, seen, out = Mem(), collections.Counter(), []
    for t in traces:
        out.append(score(store, t, seen[t["servers"]])); store.ingest(t); seen[t["servers"]] += 1
    return out

def report(res, label):
    B = lambda p: next(b for b in (0, 5, 10, 20, 50, 100, 200) [::-1] if p >= b)
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in res:
        for b in (B(r["prior"]), "all"): agg[b].update({**r, "n": 1, "n_dep": r["producers"] > 0})
    print(f"== {label}: bucket(prior traces on server-set) n tasks | tasks w/ lookup | lookups | saved KV | saved Deja | KV stale | bind ok/bad")
    for b in sorted(agg, key=lambda x: (x == "all", x if x != "all" else 0)):
        a = agg[b]
        print(f"  {str(b):>4} {a['n']:6} {a['n_dep']:6} {a['producers']:6} | {a['kv']:5} ({a['kv']/max(a['producers'],1):5.1%}) | {a['deja']:5} ({a['deja']/max(a['producers'],1):5.1%}) | {a['kv_bad']:4} | {a['hs_ok']}/{a['hs_bad']} prec {a['hs_ok']/max(a['hs_ok']+a['hs_bad'],1):.0%}")

if __name__ == "__main__":
    T = load(sys.argv[1] if len(sys.argv) > 1 else "data/raw/kimi_*.parquet")
    seen, D = set(), []
    for t in T:
        if t["prompt_id"] in seen: continue
        seen.add(t["prompt_id"]); D.append(t)
    sets = collections.Counter(t["servers"] for t in D)
    print(f"traces {len(T)}, unique prompt_id {len(D)}, server-sets {len(sets)}, multi-server {sum(len(t['servers'])>1 for t in D)/len(D):.0%}, "
          f"with flows {sum(bool(t['flows']) for t in D)/len(D):.0%}; sets with >=20 traces: {sum(v>=20 for v in sets.values())} covering {sum(v for v in sets.values() if v>=20)} traces")
    report(replay(D), "dataset order")
    for s in (1, 2, 3): report(replay(D, s), f"shuffle {s}")
