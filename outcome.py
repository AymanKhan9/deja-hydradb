"""C: record whether each call worked. A call failed if its response opens with an error (toucan.call_ok).

Variants, same prequential stream as retr.py (score, then learn):
  as-is      every called tool votes / counts (what's deployed)
  -ok        only tools with a successful call in that task vote / count
  -ok+pen    -ok, plus a penalty for tools with a poor track record so far: score -= PEN * failure_rate (>= 3 tasks seen)
Each is scored three ways: every CALLED tool loaded (old metric), every tool that SUCCEEDED loaded (new metric),
and how often the loadout contains a chronically broken tool (failed in >= 80% of the >= 10 tasks that called it).
"""
import collections, json, random, sys
import numpy as np
import retr
from retr import Learned, deja_rank, rank, pack

BUDGET, PEN = 2000, 0.3

class LearnedOK(Learned):
    """Learns only from calls that worked; keeps per-tool outcome counts (graph: Task-USED->Tool for successes,
    Task-FAILED->Tool for failures)."""
    def __init__(s):
        super().__init__(); s.okc, s.failc = collections.Counter(), collections.Counter()
    def ingest(s, task):
        for g in task["gold_ok"]: s.okc[g] += 1
        for g in task["failed"]: s.failc[g] += 1
        t2 = dict(task, gold=task["gold_ok"], flows=[f for f in task["flows"] if f[0] in task["gold_ok"] and f[1] in task["gold_ok"]])
        super().ingest(t2)
    def fail_rate(s, n):
        tot = s.okc[n] + s.failc[n]
        return s.failc[n] / tot if tot >= 3 else 0.0

def run(seed):
    cat, tasks = retr.build(); names = list(cat); nix = {n: k for k, n in enumerate(names)}
    TE, Q = retr.embed(None, "data/tool_emb.npy"), retr.embed(None, "data/q_emb.npy"); Learned.Q = Q
    calls, fails = collections.Counter(), collections.Counter()
    for t in tasks:
        for g in t["gold"]: calls[g] += 1
        for g in t["failed"]: fails[g] += 1
    broken = {g for g in calls if calls[g] >= 10 and fails[g] / calls[g] >= 0.8}
    idx = list(range(len(tasks))); random.Random(seed).shuffle(idx)
    L, LO = Learned(), LearnedOK(); res = []
    for pos, i in enumerate(idx):
        t = tasks[i]; q = Q[i]; sim = TE @ q; V = [names[k] for k in np.argsort(-sim, kind="stable")[:400]]
        r = {"pos": pos, "multi": len(t["gold_servers"]) > 1, "has_ok": bool(t["gold_ok"]), "had_failure": bool(t["failed"])}
        variants = {"vector": V, "search": retr.search_rank(t["q"], sim, V, nix)}     # search = hybrid tool search
        for tag, St in (("", L), ("-ok", LO)):
            variants["knn" + tag] = retr.memory_rank(St, t["q"], q, sim, V, nix)
            variants["deja" + tag] = deja_rank(St, t["q"], q, sim, V, nix)[0]
        qset = LO.quarantined(); act = set(names) - qset                      # quarantine: never scored while held
        variants["deja-ok+quar"] = deja_rank(LO, t["q"], q, sim, [n for n in V if n in act], nix, active=act)[0]
        r["quarantined"] = len(qset)
        fr = {n: LO.fail_rate(n) for n in variants["deja-ok"][:400]}
        pos_of = {n: k for k, n in enumerate(variants["deja-ok"])}
        # penalty re-rank: keep Deja's order as the base score proxy (rank position), push down tools that keep failing
        variants["deja-ok+pen"] = sorted(variants["deja-ok"][:400], key=lambda n: (pos_of[n] / 400 + PEN * fr.get(n, 0), n))
        for name, ranked in variants.items():
            lo = set(pack(ranked, cat, BUDGET))
            r[name] = t["gold"] <= lo; r[name + "@ok"] = t["gold_ok"] <= lo if t["gold_ok"] else None
            r[name + "@broken"] = bool(lo & broken)
        res.append(r); L.ingest(t); LO.ingest(t)
    return res, len(broken)

if __name__ == "__main__":
    out = {}
    for seed in (1, 2, 3):
        res, nb = run(seed); M = [k for k in res[0] if k in ("vector", "search") or k.startswith(("knn", "deja")) and "@" not in k and k != "quarantined"]
        agg = {m: {"called": float(np.mean([r[m] for r in res])), "succeeded": float(np.mean([r[m + "@ok"] for r in res if r["has_ok"]])),
                   "called_on_tasks_with_a_failure": float(np.mean([r[m] for r in res if r["had_failure"]])),
                   "succeeded_on_tasks_with_a_failure": float(np.mean([r[m + "@ok"] for r in res if r["had_failure"] and r["has_ok"]])),
                   "broken_tool_loaded": float(np.mean([r[m + "@broken"] for r in res]))} for m in M}
        agg["_quarantined_per_task"] = float(np.mean([r["quarantined"] for r in res]))
        out[seed] = agg; print(f"== seed {seed} ({nb} chronically broken tools)")
        for m, a in ((m, a) for m, a in agg.items() if not m.startswith("_")):
            print(f"  {m:<12} called {a['called']:.1%}  succeeded {a['succeeded']:.1%}  | on tasks with a failed call: called "
                  f"{a['called_on_tasks_with_a_failure']:.1%} succeeded {a['succeeded_on_tasks_with_a_failure']:.1%} | broken tool loaded {a['broken_tool_loaded']:.1%}")
        sys.stdout.flush()
    json.dump(out, open("data/outcome.json", "w"), indent=1)
