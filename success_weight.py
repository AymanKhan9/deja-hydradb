"""Does weighting each past task's votes by whether the task succeeded help? (@OctoHallHQ's suggestion.)
Task success = Toucan's LLM-judge completeness score (1-5, response_quality_assessment). Served model (votes come only
from calls that worked), prequential shuffle 1. Also: for misses where every needed tool is in memory but ranked too
low, did any similar past task (or step) vote for it?      uv run python success_weight.py -> data/success_weight.json
"""
import collections, glob, json, random
import numpy as np, pyarrow.parquet as pq
import retr
from retr import Learned, deja_rank, pack, embed
T = json.load(open("data/traces_slim_ok.json"))
rq = {}
for f in sorted(glob.glob("data/raw/kimi_*.parquet")):
    for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["uuid", "response_quality_assessment"]):
        for r in b.to_pylist():
            try: a = json.loads(r["response_quality_assessment"] or "null")
            except Exception: a = None
            rq[r["uuid"]] = a
cat, tasks = retr.build()
seen, D = set(), []
for t in T:
    if t["prompt_id"] not in seen: seen.add(t["prompt_id"]); D.append(t)
D = [t for t in D if any(c["tool"] in cat for c in t["calls"])]
assert len(D) == len(tasks) and all(d["question"] == t["q"] for d, t in zip(D, tasks))
out = []
for d in D:
    a = rq.get(d["uuid"]) or {}
    out.append({"completeness": (a.get("completeness") or {}).get("score"), "overall": a.get("overall_score"), "desired_pct": a.get("desired_tools_used_percentage")})
QL = out
Learned.OUTCOME = True

VAR = {"current": None,
       "weight c/5": lambda c: c / 5,
       "drop c<=2": lambda c: 0.0 if c <= 2 else 1.0,
       "drop c=1": lambda c: 0.0 if c == 1 else 1.0}
W = {}
def tally(nb, used):                 # same normalisation as retr.tally, each past task's vote scaled by W[i]
    z = sum(w for _, w in nb); out = collections.Counter()
    if z:
        for i, w in nb:
            for g in used.get(i, ()): out[g] += w * W.get(i, 1.0) / z
    return out
retr.tally = tally
names = list(cat); nix = {n: k for k, n in enumerate(names)}
TE = embed(None, "data/tool_emb.npy"); Q = embed(None, "data/q_emb.npy"); Learned.Q = Q
idx = list(range(len(tasks))); random.Random(1).shuffle(idx)
L = Learned(); hits = collections.Counter(); hits_all = collections.Counter(); n_ok = 0; miss = []
for pos, i in enumerate(idx):
    t = tasks[i]; sim = TE @ Q[i]; V = [names[k] for k in np.argsort(-sim, kind="stable")[:400]]
    for name, f in VAR.items():
        W.clear()
        if f: W.update({j: f(QL[j]["completeness"] or 3) for j in L.past})
        ranked, parts = deja_rank(L, t["q"], Q[i], sim, V, nix)
        lo = set(pack(ranked, cat, 2000))
        hits_all[name] += t["gold"] <= lo
        if t["gold_ok"]: hits[name] += t["gold_ok"] <= lo
        if name == "current" and t["gold_ok"] and not t["gold_ok"] <= lo and pos >= 400:
            g_miss = t["gold_ok"] - lo
            fits = sum(cat[g]["tokens"] for g in t["gold_ok"]) <= 2000
            seen = all(L.uses[g] > 0 for g in t["gold_ok"])
            if fits and seen:
                pos_r = {n: k for k, n in enumerate(ranked)}
                miss.append({"voted": all(parts["knn"].get(g, 0) > 0 or parts["clause"].get(g, 0) > 0 for g in g_miss),
                             "any_neighbour_used_it_ok": all(any(g in L.pastgold[j] for j, _ in L.neighbours(Q[i])) for g in g_miss),
                             "rank": max(pos_r.get(g, 999) for g in g_miss), "n_loaded": len(lo),
                             "nb_quality": float(np.mean([QL[j]["completeness"] or 3 for j, _ in L.neighbours(Q[i])[:5]]))})
    n_ok += bool(t["gold_ok"]); L.ingest(t)
    if pos % 1000 == 999: print(pos + 1, {k: round(v / n_ok, 4) for k, v in hits.items()}, flush=True)
out = {"n": len(tasks), "n_scored": n_ok, "worked_tools_loaded": {k: v / n_ok for k, v in hits.items()},
       "called_tools_loaded": {k: v / len(tasks) for k, v in hits_all.items()},
       "ranked_low_misses_after_400": len(miss),
       "share_voted_by_a_neighbour": float(np.mean([m["voted"] for m in miss])),
       "share_used_by_a_top20_neighbour": float(np.mean([m["any_neighbour_used_it_ok"] for m in miss])),
       "rank_median": float(np.median([m["rank"] for m in miss])), "loaded_median": float(np.median([m["n_loaded"] for m in miss])),
       "rank_hist": collections.Counter(min(m["rank"] // 5 * 5, 50) for m in miss)}
json.dump(out, open("data/success_weight.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
