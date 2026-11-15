"""Prequential first-loadout eval (in-memory oracle): which tools go into the agent's context at a fixed token budget.
Score each task before learning from it. Gold = tools the agent actually called."""
import collections, json, math, os, random, sys
import numpy as np
from toucan import load, norm

STOP = set("a an the and or of to in on for with by from at is are be this that these those i me my we our you your it its as "
           "can could would should will do does did have has had need want help please also then than what which who how when where "
           "about into using use get find give make show tell any some all each other more most just like so if not no".split())
def terms(q): return {w for w in norm(q).split() if w not in STOP and len(w) > 2}

def build(path="data/raw/kimi_*.parquet", cache="data/traces_slim.json"):
    if os.path.exists(cache): T = json.load(open(cache))
    else: T = load(path, slim=True); json.dump(T, open(cache, "w"))
    seen = set(); D = []
    for t in T:
        if t["prompt_id"] not in seen: seen.add(t["prompt_id"]); D.append(t)
    cat, server = {}, {}
    for t in D:
        for td in t["tools"]:
            n = td["name"]
            if n not in cat: cat[n] = {"name": n, "text": f"{n}: {td['desc']}", "tokens": td["tokens"]}
        server.update(t["tool_server"])
    for n in cat: cat[n]["server"] = server.get(n, n.split("-")[0])
    tasks = []
    for t in D:
        gold = {c["tool"] for c in t["calls"] if c["tool"] in cat}
        if gold: tasks.append({"q": t["question"], "gold": gold, "servers": t["servers"], "gold_servers": {cat[g]["server"] for g in gold},
                               "flows": [(t["calls"][i]["tool"], t["calls"][j]["tool"]) for i, j, _, _ in t["flows"]]})
    return cat, tasks

def embed(texts, cache):
    if os.path.exists(cache): return np.load(cache)
    from fastembed import TextEmbedding
    m = TextEmbedding("BAAI/bge-small-en-v1.5", threads=4)
    E = np.array(list(m.embed(texts, batch_size=32)), dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True); np.save(cache, E); return E

class Learned:
    """What the graph stores: Term-LED_TO->Tool and Tool-CO_USED->Tool counts. In-memory twin of the HydraDB store.
    Served graph keeps only term links seen >= MIN_N times, top K per term (HydraDB reads cost ~1 ms/row; this cuts
    a typical loadout query from ~5.8k rows to ~200)."""
    K, MIN_N = int(os.environ.get("DEJA_K", 10)), int(os.environ.get("DEJA_MIN_N", 2))
    def links(s, w): return [(g, n) for g, n in sorted(s.led[w].items(), key=lambda x: (-x[1], x[0]))[:s.K] if n >= s.MIN_N]  # deterministic ties
    def __init__(s): s.led, s.tn, s.co, s.uses, s.df, s.N = collections.defaultdict(collections.Counter), collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter(), collections.Counter(), 0
    def ingest(s, task):
        ts = terms(task["q"]); s.N += 1
        for w in ts:
            s.df[w] += 1; s.tn[w] += 1
            for g in task["gold"]: s.led[w][g] += 1
        for g in task["gold"]:
            s.uses[g] += 1
            for h in task["gold"]:
                if g != h: s.co[g][h] += 1
        for a, b in task["flows"]: s.co[a][b] += 1          # observed data flow counts double
    def term_scores(s, q):
        ts = [w for w in terms(q) if w in s.led and s.links(w)]; out = collections.Counter(); z = 0   # same terms the graph can see
        for w in ts:
            idf = math.log((s.N + 1) / (s.df[w] + 0.5)); z += idf
            for g, n in s.links(w): out[g] += idf * n / s.tn[w]
        return {g: v / z for g, v in out.items()} if z else {}
    def co_scores(s, seeds):
        out = collections.Counter()
        for r, a in enumerate(seeds):
            if not s.uses[a]: continue
            for b, n in s.co[a].items(): out[b] = max(out[b], n / s.uses[a] / (1 + r))
        return out

def pack(ranked, cat, budget):
    out, used = [], 0
    for n in ranked:
        if used + cat[n]["tokens"] <= budget: out.append(n); used += cat[n]["tokens"]
    return out

def run(cat, tasks, Q, TE, names, budget, seed=None, beta=0.5, gamma=0.2, seeds_k=5, store=Learned):
    idx = list(range(len(tasks)))
    if seed is not None: random.Random(seed).shuffle(idx)
    L = store(); res = []; names_ix = {n: k for k, n in enumerate(names)}
    for pos, i in enumerate(idx):
        t = tasks[i]; sim = TE @ Q[i]; order = np.argsort(-sim)
        V = [names[k] for k in order[:400]]
        s = {names[k]: float(sim[k]) for k in order[:400]}
        # server baseline: load whole servers ranked by their best tool
        bysrv = {}
        for n in V: bysrv.setdefault(cat[n]["server"], []).append(n)   # V is sim-ordered, so servers rank by best tool
        S = [n for ns in bysrv.values() for n in ns]
        P = sorted(V, key=lambda n: -(s[n] + 0.02 * math.log1p(L.uses[n])))
        ts, cs = L.term_scores(t["q"]), L.co_scores(V[:seeds_k])
        def deja(b, g):
            cand = set(V[:200]) | set(ts) | set(cs)
            return sorted(cand, key=lambda n: (-round(float(sim[names_ix[n]] + b * ts.get(n, 0) + g * cs.get(n, 0)), 9), n))
        r = {"pos": pos, "multi": len(t["gold_servers"]) > 1, "n_gold": len(t["gold"])}
        for name, ranked in (("vector", V), ("servers", S), ("popularity", P), ("deja", deja(beta, gamma)),
                             ("deja-terms", deja(beta, 0)), ("deja-couse", deja(0, gamma))):
            lo = set(pack(ranked, cat, budget)); r[name] = t["gold"] <= lo; r[name + "_rec"] = len(t["gold"] & lo) / len(t["gold"])
        res.append(r); L.ingest(t)
    return res

M = ["vector", "servers", "popularity", "deja", "deja-terms", "deja-couse"]
def aggregate(res, windows=10):
    w = len(res) // windows
    return {"overall": {m: float(np.mean([r[m] for r in res])) for m in M},
            "multi": {m: float(np.mean([r[m] for r in res if r["multi"]])) for m in M},
            "curve": {m: [float(np.mean([r[m] for r in res[k * w:(k + 1) * w]])) for k in range(windows)] for m in M}}

def summarize(res, label):
    def line(rs, tag):
        return f"  {tag:<22} n={len(rs):5} " + " ".join(f"{m}={np.mean([r[m] for r in rs]):.1%}" for m in M)
    print(f"== {label}  (all gold tools in first loadout)")
    print(line(res, "all")); print(line([r for r in res if r["multi"]], "multi-server gold"))
    n = len(res); w = n // 5
    for k in range(5): print(line(res[k * w:(k + 1) * w], f"stream {k*20}-{k*20+20}%"))

if __name__ == "__main__":
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    cat, tasks = build(); names = list(cat); names_ix = {n: k for k, n in enumerate(names)}
    print(f"catalog {len(cat)} tools / {len(set(c['server'] for c in cat.values()))} servers; tasks {len(tasks)}; multi-server gold {np.mean([len(t['gold_servers'])>1 for t in tasks]):.0%}; "
          f"median tool tokens {np.median([c['tokens'] for c in cat.values()])}, budget {budget}")
    TE = embed([cat[n]["text"] for n in names], "data/tool_emb.npy"); Q = embed([t["q"] for t in tasks], "data/q_emb.npy")
    out = {"budget": budget, "n_tasks": len(tasks), "n_tools": len(cat), "orders": {}}
    for sd in (None, 1, 2, 3):
        res = run(cat, tasks, Q, TE, names, budget, sd); summarize(res, f"order={sd or 'dataset'}")
        out["orders"][str(sd or "dataset")] = aggregate(res)
    json.dump(out, open(f"data/results_{budget}_k{Learned.K}_n{Learned.MIN_N}.json", "w"), indent=1)
