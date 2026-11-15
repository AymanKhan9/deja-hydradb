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
        if gold: tasks.append({"i": len(tasks), "q": t["question"], "gold": gold, "servers": t["servers"], "gold_servers": {cat[g]["server"] for g in gold},
                               "flows": [(t["calls"][i]["tool"], t["calls"][j]["tool"]) for i, j, _, _ in t["flows"]]})
    return cat, tasks

def embed(texts, cache):
    if os.path.exists(cache): return np.load(cache)
    from fastembed import TextEmbedding
    m = TextEmbedding("BAAI/bge-small-en-v1.5", threads=4)
    E = np.array(list(m.embed(texts, batch_size=32)), dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True); np.save(cache, E); return E

class Learned:
    """What the graph stores: Task-USED->Tool (episodic memory), Term-LED->Tool and Tool-CO->Tool counts.
    In-memory twin of the HydraDB store; HydraLearned overrides the three *_scores reads with graph traversals.
    Served graph keeps only term links seen >= MIN_N times, top K per term (HydraDB reads cost ~1 ms/row; this cuts
    a typical loadout query from ~5.8k rows to ~200)."""
    K, MIN_N = int(os.environ.get("DEJA_K", 10)), int(os.environ.get("DEJA_MIN_N", 2))
    TOPT, KNN = 8, 20          # query only the 8 highest-idf terms; vote with the 20 most similar past tasks
    Q = None                   # task embedding matrix, set by the caller
    def links(s, w): return [(g, n) for g, n in sorted(s.led[w].items(), key=lambda x: (-x[1], x[0]))[:s.K] if n >= s.MIN_N]  # deterministic ties
    def __init__(s):
        s.led, s.tn, s.co, s.uses, s.df, s.N = collections.defaultdict(collections.Counter), collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter(), collections.Counter(), 0
        s.past, s.pastgold = [], {}                         # past task ids (rows of Q) and the tools each one used
    def neighbours(s, qvec):
        """K most similar past tasks by embedding -> [(task id, weight)]. Plain vector search (numpy), no graph."""
        if not s.past: return []
        ids = np.array(s.past); sc = s.Q[ids] @ qvec
        top = np.argsort(-sc, kind="stable")[:s.KNN]
        return [(int(ids[j]), max(float(sc[j]), 0.0)) for j in top]
    def votes(s, nb):            # Task -USED-> Tool, weighted by similarity (graph version reads the USED edges)
        return {i: s.pastgold[i] for i, _ in nb}
    def knn_scores(s, qvec):
        nb = s.neighbours(qvec); z = sum(w for _, w in nb)
        if not z: return {}
        used = s.votes(nb); out = collections.Counter()
        for i, w in nb:
            for g in used.get(i, ()): out[g] += w / z
        return dict(out)
    def ingest(s, task):
        s.past.append(task["i"]); s.pastgold[task["i"]] = sorted(task["gold"])
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
        ts = sorted((w for w in terms(q) if w in s.led and s.links(w)), key=lambda w: (s.df[w], w))[:s.TOPT]   # highest idf first
        out = collections.Counter(); z = 0
        for w in ts:
            idf = math.log((s.N + 1) / (s.df[w] + 0.5)); z += idf
            for g, n in s.links(w): out[g] += idf * n / s.tn[w]
        return {g: v / z for g, v in out.items()} if z else {}
    def term_scores_all(s, q):   # v1 (shipped before the review): every query term, for the comparison row only
        ts = [w for w in terms(q) if w in s.led and s.links(w)]; out = collections.Counter(); z = 0
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

LAM, BETA, GAMMA, SEEDS = 1.0, 0.5, 0.2, 5

def rank(cand, score):
    return sorted(cand, key=lambda n: (-round(float(score(n)), 9), n))   # deterministic ties

def deja_rank(L, q, qvec, sim, V, nix, use_knn=True, use_terms=True, use_co=True):
    """Fused ranking. Returns (ranked tools, parts) so the UI can explain each pick."""
    kn = L.knn_scores(qvec) if use_knn else {}
    fused = lambda n: sim[nix[n]] + LAM * kn.get(n, 0)
    seeds = rank(set(V[:200]) | set(kn), fused)[:SEEDS]           # co-use hop starts from the memory-informed top 5
    ts = L.term_scores(q) if use_terms else {}
    cs = L.co_scores(seeds) if use_co else {}
    final = lambda n: fused(n) + BETA * ts.get(n, 0) + GAMMA * cs.get(n, 0)
    return rank(set(V[:200]) | set(kn) | set(ts) | set(cs), final), {"knn": kn, "terms": ts, "co": cs}

def run(cat, tasks, Q, TE, names, budget, seed=None, store=Learned):
    idx = list(range(len(tasks)))
    if seed is not None: random.Random(seed).shuffle(idx)
    store.Q = Q; L = store(); res = []; nix = {n: k for k, n in enumerate(names)}
    for pos, i in enumerate(idx):
        t = tasks[i]; sim = TE @ Q[i]; order = np.argsort(-sim, kind="stable")
        V = [names[k] for k in order[:400]]
        bysrv = {}
        for n in V: bysrv.setdefault(cat[n]["server"], []).append(n)   # V is sim-ordered, so servers rank by best tool
        S = [n for ns in bysrv.values() for n in ns]
        P = rank(V, lambda n: sim[nix[n]] + 0.02 * math.log1p(L.uses[n]))
        kn = L.knn_scores(Q[i])
        K = rank(set(V[:200]) | set(kn), lambda n: sim[nix[n]] + LAM * kn.get(n, 0))
        ts1, cs1 = Learned.term_scores_all(L, t["q"]), L.co_scores(V[:SEEDS])
        v1 = rank(set(V[:200]) | set(ts1) | set(cs1), lambda n: sim[nix[n]] + 0.5 * ts1.get(n, 0) + 0.2 * cs1.get(n, 0))
        r = {"pos": pos, "multi": len(t["gold_servers"]) > 1, "n_gold": len(t["gold"])}
        for name, ranked in (("vector", V), ("servers", S), ("popularity", P), ("knn", K), ("deja-v1", v1),
                             ("deja", deja_rank(L, t["q"], Q[i], sim, V, nix)[0]),
                             ("deja-no-terms", deja_rank(L, t["q"], Q[i], sim, V, nix, use_terms=False)[0]),
                             ("deja-no-couse", deja_rank(L, t["q"], Q[i], sim, V, nix, use_co=False)[0])):
            lo = set(pack(ranked, cat, budget)); r[name] = t["gold"] <= lo
        res.append(r); L.ingest(t)
    return res

M = ["vector", "servers", "popularity", "knn", "deja-v1", "deja", "deja-no-terms", "deja-no-couse"]
EDGES = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400]          # log-spaced: the cold start is visible
def aggregate(res):
    e = EDGES + [len(res)]
    return {"overall": {m: float(np.mean([r[m] for r in res])) for m in M},
            "multi": {m: float(np.mean([r[m] for r in res if r["multi"]])) for m in M},
            "curve_edges": e, "curve": {m: [float(np.mean([r[m] for r in res[a:b]])) for a, b in zip(e, e[1:])] for m in M}}

def summarize(res, label):
    def line(rs, tag):
        return f"  {tag:<22} n={len(rs):5} " + " ".join(f"{m}={np.mean([r[m] for r in rs]):.1%}" for m in M)
    print(f"== {label}  (all gold tools in first loadout)")
    print(line(res, "all")); print(line([r for r in res if r["multi"]], "multi-server gold"))
    e = EDGES + [len(res)]
    for a, b in zip(e, e[1:]): print(line(res[a:b], f"tasks {a}-{b}"))

if __name__ == "__main__":
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    cat, tasks = build(); names = list(cat); names_ix = {n: k for k, n in enumerate(names)}
    print(f"catalog {len(cat)} tools / {len(set(c['server'] for c in cat.values()))} servers; tasks {len(tasks)}; multi-server gold {np.mean([len(t['gold_servers'])>1 for t in tasks]):.0%}; "
          f"median tool tokens {np.median([c['tokens'] for c in cat.values()])}, budget {budget}")
    TE = embed([cat[n]["text"] for n in names], "data/tool_emb.npy"); Q = embed([t["q"] for t in tasks], "data/q_emb.npy")
    orders = [None if x == "dataset" else int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["dataset", "1", "2", "3"])]
    out = {"budget": budget, "n_tasks": len(tasks), "n_tools": len(cat), "weights": {"lambda": LAM, "beta": BETA, "gamma": GAMMA, "knn": Learned.KNN, "top_terms": Learned.TOPT}, "orders": {}}
    for sd in orders:
        res = run(cat, tasks, Q, TE, names, budget, sd); summarize(res, f"order={sd or 'dataset'}")
        out["orders"][str(sd or "dataset")] = aggregate(res)
    json.dump(out, open(f"data/results_v2_{budget}.json", "w"), indent=1)
