"""A + B: abstain when nothing in the catalog fits; fall back to plain tool search when memory is weak (cold start).

Benchmark (Toucan has no "no suitable tool" tasks, so we make them): hold out whole capability clusters of servers —
a cluster of near-duplicate servers (tool-embedding cosine >= 0.85, single-linkage) plus every server within 0.80 of it,
so a removed weather server doesn't leave ten substitute weather servers behind. Their tools leave the catalog; tasks
whose needed servers were ALL removed AND where no remaining tool resembles any needed tool (cosine < SUBST) become
"unanswerable"; tasks needing none of them stay "answerable"; everything in between is dropped as ambiguous. Prequential stream as in retr.py: score each task, then learn from it (answerable only:
an unanswerable task has nothing in the catalog to learn).

Thresholds/weights are fit on seed 1 (its own held-out clusters) and evaluated on seeds 2 and 3 (different clusters).
"""
import collections, json, random, sys
import numpy as np
import retr
from retr import Learned, deja_rank, rank, pack

BUDGET = 2000
cat, tasks = retr.build(); names = list(cat); nix = {n: k for k, n in enumerate(names)}
TE, Q = retr.embed(None, "data/tool_emb.npy"), retr.embed(None, "data/q_emb.npy"); Learned.Q = Q
srv = sorted({cat[n]["server"] for n in cat}); by = collections.defaultdict(list)
for k, n in enumerate(names): by[cat[n]["server"]].append(k)
SE = np.array([TE[by[s]].mean(0) for s in srv]); SE /= np.linalg.norm(SE, axis=1, keepdims=True); SC = SE @ SE.T

def clusters(tau=0.85):
    par = list(range(len(srv)))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in zip(*np.nonzero(np.triu(SC >= tau, 1))): par[f(i)] = f(j)
    cl = collections.defaultdict(set)
    for i in range(len(srv)): cl[f(i)].add(i)
    return sorted(cl.values(), key=lambda c: min(c))

def held_out_servers(seed, frac=0.10, near=0.80):
    cl = clusters(); random.Random(seed).shuffle(cl); out = set()
    for c in cl:
        if len(out) >= frac * len(srv): break
        out |= c | {j for i in c for j in np.nonzero(SC[i] >= near)[0]}       # take the substitutes with it
    return {srv[i] for i in out}

SUBST = 0.80   # tool-tool cosine: substitutes measured 0.83-0.93, non-substitutes 0.52-0.78 (hand-checked pairs)
FEATS = ["s_mem", "s_tool", "vote_top", "vote_agree", "srv_spread", "co_coh"]
SIM_ONLY = ["s_mem", "s_tool"]

def stream(seed):
    out = held_out_servers(seed)
    active = {n for n in names if cat[n]["server"] not in out}; act_ix = np.array([nix[n] for n in names if n in active])
    ans = [t for t in tasks if not (t["gold_servers"] & out)]
    sub = lambda t: max(float((TE[act_ix] @ TE[nix[g]]).max()) for g in t["gold"])        # closest remaining substitute
    un = [t for t in tasks if t["gold_servers"] <= out and sub(t) < SUBST]                 # nothing left resembles any needed tool
    order = ans + un; random.Random(seed).shuffle(order); L = Learned(); rows = []
    for pos, t in enumerate(order):
        q = Q[t["i"]]; sim = TE @ q; a_sim = sim[act_ix]
        V = [names[act_ix[k]] for k in np.argsort(-a_sim, kind="stable")[:400]]
        nb = L.neighbours(q); ranked, parts = deja_rank(L, t["q"], q, sim, V, nix, active=active)
        kn = parts["knn"]; top5 = ranked[:5]
        votes = sorted(kn.values(), reverse=True)
        # agreement: how many of the 20 neighbours used one of the top-3 voted tools (graph read: Task-USED->Tool)
        top3 = set(sorted(kn, key=lambda n: -kn[n])[:3]); used = L.votes(nb) if nb else {}
        agree = np.mean([bool(set(used.get(i, ())) & top3) for i, _ in nb]) if nb else 0.0
        pairs = [(a, b) for a in top5 for b in top5 if a != b]
        r = {"pos": pos, "un": t["gold_servers"] <= out, "multi": len(t["gold_servers"]) > 1,
             "s_mem": nb[0][1] if nb else 0.0, "s_tool": float(a_sim.max()), "vote_top": votes[0] if votes else 0.0,
             "vote_agree": float(agree), "srv_spread": -len({cat[n]["server"] for n in top5}),       # fewer servers = more coherent
             "co_coh": np.mean([L.co[a][b] > 0 for a, b in pairs]) if pairs else 0.0}
        if not r["un"]:
            r["deja_ok"] = t["gold"] <= set(pack(ranked, cat, BUDGET)); r["vec_ok"] = t["gold"] <= set(pack(V, cat, BUDGET))
            kr = rank(set(V[:200]) | set(kn), lambda n: sim[nix[n]] + retr.LAM * kn.get(n, 0))
            r["knn_ok"] = t["gold"] <= set(pack(kr, cat, BUDGET))
            L.ingest(t)
        rows.append(r)
    return {"seed": seed, "held_servers": len(out), "n_ans": len(ans), "n_un": len(un), "rows": rows}

def auroc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort(kind="stable").argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def fit(X, y, iters=3000, lr=0.5):
    """Logistic regression (numpy, standardized features). y=1 means 'answerable'."""
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - y; w -= lr * Z.T @ g / len(y); b -= lr * g.mean()
    return lambda X2: 1 / (1 + np.exp(-(((X2 - mu) / sd) @ w + b))), dict(zip(range(len(w)), w.round(3)))

def evaluate(res, scorer, feats, thr, mode):
    """mode 'deja' / 'knn' / 'vec': which loadout answerable tasks get when Deja does not abstain."""
    R = res["rows"]; X = np.array([[r[f] for f in feats] for r in R]); p = scorer(X); abst = p < thr
    un = np.array([r["un"] for r in R]); ok = np.array([r.get(f"{mode}_ok", False) for r in R])
    ans = ~un
    return {"abstain_recall_unanswerable": float(abst[un].mean()), "false_abstain_answerable": float(abst[ans].mean()),
            "answerable_success": float((ok & ~abst)[ans].mean()), "answerable_success_no_abstain": float(ok[ans].mean()),
            "correct_outcome": float(np.where(un, abst, ok & ~abst).mean())}

if __name__ == "__main__":
    runs = {s: stream(s) for s in (1, 2, 3)}
    out = {"setup": {s: {k: v for k, v in r.items() if k != "rows"} for s, r in runs.items()}}
    print("setup:", out["setup"])
    # 1) separability of each signal (unanswerable vs answerable), all seeds
    out["auroc"] = {}
    for s, r in runs.items():
        R = r["rows"]; un = [x for x in R if x["un"]]; an = [x for x in R if not x["un"] and x["pos"] >= 200]   # warm memory
        out["auroc"][s] = {f: auroc(np.array([x[f] for x in an]), np.array([x[f] for x in un])) for f in FEATS}
        print(f"seed {s} AUROC answerable>unanswerable:", {f: round(v, 3) for f, v in out["auroc"][s].items()})
    # 2) abstainers fit on seed 1 at <=5% false abstain, evaluated on seeds 2,3
    out["policies"] = {}
    for label, feats in (("similarity only", SIM_ONLY), ("similarity + graph", FEATS)):
        R1 = runs[1]["rows"]; X1 = np.array([[r[f] for f in feats] for r in R1]); y1 = np.array([not r["un"] for r in R1], float)
        sc, w = fit(X1, y1); p1 = sc(X1); thr = float(np.quantile(p1[y1 == 1], 0.05))
        out["policies"][label] = {"weights": {feats[k]: v for k, v in w.items()}, "threshold": thr,
                                  "eval": {s: {m: evaluate(runs[s], sc, feats, thr, m) for m in ("deja", "knn", "vec")} for s in (2, 3)},
                                  "tradeoff": {str(fa): {s: evaluate(runs[s], sc, feats, float(np.quantile(p1[y1 == 1], fa)), "deja") for s in (2, 3)}
                                               for fa in (0.02, 0.05, 0.10, 0.20)}}
        for fa, ev in out["policies"][label]["tradeoff"].items():
            print(f"   [{label}] budget {fa} wrong-abstain on seed 1 -> seeds 2/3: catches " + " / ".join(f"{ev[s]['abstain_recall_unanswerable']:.0%}" for s in (2, 3)) +
                  ", wrongly abstains " + " / ".join(f"{ev[s]['false_abstain_answerable']:.0%}" for s in (2, 3)))
        for s in (2, 3):
            e = out["policies"][label]["eval"][s]
            print(f"[{label}] seed {s}: abstains on {e['deja']['abstain_recall_unanswerable']:.1%} of unanswerable, wrongly on "
                  f"{e['deja']['false_abstain_answerable']:.1%} of answerable; answerable success Deja {e['deja']['answerable_success']:.1%} "
                  f"(no abstain {e['deja']['answerable_success_no_abstain']:.1%}), memory-only {e['knn']['answerable_success']:.1%}")
    # 3) cold-start fallback: use tool search when the nearest past task is far; switch threshold fit on seed 1
    def fb(R, tau, mode="deja"): return np.mean([(r["vec_ok"] if r["s_mem"] < tau else r[f"{mode}_ok"]) for r in R if not r["un"]])
    R1 = runs[1]["rows"]; taus = np.round(np.arange(0.0, 0.96, 0.01), 2)
    tau = float(taus[int(np.argmax([fb(R1, t) for t in taus]))]); out["fallback"] = {"tau": tau, "eval": {}}
    edges = retr.EDGES
    for s in (2, 3):
        A = [r for r in runs[s]["rows"] if not r["un"]]
        bucket = lambda key: [float(np.mean([(r["vec_ok"] if (key == "fb" and r["s_mem"] < tau) else r["deja_ok" if key != "vec" else "vec_ok"])
                                             for r in A if a <= r["pos"] < b] or [np.nan])) for a, b in zip(edges, edges[1:] + [10 ** 9])]
        out["fallback"]["eval"][s] = {"deja": fb(A, -1), "deja+fallback": fb(A, tau), "knn": fb(A, -1, "knn"), "knn+fallback": fb(A, tau, "knn"),
                                      "vector": float(np.mean([r["vec_ok"] for r in A])), "curve_edges": edges,
                                      "curve": {"deja": bucket("deja"), "deja+fallback": bucket("fb"), "vector": bucket("vec")}}
        e = out["fallback"]["eval"][s]
        print(f"[fallback tau={tau}] seed {s}: Deja {e['deja']:.1%} -> {e['deja+fallback']:.1%}; memory-only {e['knn']:.1%} -> {e['knn+fallback']:.1%}; "
              f"tool search {e['vector']:.1%}; first buckets deja {np.round(e['curve']['deja'][:4], 2)} fb {np.round(e['curve']['deja+fallback'][:4], 2)} vec {np.round(e['curve']['vector'][:4], 2)}")
    json.dump(out, open("data/abstain.json", "w"), indent=1, default=float)
    # label audit sample: unanswerable tasks with the best remaining tools, for a by-eye check of substitutes
    rnd = random.Random(0); s = runs[2]; outs = held_out_servers(2); act = [k for k, n in enumerate(names) if cat[n]["server"] not in outs]
    with open("data/abstain_audit.txt", "w") as f:
        acts = np.array(act)
        for t in rnd.sample([t for t in tasks if t["gold_servers"] <= outs and max(float((TE[acts] @ TE[nix[g]]).max()) for g in t["gold"]) < SUBST], 25):
            sim = TE[act] @ Q[t["i"]]; top = [names[act[k]] for k in np.argsort(-sim)[:3]]
            f.write(f"Q: {t['q'][:220]}\n   needed: {sorted(t['gold'])}\n   best remaining: {top}\n\n")
    print("wrote data/abstain.json, data/abstain_audit.txt")
