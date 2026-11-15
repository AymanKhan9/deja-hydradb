"""Regenerate README.md's result tables from the eval JSON, so the prose never drifts from the numbers."""
import json, os
R = json.load(open("data/results_v2_2000.json")); O = [R["orders"][k] for k in ("dataset", "1", "2", "3")]
W = json.load(open("web/results.json")) if os.path.exists("web/results.json") else {}
H, L = W.get("hydra", {}), W.get("load", {})
def rng(m, part):
    v = [o[part][m] for o in O]; a, b = 100 * min(v), 100 * max(v)
    return f"{a:.1f}%" if f"{a:.1f}" == f"{b:.1f}" else f"{a:.1f}–{b:.1f}%"
rows = [("Tool search: vector top-k over tool descriptions", "vector"), ("Load whole servers, vector-ranked", "servers"),
        ("Vector + learned popularity (non-graph learner)", "popularity"),
        ("**Past-task memory, no graph** (20 nearest past tasks vote for their tools)", "knn"),
        ("Deja v1, first shipped: word + co-use edges, no task memory", "deja-v1"),
        ("**Deja**: task memory in HydraDB + co-use hop + word hop", "deja"),
        ("Deja without word edges (ablation)", "deja-no-terms"), ("Deja without co-use edges (ablation)", "deja-no-couse")]
main = "\n".join(f"| {n} | {rng(m, 'overall')} | {rng(m, 'multi')} |" for n, m in rows)
B = {b: json.load(open(f"data/results_v2_{b}.json"))["orders"]["1"]["overall"] for b in (1000, 2000, 4000) if os.path.exists(f"data/results_v2_{b}.json")}
budget = "| token budget | " + " | ".join(f"{b:,}" for b in B) + " |\n|---|" + "---|" * len(B) + "\n" + "\n".join(
    f"| {n} | " + " | ".join(f"{100*B[b][m]:.1f}%" for b in B) + " |" for n, m in (("Tool search", "vector"), ("Past-task memory, no graph", "knn"), ("Deja", "deja")))
o1 = R["orders"]["1"]; e = o1["curve_edges"]
curve = "| tasks already served | " + " | ".join(f"{a:,}–{b:,}" for a, b in zip(e, e[1:])) + " |\n|---|" + "---|" * (len(e) - 1) + "\n" + "\n".join(
    f"| {n} | " + " | ".join(f"{100*v:.0f}%" for v in o1["curve"][m]) + " |" for n, m in (("Tool search", "vector"), ("Past-task memory, no graph", "knn"), ("Deja v1", "deja-v1"), ("Deja", "deja")))
held = (f"**Held out, live from HydraDB:** on {H['n']} tasks the served graph never learned from: Deja {100*H['deja']:.1f}%, past-task memory alone "
        f"{100*H['knn']:.1f}%, tool search {100*H['vector']:.1f}%. Graph reads per loadout p50 {H['latency_ms']['p50']:.0f} ms / p95 {H['latency_ms']['p95']:.0f} ms. "
        f"Every served loadout is identical to the in-memory twin (`matches_oracle: {H['matches_oracle']}`).") if H else ""
load = (f"{L['readers']} concurrent planners computing full loadouts (task-vote, word and co-use reads) from one single-node HydraDB while a learner "
        f"keeps writing new tasks: **{L['errors']} errors**, {L['qps']:.1f} loadouts/s, p50 {L['p50']:.0f} ms, p95 {L['p95']:.0f} ms, {L['writes']} tasks learned meanwhile. A single planner gets {L['single_reader_qps']:.1f}/s at p50 {L['single_reader_p50']:.0f} ms, so 16 planners give "
        f"{L['qps']/L['single_reader_qps']:.1f}× the throughput.") if L else ""

readme = f"""# Deja: learned tool loadouts for agents, on HydraDB

**Live demo: http://140.245.206.42/** · Loom: _(link)_ · MIT licensed

Agents now sit in front of hundreds or thousands of MCP tools. Preloading them all burns the context window, and plain
tool search over descriptions misses the tools a task *actually* ends up calling, especially the second server a task
needs. **Deja remembers, in a HydraDB graph, what past tasks needed, and gets sharper with every task it serves.**

## Results

Toucan-1.5M real MCP traces · 8,612 unique tasks · 2,236 tools across 367 servers · 2,000-token budget.
Metric: share of tasks where **every** tool the agent called was already in its **first** loadout.

| Method | All tasks | Called tools on 2+ servers (31%) |
|---|---|---|
{main}

{held}

**Read this table honestly.** Memory of past tasks does most of the work: a plain vector store of past tasks, with no
graph, already gets most of the lift over tool search. The graph adds on top of that: Deja keeps the memory as
`(:Task)-[:USED]->(:Tool)` in HydraDB and adds a word hop and a co-use hop, worth about 2.5 points overall and about 3
on multi-server tasks. Per the ablations, the word hop carries the multi-server gain; the co-use hop adds about 1 point
overall and roughly nothing on multi-server tasks. Our first shipped version (v1) had no task memory and lost to this baseline. A reviewer added the
baseline, we reproduced it, and we rebuilt around it. The v1 row stays in the table.

Evaluation is **prequential**: every task is scored *before* Deja learns from it, so the x-axis is literally "tasks
served". Ranges cover 4 orderings (dataset order and 3 shuffles). One row per Toucan `prompt_id`; only 36 of 8,612
question texts are exact duplicates, and 1.8% of tasks have an earlier task at cosine ≥ 0.95. Weights are not tuned per
ordering: vote weight 1.0 and K=20 come from one check on shuffle 1, and word weight 0.5 and co-use weight 0.2 are
unchanged from v1. The stream numbers use an in-memory twin of exactly the served model; the twin is checked equal to
HydraDB reads on the held-out set and in `python hydra_store.py`.

**Learning curve** (shuffle 1, log-spaced buckets so the cold start is visible):

{curve}

**Other token budgets** (shuffle 1):

{budget}

## How it works

```
(:Task) -[:USED]-> (:Tool)                                   every past task and the tools it called
(:Term {{df}}) -[:LED]-> (:Link {{n}}) -[:TO]-> (:Tool {{uses}})   a word in past tasks → the tool that solved them
(:Tool) -[:CO]-> (:CoLink {{n}}) -[:TO]-> (:Tool)              tools used together in one task
```

1. **Recall similar tasks:** embed the task (bge-small) and pick the 20 most similar past tasks by cosine (numpy, a plain
   vector index). Read their `USED` edges from HydraDB in one id-anchored batched read; each votes for its tools,
   weighted by similarity. `fused = cosine(task, tool) + 1.0 · votes`.
2. **Co-use hop:** from the top 5 of the fused ranking, walk `CO` → `CoLink` → `Tool` to the tools those are
   usually used with (about +1 point overall).
3. **Word hop:** from the task's 8 rarest words (highest idf), walk `LED` → `Link` → `Tool`, weighted by learned counts.
   Limiting it to 8 words cut the rows read about 4× and slightly *raised* accuracy. This hop carries most of the
   multi-server gain (about +3 points).
4. **Rank and pack:** `fused + 0.5 · words + 0.2 · co-use`, packed greedily into the token budget.
5. **Learn:** after the task, write a `Task` node with its `USED` edges and update the word and co-use counts.

HydraDB specifics that shaped the design:
- Batch writes can set node properties but not edge properties, so counts live on reified `Link` and `CoLink` nodes.
- List parameters are rejected outside `UNWIND`, so multi-key reads use scalar parameters.
- The vote hop uses the id-only batched read shape, which skips per-row property reads.
- The served graph keeps word links seen ≥ 2 times, at most 10 per word.

## Under load

{load} Correct under contention; throughput is the ceiling of one node on one machine. HydraDB scales reads by adding
query nodes over the same object store, which we haven't done here.

## What we measured and threw away

- **Shared entity memory** ("reuse IDs other agents already resolved") was our first headline, with kill criteria set
  before building. It failed them: binding precision was 21%, and even a perfect binder would save only 6.3% of lookups
  beyond a plain `(tool, args)` cache.
- **Data-flow edges** (a value from one tool's output reappearing verbatim in a later tool's arguments) are real but rare.
  Removing them moves the result by 0.1 points, so they aren't part of the claim.
- **Real tool chains are shallow:** only 2.8% of traces have 3 or more calls linked by data flow. One-hop traversals are
  enough; deep path search isn't needed.
- **Offered vs called:** 43% of tasks *offer* tools from 2 or more servers, but 31% *call* tools on 2 or more servers.
  The tables use "called".
- **HydraDB on the local-filesystem store** can't publish index generations or hand a restarted node its write lease,
  because it lacks conditional PUT. Reads crawled (about 4 s) until we moved it onto S3 (RustFS) with `graph-indexer`
  running. `hydra.sh` runs all three containers.

## Run it

```bash
./hydra.sh                          # RustFS (S3) + HydraDB graph-node + graph-indexer; BUCKET=name picks a graph
uv sync
# data: two Kimi-K2 shards of Toucan-1.5M (about 310 MB)
for i in 00000 00001; do curl -L -o data/raw/kimi_$i.parquet \\
  https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/resolve/main/Kimi-K2/train-$i-of-00040.parquet; done
uv run python retr.py 2000          # prequential eval, 4 orderings -> data/results_v2_2000.json
uv run python hydra_store.py        # HydraDB loadouts == in-memory twin (asserts)
uv run python load.py 16            # 16 concurrent planners + 1 learner writing
uv run python server.py             # bulk-load graph, held-out check, page + /api/loadout on :8090
./deploy.sh ubuntu@<vm>             # Ubuntu VM (x86_64 or arm64), opens port 80
uv run python make_readme.py        # regenerate this README's tables from the results
```

`sim.py` is the killed entity-memory experiment, kept so those numbers reproduce. `spike_paths.py` is the hour-one
HydraDB spike (weighted paths, multi-type `MSpaths`, a concurrent writer).
"""
open("README.md", "w").write(readme)
print("README.md written")
