# Deja — learned tool loadouts for agents, on HydraDB

Agents now sit in front of hundreds or thousands of MCP tools. Preloading them all burns the context window; plain
vector search over tool descriptions misses the tools a task *actually* ends up calling — especially the second
server a task needs. **Deja keeps a graph in HydraDB of what real tasks needed, and gets sharper with every task it serves.**

**Live demo: http://140.245.206.42/** · 

**Result** (Toucan-1.5M real MCP traces · 8,612 unique tasks · 2,236 tools across 367 servers · 2,000-token budget):

| Share of tasks where every tool the agent called was in its **first** loadout | All tasks | Tools span 2+ servers |
|---|---|---|
| Vector top-k (tool search) | 37.4% | 19.0% |
| Load whole servers (vector-ranked) | 43.5% | 12.6% |
| Vector + learned popularity (non-graph learner) | 43.0–43.5% | 22.8–24.6% |
| **Deja (HydraDB graph)** | **52.4–53.2%** | **27.3–28.4%** |
| Deja, term edges only (ablation) | 48.3–49.1% | 26.1–27.1% |
| Deja, co-use edges only (ablation) | 46.4–46.8% | 21.7–22.3% |

**Held out, live from HydraDB:** on 200 tasks the served graph never learned from, Deja 52.5% vs vector 39.0%; graph read p50 220 ms / p95 604 ms; loadouts identical to the in-memory twin (`matches_oracle: True`).

Ranges are over 4 orderings of the stream (dataset order + 3 shuffles). The served graph keeps term links seen ≥2 times, top 10 per term; the stream numbers are computed with an in-memory twin of exactly that model (a per-task HydraDB replay of 8.6k tasks takes over an hour on a laptop), and the twin is checked equal to HydraDB reads on the held-out set. Evaluation is **prequential**: every task is
scored *before* Deja learns from it, so nothing is tested on data it trained on, and the x-axis of the learning curve is
literally "tasks served". One row per Toucan `prompt_id` so near-duplicate prompts don't inflate the score.

## How it works

```
(:Term {df}) -[:LED]-> (:Link {n}) -[:TO]-> (:Tool {uses})     a word in past tasks → the tool that solved them
(:Tool) -[:CO]-> (:CoLink {n}) -[:TO]-> (:Tool)                used together, or a value flowed from one to the other
```

1. **Seed** — embed the task (bge-small), rank the pooled catalog by cosine.
2. **Traverse** (HydraDB) — task terms → `Link` → `Tool`, weighted by learned counts × idf; and the top seed tools →
   `CoLink` → `Tool`, which is what pulls in the *other* server a multi-server task needs.
3. **Rank & pack** — `cosine + 0.5·term + 0.2·co-use`, packed greedily into the token budget.
4. **Learn** — after the task, the tools it actually called update the counts. Data-flow edges are *observed*, not
   labelled: a value from one tool's output reappearing verbatim in a later tool's arguments (IDs only; values the user
   typed or the model wrote before the lookup are excluded).

HydraDB specifics that shaped the design: batch writes can set node properties but not edge properties, so counts live on
reified `Link`/`CoLink` nodes; list parameters are rejected outside `UNWIND`, so multi-key reads use scalar parameters.
A single learner keeps an in-memory mirror of the counts so each write is one absolute-value batch; the mirror doubles as
the oracle the graph reads are checked against (`python hydra_store.py` asserts zero mismatches).

## Under load

16 concurrent planners reading loadouts from one single-node HydraDB while a learner keeps writing: **0 errors**, 6.8 loadouts/s,
p50 2.4 s, p95 3.9 s, 45 tasks learned meanwhile. Correct under contention; throughput is the ceiling of one
laptop node (raising `GRAPH_MAX_CONCURRENT_HYDRATIONS` and switching to id-only batched reads both helped < 2×). HydraDB's answer is more query nodes
over the same object store — not done here.

## What we measured and threw away

We started from a different headline — **shared entity memory** (reuse IDs other agents already resolved) — and set kill
criteria before building. It failed them: binding precision 21%, and even a perfect binder would save only 6.3% of
lookups beyond a plain `(tool, args)` cache. Other findings from the traces:

- Only 2.8% of traces have 3+ calls linked by data flow — real tool chains are shallow, so Deja uses 1–2-hop traversals
  instead of deep path search.
- 43% of tasks involve more than one MCP server.
- 40% of repeated identical tool calls return a different result — blind output caching is unsafe.
- HydraDB on the local-filesystem store can't publish index generations or hand a restarted node its write lease (no conditional PUT); reads crawled (~4 s) until we moved it onto S3 (RustFS) with `graph-indexer` running. `hydra.sh` runs all three.

## Run it

```bash
./hydra.sh                                  # HydraDB 0.4.0 (ghcr.io/hydra-db/hydradb), fresh local store
uv sync
# data: two Kimi-K2 shards of Agent-Ark/Toucan-1.5M into data/raw/kimi_0000{0,1}.parquet
uv run python retr.py 2000                  # prequential eval, 4 orderings -> data/results_2000_k10_n2.json
uv run python hydra_store.py                # graph reads == in-memory oracle (asserts)
uv run python load.py 16                    # 16 concurrent planners + 1 learner writing
uv run python server.py                     # bulk-load graph, held-out check, live page + /api/loadout on :8090
```

`sim.py` is the (killed) entity-memory experiment, kept so the numbers above are reproducible.
