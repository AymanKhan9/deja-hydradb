# Deja: learned tool loadouts for agents, on HydraDB

**Live demo: https://deja-hydradb.vercel.app** · [Article on X](https://x.com/AymanKhan90/status/2122198078967226712) · MIT licensed

![Deja](docs/img/1_cover.png)

Agents now sit in front of hundreds or thousands of MCP tools. Preloading them all burns the context window, and plain
tool search over descriptions misses the tools a task *actually* ends up calling, especially the second server a task
needs. **Deja remembers, in a HydraDB graph, what past tasks needed, and gets sharper with every task it serves.**

## Results

**The served model learns only from tool calls that worked** (12–15% of recorded calls errored), abstains when nothing in
the catalog fits, falls back to plain tool search when memory has nothing close, and reads the live tool catalog from the
graph. Each of those is measured below. First, the core comparison, which learns from and scores *every called tool*:

8,612 tasks run against real MCP servers (Toucan-1.5M: questions LLM-generated from each server's tools, tool calls
really executed) · 2,236 tools across 367 servers · 2,000-token budget.
Metric: share of tasks where **every** tool the agent called was already in its **first** loadout.

| Method | All tasks | Called tools on 2+ servers (31%) |
|---|---|---|
| Tool search: vector (bge-small) over tool descriptions | 37.4% | 19.0% |
| Tool search: BM25 keywords | 34.0% | 21.1% |
| Tool search: hybrid, reciprocal-rank fusion of vector + BM25 | 42.8% | 27.5% |
| **Tool search: Deja's base alone** (cosine + 0.15 · BM25) | 43.6% | 27.2% |
| Load whole servers, vector-ranked | 43.5% | 12.6% |
| Vector + learned popularity (non-graph learner) | 43.0–43.5% | 22.8–24.6% |
| **Past-task memory, no graph** (same base; 20 nearest past tasks vote for their tools) | 68.1–68.7% | 42.2–43.4% |
| Deja v1, first shipped: word + co-use edges, no task memory | 52.4–53.2% | 27.3–28.4% |
| **Deja**: hybrid base + task memory in HydraDB + co-use hop + word hop | 70.4–70.9% | 45.4–45.8% |
| Deja without word edges (ablation) | 69.6–70.0% | 43.0–43.9% |
| Deja without co-use edges (ablation) | 69.4–69.8% | 45.2–45.8% |

**Held out, live from HydraDB:** on 200 tasks the served graph never learned from (all answerable), every tool that worked was in the first loadout for **77.4%** vs tool search 45.7%; loadouts carried a chronically broken tool 0.0% of the time vs 40.0% for tool search. The served policy answered with Deja on 198, fell back to tool search on 0 and abstained on 2. Graph reads per loadout p50 93 ms / p95 172 ms. Every served decision and loadout is identical to the in-memory twin (`matches_oracle: True`).

**Read this table honestly.** Memory of past tasks does most of the work: a plain vector store of past tasks, with no
graph, already gets most of the lift over tool search. The graph adds on top of that: Deja keeps the memory as
`(:Task)-[:USED]->(:Tool)` in HydraDB and adds a word hop and a co-use hop, worth 2.2 points overall and
2.7 on multi-server tasks (means over the 4 orderings; see the ablation rows for each hop). Our first shipped version (v1) had no task memory and lost to this baseline. A reviewer added the
baseline, we reproduced it, and we rebuilt around it. The v1 row stays in the table.

Evaluation is **prequential**: every task is scored *before* Deja learns from it, so the x-axis is literally "tasks
served". Ranges cover 4 orderings (dataset order and 3 shuffles). One row per Toucan `prompt_id`; only 36 of 8,612
question texts are exact duplicates, and 1.8% of tasks have an earlier task at cosine ≥ 0.95. Weights are not tuned per
ordering. Model choices were made on shuffle 1 only: K=20 neighbours; votes weighted by similarity⁸ (tried 1 and 8); the
base adds 0.15 · BM25 to cosine (tried 0.05, 0.15, 0.25, 0.35). Word weight 0.5 and co-use weight 0.2 are unchanged from
v1. The memory-only baseline gets the same base and vote weighting, so the graph's gain is measured like for like.

Paired over shuffle 1, Deja minus memory-only is **+2.66 points (95% bootstrap CI +2.22 to +3.10)**, and +3.03 on multi-server tasks (+2.12 to +3.94); Deja alone gets 308 tasks right that memory-only misses, against 79 the other way.

**Where Deja still misses** (shuffle 1, 2,504 misses): 3% need more tools than fit in 2,000 tokens, 21% need a tool that no past task has used yet, and 75% had every tool in memory but ranked it too low. On the 10% of tasks that need a never-used tool, Deja gets 38.5% against 38.3% for its own base alone: memory can't help with a tool it has never seen. The stream numbers use an in-memory twin of exactly the served model; the twin is checked equal to
HydraDB reads on the held-out set and in `python hydra_store.py`, which also checks the per-task learning path: bulk-load
300 tasks, ingest 100 one at a time, and require identical loadouts on the next 100.

![results](docs/img/2_results.png)

**Learning curve** (shuffle 1, log-spaced buckets so the cold start is visible):

![learning curve](docs/img/3_learning_curve.png)

| tasks already served | 0–50 | 50–100 | 100–200 | 200–400 | 400–800 | 800–1,600 | 1,600–3,200 | 3,200–6,400 | 6,400–8,612 |
|---|---|---|---|---|---|---|---|---|---|
| Tool search (vector) | 34% | 30% | 38% | 38% | 41% | 36% | 37% | 38% | 38% |
| Tool search (Deja's base) | 46% | 36% | 48% | 42% | 48% | 42% | 42% | 43% | 45% |
| Past-task memory, no graph | 42% | 32% | 54% | 55% | 62% | 62% | 66% | 71% | 73% |
| Deja v1 | 38% | 30% | 45% | 42% | 49% | 50% | 51% | 54% | 55% |
| Deja | 44% | 30% | 51% | 56% | 62% | 65% | 69% | 74% | 76% |

**Other token budgets** (shuffle 1):

| token budget | 1,000 | 2,000 | 4,000 |
|---|---|---|---|
| Tool search (vector) | 26.5% | 37.4% | 48.2% |
| Tool search (Deja's base) | 30.8% | 43.6% | 55.7% |
| Past-task memory, no graph | 54.1% | 68.3% | 77.5% |
| Deja | 55.6% | 70.9% | 79.7% |

### Learning only from calls that worked

`(:Task)-[:USED]->(:Tool)` used to record that a tool was *called*, so a call that errored still earned votes. Toucan
stores each executed call's response; a call counts as failed if its response opens with an error (HTTP 4xx/5xx,
missing API key, crash, "unknown tool"). Hand-check: 40/40 sampled flags were real failures, and 28/30 unflagged
responses were real successes (the two misses' patterns were then added). 21% of tasks contain a call that never
succeeded. The served model writes the outcome as the edge type, `USED` for successes and `FAILED` for failures, and
only successes vote or count. Mean of 3 shuffles; "chronically broken" = fails in ≥ 80% of the ≥ 10 tasks that call it:

| | Every tool that worked loaded | Every called tool loaded | Loadout has a chronically broken tool |
|---|---|---|---|
| Tool search (vector) | 40.0% | 37.4% | 35.0% |
| Tool search (Deja's base) | 46.6% | 43.6% | 35.1% |
| Past-task memory, learning from every call | 70.6% | 68.3% | 40.1% |
| Deja, learning from every call | 72.7% | 70.7% | 41.1% |
| Past-task memory, successes only | 70.8% | 64.4% | 26.0% |
| Deja, successes only | 73.2% | 66.3% | 23.8% |
| **Deja, successes only + quarantine (served)** | 73.0% | 62.1% | 7.2% |
| Deja, successes only + reliability penalty (dropped) | 66.2% | 54.1% | 4.4% |

On the tools that worked it is as good or slightly better, and loadouts carrying a broken tool drop by more than half.

**Quarantine (served).** The `FAILED` edges are read, too: a tool goes into quarantine when at least 4 of its last 5
recorded outcomes failed, stored as `(:Quarantine)-[:HOLDS]->(:Tool)` and kept in step by the learner. A held tool is
never scored, except for one task in every 200 where it gets a trial, so a tool that gets fixed can earn its way back
(two successes in its last five releases it). Two caveats: the "broken tool" measure counts tools that fail in most runs,
which is close to what the rule removes, so the drop is partly by construction; and the every-called-tool number falls
further, because quarantined tools that the agent still tried no longer appear.
It no longer loads tools whose calls failed, so the every-called-tool number falls; both are reported. The graph keeps
its lead over the memory-only baseline built the same way.

### Knowing when nothing fits

Toucan has no "no suitable tool" tasks, so we built them. Cluster servers by tool embeddings (single-linkage at cosine
0.85), hold out whole clusters plus every server within 0.80 of them (3 seeds, about 37 servers each), and call a task
unanswerable only if all its servers were removed **and** no remaining tool resembles any tool it needed (tool-tool
cosine < 0.80; hand-checked substitutes score 0.83–0.93, non-substitutes 0.52–0.78). Without that substitute check,
about half the "unanswerable" weather tasks still had a working weather server left. Unanswerable tasks per seed:
431, 317, 368.

How well each signal separates answerable from unanswerable tasks (AUROC, 3 seeds, warm memory): closest past task 0.84–0.89, best tool match 0.66–0.78, top vote share 0.73–0.84, neighbour agreement 0.68–0.76, server spread 0.44–0.49, co-use coherence 0.54–0.56.
The abstain rule is a two-feature logistic regression (closest past task, best tool match) fit on seed 1 and tested on
seeds 2 and 3:

| Allowed wrong abstentions (fit on seed 1) | No-fit tasks caught (seed 2 / 3) | Answerable wrongly abstained (seed 2 / 3) | Same, with graph signals added: caught |
|---|---|---|---|
| 2% | 21% / 12% | 2% / 2% | 24% / 15% |
| 5% (served) | 46% / 31% | 5% / 5% | 50% / 32% |
| 10% | 67% / 49% | 11% / 11% | 69% / 51% |
| 20% | 83% / 71% | 22% / 21% | 85% / 68% |

**The graph doesn't help here.** Adding the graph signals (vote share, neighbour agreement, server spread, co-use
coherence) changed the catch rate by −2 to +2 points, so the served rule uses similarity only. Cold-start fallback
(plain tool search when the closest past task is below 0.66): 72.5% → 72.6% (seed 2), 73.1% → 73.2% (seed 3); it helps in the first few hundred tasks and is
neutral after.

**Remembering what couldn't be served.** The shipped rule forgets every task it abstained on, so the 50th request for a
missing capability looks as new as the first. Remembering each unserved task (a `Task` node with no `USED` edge) adds
two signals: how close the nearest unserved past task is, and the unserved share of the 20 nearest past tasks. We report
two views, because the benchmark's unserved tasks come from held-out servers and naturally repeat: "gap repeats" lets an
unserved task see earlier unserved tasks from its own servers; "new gaps only" doesn't, which is what the first request
for a new missing capability looks like. The combined rule abstains if either the shipped rule or the memory rule says so.

| Allowed wrong abstentions (fit on seed 1) | Shipped rule: caught | Unserved memory: caught, gap repeats | Unserved memory: caught, new gaps only | Combined rule: caught, repeats / new | Combined: wrongly abstained | Combined: caught among first 20 unserved |
|---|---|---|---|---|---|---|
| 2% | 21% / 12% | 86% / 88% | 41% / 26% | 81% / 86% ; 41% / 26% | 2% / 2% | 60% / 60% |
| 5% | 46% / 31% | 89% / 90% | 58% / 37% | 89% / 90% ; 60% / 37% | 4% / 5% | 80% / 75% |
| 10% | 67% / 49% | 92% / 92% | 72% / 50% | 93% / 93% ; 76% / 53% | 8% / 9% | 90% / 90% |

Most of the gain is the same missing capability being asked for again; there is a smaller real gain on new gaps. This
needs a signal that a task went unserved (the agent called nothing, or the user said so). The live demo has no such
signal, every Toucan task is answerable, so it ships in the evaluation and the model, not on the page.

### Tools come and go

The catalog is graph state: `(:Catalog)-[:LISTS]->(:Tool)`. When a server drops a tool from `tools/list`, Deja deletes
one edge; every read is filtered to the live catalog, whatever memory says.

**The removal test, on HydraDB:** learn a task that used tool A → remove A from the catalog → repeat the task. Over 40 cases, A was never loaded again (40/40) and the token budget was still packed (40/40).

**Removing 10% of tools mid-stream** (3 seeds, tasks still answerable afterwards): reading the live catalog keeps 76.5% success, against 75.4% when trusting stale memory, which also spends 129 tokens per loadout on tools that no longer exist. Tool search: 40.6%.

## How it works

![how it works](docs/img/4_how_it_works.png)

```
(:Task) -[:USED]-> (:Tool)                                   every past task and the tools that worked for it
(:Task) -[:FAILED]-> (:Tool)                                 calls that errored (recorded, never vote)
(:Catalog) -[:LISTS]-> (:Tool)                               the live tool catalog
(:Term {df}) -[:LED]-> (:Link {n}) -[:TO]-> (:Tool {uses})   a word in past tasks → the tool that solved them
(:Tool) -[:CO]-> (:CoLink {n}) -[:TO]-> (:Tool)              tools used together in one task
```

0. **Abstain or fall back:** a two-feature rule (closest past task, best tool match) decides first. If nothing fits,
   load nothing and say why; if no past task is close, use plain tool search; otherwise continue.
1. **Recall similar tasks:** embed the task (bge-small) and pick the 20 most similar past tasks by cosine (numpy, a plain
   vector index). Read their `USED` edges from HydraDB in one id-anchored batched read; each votes for its tools,
   weighted by similarity. `fused = base + 1.0 · votes`, where `base = cosine + 0.15 · BM25/max(BM25)` and each neighbour's vote is
   weighted by similarity⁸ (near neighbours dominate).
2. **Co-use hop:** from the top 5 of the fused ranking, walk `CO` → `CoLink` → `Tool` to the tools those are
   usually used with (about +1 point overall).
3. **Word hop:** from the task's 8 rarest words (highest idf), walk `LED` → `Link` → `Tool`, weighted by learned counts.
   Limiting it to 8 words cut the rows read about 4× and slightly *raised* accuracy. This hop carries most of the
   multi-server gain (about +3 points).
4. **Rank and pack:** `fused + 0.5 · words + 0.2 · co-use`, packed greedily into the token budget.
5. **Learn:** after the task, write a `Task` node with `USED` (worked) and `FAILED` (errored) edges, and update the word
   and co-use counts from the calls that worked. All reads are filtered to the tools the `Catalog` node currently lists.

HydraDB specifics that shaped the design:
- Batch writes can set node properties but not edge properties, so counts live on reified `Link` and `CoLink` nodes.
- List parameters are rejected outside `UNWIND`, so multi-key reads use scalar parameters.
- The vote hop uses the id-only batched read shape, which skips per-row property reads.
- The served graph keeps word links seen ≥ 2 times, at most 10 per word.

## Under load

16 concurrent planners computing full loadouts (task-vote, word and co-use reads) from one single-node HydraDB while a learner keeps writing new tasks: **0 errors**, 10.3 loadouts/s, p50 1528 ms, p95 2413 ms, 36 tasks learned meanwhile. A single planner gets 2.4/s at p50 418 ms, so 16 planners give 4.4× the throughput. Correct under contention; throughput is the ceiling of one node on one machine. HydraDB scales reads by adding
query nodes over the same object store, which we haven't done here.

## With an agent in the loop

Every number above is "the right tools were in the loadout". To check that this changes what an agent does, Claude Opus 5.5 (via OpenRouter, medium effort) got each of the 199 held-out tasks twice, once with Deja's live loadout and once with the hybrid tool search's, and we scored the **first tool it called** (`agent_check.py`).

| | Deja's loadout | Tool search's loadout |
|---|---|---|
| First call was a tool that worked in the recorded run | **63.3%** | 44.2% |
| First call matched the recorded agent's first working call | 55.3% | 38.2% |
| Model called no tool (said none fits) | 4.0% | 9.5% |
| Prompt tokens per request | 3,256 | 3,279 |

Paired: **+19.1 points (95% bootstrap CI +13.1 to +25.1)**; Deja's loadout was better on 41 tasks and worse on 3, at the same prompt size. 2 of 398 requests failed on the provider's credit limit and count as misses for both. One model, one run, first tool call only: it shows the loadout changes what the agent does, not end-to-end task success.

## Not done yet

- **An MCP proxy.** As a proxy in front of real MCP servers Deja would get call outcomes from `isError`, catalog changes
  from `tools/list_changed`, and the unserved signal from tasks that end without a call, and could learn from real
  agent traffic instead of Toucan's generated questions.

## Limitations

- **Synthetic questions, real execution.** Toucan's questions were generated by an LLM from each server's tool set, so
  similar tasks recur by construction, which favours any past-task memory (ours and the baseline's). Organic agent
  traffic may recur less. The tool calls and their results are real.
- **Cold start.** Over its first 100 tasks Deja is level with or slightly below its own hybrid base used as plain tool
  search (37% vs 41%, shuffle 1); after that memory pays off.
  The fallback to tool search softens it but doesn't remove it. A hosted version would start warm from memory shared
  across a team's agents.
- **Abstaining is partial.** At 5% wrong abstentions it catches 31–46% of tasks with no fitting tool. The benchmark's
  "no fitting tool" labels come from held-out server clusters and an embedding-based substitute check, not from people.
- **Failure detection is a text heuristic** on the recorded responses, hand-checked on a sample, not ground truth.

## What we measured and threw away

- **Shared entity memory** ("reuse IDs other agents already resolved") was our first headline, with kill criteria set
  before building. It failed them: binding precision was 21%, and even a perfect binder would save only 6.3% of lookups
  beyond a plain `(tool, args)` cache.
- **Graph signals for abstaining** (neighbour agreement, server spread, co-use coherence) added nothing over plain
  similarity; the abstain rule doesn't use them.
- **A reliability penalty** on top of learning from successes cut broken tools to 3% of loadouts but cost about 6 points.
  Too blunt; dropped.
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
for i in 00000 00001; do curl -L -o data/raw/kimi_$i.parquet \
  https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/resolve/main/Kimi-K2/train-$i-of-00040.parquet; done
uv run python retr.py 2000          # prequential eval, 4 orderings -> data/results_v2_2000.json
uv run python hydra_store.py        # HydraDB loadouts == in-memory twin (asserts)
uv run python load.py 16            # 16 concurrent planners + 1 learner writing
uv run python outcome.py            # learning only from calls that worked -> data/outcome.json
uv run python abstain.py            # abstention benchmark + cold-start fallback -> data/abstain.json
uv run python lifecycle.py          # tool removal on HydraDB + mid-stream removal -> data/lifecycle.json
uv run --with matplotlib python charts.py docs/img
uv run python server.py             # bulk-load graph, held-out check, page + /api/loadout on :8090
./deploy.sh ubuntu@<vm>             # Ubuntu VM (x86_64 or arm64), opens port 80
uv run python make_readme.py        # regenerate this README's tables from the results
```

`sim.py` is the killed entity-memory experiment, kept so those numbers reproduce. `spike_paths.py` is the hour-one
HydraDB spike (weighted paths, multi-type `MSpaths`, a concurrent writer).
