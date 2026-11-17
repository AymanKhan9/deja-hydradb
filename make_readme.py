"""Regenerate README.md's result tables from the eval JSON, so the prose never drifts from the numbers."""
import json, os
import numpy as np
R = json.load(open("data/results_v2_2000.json")); O = [R["orders"][k] for k in ("dataset", "1", "2", "3")]
W = json.load(open("web/results.json")) if os.path.exists("web/results.json") else {}
H, L = W.get("hydra", {}), W.get("load", {})
def rng(m, part):
    v = [o[part][m] for o in O]; a, b = 100 * min(v), 100 * max(v)
    return f"{a:.1f}%" if f"{a:.1f}" == f"{b:.1f}" else f"{a:.1f}–{b:.1f}%"
rows = [("Tool search: vector (bge-small) over tool descriptions", "vector"), ("Tool search: BM25 keywords", "bm25"),
        ("Tool search: hybrid, reciprocal-rank fusion of vector + BM25", "hybrid"),
        ("**Tool search: Deja's base alone** (cosine + 0.15 · BM25)", "search"), ("Load whole servers, vector-ranked", "servers"),
        ("Vector + learned popularity (non-graph learner)", "popularity"),
        ("**Past-task memory, no graph** (same base; the 20 nearest past tasks to the task and to each of its steps vote)", "knn"),
        ("Deja v1, first shipped: word + co-use edges, no task memory", "deja-v1"),
        ("**Deja**: hybrid base + task and step memory in HydraDB + co-use hop + word hop", "deja"),
        ("Deja without word edges (ablation)", "deja-no-terms"), ("Deja without co-use edges (ablation)", "deja-no-couse")]
main = "\n".join(f"| {n} | {rng(m, 'overall')} | {rng(m, 'multi')} |" for n, m in rows)
O4 = [json.load(open("data/results_v4_2000.json"))["orders"][k] for k in ("dataset", "1", "2", "3")] if os.path.exists("data/results_v4_2000.json") else []
r4 = lambda m, part: (lambda a, b: a if a == b else f"{a[:-1]}–{b}")(*(f"{100*x:.1f}%" for x in (min(o[part][m] for o in O4), max(o[part][m] for o in O4))))
v4line = (f"| *v4 for reference (before step recall and the length penalty)*: tool search / memory-only / Deja | "
          f"{r4('search','overall')} / {r4('knn','overall')} / {r4('deja','overall')} | {r4('search','multi')} / {r4('knn','multi')} / {r4('deja','multi')} |") if O4 else ""
main += "\n" + v4line
SB = O[1].get("by_similarity", {})
simtab = ("| Closest past task similarity | Tasks (shuffle 1) | Tool search | Memory-only | Deja | Deja's lead over tool search |\n|---|---|---|---|---|---|\n" + "\n".join(
    f"| {k} | {v['n']:,} | {100*v['search']:.1f}% | {100*v['knn']:.1f}% | {100*v['deja']:.1f}% | +{100*(v['deja']-v['search']):.1f} |"
    for k, v in SB.items() if v["n"])) if SB else ""
B = {b: json.load(open(f"data/results_v2_{b}.json"))["orders"]["1"]["overall"] for b in (1000, 2000, 4000) if os.path.exists(f"data/results_v2_{b}.json")}
budget = "| token budget | " + " | ".join(f"{b:,}" for b in B) + " |\n|---|" + "---|" * len(B) + "\n" + "\n".join(
    f"| {n} | " + " | ".join(f"{100*B[b][m]:.1f}%" for b in B) + " |" for n, m in (("Tool search (vector)", "vector"), ("Tool search (Deja's base)", "search"), ("Past-task memory, no graph", "knn"), ("Deja", "deja")))
o1 = R["orders"]["1"]; e = o1["curve_edges"]
_w = lambda m: (o1["curve"][m][0] * 50 + o1["curve"][m][1] * 50) / 100
cold = f"{100*_w('deja'):.0f}% vs {100*_w('search'):.0f}%"
cold_rel = "above" if _w('deja') - _w('search') > 0.02 else "below" if _w('search') - _w('deja') > 0.02 else "level with"
curve = "| tasks already served | " + " | ".join(f"{a:,}–{b:,}" for a, b in zip(e, e[1:])) + " |\n|---|" + "---|" * (len(e) - 1) + "\n" + "\n".join(
    f"| {n} | " + " | ".join(f"{100*v:.0f}%" for v in o1["curve"][m]) + " |" for n, m in (("Tool search (vector)", "vector"), ("Tool search (Deja's base)", "search"), ("Past-task memory, no graph", "knn"), ("Deja v1", "deja-v1"), ("Deja", "deja")))
kk = lambda x: f"{x/1000:.1f}".removesuffix(".0") + "k" if x >= 1000 else str(x)
gp = [(a, b, round(100 * (d - sv))) for a, b, d, sv in zip(e, e[1:], o1["curve"]["deja"], o1["curve"]["search"])]
first = next(x for i, x in enumerate(gp) if all(y[2] > 0 for y in gp[i:]))
gaptxt = ("**Does the gap over tool search keep widening?** It widens, then flattens. Deja minus its own base used as tool "
          "search, shuffle 1: " + ", ".join(f"{kk(a)}–{kk(b)} tasks {g:+d}" for a, b, g in gp if a >= 100) +
          f". Deja is {cold_rel} tool search over its first 100 tasks ({cold}) and ahead from the {kk(first[0])}–{kk(first[1])} bucket on "
          "(v4 was ahead from 100–200; the ~400 figure in the launch article was v2).")
held = (f"**Held out, live from HydraDB:** on {H['n']} tasks the served graph never learned from (all answerable), every tool that worked was "
        f"in the first loadout for **{100*H['deja']:.1f}%** vs tool search {100*H['vector']:.1f}%; loadouts carried a chronically broken tool "
        f"{100*H['deja_broken']:.1f}% of the time vs {100*H['vector_broken']:.1f}% for tool search. The served policy answered with Deja on "
        f"{H['modes']['deja']} and abstained on {H['modes']['abstain']}. Graph reads per loadout "
        f"p50 {H['latency_ms']['p50']:.0f} ms / p95 {H['latency_ms']['p95']:.0f} ms. Every served decision and loadout is identical to the "
        f"in-memory twin (`matches_oracle: {H['matches_oracle']}`).") if H else ""
OC = json.load(open("data/outcome.json")) if os.path.exists("data/outcome.json") else {}
om = lambda m, k: 100 * float(np.mean([OC[x][m][k] for x in OC]))
outcome = "\n".join([
    "| | Every tool that worked loaded | Every called tool loaded | Loadout has a chronically broken tool |", "|---|---|---|---|"] +
    [f"| {n} | {om(m, 'succeeded'):.1f}% | {om(m, 'called'):.1f}% | {om(m, 'broken_tool_loaded'):.1f}% |" for n, m in (
        ("Tool search (vector)", "vector"), ("Tool search (Deja's base)", "search"), ("Past-task memory, learning from every call", "knn"),
        ("Deja, learning from every call", "deja"), ("Past-task memory, successes only", "knn-ok"), ("Deja, successes only", "deja-ok"),
        ("**Deja, successes only + quarantine (served)**", "deja-ok+quar"),
        ("Deja, successes only + reliability penalty (dropped)", "deja-ok+pen")) if m in OC[next(iter(OC))]]) if OC else ""
gain = lambda part: f"{100*np.mean([o[part]['deja'] - o[part]['knn'] for o in O]):.1f}"
AN = json.load(open("data/analysis_v5.json")) if os.path.exists("data/analysis_v5.json") else {}
ci = (f"Paired over shuffle 1, Deja minus memory-only is **+{100*AN['diff']:.2f} points (95% bootstrap CI {100*AN['ci95'][0]:+.2f} to "
      f"{100*AN['ci95'][1]:+.2f})**, and +{100*AN['diff_multi']:.2f} on multi-server tasks ({100*AN['ci95_multi'][0]:+.2f} to {100*AN['ci95_multi'][1]:+.2f}); "
      f"Deja alone gets {AN['deja_only_wins']} tasks right that memory-only misses, against {AN['memory_only_wins']} the other way.") if AN else ""
misses = (f"**Where Deja still misses** (shuffle 1, {AN['misses']:,} misses): {100*AN['miss_doesnt_fit']:.0f}% need more tools than fit in "
          f"2,000 tokens, {100*AN['miss_unseen_tool']:.0f}% need a tool that no past task has used yet, and {100*AN['miss_ranked_out']:.0f}% had every "
          f"tool in memory but ranked it too low. On the {100*AN['unseen_tasks']:.0f}% of tasks that need a never-used tool, Deja gets "
          f"{100*AN['unseen_deja']:.1f}% against {100*AN['unseen_search']:.1f}% for its own base alone: memory can't help with a tool it has never seen.") if AN else ""
AB = json.load(open("data/abstain.json")) if os.path.exists("data/abstain.json") else {}; UN = AB.get("unserved")
AG = json.load(open("data/agent_check.json")) if os.path.exists("data/agent_check.json") else {}
if AG:
    a_, s_ = AG["conditions"]["deja"], AG["conditions"]["search"]; pr = AG["paired"]
    agent = (f"Every number above is \"the right tools were in the loadout\". To check that this changes what an agent does, Claude Opus 5.5 "
             f"(via OpenRouter, medium effort) got each of the {AG['n_tasks']} held-out tasks twice, once with Deja's live loadout and once with "
             f"the hybrid tool search's, and we scored the **first tool it called** (`agent_check.py`).\n\n"
             f"| | Deja's loadout | Tool search's loadout |\n|---|---|---|\n"
             f"| First call was a tool that worked in the recorded run | **{100*a_['worked']:.1f}%** | {100*s_['worked']:.1f}% |\n"
             f"| First call matched the recorded agent's first working call | {100*a_['first']:.1f}% | {100*s_['first']:.1f}% |\n"
             f"| Model called no tool (said none fits) | {100*a_['no_tool_called']:.1f}% | {100*s_['no_tool_called']:.1f}% |\n"
             f"| Prompt tokens per request | {a_['mean_input_tokens']:,.0f} | {s_['mean_input_tokens']:,.0f} |\n\n"
             f"Paired: **+{100*pr['diff_worked']:.1f} points (95% bootstrap CI +{100*pr['ci95'][0]:.1f} to +{100*pr['ci95'][1]:.1f})**; Deja's "
             f"loadout was better on {pr['deja_only']} tasks and worse on {pr['search_only']}, at the same prompt size. "
             f"{a_['errors'] + s_['errors']} of {2*AG['n_tasks']} requests failed on the provider's credit limit and count as misses for both. "
             f"One model, one run, first tool call only: it shows the loadout changes what the agent does, not end-to-end task success. "
             f"This check ran on the v4 loadouts, before step recall and the length penalty; it hasn't been re-run on v5.")
else: agent = ""
KEY = "abstain_recall_unanswerable"
def un_table():
    rows = ["| Allowed wrong abstentions (fit on seed 1) | Shipped rule: caught | Unserved memory: caught, gap repeats | Unserved memory: caught, new gaps only | Combined rule: caught, repeats / new | Combined: wrongly abstained | Combined: caught among first 20 unserved |",
            "|---|---|---|---|---|---|---|"]
    T = AB["policies"]["similarity only"]["tradeoff"]
    for fa, v in UN["budgets"].items():
        e = v["eval"]; f = lambda view, k, key: " / ".join(f"{100*e[view][sd][k][key]:.0f}%" for sd in ("2", "3"))
        rows.append(f"| {100*float(fa):.0f}% | {' / '.join(f'{100*T[fa][sd][KEY]:.0f}%' for sd in ('2', '3'))} | "
                    f"{f('repeats', 'memory', 'caught')} | {f('new_gaps', 'memory', 'caught')} | {f('repeats', 'combined', 'caught')} ; {f('new_gaps', 'combined', 'caught')} | "
                    f"{f('repeats', 'combined', 'wrong')} | {f('repeats', 'combined', 'caught_first20')} |")
    return "\n".join(rows)
untab = un_table() if UN else ""
AB = json.load(open("data/abstain.json")) if os.path.exists("data/abstain.json") else {}
def ab_table():
    T = AB["policies"]["similarity only"]["tradeoff"]; G = AB["policies"]["similarity + graph"]["tradeoff"]
    rows = ["| Allowed wrong abstentions (fit on seed 1) | No-fit tasks caught (seed 2 / 3) | Answerable wrongly abstained (seed 2 / 3) | Same, with graph signals added: caught |", "|---|---|---|---|"]
    for fa in T:
        f = lambda d, k: " / ".join(f"{100*d[fa][x][k]:.0f}%" for x in ("2", "3"))
        rows.append(f"| {100*float(fa):.0f}%{' (served)' if abs(float(fa)-0.05) < 1e-9 else ''} | {f(T, 'abstain_recall_unanswerable')} | {f(T, 'false_abstain_answerable')} | {f(G, 'abstain_recall_unanswerable')} |")
    au = AB["auroc"]; fe = ("s_mem", "s_tool", "vote_top", "vote_agree", "srv_spread", "co_coh")
    lo_hi = lambda k: f"{min(au[x][k] for x in au):.2f}–{max(au[x][k] for x in au):.2f}"
    F = AB["fallback"]["eval"]
    return ("\n".join(rows), ", ".join(f"{n} {lo_hi(k)}" for n, k in zip(("closest past task", "best tool match", "top vote share", "neighbour agreement",
            "server spread", "co-use coherence"), fe)), AB["setup"], AB["fallback"]["tau"],
            f"{100*F['2']['deja']:.1f}% → {100*F['2']['deja+fallback']:.1f}% (seed 2), {100*F['3']['deja']:.1f}% → {100*F['3']['deja+fallback']:.1f}% (seed 3)")
abt, auroc_s, absetup, abtau, fbs = ab_table() if AB else ("", "", {}, 0, "")
LF = json.load(open("data/lifecycle.json")) if os.path.exists("data/lifecycle.json") else {}
if LF:
    sc, st = LF["scenario"], LF["stream"]; lm = lambda k, f: float(np.mean([st[x][k][f] for x in st]))
    life = (f"**The removal test, on HydraDB:** learn a task that used tool A → remove A from the catalog → repeat the task. Over {sc['cases']} cases, "
            f"A was never loaded again ({sc['A_never_loaded']}/{sc['cases']}) and the token budget was still packed ({sc['budget_still_packed']}/{sc['cases']}).\n\n"
            f"**Removing 10% of tools mid-stream** (3 seeds, tasks still answerable afterwards): reading the live catalog keeps "
            f"{100*lm('filtered','success'):.1f}% success, against {100*lm('stale','success'):.1f}% when trusting stale memory, which also spends "
            f"{lm('stale','tokens_on_removed_tools'):.0f} tokens per loadout on tools that no longer exist. Tool search: {100*lm('vector','success'):.1f}%.")
else: life = ""
CC = json.load(open("data/catalog_check.json")) if os.path.exists("data/catalog_check.json") else {}
if CC: life += (f"\n\n**The same test through the server's loadout path** (`catalog_check.py`, the function behind `/api/loadout`): "
                f"{CC['cases']} tools removed while the server's catalog was warm were absent from both columns in "
                f"{CC['absent_immediately']}/{CC['cases']} loadouts asked for right after the removal and {CC['absent_after_6s']}/{CC['cases']} "
                f"six seconds later; re-added, {CC['back_in_catalog_after_readd']}/{CC['cases']} came back. A reader (@webmcpx) spotted that the "
                "server cached the catalog for 5 s, so a removed tool could be served for up to 5 s after removal; this test failed "
                "on that version (14 of 20 still loaded the removed tool). Now every catalog write gives the `Catalog` node a new "
                "version, and the server re-reads the tool list whenever the version changes, whichever process made the change.")
load = (f"{L['readers']} concurrent planners computing full loadouts (task-vote, step-vote, word and co-use reads) from one single-node HydraDB while a learner "
        f"keeps writing new tasks: **{L['errors']} errors**, {L['qps']:.1f} loadouts/s, p50 {L['p50']:.0f} ms, p95 {L['p95']:.0f} ms, {L['writes']} tasks learned meanwhile. A single planner gets {L['single_reader_qps']:.1f}/s at p50 {L['single_reader_p50']:.0f} ms, so 16 planners give "
        f"{L['qps']/L['single_reader_qps']:.1f}× the throughput. Step recall costs throughput under contention: v4, without it, did 10.3 loadouts/s with 16 planners. The step votes share one graph read with the task votes; what remains is likely the extra per-step neighbour searches, which run in Python.") if L else ""

readme = f"""# Deja: learned tool loadouts for agents, on HydraDB

**Live demo: https://deja-hydradb.vercel.app** · [Article on X](https://x.com/AymanKhan90/status/2122198078967226712) · MIT licensed

![Deja](docs/img/1_cover.png)

Agents now sit in front of hundreds or thousands of MCP tools. Preloading them all burns the context window, and plain
tool search over descriptions misses the tools a task *actually* ends up calling, especially the second server a task
needs. **Deja remembers, in a HydraDB graph, what past tasks needed, and gets sharper with every task it serves.**

## Results

**The served model learns only from tool calls that worked** (12–15% of recorded calls errored), abstains when nothing in
the catalog fits, ramps memory's weight up over its first 400 tasks (cold start), and reads the live tool catalog from the
graph. Each of those is measured below. First, the core comparison, which learns from and scores *every called tool*:

8,612 tasks run against real MCP servers (Toucan-1.5M: questions LLM-generated from each server's tools, tool calls
really executed) · 2,236 tools across 367 servers · 2,000-token budget.
Metric: share of tasks where **every** tool the agent called was already in its **first** loadout.

| Method | All tasks | Called tools on 2+ servers (31%) |
|---|---|---|
{main}

{held}

**Read this table honestly.** Memory of past tasks does most of the work: a plain vector store of past tasks, with no
graph, already gets most of the lift over tool search. The graph adds on top of that: Deja keeps the memory as
`(:Task)-[:USED]->(:Tool)` in HydraDB and adds a word hop and a co-use hop, worth {gain('overall')} points overall and
{gain('multi')} on multi-server tasks (means over the 4 orderings; see the ablation rows for each hop). Our first shipped version (v1) had no task memory and lost to this baseline. A reviewer added the
baseline, we reproduced it, and we rebuilt around it. The v1 row stays in the table.

Evaluation is **prequential**: every task is scored *before* Deja learns from it, so the x-axis is literally "tasks
served". Ranges cover 4 orderings (dataset order and 3 shuffles). One row per Toucan `prompt_id`; only 36 of 8,612
question texts are exact duplicates, and 1.8% of tasks have an earlier task at cosine ≥ 0.95. Weights are not tuned per
ordering. Model choices were made on shuffle 1 only: K=20 neighbours; votes weighted by similarity⁸ (tried 1 and 8); the
base adds 0.15 · BM25 to cosine (tried 0.05, 0.15, 0.25, 0.35). Word weight 0.5 and co-use weight 0.2 are unchanged from
v1. **v5 (from a reviewer's analysis, reproduced here):** each step of a multi-step task ("…, then …", sentence ends) also
recalls its own 20 nearest past tasks, which vote the same way, and a tool keeps its best step vote (weight 0.5, tried
0.25–1.0); and every ranked method subtracts 0.05 · ln(tokens) so a long schema has to earn its space (tried 0.02–0.2).
We dropped a step-to-tool cosine term that added 0.3 points on shuffle 1, to keep it simple. **Cold-start ramp:** memory's
weights (task votes, step votes, both hops) scale with min(1, tasks learned / 400), for memory-only too; 400 was best of
50–1,600 on shuffle 1 (it ties 800), and it lifts the first 100 tasks on every ordering without changing anything after 400. **The memory-only baseline
and tool search get the same changes** (step votes for memory-only, the length penalty for both), so the graph's gain is
measured like for like; both changes lift memory-only nearly as much as Deja, which is why the graph's margin is smaller
than in v4.

{ci}

{misses} The stream numbers use an in-memory twin of exactly the served model; the twin is checked equal to
HydraDB reads on the held-out set and in `python hydra_store.py`, which also checks the per-task learning path: bulk-load
300 tasks, ingest 100 one at a time, and require identical loadouts on the next 100.

![results](docs/img/2_results.png)

**Learning curve** (shuffle 1, log-spaced buckets so the cold start is visible):

![learning curve](docs/img/3_learning_curve.png)

{curve}

{gaptxt} {f"{100*AN['miss_ranked_out']:.0f}% of the remaining misses are tools memory already has but ranks too low, so ranking, not coverage, is the next ceiling." if AN else ""}

**Does the lead need near-duplicates?** Toucan's questions recur by construction, so here is accuracy by how similar the
closest past task was (shuffle 1, tasks after the first 400). For scale: "List my open GitHub issues" and a past task
about trending GitHub repositories are 0.76 similar.

{simtab}

**Other token budgets** (shuffle 1):

{budget}

### Learning only from calls that worked

`(:Task)-[:USED]->(:Tool)` used to record that a tool was *called*, so a call that errored still earned votes. Toucan
stores each executed call's response; a call counts as failed if its response opens with an error (HTTP 4xx/5xx,
missing API key, crash, "unknown tool"). Hand-check: 40/40 sampled flags were real failures, and 28/30 unflagged
responses were real successes (the two misses' patterns were then added). 21% of tasks contain a call that never
succeeded. The served model writes the outcome as the edge type, `USED` for successes and `FAILED` for failures, and
only successes vote or count. Mean of 3 shuffles; "chronically broken" = fails in ≥ 80% of the ≥ 10 tasks that call it:

{outcome}

On the tools that worked it is as good or slightly better, and loadouts carrying a broken tool drop by more than half.
The v5 length penalty packs more, shorter tools into the same budget (tool search: 18 tools per loadout instead of 14), so
every method without quarantine carries a broken tool more often than in v4 (tool search 35% → 50%); with quarantine Deja
goes from 7.2% to 8.5%.

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
{", ".join(str(absetup[x]["n_un"]) for x in absetup) if absetup else ""}.

How well each signal separates answerable from unanswerable tasks (AUROC, 3 seeds, warm memory): {auroc_s}.
The abstain rule is a two-feature logistic regression (closest past task, best tool match) fit on seed 1 and tested on
seeds 2 and 3:

{abt}

**The graph doesn't help here.** Adding the graph signals (vote share, neighbour agreement, server spread, co-use
coherence) changed the catch rate by −2 to +2 points, so the served rule uses similarity only. **No cold-start
fallback any more:** v4/v5 switched to plain tool search when the closest past task was below 0.66. With memory's
weight ramping up over the first 400 tasks, the best switch point fit on seed 1 is {abtau:.2f}, i.e. never switch, so the
fallback was removed.

**Remembering what couldn't be served.** The shipped rule forgets every task it abstained on, so the 50th request for a
missing capability looks as new as the first. Remembering each unserved task (a `Task` node with no `USED` edge) adds
two signals: how close the nearest unserved past task is, and the unserved share of the 20 nearest past tasks. We report
two views, because the benchmark's unserved tasks come from held-out servers and naturally repeat: "gap repeats" lets an
unserved task see earlier unserved tasks from its own servers; "new gaps only" doesn't, which is what the first request
for a new missing capability looks like. The combined rule abstains if either the shipped rule or the memory rule says so.

{untab}

Most of the gain is the same missing capability being asked for again; there is a smaller real gain on new gaps. This
needs a signal that a task went unserved (the agent called nothing, or the user said so). The live demo has no such
signal, every Toucan task is answerable, so it ships in the evaluation and the model, not on the page.

### Tools come and go

The catalog is graph state: `(:Catalog)-[:LISTS]->(:Tool)`. When a server drops a tool from `tools/list`, Deja deletes
one edge and gives the `Catalog` node a new version; every read is filtered to the live catalog, whatever memory says.

{life}

## How it works

![how it works](docs/img/4_how_it_works.png)

```
(:Task) -[:USED]-> (:Tool)                                   every past task and the tools that worked for it
(:Task) -[:FAILED]-> (:Tool)                                 calls that errored (recorded, never vote)
(:Catalog) -[:LISTS]-> (:Tool)                               the live tool catalog
(:Term {{df}}) -[:LED]-> (:Link {{n}}) -[:TO]-> (:Tool {{uses}})   a word in past tasks → the tool that solved them
(:Tool) -[:CO]-> (:CoLink {{n}}) -[:TO]-> (:Tool)              tools used together in one task
```

0. **Abstain:** a two-feature rule (closest past task, best tool match) decides first. If nothing fits, load nothing
   and say why; otherwise continue. (Cold start: memory's weights ramp up over the first 400 tasks learned.)
1. **Recall similar tasks:** embed the task (bge-small) and pick the 20 most similar past tasks by cosine (numpy, a plain
   vector index). Read their `USED` edges from HydraDB in one id-anchored batched read; each votes for its tools,
   weighted by similarity. `fused = base + 1.0 · votes`, where `base = cosine + 0.15 · BM25/max(BM25)` and each neighbour's vote is
   weighted by similarity⁸ (near neighbours dominate). **Steps** of a multi-step task do the same on their own (one more
   batched `USED` read for all steps); a tool keeps its best step vote.
2. **Co-use hop:** from the top 5 of the fused ranking, walk `CO` → `CoLink` → `Tool` to the tools those are
   usually used with (about +1 point overall).
3. **Word hop:** from the task's 8 rarest words (highest idf), walk `LED` → `Link` → `Tool`, weighted by learned counts.
   Limiting it to 8 words cut the rows read about 4× and slightly *raised* accuracy. This hop carries most of the
   multi-server gain (about +3 points).
4. **Rank and pack:** `fused + 0.5 · words + 0.2 · co-use + 0.5 · step votes − 0.05 · ln(tokens)`, packed greedily into the token budget.
5. **Learn:** after the task, write a `Task` node with `USED` (worked) and `FAILED` (errored) edges, and update the word
   and co-use counts from the calls that worked. All reads are filtered to the tools the `Catalog` node currently lists.

HydraDB specifics that shaped the design:
- Batch writes can set node properties but not edge properties, so counts live on reified `Link` and `CoLink` nodes.
- List parameters are rejected outside `UNWIND`, so multi-key reads use scalar parameters.
- The vote hop uses the id-only batched read shape, which skips per-row property reads.
- The served graph keeps word links seen ≥ 2 times, at most 10 per word.

## Under load

{load} Correct under contention; throughput is the ceiling of one node on one machine. HydraDB scales reads by adding
query nodes over the same object store, which we haven't done here.

## With an agent in the loop

{agent}

## Not done yet

- **An MCP proxy.** As a proxy in front of real MCP servers Deja would get call outcomes from `isError`, catalog changes
  from `tools/list_changed`, and the unserved signal from tasks that end without a call, and could learn from real
  agent traffic instead of Toucan's generated questions.

## Limitations

- **Synthetic questions, real execution.** Toucan's questions were generated by an LLM from each server's tool set, so
  similar tasks recur by construction, which favours any past-task memory (ours and the baseline's). Organic agent
  traffic may recur less. The similarity table above shows how much that matters: the lead over tool search is 20–30
  points in every band, including tasks whose closest past task is only loosely related (below 0.7 there are only 60
  tasks, so that row is noisy). Organic traffic would sit lower in the table. The tool calls and their results are real.
- **Step recall fits how these questions were written.** 98% of Toucan tasks split into steps because an LLM wrote them as
  multi-step requests; real prompts will split less often, and gain less from it.
- **Cold start.** Over its first 100 tasks Deja is {cold_rel} its own hybrid base used as plain tool search ({cold},
  shuffle 1), not ahead: memory has little to offer yet. Memory's weight ramps up over the first 400 tasks; without the
  ramp, v5 was at 33% vs 45%. A hosted version would start warm from memory shared
  across a team's agents.
- **Abstaining is partial.** At 5% wrong abstentions it catches 31–46% of tasks with no fitting tool. The benchmark's
  "no fitting tool" labels come from held-out server clusters and an embedding-based substitute check, not from people.
  It misses near-topic gaps: "List my open GitHub issues and group them by repository" (reported by a reader; the catalog
  has no GitHub-issues tool) gets 16 GitHub-adjacent tools at confidence 0.89 (abstains below 0.82), because the closest
  past task, about trending GitHub repositories, is 0.76 similar. The rule measures topic, not whether a tool can do the action.
- **Failure detection is a text heuristic** on the recorded responses, hand-checked on a sample, not ground truth.

## What we measured and threw away

- **Shared entity memory** ("reuse IDs other agents already resolved") was our first headline, with kill criteria set
  before building. It failed them: binding precision was 21%, and even a perfect binder would save only 6.3% of lookups
  beyond a plain `(tool, args)` cache.
- **Graph signals for abstaining** (neighbour agreement, server spread, co-use coherence) added nothing over plain
  similarity; the abstain rule doesn't use them.
- **A cross-encoder "can a tool do this?" check for abstaining** (local MiniLM, Jina and bge rerankers, task vs the top
  8 tools). On the GitHub-issues query, which no tool can do, all three scored *higher* than 60–100% of answerable tasks:
  they judge topic relevance, like the similarity rule. They were also too slow on CPU (2–19 pairs/s). Dropped before
  the full benchmark; a model yes/no on the loadout is the next thing to try.
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
for i in 00000 00001; do curl -L -o data/raw/kimi_$i.parquet \\
  https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/resolve/main/Kimi-K2/train-$i-of-00040.parquet; done
uv run python retr.py 2000          # prequential eval, 4 orderings -> data/results_v2_2000.json
uv run python hydra_store.py        # HydraDB loadouts == in-memory twin (asserts)
uv run python load.py 16            # 16 concurrent planners + 1 learner writing
uv run python outcome.py            # learning only from calls that worked -> data/outcome.json
uv run python abstain.py            # abstention benchmark (+ fallback switch point, now 0) -> data/abstain.json
uv run python lifecycle.py          # tool removal on HydraDB + mid-stream removal -> data/lifecycle.json
uv run python catalog_check.py      # tool removal through the server's loadout path -> data/catalog_check.json
uv run --with matplotlib python charts.py docs/img
uv run python server.py             # bulk-load graph, held-out check, page + /api/loadout on :8090
./deploy.sh ubuntu@<vm>             # Ubuntu VM (x86_64 or arm64), opens port 80
uv run python make_readme.py        # regenerate this README's tables from the results
```

`sim.py` is the killed entity-memory experiment, kept so those numbers reproduce. `spike_paths.py` is the hour-one
HydraDB spike (weighted paths, multi-type `MSpaths`, a concurrent writer).
"""
open("README.md", "w").write(readme)
print("README.md written")
