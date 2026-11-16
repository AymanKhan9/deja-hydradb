"""Agent in the loop: does a real model pick the right tool more often with Deja's loadout than with tool search's?

For each of the 200 held-out tasks (never learned from), fetch both loadouts from the live API, give Claude the task with
that loadout's tool definitions, and score the FIRST tool it calls:
  worked   - a tool that worked in the recorded run (the main score)
  first    - the recorded agent's first working tool
  called   - any tool the recorded agent called
Also records prompt tokens, so the comparison shows cost as well as accuracy. Deja abstaining counts as a miss.

  uv run python agent_check.py --dry-run     # build every request, call nothing
  OPENROUTER_API_KEY=... uv run python agent_check.py   (Claude Opus 5.5 via OpenRouter's chat-completions API)
"""
import concurrent.futures, hashlib, json, os, re, sys, time, urllib.error, urllib.request
import numpy as np
import retr

API, MODEL, PRICE_IN, PRICE_OUT = "http://140.245.206.42", "anthropic/claude-opus-5.5", 4.0, 20.0   # live server (the VM behind deja-hydradb.vercel.app); $ per 1M tokens
SYSTEM = ("You are an agent completing the user's task with the tools provided. Start the task now: call the single tool "
          "you need first. If none of the provided tools fits the task, reply in one sentence instead of calling a tool.")
cat, tasks = retr.build(); S = json.load(open("data/tool_schemas.json"))

def api_name(n):
    """Tool names must match ^[a-zA-Z0-9_-]{1,64}$; rename the others reversibly."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", n)
    return s if s == n and len(s) <= 64 else s[:55] + "_" + hashlib.md5(n.encode()).hexdigest()[:8]

def schema(p):
    p = dict(p) if isinstance(p, dict) else {}
    p.pop("$schema", None); p["type"] = "object"
    if not isinstance(p.get("properties"), dict): p["properties"] = {}
    return p

def tools_for(names, permissive=False):
    return [{"name": api_name(n), "description": S[n]["description"] or n,
             "input_schema": {"type": "object", "properties": {}} if permissive else schema(S[n]["parameters"])} for n in names]

def loadouts(i):
    for attempt in range(4):
        try: r = json.load(urllib.request.urlopen(f"{API}/api/loadout?id={i}", timeout=60)); break
        except OSError:
            if attempt == 3: raise
    return r["mode"], [t["name"] for t in r["deja"]], [t["name"] for t in r["vector"]]

def openrouter(body, key):
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(5):
        try: return json.load(urllib.request.urlopen(req, timeout=180))
        except urllib.error.HTTPError as e:
            if e.code == 400 or attempt == 4: raise
            time.sleep(2 ** attempt)             # 429 / 5xx: back off and retry
        except OSError:
            if attempt == 4: raise
            time.sleep(2 ** attempt)

def ask(key, t, names):
    if not names: return {"called": None, "abstained": True, "input_tokens": 0, "output_tokens": 0}
    back = {api_name(n): n for n in names}; note = None
    for permissive in (False, True):
        body = {"model": MODEL, "max_tokens": 4000, "reasoning": {"effort": "medium"}, "tool_choice": "auto",
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t["q"]}],
                "tools": [{"type": "function", "function": {"name": d["name"], "description": d["description"], "parameters": d["input_schema"]}}
                          for d in tools_for(names, permissive)]}
        try: resp = openrouter(body, key); break
        except urllib.error.HTTPError as e:      # a malformed community schema: retry once with permissive schemas
            if e.code != 400 or permissive: raise
            note = "permissive_schemas"
    ch = resp["choices"][0]; calls = ch["message"].get("tool_calls") or []
    first = calls[0]["function"]["name"] if calls else None
    return {"called": back.get(first, first), "stop_reason": ch.get("finish_reason"), "model": resp.get("model"), "note": note,
            "input_tokens": resp["usage"]["prompt_tokens"], "output_tokens": resp["usage"]["completion_tokens"]}

def score(t, out):
    c = out["called"]
    return {"worked": c in t["gold_ok"], "first": c is not None and c == t["first_ok"], "any_called": c in t["gold"]}

if __name__ == "__main__":
    held = [p["id"] for p in json.load(urllib.request.urlopen(API + "/api/presets", timeout=60))["presets"]]
    held = [i for i in held if tasks[i]["gold_ok"]][:int(os.environ.get("LIMIT", 10 ** 6))]   # LIMIT=3 for a smoke test
    L = {i: loadouts(i) for i in held}
    if "--dry-run" in sys.argv:
        bad = [n for i in held for n in L[i][1] + L[i][2] if n not in S]
        names = {api_name(n) for i in held for n in L[i][1] + L[i][2]}
        print(f"{len(held)} tasks; tools without a schema: {len(bad)}; distinct API tool names {len(names)}; "
              f"renamed: {sum(1 for i in held for n in L[i][1] + L[i][2] if api_name(n) != n)}; Deja abstains on "
              f"{sum(1 for i in held if L[i][0] == 'abstain')}")
        sys.exit()
    key = os.environ["OPENROUTER_API_KEY"]
    jobs = {(i, c): names for i in held for c, names in (("deja", L[i][1]), ("search", L[i][2]))}
    res = {}
    if "--retry-errors" in sys.argv:      # keep the good rows, re-ask only the requests that errored
        old = json.load(open("data/agent_check.json"))
        for r in old["rows"]:
            k = (r["task"], r["condition"])
            if "error" not in r: res[k] = {kk: v for kk, v in r.items() if kk not in ("task", "condition")}
        jobs = {k: v for k, v in jobs.items() if k not in res}
        print(f"re-asking {len(jobs)} errored requests", flush=True)
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        fut = {ex.submit(ask, key, tasks[i], names): (i, c) for (i, c), names in jobs.items()}
        for k, f in enumerate(concurrent.futures.as_completed(fut)):
            i, c = fut[f]
            try: o = f.result()
            except urllib.error.HTTPError as e:
                o = {"called": None, "error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}", "input_tokens": 0, "output_tokens": 0}
            except Exception as e: o = {"called": None, "error": f"{type(e).__name__}: {e}"[:300], "input_tokens": 0, "output_tokens": 0}
            res[(i, c)] = {**o, **score(tasks[i], o)}
            if k % 50 == 0: print(f"{k}/{len(jobs)}", flush=True)
    out = {"model": MODEL, "n_tasks": len(held), "conditions": {}}
    for c in ("deja", "search"):
        R = [res[(i, c)] for i in held]
        out["conditions"][c] = {m: float(np.mean([r.get(m, r.get("called") if m == "any_called" else False) for r in R])) for m in ("worked", "first", "any_called")} | {
            "mean_input_tokens": float(np.mean([r["input_tokens"] for r in R if r["input_tokens"]])),
            "no_tool_called": float(np.mean([r.get("stop_reason") not in ("tool_calls", "tool_use") and "error" not in r for r in R])),
            "errors": sum(1 for r in R if "error" in r),
            "served_by_other_model": sum(1 for r in R if r.get("model") and not r["model"].startswith(MODEL)),
            "permissive_schema_retries": sum(1 for r in R if r.get("note"))}
    d = [res[(i, "deja")]["worked"] - res[(i, "search")]["worked"] for i in held]
    bs = [np.random.default_rng(k).choice(d, len(d)).mean() for k in range(2000)]
    out["paired"] = {"diff_worked": float(np.mean(d)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     "deja_only": int(sum(x > 0 for x in d)), "search_only": int(sum(x < 0 for x in d))}
    tin, tout = sum(r["input_tokens"] for r in res.values()), sum(r["output_tokens"] for r in res.values())
    out["cost_usd"] = round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 2)
    out["rows"] = [{"task": i, "condition": c, **{k: v for k, v in res[(i, c)].items()}} for i, c in sorted(res)]
    json.dump(out, open("data/agent_check.json" if "LIMIT" not in os.environ else "data/agent_check_smoke.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
