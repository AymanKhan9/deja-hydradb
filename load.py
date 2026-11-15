"""Holds up under load? N concurrent planners read loadouts from HydraDB while one learner keeps writing new tasks."""
import json, random, sys, threading, time
import numpy as np
import retr
from hydra_store import Hydra, HydraLearned

READERS, SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 16, 30
cat, tasks = retr.build()
order = list(range(len(tasks))); random.Random(1).shuffle(order)
learner = HydraLearned(); learner.bulk_load([tasks[i] for i in order[:-200]])
stream = [tasks[i] for i in order[-200:]]
lat, errs, writes, stop = [], [], [0], threading.Event()

def reader(k):
    me = HydraLearned(Hydra()); rnd = random.Random(k)            # own Bolt session per planner
    while not stop.is_set():
        t = rnd.choice(tasks); s = time.perf_counter()
        try: me.graph_term_scores(t["q"]); me.graph_co_scores(sorted(t["gold"])[:5])
        except Exception as e: errs.append(type(e).__name__)
        lat.append(time.perf_counter() - s)

def writer():
    for t in stream:
        if stop.is_set(): break
        learner.ingest(t); writes[0] += 1

ts = [threading.Thread(target=reader, args=(k,)) for k in range(READERS)] + [threading.Thread(target=writer)]
t0 = time.time(); [t.start() for t in ts]; time.sleep(SECONDS); stop.set(); [t.join() for t in ts]
l = np.array(lat) * 1000
print(f"{READERS} planners + 1 learner, {SECONDS}s: {len(l)} loadout reads ({len(l)/SECONDS:.0f}/s), p50 {np.percentile(l,50):.1f}ms "
      f"p95 {np.percentile(l,95):.1f}ms p99 {np.percentile(l,99):.1f}ms, errors {len(errs)}, learner wrote {writes[0]} tasks meanwhile")
json.dump({"readers": READERS, "seconds": SECONDS, "reads": len(l), "qps": len(l) / SECONDS, "p50": float(np.percentile(l, 50)),
           "p95": float(np.percentile(l, 95)), "p99": float(np.percentile(l, 99)), "errors": len(errs), "writes": writes[0]},
          open("data/load.json", "w"))
