"""Spike: weighted paths + multi-type MSpaths + concurrent writer on HydraDB. Throwaway ids 990000+."""
import threading, time
from neo4j import GraphDatabase

TOKEN = open(".hydradb/auth-token").read().strip()
drv = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", TOKEN))
B = 990000
def run(q, **p):
    with drv.session(database="default") as s:
        return [r.data() for r in s.run(q, **p)]

# toy graph: 0->1 (w1), 1->3 (w1), 0->2 (w5), 2->3 (w1)  => cheapest 0->3 is via 1 (cost 2)
for a, b, w in [(0, 1, 1.0), (1, 3, 1.0), (0, 2, 5.0), (2, 3, 1.0)]:
    run("MERGE (a:Toy {id: $a, key: $ka})-[:T {w: $w}]->(b:Toy {id: $b, key: $kb})", a=B + a, b=B + b, ka=f"t{a}", kb=f"t{b}", w=w)

def sp():
    r = run("CALL algo.SPpaths({sourceNode: $s, targetNode: $t, relTypes: ['T'], relDirection: 'outgoing', maxLen: 3, pathCount: 1, weightProp: 'w'}) YIELD path, pathWeight RETURN path, pathWeight", s=B, t=B + 3)
    return [([n.id - B for n in x["path"].nodes] if hasattr(x["path"], "nodes") else x["path"], x.get("pathWeight")) for x in r]

print("weighted SP 0->3 (expect via 1):", sp())
run("MATCH (a {id: $a})-[r:T]->(b {id: $b}) SET r.w = 10.0", a=B, b=B + 1)
print("after raising 0->1 to 10 (expect via 2):", sp())

# two relationship types in one MSpaths: Mention -REFERS_TO-> Entity -BINDS-> Param
run("MERGE (m:Mention {id: $m, key: 'sao paulo'})-[:REFERS_TO]->(e:Entity {id: $e, key: '3550308'})", m=B + 10, e=B + 11)
run("MERGE (e:Entity {id: $e})-[:BINDS]->(p:Param {id: $p, key: 'ibge.GetDistritos.codigoMunicipio'})", e=B + 11, p=B + 12)
r = run("CALL algo.MSpaths({sourceLabel: 'Mention', sourceProperty: 'key', sourceValues: ['sao paulo', 'rio'], "
        "targetValues: ['ibge.GetDistritos.codigoMunicipio'], relTypes: ['REFERS_TO', 'BINDS'], relDirection: 'outgoing', "
        "maxLen: 2, pathCount: 1, pairwise: false, resultLimit: 50}) YIELD path RETURN path")
print("MSpaths mention->entity->param:", [[(n.get("key")) for n in x["path"].nodes] for x in r])

# concurrent: 8 readers running SP while 1 writer flips weights
lat, errs, stop = [], [], False
def reader():
    while not stop:
        t = time.perf_counter()
        try: sp()
        except Exception as e: errs.append(repr(e)[:200])
        lat.append(time.perf_counter() - t)
def writer():
    i = 0
    while not stop:
        run("MATCH (a {id: $a})-[r:T]->(b {id: $b}) SET r.w = $w", a=B, b=B + 1, w=float(1 + i % 20)); i += 1
ts = [threading.Thread(target=reader) for _ in range(8)] + [threading.Thread(target=writer)]
[t.start() for t in ts]; time.sleep(5); stop = True; [t.join() for t in ts]
lat.sort()
print(f"load: {len(lat)} path queries in 5s, p50 {lat[len(lat)//2]*1000:.1f}ms p95 {lat[int(len(lat)*.95)]*1000:.1f}ms, errors {len(errs)}", errs[:2])
drv.close()
