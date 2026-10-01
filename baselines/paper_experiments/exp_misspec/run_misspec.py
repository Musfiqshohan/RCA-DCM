"""GRAPH MISSPECIFICATION: same data, deliberately wrong graph.

Holds the datasets FIXED and perturbs only the graph handed to DCM, so any drop
in accuracy is attributable to the graph error alone. This is what the
edge_density sweep could NOT show: changing density regenerates the true graph
AND the data from it, and DCM still receives the correct graph.

Datasets are the 100 already generated for Exp 2 v10 (10 observed vars, no
latents, linear_dec, 5 root causes, 5000+5000 samples). Each dataset dir carries
its own graph.yaml, so the true graph is known per instance.

Arms (edges are perturbed per-instance, seeded by iteration so arms are
comparable across runs):
  true    -- unperturbed control; reproduces the Exp 2 v10 number (0.54)
  sparse  -- delete a fraction of true directed edges   (MISSING parents)
  dense   -- add that fraction of spurious edges        (PHANTOM parents)
Spurious edges respect the topological order of the true DAG, so the perturbed
graph is still acyclic and DCM's top-sort stays valid -- the error is a wrong
parent set, not a malformed graph.

DCM only: it is the one method whose input graph we are corrupting.
"""
import argparse, copy, json, os, random, sys, glob
import pandas as pd, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "dcm"))
from dcm_rca import DCM_RCA  # noqa: E402


def perturb(dag, conf, mode, frac, rng):
    """Return (new dag, new confounders, n_changed).

    dense      : ADD frac of spurious DIRECTED edges. Parents are drawn only
                 from nodes earlier in the true topological order, so the graph
                 stays acyclic and DCM's top-sort remains valid. Note a
                 supergraph is still Markov to the data -- this arm costs
                 parameters, not correctness.
    bi_sparse  : DELETE frac of the BIDIRECTED (confounder) edges, i.e. tell DCM
                 that some genuinely confounded pairs are unconfounded. This is
                 the arm that can inject false constraints.
    """
    dag = {k: list(v) for k, v in dag.items()}
    conf = {k: list(v) for k, v in (conf or {}).items()}
    nodes = list(dag.keys())
    pos = {n: i for i, n in enumerate(nodes)}
    if mode == "true" or frac <= 0:
        return dag, conf, 0
    if mode == "dense":
        # add frac of BOTH kinds: spurious directed edges AND spurious confounders
        edges = [(c, p) for c, ps in dag.items() for p in ps]
        kd = max(1, int(round(len(edges) * frac)))
        cand = [(c, p) for c in nodes for p in nodes
                if pos[p] < pos[c] and p not in dag[c]]
        for c, p in rng.sample(cand, min(kd, len(cand))):
            dag[c].append(p)
        nd = min(kd, len(cand))
        # spurious bidirected: pairs not already sharing a confounder group
        kb = max(1, int(round(len(conf) * frac))) if conf else 0
        have = {frozenset(v) for v in conf.values() if len(v) == 2}
        bcand = [ (a, b) for a, b in
                  [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
                  if frozenset((a, b)) not in have ]
        nb_added = 0
        for a, b in rng.sample(bcand, min(kb, len(bcand))):
            conf[f"Lx{nb_added}"] = [a, b]; nb_added += 1
        return dag, conf, nd + nb_added
    if mode == "both_sparse":
        # DELETE frac of BOTH kinds. Removing directed edges strips genuine
        # parents (the candidate's mechanism is then fit against an incomplete
        # parent set); removing confounders decorrelates genuinely shared noise.
        # Deleting edges always leaves a DAG, so the top-sort stays valid.
        edges = [(c, p) for c, ps in dag.items() for p in ps]
        kd = max(1, int(round(len(edges) * frac)))
        for c, p in rng.sample(edges, min(kd, len(edges))):
            dag[c].remove(p)
        nd = min(kd, len(edges))
        keys = list(conf.keys())
        kb = max(1, int(round(len(keys) * frac))) if keys else 0
        for key in rng.sample(keys, min(kb, len(keys))):
            del conf[key]
        return dag, conf, nd + min(kb, len(keys))
    if mode == "bi_sparse":
        keys = list(conf.keys())
        if not keys:
            return dag, conf, 0
        k = max(1, int(round(len(keys) * frac)))
        for key in rng.sample(keys, min(k, len(keys))):
            del conf[key]
        return dag, conf, min(k, len(keys))
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--mode", required=True, choices=["true", "bi_sparse", "both_sparse", "dense"])
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--graph_name", default="lit_random_v10_lat4")
    ap.add_argument("--num_epc", type=int, default=50)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--thr_num", type=int, default=6)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(a.data_root, "dataset*")),
                  key=lambda p: int(os.path.basename(p).split("_")[0].replace("dataset", "")))
    rows, csv = [], os.path.join(a.out, f"misspec_{a.mode}.csv")
    print(f"mode={a.mode} frac={a.frac} on {min(len(dirs), a.limit)} datasets", flush=True)
    for i, d in enumerate(dirs[:a.limit]):
        rng = random.Random(1000 + i)                     # same perturbation per i across arms
        g = yaml.safe_load(open(os.path.join(d, "graph.yaml")))
        key = list(g["graphs"].keys())[0]
        entry = copy.deepcopy(g["graphs"][key])
        n_true = sum(len(v) for v in entry["dag"].values())
        n_bi = len(entry.get("confounders") or {})
        entry["dag"], entry["confounders"], n_ch = perturb(
            entry["dag"], entry.get("confounders"), a.mode, a.frac, rng)
        gy = os.path.join(a.out, "graphs", f"{a.mode}_{i}.yaml")
        os.makedirs(os.path.dirname(gy), exist_ok=True)
        yaml.safe_dump({"graphs": {key: entry}}, open(gy, "w"))

        true_rc = {f"X{j}" for j in json.load(open(os.path.join(d, "true_rc.json")))["true_rc"]}
        nrm = pd.read_csv(os.path.join(d, "disc_nrm.csv"))
        anm = pd.read_csv(os.path.join(d, "disc_anm.csv"))
        rca = DCM_RCA(
            config_path=os.path.join(ROOT, "dcm", "conf.yaml"),
            grp=key, grp_yml=gy,
            **{"pth.rot_dir": ROOT, "pth.out_dir": os.path.join(a.out, "runs", f"{a.mode}_{i}")},
            dst=f"misspec/{a.mode}_{i}", num_epc=a.num_epc, thr_num=a.thr_num,
            mdl="dcm_flow", data_type="cont", lat_dim=10,
            upd="shared", snk_grp=False, gpu_id=a.gpu_id,
        )
        r = rca.run(normal_data=nrm, anomalous_data=anm)
        k = len(true_rc)
        got = {m: int(set(r.ranking(metric=m, agg="add").index.tolist()[:k]) == true_rc)
               for m in ["wasserstein", "mmd", "flow"]}
        rows.append({"iter": i, "mode": a.mode, "frac": a.frac, "k": k,
                     "n_true_edges": n_true, "n_true_bidirected": n_bi,
                     "n_changed": n_ch, **got})
        pd.DataFrame(rows).to_csv(csv, index=False)
        df = pd.DataFrame(rows)
        print(f"  [{i+1}] dir={n_true} bi={n_bi} changed={n_ch}  wass={got['wasserstein']}  "
              f"running PRR={df['wasserstein'].mean():.3f}", flush=True)
    df = pd.DataFrame(rows)
    print(f"\n=== {a.mode} frac={a.frac}  n={len(df)} ===")
    for m in ["wasserstein", "mmd", "flow"]:
        print(f"  {m:12s} {df[m].mean():.3f}")


if __name__ == "__main__":
    main()
