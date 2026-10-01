
import os, sys, json, shutil, pandas as pd, numpy as np
from pathlib import Path
import yaml
import networkx as nx
import random
import argparse
import math
import matplotlib.pyplot as plt
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dcm"))
sys.path.insert(0, str(PROJECT_ROOT / "dcm" / "dat_gen"))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "RCAEval"))

import logging
lgr = logging.getLogger(__name__)

from RCAEval.e2e.rcd import rcd

from RCAEval.utility import dump_json

from RCAEval.e2e.baro import baro

from RCAEval.e2e.rcg import rcg

from RCAEval.e2e.circa import circa

from RCAEval.e2e import nsigma

from admg_utils import create_graph, plot_dag_with_latents

# Import LIT data generation modules
from data.sem_latent import generate_artificial_data, sample_ER

# DCM_RCA runs in-process now (see the removed DCM_PYTHON/subprocess machinery below
# this diff, and the turn's report for why: empirically verified that everything
# rca_in_lit_.py + DCM_RCA needs -- RCAEval.e2e.{rcd,baro}, admg_utils (pyagrum),
# data.sem_latent, and torch/causalflows (transitively via dcm_rca) -- imports
# together under one interpreter. The old DCM_PYTHON escape hatch existed because
# rca_in_lit_.py itself never used to need torch; now that it constructs DCM_RCA
# directly, it does, unconditionally, at import time. If you run this script under
# an interpreter that lacks torch/causalflows, it will fail here, at this import,
# not later at the old subprocess boundary.
from dcm_rca import DCM_RCA






def get_G_from_graphs_yaml(graph_name, graphs_config):
    """
    Build DAG matrix G from a graph in graphs.yaml (same shape as LIT's generate_artificial_data expects).

    Args:
        graph_name: Key in graphs_config['graphs']
        graphs_config: Loaded from dcm/dat_gen/graphs.yaml

    Returns:
        G: DAG matrix of shape [num_obs_vars, num_vars] with num_vars = num_latent_vars + num_obs_vars
           First num_latent_vars columns: latent->observed (from confounders)
           Remaining columns: observed->observed (from dag)
        obs_names: List of observed variable names (order matches rows/cols of G)
        num_obs_vars: int
        num_latent_vars: int
    """
    graph_cfg = graphs_config['graphs'][graph_name]
    dag = graph_cfg['dag']  # child -> list of observed parent names
    confounders = graph_cfg.get('confounders', {})  # lat_name -> list of observed children

    obs_names = list(dag.keys())
    num_obs_vars = len(obs_names)
    num_latent_vars = len(confounders)

    print(f'dag: {dag}')
    print(f'confounders: {confounders}')
    print(f'obs_names: {obs_names}')
    print(f'num_obs_vars: {num_obs_vars}')
    print(f'num_latent_vars: {num_latent_vars}')
    print('--------------------------------')

    # Observed->observed DAG
    G_obs = np.zeros((num_obs_vars, num_obs_vars), dtype=np.int32)
    for i, child in enumerate(obs_names):
        for parent in dag[child]:
            if parent in obs_names:
                G_obs[i, obs_names.index(parent)] = 1

    # Latent->observed (confounders)
    mat_B = np.zeros((num_obs_vars, num_latent_vars), dtype=np.int32)
    for lat_idx, (lat_name, children) in enumerate(confounders.items()):
        for obs in children:
            if obs in obs_names:
                mat_B[obs_names.index(obs), lat_idx] = 1

    G = np.hstack((mat_B, G_obs))
    return G, obs_names, num_obs_vars, num_latent_vars


def build_nx_graph_from_yaml(graph_name, graphs_config):
    """Build networkx DiGraph for RCA from graphs.yaml (same as rcd_baro_lit_data)."""
    dag = graphs_config["graphs"][graph_name]["dag"]
    G = nx.DiGraph()
    for node, parents in dag.items():
        G.add_node(node)
        for parent in parents:
            G.add_edge(parent, node)
    return G


def preprocess_lit_data(data, inject_time, length_min=20, tdelta=0):
    """Same preprocessing as run_single for merged_timeseries LIT data (same as rcd_baro_lit_data)."""
    inject_time = int(inject_time) + tdelta
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill().fillna(0)
    if "time" in data.columns:
        normal_df = data[data["time"] < inject_time].tail(length_min * 60 // 2)
        anomal_df = data[data["time"] >= inject_time].head(length_min * 60 // 2)
    else:
        normal_df = data.iloc[:inject_time].tail(length_min * 60 // 2)
        anomal_df = data.iloc[inject_time:].head(length_min * 60 // 2)
    return pd.concat([normal_df, anomal_df], ignore_index=True), inject_time


# Option 2: Generate a dense DAG manually (same format as LIT's generate_artificial_data expects)
def generate_dense_dag(num_obs_vars, num_latent_vars, edge_density=0.4, random_seed=42):
    """
    Generate a dense DAG with specified edge density.
    Returns G of shape [num_obs_vars, num_vars] with num_vars = num_latent_vars + num_obs_vars.
    """
    np.random.seed(random_seed)
    mat_B = np.zeros((num_obs_vars, num_latent_vars), dtype=np.int32)
    for lat_idx in range(num_latent_vars):
        min_connections = 2
        connections = np.random.choice(num_obs_vars, size=min(num_obs_vars, min_connections), replace=False)
        mat_B[connections, lat_idx] = 1
    dag_obs = np.zeros((num_obs_vars, num_obs_vars), dtype=np.int32)
    for child_idx in range(num_obs_vars):
        max_parents = max(1, int((child_idx) * edge_density))
        if max_parents > 0:
            possible_parents = list(range(child_idx))
            if len(possible_parents) > 0:
                num_parents = min(max_parents, len(possible_parents))
                parents = np.random.choice(possible_parents, size=num_parents, replace=False)
                dag_obs[child_idx, parents] = 1
    G = np.hstack((mat_B, dag_obs))
    return G


def plot_marginal(normal_df, anomalous_df, selected_var_anomalous, dat_dir_full):
    ## Plot Marginal Distributions

    # Plot marginal distributions for each variable
    # Compare normal vs anomalous data
    fig, axes = plt.subplots(2, (normal_df.shape[1] + 1) // 2, figsize=(15, 8))
    axes = axes.flatten() if normal_df.shape[1] > 1 else [axes]

    for idx, col in enumerate(normal_df.columns):
        ax = axes[idx]
        
        # Plot histograms for normal and anomalous data
        ax.hist(normal_df[col], bins=50, alpha=0.6, label='Normal', density=True, color='blue', edgecolor='black')
        ax.hist(anomalous_df[col], bins=50, alpha=0.6, label='Anomalous', density=True, color='red', edgecolor='black')
        
        ax.set_title(f'{col} Marginal Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(normal_df.shape[1], len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Marginal Distributions: Normal vs Anomalous Data ({normal_df.shape[1]} variables)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()  
    plt.savefig(dat_dir_full / f'marginal_distributions_{normal_df.shape[1]}.png')
    plt.close()

    # Print statistics for intervened variables
    if len(selected_var_anomalous) > 0:
        print(f"\nIntervened variables (observed indices): {selected_var_anomalous}")
        print("Statistics for intervened variables:")
        for var_idx in range(normal_df.shape[1]):
            var_name = f'X{var_idx+1}'
            if var_name in normal_df.columns:
                print(f"\n{var_name}:")
                print(f"  Normal - Mean: {normal_df[var_name].mean():.4f}, Std: {normal_df[var_name].std():.4f}")
                print(f"  Anomalous - Mean: {anomalous_df[var_name].mean():.4f}, Std: {anomalous_df[var_name].std():.4f}")
                print(f"  Mean difference: {abs(anomalous_df[var_name].mean() - normal_df[var_name].mean()):.4f}")
                print(f"  Std difference: {abs(anomalous_df[var_name].std() - normal_df[var_name].std()):.4f}")


def run_rca(method, data, inject_time, graph, target_node, sli, output_dir=None, result_name=None):
    """Run RCD or Baro; return ranks and optionally save JSON."""
    run_args = argparse.Namespace(root_path=os.getcwd(), data_path=None)
    num_node = len([c for c in data.columns if c != "time"])
    func = {"baro": baro, "rcd": rcd, "rcg": rcg, "circa": circa, "nsigma": nsigma}[method]
    if func is None:
        print(f"{method} not available in this Python version")
        return None
    out = func(
        data,
        inject_time,
        dataset="online-boutique",
        anomalies=None,
        dk_select_useful=False,
        sli=sli,
        verbose=False,
        n_iter=num_node,
        args=run_args,
        graph=graph,
        target_node=target_node,
    )
    ranks = out.get("ranks", [])
    scores = out.get("scores", {})
    if output_dir and result_name:
        os.makedirs(output_dir, exist_ok=True)
        rp = os.path.join(output_dir, f"{method}_results", f"{result_name}.json")
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        dump_json(filename=rp, data={0: ranks})
        print(f"Results saved to {rp}")
    return ranks, scores




def run_dcm(dcm, result, dat_dir_full, selected_var_anomalous, true_root_causes, succ, iter):
    """Score a DCM_RCA Result against ground truth, updating `succ` in place.

    Takes the Result object directly instead of reading root_cause_results.csv /
    all_ranking.csv back off disk -- no CSV round-trip, and no exposure to
    BUG(A1-11)'s old write race (already fixed at the DCM_RCA layer; nothing
    written by a concurrent worker to race on in the first place now).

    Restores the per-metric x per-agg accounting (mmd/flow/wasserstein/baro x
    add/sub) that used to be commented out and silently replaced by
    root_cause_results.csv's single-column fallback (which only ever reflected
    cfg.exp.metric's 'add' aggregate). This also resolves the previous BUG(A1-09)
    (a warning that named a file whose check was commented out) as a natural
    consequence -- that whole code path no longer exists to be inaccurate about.
    result.ranking(metric=, agg=) gives each ordered node list directly; no CSV
    parsing needed. This is the MMD-vs-Wasserstein comparison the paper needs.
    """
    k = len(selected_var_anomalous)

    if result.scores.empty:
        lgr.warning(f'{dcm}: no nodes scored (see per-node logs for errors)')
        any_fail = True
    else:
        for metric in ['mmd', 'flow', 'wasserstein', 'baro']:
            for agg in ['add', 'sub']:
                dcm_rc = result.ranking(metric=metric, agg=agg).index.tolist()[:k]
                correct = len(set(dcm_rc) & set(true_root_causes))
                miss = str(k - correct)

                metric_key = f'{dcm}_{metric}_{agg}'
                if metric_key not in succ:
                    succ[metric_key] = {'res': [], 'total': 0, 'miss': {}, 'dist': {}}
                succ[metric_key]['total'] += 1
                succ[metric_key]['res'].append(f'{correct}/{k}')
                if miss not in succ[metric_key]['miss']:
                    succ[metric_key]['miss'][miss] = 0
                succ[metric_key]['miss'][miss] += 1
                for mk in succ[metric_key]['miss']:
                    succ[metric_key]['dist'][mk] = round(succ[metric_key]['miss'][mk] / succ[metric_key]['total'], 4)
                succ[metric_key]['miss'] = dict(sorted(succ[metric_key]['miss'].items()))
                succ[metric_key]['dist'] = dict(sorted(succ[metric_key]['dist'].items()))

                print(f'{metric_key}: {set(sorted(dcm_rc))} vs true: {set(sorted(true_root_causes))} -> {correct}/{k}')

        # Headline number for this method, using its own primary metric (cfg.exp.metric),
        # matching what root_cause_results.csv used to report before this rewrite.
        primary_metric = result.cfg.exp.metric
        dcm_rc = result.ranking(metric=primary_metric, agg='add').index.tolist()[:k]
        correct = len(set(dcm_rc) & set(true_root_causes))
        miss = k - correct
        miss = str(miss)
        any_fail = correct < k

        succ[dcm]['total'] += 1
        succ[dcm]['res'].append(f'{correct}/{k}')
        if miss not in succ[dcm]['miss']:
            succ[dcm]['miss'][miss] = 0
        succ[dcm]['miss'][miss] += 1
        for mk in succ[dcm]['miss']:
            succ[dcm]['dist'][mk] = round(succ[dcm]['miss'][mk] / succ[dcm]['total'], 4)
        succ[dcm]['miss'] = dict(sorted(succ[dcm]['miss'].items()))
        succ[dcm]['dist'] = dict(sorted(succ[dcm]['dist'].items()))
        print(f'{dcm}: {set(sorted(dcm_rc))} vs true: {set(sorted(true_root_causes))} -> {correct}/{k}')

    if any_fail:
        fail_dir = dat_dir_full / f"{dcm}fail/fail{iter+1}"
        os.makedirs(fail_dir, exist_ok=True)
        for f in dat_dir_full.iterdir():
            if f.is_file():
                shutil.copy2(f, fail_dir / f.name)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LIT data and run DCM RCA training")
    # Graph configuration
    # parser.add_argument("--use_graphs_yaml", action="store_true",help="Use graph from dcm/dat_gen/graphs.yaml; otherwise generate dense DAG")
    parser.add_argument("--graphs_yaml_pth", default=None, choices=[None, "dcm/dat_gen/graphs.yaml", "dcm/dat_gen/admgs_from_pag.yaml", "baselines/_archive/lit_chain_demo/graphs.yaml", "baselines/lit_iv_demo/graphs.yaml", "baselines/lit_iv_noconf_demo/graphs.yaml", "baselines/lit_iv_4node/graphs.yaml", "baselines/lit_iv_rcdtrap/graphs.yaml"], help="location of the graph object; if None, generate dense DAG")
    parser.add_argument("--sample_imb", action="store_true",help="Only use datasets that are imbalanced for benchmarking")
    parser.add_argument("--fixed_interv_targets", type=str, nargs='+', default=None,
                         help="Named observed variables (e.g. X1 X2) to intervene on every "
                              "iteration, overriding the random.sample() selection below -- for "
                              "a fixed hand-designed graph (--graphs_yaml_pth) where the "
                              "intervened set must be the SAME nodes every iteration (only the "
                              "underlying noise/SEM weights vary run to run), not a random "
                              "subset. Only meaningful with --graphs_yaml_pth set (needs obs_names).")
    parser.add_argument("--intv_wgt_custom", type=str, default=None,
                         help="explicit comma-separated per-noise-slot intervention weights "
                              "(length = num_latent + num_obs), overriding --intv_wgt_func. "
                              "The built-in ramps only give evenly-spaced ratios; this allows "
                              "pushing one variable's intervention far below the others.")
    parser.add_argument("--tot", type=int, default=200, help="number of iterations (datasets) to generate/evaluate")

    parser.add_argument("--intv_wgt_func", type=str, default="exp",choices=["linear_inc", "linear_dec", "log_inc", "log_dec", "none"],help="Function to control intervention strength")           
    parser.add_argument("--graphs", type=str, nargs='+', default=["admg_01", "admg_02", "admg_03"],help="Graph names from graphs.yaml (e.g. bow, dbl_bow, star_bow). Used when --graphs_yaml_pth is not None")
    parser.add_argument("--num_vars", type=int, default=20,help="Number of variables. Used when not --graphs_yaml_pth is not None")
    parser.add_argument("--latent_vars_perc", type=float, default=None,help="Percentage of latent variables. Used when not --graphs_yaml_pth is not None")
    parser.add_argument("--edge_density", type=float, default=0.5, help="what percent of ancestors does have an edge to a node")
    # Intervention
    parser.add_argument("--obs_interv_perc", type=float, default=0.5,help="Percentage of observed variables to intervene on")
    parser.add_argument("--latent_interv_perc", type=float, default=0.0,help="Fraction of interventions on latent vars")
    # Data generation
    parser.add_argument("--num_env", type=int, default=1,help="Number of environments for anomalous data")
    parser.add_argument("--num_obs_normal", type=int, default=5000,help="Number of normal samples")
    parser.add_argument("--num_obs_anomalous", type=int, default=5000,help="Number of anomalous samples per environment")
    # SEM parameters
    parser.add_argument("--num_layers", type=int, default=2,help="Number of SEM mixing layers (input -> hidden -> output)")
    parser.add_argument("--activation", type=str, default="leakyrelu",choices=["leakyrelu", "tanh"],help="Activation function")
    parser.add_argument("--sourcetype", type=str, default="laplace",choices=["laplace", "gaussian"],help="Noise type")
    parser.add_argument("--lrange_min", type=float, default=0.,help="Min noise scale (normal)")
    parser.add_argument("--lrange_max", type=float, default=3.,help="Max noise scale (normal)")
    parser.add_argument("--lrange_min_interv", type=float, default=2.,help="Min intervention strength (anomalous)")
    parser.add_argument("--lrange_max_interv", type=float, default=12., help="Max intervention strength (anomalous)")
    parser.add_argument("--latent_edge_strength", type=float, default=0.0, help="Scale factor for latent->observed edge strength (stronger confounder)")
    # Discretization
    parser.add_argument("--n_bins", type=int, default=30,help="Number of bins for discretization")
    # Training (passed to train.py)
    parser.add_argument("--num_epc", type=int, default=50,help="Number of training epochs")
    parser.add_argument("--thr_num", type=int, default=10,help="Threshold number for RCA")
    parser.add_argument("--gpu_id", type=int, default=0,help="GPU ID for training")
    parser.add_argument("--seed", type=int, default=51,help="Random seed")

    parser.add_argument("--methods", type=str, nargs='+', default=["dcm_flow"], choices=["rcd", "baro", "rcg", "circa", "nsigma", "dcm_flow"], help="Method(s) to use for RCA")
    parser.add_argument("--early_stop", action="store_true",
                         help="stop each candidate's training once its wasserstein tvd_diff "
                              "stabilizes, training up to --max_epc instead of the fixed "
                              "--num_epc (cfg.trn.early_stop_tol/early_stop_patience). Off by "
                              "default, preserving the previous fixed-num_epc behavior.")
    parser.add_argument("--max_epc", type=int, default=50,
                         help="early-stop ceiling, used instead of --num_epc when --early_stop is set")
    parser.add_argument("--data_root", type=str, default="baselines/LIT/data",
                         help="where per-iteration generated data/graph.yaml/succ.json land, relative "
                              "to PROJECT_ROOT (default preserves existing behavior)")
    parser.add_argument("--out_root", type=str, default="dcm/out",
                         help="where DCM_RCA's per-iteration Result.to_csv() output lands, relative "
                              "to PROJECT_ROOT (default preserves existing behavior)")


    print_exp_args(parser.parse_args(), parser)

    
    return parser.parse_args()


def print_exp_args(args, parser):
    """Write current CLI values + help text next to this script (compact)."""
    out_path = SCRIPT_DIR / "rca_in_lit_arg_log.txt"
    lines = []
    for action in parser._actions:
        if not action.option_strings or action.dest in ("help",):
            continue
        val = getattr(args, action.dest, None)
        flag = action.option_strings[-1]  # prefer long form, e.g. --num_vars
        help_txt = (action.help or "").strip()
        lines.append(f"{flag}={val}  # {help_txt}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote experiment args status to: {out_path}")


def main():
    args = parse_args()



    baselines = args.methods

    # dcm_mlp (DCM_MLP class removed from functions/models.py in the earlier prune)
    # and dcm_vtoy/dcm_vtoy_trn_tst (BUG(A1-06): dcm/train_one_mdl.py and
    # dcm/train_one_mdl_trn_tst.py were never ported into this tree) are no longer
    # selectable at all -- removed from --methods' choices above rather than
    # gated at runtime. See dependency_trace.md and BUGS.md for what they used to do.

    print(f'Running RCA for methods: {baselines}')

    succ = {'rcd': {'res': [], 'total': 0, 'miss':{}, 'dist':{}},
        'baro': {'res': [], 'total': 0, 'miss':{}, 'dist':{}},
        'rcg': {'res': [], 'total': 0, 'miss':{}, 'dist':{}},
        'circa': {'res': [], 'total': 0, 'miss':{}, 'dist':{}},
        'nsigma': {'res': [], 'total': 0, 'miss':{}, 'dist':{}},
        'dcm_flow': {'res': [], 'total': 0, 'miss':{}, 'dist':{}},
        }

    
    # admg_pth = PROJECT_ROOT / "baselines" / "LIT" / "data" / "pag_to_admg"
    admg_pth = PROJECT_ROOT / args.data_root
    
    
    # succ_pth = admg_pth
    # if os.path.exists(succ_pth / "succ.json"):
    #     with open(succ_pth / "succ.json", "r") as f:
    #         succ_old = json.load(f)
    #     succ.update(succ_old)

        

    print(f'succ: {succ}')

    fail = 0
    tot = args.tot
    for iter in range(tot):


        args.seed += 1

        rnd = random.randint(0, 1000)


        


        # ============================================================================
        # COMPLEX CONFIGURATION - Choose: graphs.yaml OR dense DAG
        # ============================================================================
        if args.graphs_yaml_pth is not None:
            # Load graphs from dcm/dat_gen/graphs.yaml -- READ-ONLY. This file is
            # user-authored; it is never written back to (see the per-iteration
            # YAML write further down, which is the only thing this script writes).
            GRAPHS_YAML_SOURCE = PROJECT_ROOT / args.graphs_yaml_pth
            with open(GRAPHS_YAML_SOURCE, 'r') as f:
                graphs_config = yaml.safe_load(f)
            
            
            # 4 experiments for admg.
            # graph_name = args.graphs[math.floor(iter/4)] if len(args.graphs) > 1 else args.graphs[0]
            # iterating over graphs
            graph_name = args.graphs[iter%len(args.graphs)] if len(args.graphs) > 1 else args.graphs[0]
            G_dense, obs_names, num_obs_vars, num_latent_vars = get_G_from_graphs_yaml(graph_name, graphs_config)
            num_vars = num_latent_vars + num_obs_vars

            print(f'Chosen graph: {graph_name}')
            print(f'graphs_config: {graphs_config}')
            print(f'obs_names: {obs_names}')
            print(f'num_obs_vars: {num_obs_vars}')
            print(f'num_latent_vars: {num_latent_vars}')
            print(f'num_vars: {num_vars}')
            print(f'num_vars: {num_vars}')
        else:
            num_vars = args.num_vars
            num_latent_vars = int(args.num_vars*0.5) if args.latent_vars_perc is None else int(args.num_vars*args.latent_vars_perc)
            num_obs_vars = num_vars - num_latent_vars
            obs_names = [f'X{i+1}' for i in range(num_obs_vars)]
            graph_name = f"lit_random_v{num_vars}_lat{num_latent_vars}"
            
            print(f'edge_density: {args.edge_density}')
            G_dense = generate_dense_dag(num_obs_vars, num_latent_vars, edge_density= args.edge_density, random_seed=args.seed)
            graphs_config = {}


        print(f'num_vars: {num_vars}, num_latent_vars: {num_latent_vars}')



        # Data directory (same structure as bnch_mrk_lit)
        dat_dir_top_level = admg_pth / f"{graph_name}_{args.intv_wgt_func}_latnstrg{args.latent_edge_strength}_Ns{args.num_obs_normal}As{args.num_obs_anomalous}"
        dat_dir_full = dat_dir_top_level / f"dataset{iter+1}_seed{rnd:03d}"
        os.makedirs(dat_dir_top_level, exist_ok=True)
        os.makedirs(dat_dir_full, exist_ok=True)


        # Plot DAG and save to baselines/LIT/data/<graph_name>/
        dag_plot_path = dat_dir_full / "dag_plot.png"
        plot_dag_with_latents(G_dense, obs_names, num_obs_vars, num_latent_vars, graph_name, save_path=dag_plot_path)
        print(f'DAG plot saved to: {dag_plot_path}')






        # Interventions: |T| on observed variables
        if args.fixed_interv_targets:
            # Fixed hand-designed graph: intervene on the SAME named nodes every
            # iteration (e.g. always X1,X2 -- Z,X in chain_confound_demo's
            # semantic mapping), not a random subset of the observed variables.
            interv_targets = [obs_names.index(n) + num_latent_vars for n in args.fixed_interv_targets]
            num_vars_intervened = len(interv_targets)
        else:
            num_vars_intervened = int((num_vars- num_latent_vars+1)* args.obs_interv_perc)  #args.num_vars_intervened
            interv_targets = random.sample(list(range(num_latent_vars, num_vars)), num_vars_intervened)
        latent_interv_perc = args.latent_interv_perc

        # Data generation parameters
        num_env = args.num_env
        num_obs_normal = args.num_obs_normal
        num_obs_anomalous = args.num_obs_anomalous

        
        
        # Generate normal data (no interventions) with dense graph
        sensor_normal, noise_normal, label_normal, G_normal, selected_var_normal, selected_var_all_normal = generate_artificial_data(
            num_vars=num_vars,
            num_latent_vars=num_latent_vars,
            interv_targets=[],  # No interventions for normal data
            num_vars_intervened=0,
            latent_interv_perc=0.0,
            num_env=1,  # Single environment for normal
            num_obs=num_obs_normal * num_env,  # Total samples
            G=G_dense,  # Use our dense graph
            num_layer=args.num_layers,
            activation=args.activation,
            sourcetype=args.sourcetype,
            lrange_min=args.lrange_min,
            lrange_max=args.lrange_max,
            lrange_min_interv=args.lrange_min,
            lrange_max_interv=args.lrange_max,
            random_seed=args.seed,
            latent_edge_strength=args.latent_edge_strength
        )

        # arr_min, arr_max = sensor_normal.min(axis=0), sensor_normal.max(axis=0)
        # sensor_normal = (sensor_normal - arr_min) / (arr_max - arr_min + 1e-8)

        # Generate anomalous data (with interventions on latents) using same dense graph
        sensor_anomalous, noise_anomalous, label_anomalous, G_anomalous, selected_var_anomalous, selected_var_all_anomalous = generate_artificial_data(
            num_vars=num_vars,
            num_latent_vars=num_latent_vars,
            interv_targets=interv_targets,  # Intervene on latent variables
            num_vars_intervened=num_vars_intervened,
            latent_interv_perc=latent_interv_perc,
            num_env=num_env,  # Multiple environments with different interventions
            num_obs=num_obs_anomalous,  # Samples per environment
            G=G_normal,  # Use same dense graph structure as normal
            num_layer=args.num_layers,
            activation=args.activation,
            sourcetype=args.sourcetype,
            lrange_min=args.lrange_min,
            lrange_max=args.lrange_max,
            lrange_min_interv=args.lrange_min_interv,
            lrange_max_interv=args.lrange_max_interv,
            random_seed=args.seed,
            intv_wgt_func=args.intv_wgt_func,
            latent_edge_strength=args.latent_edge_strength,
            intv_wgt_vec=([float(x) for x in args.intv_wgt_custom.split(',')]
                          if args.intv_wgt_custom else None)
        )

        # arr_min, arr_max = sensor_anomalous.min(axis=0), sensor_anomalous.max(axis=0)
        # sensor_anomalous = (sensor_anomalous - arr_min) / (arr_max - arr_min + 1e-8)

        # Identify true root causes (intervened variables)
        # Convert observed indices to variable names
        true_root_causes = set()
        if len(selected_var_anomalous) > 0:
            for var_idx in selected_var_anomalous:
                var_name = f'X{var_idx+1}'
                true_root_causes.add(var_name)

        print('true_root_causes', true_root_causes)

        # Convert to DataFrames (only observed variables, not latents)
        # sensor_normal and sensor_anomalous contain only observed variables
        normal_df = pd.DataFrame(sensor_normal, columns=[f'X{i+1}' for i in range(num_obs_vars)])
        anomalous_df = pd.DataFrame(sensor_anomalous, columns=[f'X{i+1}' for i in range(num_obs_vars)])


        # 
        # Print statistics for intervened variables
        if len(selected_var_anomalous) > 0:
            std_diff ={}
            print(f"\nIntervened variables (observed indices): {selected_var_anomalous}")
            print("Statistics for intervened variables:")
            for var_idx in range(normal_df.shape[1]):
                var_name = f'X{var_idx+1}'
                if var_name in normal_df.columns:
                    print(f"\n{var_name}: GT rc: {var_idx in selected_var_anomalous}")
                    print(f"  Normal - Mean: {normal_df[var_name].mean():.4f}, Std: {normal_df[var_name].std():.4f}")
                    print(f"  Anomalous - Mean: {anomalous_df[var_name].mean():.4f}, Std: {anomalous_df[var_name].std():.4f}")
                    print(f"  Mean difference: {abs(anomalous_df[var_name].mean() - normal_df[var_name].mean()):.4f}")
                    print(f"  Std difference: {abs(anomalous_df[var_name].std() - normal_df[var_name].std()):.4f}")

                    std_diff[var_name] = abs(anomalous_df[var_name].std() - normal_df[var_name].std())

            std_diff = sorted(std_diff.items(), key=lambda x: x[1], reverse=True)
            print(std_diff)
            std_rc= [x[0] for x in std_diff][0:len(selected_var_anomalous)]
            ret = set(std_rc)== set(true_root_causes)
            if not ret:
                fail+=1
            else:
                print(f'No std imbalance detected')

                if args.sample_imb:
                    print(f'Skipping as no std imbalance and we only sample imbalanced datasets for benchmarking')
                    shutil.rmtree(dat_dir_full, ignore_errors=True)
                    continue

            

            
                


        print(f'fail: {fail}/{iter+1}')

        # 

        # Identify important columns (exclude constants)
        important_cols = [col for col in normal_df.columns if normal_df[col].std() > 0]
        n_bins = args.n_bins

        # Discretize all variables using quantile binning
        normal_processed = normal_df[important_cols].copy()
        anomalous_processed = anomalous_df[important_cols].copy()
        combined = pd.concat([normal_processed, anomalous_processed])
        
        for col in important_cols:
            _, bin_edges = pd.qcut(combined[col], q=n_bins, retbins=True, duplicates='drop', labels=False)

            # Use the maximum bin_edges for both datasets
            normal_processed[col] = pd.cut(normal_processed[col], bins=bin_edges, labels=False, include_lowest=True)
            anomalous_processed[col] = pd.cut(anomalous_processed[col], bins=bin_edges, labels=False, include_lowest=True)
            normal_processed[col] = normal_processed[col].fillna(0).astype(int)
            anomalous_processed[col] = anomalous_processed[col].fillna(0).astype(int)

        
        # Build adjacency matrix for observed variables only
        adj_matrix = np.zeros((num_obs_vars, num_obs_vars))

        # Extract observed-to-observed connections (from the right part of G). This is the DAG for observed variables
        dag_obs = G_normal[:, num_latent_vars:]


        # ------------------------------------------------------------
        # Convert to adjacency matrix (dag_obs is lower triangular in causal order)
        # We need to build the parent-child relationships
        # In dag_obs, row i represents variable i, and column j represents whether variable j is a parent
        # So dag_obs[i, j] = 1 means j -> i (j is parent of i)

        if args.graphs_yaml_pth is None:
            for child_idx in range(num_obs_vars):
                for parent_idx in range(num_obs_vars):
                    if dag_obs[child_idx, parent_idx] == 1:
                        adj_matrix[parent_idx, child_idx] = 1  # parent -> child

            # Create DataFrame for easier manipulation
            adj_df = pd.DataFrame(adj_matrix,
                                index=[f'X{i+1}' for i in range(num_obs_vars)],
                                columns=[f'X{i+1}' for i in range(num_obs_vars)])

            # Build edge list and NetworkX graph
            edge_list = []
            for from_var in adj_df.index:
                for to_var in adj_df.columns:
                    if adj_df.loc[from_var, to_var] == 1:
                        edge_list.append((from_var, to_var))

            obs_nodes = list(normal_df.columns)


        else:
            cur_dag= graphs_config['graphs'][graph_name]['dag']
            edge_list = []
            for node in cur_dag:
                for par in cur_dag[node]:
                    edge_list.append((par, node))


            obs_nodes= list(cur_dag.keys())

        
        
        G = nx.DiGraph()
        G.add_nodes_from(obs_nodes)
        G.add_edges_from(edge_list)

        # Build DAG dictionary from edge_list (if not already built)
        if 'dag' not in locals():
            graph_cols = obs_nodes
            dag = {}
            for node in graph_cols:
                parents = [p for p, c in edge_list if c == node]
                dag[node] = parents if parents else []



        # Build DAG dictionary
        graph_cols = obs_nodes
        dag = {}
        for node in graph_cols:
            parents = [p for p, c in edge_list if c == node]
            dag[node] = parents if parents else []

        # Build dim_dict from discretized data
        dim_dict = {}
        for col in graph_cols:
            combined = pd.concat([normal_processed[col], anomalous_processed[col]])
            dim_dict[col] = max(2, len(combined.unique()))

        # Create graph name
        exp_name = 'LIT'

        # Get topological order
        topo_order = list(nx.topological_sort(G))

        # Reorder dag and dim_dict in topological order
        dag_ordered = {node: dag[node] for node in topo_order}
        dim_dict_ordered = {node: 1 for node in topo_order}

        # Extract confounders from latent->observed connections
        # G_normal[:, :num_latent_vars] contains latent->observed connections
        confounders = {}
        latent_to_obs = G_normal[:, :num_latent_vars]  # Shape: (num_obs_vars, num_latent_vars)

        for lat_idx in range(num_latent_vars):
            lat_name = f'L{lat_idx+1}'
            affected_obs = []
            for obs_idx in range(num_obs_vars):
                if latent_to_obs[obs_idx, lat_idx] == 1:
                    obs_name = f'X{obs_idx+1}'
                    affected_obs.append(obs_name)
            if affected_obs:  # Only add if latent affects at least one observed variable
                confounders[lat_name] = affected_obs

        # Reorder dag and dim_dict in topological order (discrete)
        topo_order = list(nx.topological_sort(G))
        dag_ordered = {node: dag[node] for node in topo_order}
        dim_dict_ordered = {node: dim_dict[node] for node in topo_order}

        # Write this iteration's graph to its OWN per-iteration YAML (inside
        # dat_dir_full), not the shared dcm/dat_gen/graphs.yaml. That shared file
        # used to be reused across all 200 iterations: loaded, this iteration's
        # graph_name entry added, the whole thing written back. Entries
        # accumulated, and graph_name recurs constantly -- in dense-DAG mode it's
        # derived from num_vars/num_latent_vars, which don't change across a
        # sweep, so literally every iteration wrote the SAME key. The previous
        # iteration's entry got silently overwritten while its DATA directory is
        # per-iteration and never deleted, so re-reading the shared file for an
        # earlier iteration's graph returned a LATER iteration's structure
        # instead. This is very likely the cause of the "Graph is being updated
        # incorrectly" symptom the comment that used to sit here was flagging --
        # the per-iteration file removes the shared-mutable-state problem
        # entirely, regardless of the exact mechanism that produced that symptom.
        #
        # When --graphs_yaml_pth is given, that file is user-authored input --
        # READ-ONLY now (see GRAPHS_YAML_SOURCE above), never written to. Checked
        # both downstream readers of the graph config against this change:
        #   - build_nx_graph_from_yaml() (rcd/baro branches, below) reads the
        #     in-memory graphs_config dict, not a file -- graphs_config still
        #     gets this iteration's entry added below, so it's unaffected.
        #   - get_G_from_graphs_yaml() (used above, when --graphs_yaml_pth is
        #     set) already ran earlier this iteration, reading graphs_config as
        #     loaded from GRAPHS_YAML_SOURCE before any mutation -- also
        #     unaffected.
        graphs_config.setdefault('graphs', {})[graph_name] = {
            'type': 'admg',
            'dag': dag_ordered,
            'confounders': confounders,
            'dim_dict': dim_dict_ordered,
            'root_cause': ''.join([f'X{i+1},' for i in selected_var_anomalous])
        }

        iter_yaml = dat_dir_full / "graph.yaml"
        with open(iter_yaml, 'w') as f:
            yaml.dump({'graphs': {graph_name: graphs_config['graphs'][graph_name]}}, f, default_flow_style=False, sort_keys=False)

        print(f'Graph: {graph_name} saved at: {iter_yaml}')


        

        # Save data to baselines/LIT/data/ (same as bnch_mrk_lit)
        normal_df.to_csv(dat_dir_full / "cont_nrm.csv", index=False)
        anomalous_df.to_csv(dat_dir_full / "cont_anm.csv", index=False)
        normal_processed.to_csv(dat_dir_full / "disc_nrm.csv", index=False)
        anomalous_processed.to_csv(dat_dir_full / "disc_anm.csv", index=False)

        # Save merged data with inject_time (like bnch_mrk_lit)
        merged_data = pd.concat([normal_df, anomalous_df], ignore_index=True)
        merged_data["time"] = range(len(merged_data))
        merged_data.to_csv(dat_dir_full / "data.csv", index=False)
        (dat_dir_full / "inject_time.txt").write_text(str(len(normal_df)))


        plot_marginal(normal_df, anomalous_df, selected_var_anomalous, dat_dir_full)

        # Save true root causes
        with open(dat_dir_full / "true_rc.json", "w") as f:
            json.dump({"true_rc": [int(x)+1 for x in selected_var_anomalous]}, f, indent=2)

        print(f"Saved LIT data: {dat_dir_full} (inject_time={len(normal_df)})")



        if 'dcm_flow' in baselines:
            exp_dst = f"LIT/{graph_name}/latnstrg{args.latent_edge_strength}iter_{iter}_{rnd:03d}_dcm_flow"
            out_dir = PROJECT_ROOT / args.out_root / exp_dst

            # engine='original' (rca.run()'s default, below) now has its own
            # early-stopping implementation -- see train_main.py's _train().
            es_overrides = ({'trn.early_stop': True, 'trn.max_epc': args.max_epc}
                            if args.early_stop else {})
            rca = DCM_RCA(
                config_path=str(PROJECT_ROOT / 'dcm' / 'conf.yaml'),
                grp=graph_name, grp_yml=str(iter_yaml),
                **{'pth.rot_dir': str(PROJECT_ROOT), 'pth.out_dir': str(out_dir), **es_overrides},
                dst=exp_dst,
                num_epc=args.num_epc, thr_num=args.thr_num,
                mdl='dcm_flow', data_type='cont', lat_dim=10,
                upd='shared', snk_grp=False, gpu_id=args.gpu_id,
            )
            # No try/except here: construction or orchestration failures propagate
            # and halt the sweep, rather than surfacing 200 iterations later as an
            # unexplained gap in succ.json. The old subprocess path never checked
            # its return code at all -- a failed run just looked identical to
            # "no results found" downstream.
            result = rca.run(normal_data=normal_processed, anomalous_data=anomalous_processed)
            result.to_csv(out_dir)
            run_dcm('dcm_flow', result, dat_dir_top_level, selected_var_anomalous, true_root_causes, succ, iter)




        # Same parameters as rcd_baro_lit_data: preprocess data, scalar inject_time, networkx graph
        inject_time = len(normal_df)
        data_for_rca, inject_time_used = preprocess_lit_data(merged_data.copy(), inject_time)
        G_nx = build_nx_graph_from_yaml(graph_name, graphs_config)
        var_cols = [c for c in data_for_rca.columns if c != "time"]
        sli = target_node = var_cols[0]
        output_dir = str(dat_dir_full)
        result_name = f"{graph_name}_lit_20"

        
        
        for method in ["nsigma", "baro", "circa", "rcd", "rcg"]:
            
            if method not in baselines:
                continue

            print(f'Running RCA for method: {method} of {baselines}')

            if {"baro": baro, "rcd": rcd, "rcg": rcg, "circa": circa, "nsigma": nsigma}[method] is None:
                continue
            print(f"\n--- Running {method.upper()} ---")
            # Failure isolation: a method that raises must not abort a long sweep.
            # Its iteration is skipped (not counted in that method's totals), every
            # other method still runs, and the error is printed so it is visible in
            # the log rather than silently swallowed.
            try:
                ranks, scores = run_rca(
                    method, data_for_rca, inject_time_used, G_nx, target_node, sli,
                    output_dir=output_dir, result_name=result_name,
                )
            except Exception as e:
                print(f'  !! {method} FAILED on iter {iter}: {type(e).__name__}: {e}', flush=True)
                continue

            pred_rc= ranks[0:len(selected_var_anomalous)]
            print(f'pred: {set(sorted(pred_rc))} vs true: {set(sorted(true_root_causes))}')
            correct = len(set(sorted(pred_rc)) & set(sorted(true_root_causes)))
            
            
            miss= len(selected_var_anomalous) - correct
            miss = str(miss)
            
            succ[method]['total'] += 1

            if miss not in succ[method]['miss']:
                succ[method]['miss'][miss] = 0
            succ[method]['miss'][miss] += 1

            for k in succ[method]['miss']:
                succ[method]['dist'][k] = round(succ[method]['miss'][k]/ succ[method]['total'], 4)
            
            succ[method]['miss'] = dict(sorted(succ[method]['miss'].items()))
            succ[method]['dist'] = dict(sorted(succ[method]['dist'].items()))
            
            succ[method]['res'].append(f'{correct}/{len(selected_var_anomalous)}')

            if ranks:
                print(f"Top 5 root causes: {ranks[:5]}")


        print(f'---------------------------------->succ: {succ}<--------------------------------')

        

        succ_pth = dat_dir_top_level
        if os.path.exists(succ_pth / "succ.json"):
            with open(succ_pth / "succ.json", "r") as f:
                succ_old = json.load(f)
            succ_old.update(succ)
            succ = succ_old
        
        with open(succ_pth / "succ.json", "w") as f:
            json.dump(succ, f, indent= 4)


        
    

if __name__ == "__main__":
    main()


# python3 rca_in_lit_.py --edge_density=0.5 --intv_wgt_func=linear_inc --latent_vars_perc=0 --num_vars=6 --gpu_id=0 --num_epc=50 --num_obs_normal=5000 --num_obs_anomalous=5000
# python3 rca_in_lit_.py --edge_density=0.5 --intv_wgt_func=linear_inc --latent_vars_perc=0 --num_vars=50

# python3 rca_in_lit_.py --graphs_yaml_pth=dcm/dat_gen/pags.yaml --graph=admg_01 --intv_wgt_func=linear_dec --methods=dcm_flow

# Jul 29, 2026




# python3 rca_in_lit_.py --edge_density=0.5 --intv_wgt_func=linear_inc --latent_vars_perc=0.3 --methods=dcm_flow --num_vars=10 --gpu_id=0 --num_epc=50 --num_obs_normal=1000 --num_obs_anomalous=1000