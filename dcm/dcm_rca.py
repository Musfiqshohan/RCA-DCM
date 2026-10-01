"""
dcm_rca.py -- importable library for the DCM_FlOW root-cause-analysis pipeline.

Absorbs what main(), process_node(), train(), and check_root_cause() do in train.py
today. train.py is now a thin @hydra.main CLI wrapper around DCM_RCA (see train.py).

WHY NO @hydra.main HERE (requirement 2):
    @hydra.main hijacks sys.argv, chdirs into a freshly-created run directory, and
    writes its own logging/config-dump files as a side effect of decoration -- all
    unacceptable for something that gets imported. This module uses plain OmegaConf
    (OmegaConf.load / OmegaConf.merge / OmegaConf.update) instead of the Hydra
    Compose API. That's sufficient because conf.yaml's only interpolations
    (${pth.rot_dir}, ${exp.dst}, ${exp.data_type}, ${exp.name}) are standard
    OmegaConf variable references -- there are no ${hydra:...} resolvers anywhere in
    the retained config (checked), so nothing here needs Hydra's compose/resolver
    machinery, only OmegaConf's, which resolves ${a.b.c} natively on its own.

Nothing in this module writes a CSV. process_node's old read-modify-write of three
shared files from parallel workers (BUG(A1-11): TOCTOU race + no locking at all on
two of the three files) is gone by construction: _process_node returns its
rank_metrics, DCM_RCA.run() collects every node's result after the pool drains, and
Result.to_csv() writes all three files once, serially, from that already-collected
data.
"""
import os
import sys
import copy
import csv
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

# Add project root to path -- same pattern train.py already used. Needed here too:
# under the spawn start method (requirement 6), each worker re-imports this module
# fresh in a new interpreter, re-running this line before functions.* is reachable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.true_scm import CausalGraph

lgr = logging.getLogger(__name__)

# NOTE: _train and _process_node used to live here. They now live in
# train_main.py (the original O(N^2) path, moved verbatim) and
# train_normal_once.py (the pretrain-normal-once path). DCM_RCA.run(engine=...)
# imports whichever one it needs, lazily (inside run(), not here at module
# scope) so that neither of those modules' own top-level
# "from dcm_rca import ..." imports (for _create_joint_models/_compute_scores/
# _plot_metric, which stayed here) creates an import cycle.


# =============================================================================
# Config helpers (requirement 1: short-name or dotted-path overrides, unknown
# keys raise, ambiguous short names raise)
# =============================================================================

def _iter_leaf_paths(cfg_node, prefix=()):
    """Yield (dotted_path, leaf_name) for every non-container key under cfg_node."""
    for key in cfg_node.keys():
        key = str(key)
        val = cfg_node[key]
        path = prefix + (key,)
        if OmegaConf.is_dict(val):
            yield from _iter_leaf_paths(val, path)
        else:
            yield (".".join(path), key)


def _build_short_name_index(cfg):
    """Map short leaf name -> list of full dotted paths ending in that name."""
    index = {}
    for full_path, leaf_name in _iter_leaf_paths(cfg):
        index.setdefault(leaf_name, []).append(full_path)
    return index


def _path_exists(cfg, dotted_path):
    node = cfg
    for part in dotted_path.split("."):
        if not (OmegaConf.is_dict(node) and part in node.keys()):
            return False
        node = node[part]
    return True


def _resolve_override_key(key, short_index, cfg):
    """Resolve an override kwarg name (short leaf name or dotted path) to a full
    dotted path. Raises ValueError on unknown or ambiguous keys."""
    if "." in key:
        if not _path_exists(cfg, key):
            raise ValueError(
                f"Unknown config override {key!r}: no such path in the config."
            )
        return key

    matches = short_index.get(key)
    if not matches:
        raise ValueError(
            f"Unknown config override {key!r}: no leaf with that name exists anywhere "
            f"in the config. Use the full dotted path (e.g. 'section.{key}') if you "
            f"meant a section, or check for a typo."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous config override {key!r}: matches {matches} -- use one of "
            f"those full dotted paths to disambiguate."
        )
    return matches[0]


def _apply_overrides(cfg, overrides):
    """Return a NEW config with `overrides` applied. Never mutates `cfg`."""
    cfg = copy.deepcopy(cfg)
    if not overrides:
        return cfg
    short_index = _build_short_name_index(cfg)
    for key, value in overrides.items():
        full_path = _resolve_override_key(key, short_index, cfg)
        OmegaConf.update(cfg, full_path, value, merge=False)
    return cfg


# =============================================================================
# Data / graph helpers
# =============================================================================

def _load_df(data, fallback_path):
    """Accept a DataFrame, a file path, or None (falls back to fallback_path) --
    requirement 3: run() must support both call styles plus the CLI's implicit one."""
    if data is None:
        return pd.read_csv(fallback_path)
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (str, os.PathLike)):
        return pd.read_csv(data)
    raise TypeError(
        f"normal_data/anomalous_data must be a pandas DataFrame, a path, or None; got {type(data)}"
    )


def _build_causal_graph(grp, grp_yml):
    """Build a CausalGraph by name from a YAML file. Fails loudly and specifically
    (requirement 4) before ever calling the real constructor:
      - YAML path doesn't exist -> FileNotFoundError
      - grp not a key under 'graphs' -> ValueError, lists the available names
    Delegates the actual construction to the unmodified CausalGraph/load_graph_config
    once both checks pass, so their behavior is otherwise untouched."""
    if not os.path.exists(grp_yml):
        raise FileNotFoundError(f"Graph YAML not found: {grp_yml}")
    with open(grp_yml, 'r') as f:
        graphs_config = yaml.safe_load(f) or {}
    available = sorted((graphs_config.get('graphs') or {}).keys())
    if grp not in available:
        raise ValueError(f"Graph {grp!r} not found in {grp_yml}. Available graphs: {available}")
    return CausalGraph(grp, grp_yml)


def _normalize(cfg, nrm_df, anm_df, nodes):
    """Median/IQR normalization -- copied verbatim from train.py's hoisted main()
    block (do not modify, per instruction). Returns (nrm_df, anm_df, diff); diff is
    an empty set when data_type != 'cont' or normalize is False, exactly matching
    the original gate."""
    diff = set()
    if cfg.exp.data_type == 'cont' and cfg.exp.normalize == True:



        joint_df = pd.concat([nrm_df, anm_df])[nodes]

        print(f'----->joint_df.std(): {joint_df.std()}')
        feature_cols= nodes
        important_cols = [col for col in feature_cols if joint_df[col].std() > 0]  # Exclude constants
        diff=set(feature_cols) - set(important_cols)

        if len(diff) > 0:
            print(f'Columns {diff} are constants and will be removed')
            nrm_df = nrm_df[important_cols]
            anm_df = anm_df[important_cols]

            joint_df = pd.concat([nrm_df, anm_df])


        for col in nrm_df.columns:
            true_median = np.median(nrm_df[col])

            q10 = np.quantile(nrm_df[col], 0.10)
            q25 = np.quantile(nrm_df[col], 0.25)
            q75 = np.quantile(nrm_df[col], 0.75)
            q90 = np.quantile(nrm_df[col], 0.90)

            iqr0 = q75 - q25
            iqr1 = q90 - q10

            iqr= iqr0

            if iqr==0:
                iqr = iqr1
                print('Reached Case 1. Extended IQR to 90th percentile')


            if iqr == 0:
                print('Reached Case 2. IQR is zero, using anomalous dataset to calculate iqr')
                q10 = np.quantile(anm_df[col], 0.10)
                q25 = np.quantile(anm_df[col], 0.25)
                q75 = np.quantile(anm_df[col], 0.75)
                q90 = np.quantile(anm_df[col], 0.90)

                iqr0 = q75 - q25
                iqr1 = q90 - q10

                print(f'{col} anomalous  q75: {q75}, q25: {q25}, iqr: {iqr}')

                iqr= iqr0

                if iqr == 0:
                    iqr = iqr1
                    print('Reached Case 3. Extended IQR to 90th percentile')

                if iqr == 0:
                    iqr = 1
                    print('Reached Case 4. IQR is zero, using 1')


            print(f'{col} true median: {true_median}, q75: {q75}, q25: {q25}, iqr: {iqr}')

            nrm_df[col] = (nrm_df[col]- true_median) / iqr
            anm_df[col] = (anm_df[col] - true_median) / iqr

            print('--------------------------------Baro test after Baro normalization--------------------------------')
            robust = float(np.max(np.abs(anm_df[col] - true_median) / iqr)) if len(anm_df[col]) else np.nan
            print(f'{col} robust: {robust}')  #anomalous median: {round(anm_med, 3)

    return nrm_df, anm_df, diff


# _train and _process_node moved to train_main.py (original path) /
# train_normal_once.py (new path) -- see the NOTE above the imports.

def _create_joint_models(cfg, Model, causal_graph, node=None):
    device = torch.device(f"cuda:{cfg.exp.gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f' torch.cuda.device_count(): {torch.cuda.device_count()}')
    nrm_mdl = Model(causal_graph, cur_node=node, latent_dim=cfg.mdl.lat_dim, hidden_dim=cfg.mdl.hid_dim, device=device)
    if cfg.trn.upd == 'shared':
        print(f'Creating shared net for node {node}')
        shared_net= nrm_mdl.get_shared_net()
    else:
        print(f'No shared net for node {node}')
        shared_net = None

    anm_mdl = Model(causal_graph, cur_node=node, latent_dim=cfg.mdl.lat_dim, hidden_dim=cfg.mdl.hid_dim, device=device, shared_net= shared_net)


    return nrm_mdl, anm_mdl


def _plot_metric(cfg, tvd, save_path, prc=None, metric=None):
    """Plot TVD over training epochs with root cause information."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    if len(tvd['epochs']) > 0:
        plt.plot(tvd['epochs'], tvd[f'nrm_{metric}'], 'o-', label=f'Normal (final: {tvd[f"nrm_{metric}"][-1]:.4f})', linewidth=2, markersize=6)
        plt.plot(tvd['epochs'], tvd[f'anm_{metric}'], 's-', label=f'Anomalous (final: {tvd[f"anm_{metric}"][-1]:.4f})', linewidth=2, markersize=6)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric, fontsize=12)

    # Build title with prc
    title = f'{metric} - {cfg.exp.dst}'
    if prc:
        title += f'\nPossible Root Cause: {prc}'
    plt.title(title, fontsize=12)

    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    lgr.info(f'{metric} plot saved to: {save_path}')


def _compute_scores(cfg, tvd):
    """Same computation as the original check_root_cause(), minus all file I/O and
    minus any pass/fail call on a node. Returns (rank_metrics, tvd_diff); both are
    None when there's insufficient epoch history -- the same gate the original
    file-write had (`if ... and tvd_diff is not None`), just expressed as a return
    value instead of a write-skip. No per-node verdict is produced -- ranking
    (Result.ranking()/top_k()) is what identifies root causes, not a per-node
    label."""
    epoch_count = len(tvd.get('epochs', []))
    if epoch_count < 2:  # Need at least 2 epochs
        return None, None

    # Get last 10 epochs (or all if less than 10)
    last_n = min(10, epoch_count)

    rank_metrics = {}
    for metric in ['mmd', 'flow', 'wasserstein', 'baro']:
        nrm_vals = tvd.get(f'nrm_{metric}', [])
        anm_vals = tvd.get(f'anm_{metric}', [])
        avg_nrm = np.mean(nrm_vals[-last_n:]) if nrm_vals else np.nan
        avg_anm = np.mean(anm_vals[-last_n:]) if anm_vals else np.nan
        add_val = avg_nrm + avg_anm if not (np.isnan(avg_nrm) or np.isnan(avg_anm)) else np.nan
        sub_val = abs(avg_nrm - avg_anm) if not (np.isnan(avg_nrm) or np.isnan(avg_anm)) else np.nan
        rank_metrics[metric] = {'nrm': avg_nrm, 'anm': avg_anm, 'add': add_val, 'sub': sub_val}

    # BUG(A1-17): training minimizes teacher-forced normalizing-flow NLL (train()'s tot_loss); the score named here (cfg.exp.metric, e.g. wasserstein/mmd/baro) is computed from generate_samples() output instead, only 'flow' would track the same quantity being optimized
    # BUG(A1-17): impact: for any cfg.exp.metric != 'flow', the model is optimized on one objective and scored on another
    primary = cfg.exp.metric
    tvd_diff = rank_metrics[primary]['add']

    return rank_metrics, tvd_diff


def _compute_scores_at_epoch(cfg, tvd, target_epoch):
    """Like _compute_scores, but reads the SINGLE eval point recorded AT
    target_epoch (tvd['epochs'] holds the epochs where `epoch % 5 == 0` fired
    during training -- 0, 5, 10, ...) instead of averaging a trailing window
    over the end of training. Lets a caller ask "what would the ranking have
    looked like if we'd stopped at epoch N", from one training run's already-
    recorded history -- no retraining needed. Returns (rank_metrics, tvd_diff);
    both None if target_epoch was never evaluated (training stopped before
    reaching it, or it isn't a multiple of 5)."""
    epochs = tvd.get('epochs', [])
    if target_epoch not in epochs:
        return None, None
    idx = epochs.index(target_epoch)

    rank_metrics = {}
    for metric in ['mmd', 'flow', 'wasserstein', 'baro']:
        nrm_vals = tvd.get(f'nrm_{metric}', [])
        anm_vals = tvd.get(f'anm_{metric}', [])
        nrm_v = nrm_vals[idx] if idx < len(nrm_vals) else np.nan
        anm_v = anm_vals[idx] if idx < len(anm_vals) else np.nan
        add_val = nrm_v + anm_v if not (np.isnan(nrm_v) or np.isnan(anm_v)) else np.nan
        sub_val = abs(nrm_v - anm_v) if not (np.isnan(nrm_v) or np.isnan(anm_v)) else np.nan
        rank_metrics[metric] = {'nrm': nrm_v, 'anm': anm_v, 'add': add_val, 'sub': sub_val}

    primary = cfg.exp.metric
    tvd_diff = rank_metrics[primary]['add']

    return rank_metrics, tvd_diff



# =============================================================================
# Result
# =============================================================================

_METRICS = ('mmd', 'flow', 'wasserstein', 'baro')
_AGGS = ('nrm', 'anm', 'add', 'sub')


class Result:
    """Collected DCM_RCA.run() output.

    No per-node verdict -- ranking (`ranking()`/`top_k()`) is what identifies root
    causes. Nodes that errored or had insufficient training history (tvd_diff is
    None) are excluded from `scores`/`ranking`/`top_k`/`to_csv` -- exactly the set
    of nodes the original per-node file-writing code would have written a row for
    (it gated on `tvd_diff is not None` too). Nodes that errored ARE present in
    `errors` (node -> exception message), so a failed run is still visible instead
    of just quietly missing from the ranking.
    """

    def __init__(self, node_results, cfg, pretrain_stopping=None):
        self._node_results = node_results
        self.cfg = cfg
        self.errors = {n: r['error_msg'] for n, r in node_results.items() if r.get('error')}

        # Phase-1 (pretrain) early-stopping metadata -- a single whole-case
        # decision (see train_normal_once.py's _pretrain_normal_base), not
        # per-node like self.stopping below. None unless engine='normal_once'
        # was run with cfg.trn.early_stop=True.
        self.pretrain_stopping = pretrain_stopping

        self._scored = {n: r for n, r in node_results.items() if r.get('rank_metrics') is not None}

        cols = [f'{m}_{k}' for m in _METRICS for k in _AGGS]
        rows = {}
        for node, r in self._scored.items():
            flat = {}
            for m in _METRICS:
                for k, v in r['rank_metrics'][m].items():
                    flat[f'{m}_{k}'] = v
            rows[node] = flat
        self.scores = pd.DataFrame.from_dict(rows, orient='index', columns=cols)
        self.scores.index.name = 'node'

        # Early-stopping metadata (only populated when engine='normal_once' was
        # run with cfg.trn.early_stop=True; empty dict otherwise -- to_csv()'s
        # extra stopping.csv write is skipped in that case, existing callers
        # unaffected).
        self.stopping = {
            n: {'stopped_epoch': r.get('stopped_epoch'), 'converged': r.get('converged')}
            for n, r in node_results.items()
            if r.get('stopped_epoch') is not None or r.get('converged') is not None
        }

    def ranking(self, metric, agg="add"):
        """Nodes sorted descending by {metric}_{agg}, as a pandas Series (node -> score)."""
        self._check_metric_agg(metric, agg)
        col = f"{metric}_{agg}"
        return self.scores[col].sort_values(ascending=False)

    def top_k(self, k, metric, agg="add"):
        """The k highest-ranked node names by {metric}_{agg}."""
        return self.ranking(metric, agg).head(k).index.tolist()

    def to_csv(self, out_dir):
        """Write root_cause_results.csv, all_metrics.csv, all_ranking.csv -- same
        format/columns/sort order as the original per-node writes, but all at once,
        serially, from already-collected results (the BUG(A1-11) fix: no shared-file
        read-modify-write from parallel workers, so no TOCTOU race and no
        dropped/corrupted rows -- see dependency_trace.md)."""
        os.makedirs(out_dir, exist_ok=True)

        # 1. root_cause_results.csv: node, tvd_diff -- same format as the original
        entries = [(n, r['tvd_diff']) for n, r in self._scored.items()]
        entries.sort(key=lambda x: (x[1] if not np.isnan(x[1]) else -np.inf), reverse=True)
        results_file = os.path.join(out_dir, 'root_cause_results.csv')
        f = open(results_file, 'w')
        try:
            csv.writer(f).writerows([['node', 'tvd_diff']] + [[n, f'{t:.6f}'] for n, t in entries])
        finally:
            f.close()

        # 2 & 3. all_metrics.csv + all_ranking.csv -- same logic as the original,
        # operating on the pre-collected `rows` instead of an incremental read-merge.
        rows = {}
        for node, r in self._scored.items():
            flat = {'node': node}
            for m in _METRICS:
                for k, v in r['rank_metrics'][m].items():
                    flat[f'{m}_{k}'] = v
            rows[node] = flat

        metrics_file = os.path.join(out_dir, 'all_metrics.csv')
        cols = ['node'] + [f'{m}_{k}' for m in _METRICS for k in _AGGS]
        with open(metrics_file, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            wr.writeheader()
            wr.writerows(rows.values())

        n_nodes = len(rows)
        ranking_file = os.path.join(out_dir, 'all_ranking.csv')
        with open(ranking_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sort_key'] + [f'r{i}_node' for i in range(1, n_nodes+1)] + [f'r{i}_score' for i in range(1, n_nodes+1)])
            for sk in [f'{m}_{k}' for m in _METRICS for k in ['add', 'sub', 'nrm', 'anm']]:
                def sort_key(r):
                    v = r.get(sk, 0)
                    try:
                        fv = float(v)
                        return fv if not np.isnan(fv) else -np.inf
                    except (ValueError, TypeError):
                        return -np.inf
                s = sorted(rows.values(), key=sort_key, reverse=True)
                score_strs = []
                for x in s:
                    v = x.get(sk, 0)
                    try:
                        fv = float(v)
                        score_strs.append(f"{fv:.6f}" if not np.isnan(fv) else "nan")
                    except (ValueError, TypeError):
                        score_strs.append("nan")
                w.writerow([sk] + [x['node'] for x in s] + score_strs)

        # 4. stopping.csv -- only when early_stop was used (self.stopping
        # non-empty). Not part of the return tuple: to_csv()'s existing 3-value
        # unpacking (main.py etc.) stays unchanged; the path is deterministic
        # (os.path.join(out_dir, 'stopping.csv')) for anyone who wants it.
        if self.stopping:
            stopping_file = os.path.join(out_dir, 'stopping.csv')
            with open(stopping_file, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['node', 'stopped_epoch', 'converged'])
                for node, info in self.stopping.items():
                    w.writerow([node, info['stopped_epoch'], info['converged']])

        # 5. pretrain_stopping.csv -- Phase-1's single whole-case stop decision
        # (not per-node like stopping.csv above). Only when early_stop was used.
        if self.pretrain_stopping:
            pretrain_stopping_file = os.path.join(out_dir, 'pretrain_stopping.csv')
            with open(pretrain_stopping_file, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['stopped_epoch', 'converged'])
                w.writerow([self.pretrain_stopping['stopped_epoch'], self.pretrain_stopping['converged']])

        return results_file, metrics_file, ranking_file

    def _check_metric_agg(self, metric, agg):
        if metric not in _METRICS:
            raise ValueError(f"Unknown metric {metric!r}; choices: {_METRICS}")
        if agg not in _AGGS:
            raise ValueError(f"Unknown agg {agg!r}; choices: {_AGGS}")


# =============================================================================
# DCM_RCA
# =============================================================================

class DCM_RCA:
    """Importable wrapper around the DCM_FlOW root-cause-analysis pipeline.

    See the module docstring for why this class uses plain OmegaConf instead of
    @hydra.main / the Hydra Compose API.

    Construction (requirement 1):
        DCM_RCA(config_path="conf.yaml", **overrides)   -- loads + merges overrides
        DCM_RCA(cfg=already_built_cfg)                   -- train.py's usage; no
                                                             overrides applied
    Overrides may be given as either the config's short leaf name (num_epc) or the
    full dotted path (via **{"trn.num_epc": 50}}, since a literal `trn.num_epc=50`
    keyword isn't valid Python syntax). An override matching no leaf anywhere in the
    config, or matching more than one (an ambiguous short name), raises immediately
    -- a silently-ignored typo is worse than a crash.

    Instances are reusable and hold no mutable state beyond `self.cfg`; run() never
    mutates it (requirement 8).
    """

    def __init__(self, config_path=None, cfg=None, **overrides):
        if (config_path is None) == (cfg is None):
            raise ValueError("DCM_RCA requires exactly one of config_path= or cfg=")

        if config_path is not None:
            base_cfg = OmegaConf.load(config_path)
        else:
            base_cfg = copy.deepcopy(cfg)

        self.cfg = _apply_overrides(base_cfg, overrides)

        # This reduced pipeline only ever had the 'shared'/'cont' code paths kept
        # (the 'seq' trn.upd branch and the discrete data_type branch were deleted
        # outright, not merely defaulted off) -- enforced here, once, for every
        # caller of the library, not just the CLI.
        assert self.cfg.trn.upd == 'shared', f"trn.upd={self.cfg.trn.upd!r} — only 'shared' is supported, the 'seq' branch was removed"
        assert self.cfg.exp.data_type == 'cont', f"exp.data_type={self.cfg.exp.data_type!r} — only 'cont' is supported, the discrete branch was removed"

    def run(self, normal_data=None, anomalous_data=None, grp=None, causal_graph=None, engine='original'):
        """Run root-cause analysis. Returns a Result.

        normal_data / anomalous_data: a DataFrame, a file path, or omitted (falls
            back to cfg.pth.nrm_dat_pth / anm_dat_pth, i.e. the CLI's own behavior).
        grp: override cfg.exp.grp for this call only -- self.cfg is never mutated.
        causal_graph: use this CausalGraph directly instead of building one from
            grp/grp_yml (for programmatically-generated graphs; skips the YAML
            lookup entirely).
        engine: 'original' (default) -- the per-candidate path from train_main.py,
            unchanged behavior from before this file was split (2N candidates x N
            mechanisms retrained from scratch every time). 'normal_once' --
            train_normal_once.py: pretrain every node's parentless normal-regime
            mechanism ONCE (shared/frozen across every candidate), keep the
            anomalous side and the tied f_Y per-candidate exactly as in
            'original'. This is a deliberate, confirmed asymmetry (see
            train_normal_once.py's module docstring and EFFICIENCY_PLAN.md) --
            not a pure speed-up of both sides. Raises for PAG graphs (parent sets
            vary per candidate there, so nothing is safe to reuse); use
            'original' for those.
        """
        if engine not in ('original', 'normal_once'):
            raise ValueError(f"engine={engine!r} — choices: 'original', 'normal_once'")
        if engine == 'normal_once' and getattr(self.cfg.exp, "snk_conf_grp", False):
            raise ValueError(
                "snk_conf_grp=True requires engine='original'. Under 'normal_once', "
                "Phase 1 pretrains one parentless mechanism per node and reuses it "
                "across every candidate Y -- but snk_conf_grp confounds each node with "
                "a DIFFERENT set of co-confounders depending on which node is currently "
                "the excluded candidate Y, so nothing is safe to share across candidates "
                "(same category of problem as PAG graphs, see train_normal_once.py)."
            )

        cfg = _apply_overrides(self.cfg, {'grp': grp} if grp is not None else {})

        nrm_df = _load_df(normal_data, cfg.pth.nrm_dat_pth)
        anm_df = _load_df(anomalous_data, cfg.pth.anm_dat_pth)

        # ---- graph (requirement 4, same order as the original main()) ----
        graph = causal_graph if causal_graph is not None else _build_causal_graph(cfg.exp.grp, cfg.pth.grp_yml)

        if cfg.exp.cmpt_grp == True:
            graph = graph.to_complete_graph()

        if getattr(cfg.exp, 'data_type', 'disc') == 'cont':
            graph.dim_dict = {node: 1 for node in graph.dag.keys()}
            lgr.info(f"data_type=cont: overrode dim_dict to all continuous: {graph.dim_dict}")

        lgr.info('printing dag and confounders')
        lgr.info(graph.dag)
        lgr.info(f'confounders: {graph.confounders}')

        nodes = list(graph.dag.keys())
        test_nodes = nodes if cfg.exp.is_rc == 'None' else [cfg.exp.is_rc]
        num_nodes = len(test_nodes)

        nrm_df, anm_df, diff = _normalize(cfg, nrm_df, anm_df, nodes)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        start_time = time.time()
        lgr.info(f"Starting execution with {cfg.trn.thr_num} processes for {test_nodes} nodes")
        num_processes = min(cfg.trn.thr_num, num_nodes)

        # Lazy imports (requirement: not at module scope) so that train_main.py's
        # / train_normal_once.py's own top-level "from dcm_rca import ..." (for
        # _create_joint_models/_compute_scores/_plot_metric, which stayed in this
        # module) never sees a partially-initialized dcm_rca -- see the NOTE near
        # the top of this file.
        
        pretrain_stop_info = None  # only set for engine='normal_once'; surfaced on Result below
        if engine == 'original':
            from train_main import _process_node
            process_args = [
                (node, cfg_dict, graph, cfg.mdl.obs_u, nrm_df, anm_df, diff)
                for node in test_nodes
            ]
        else:
            from train_normal_once import _process_node_normal_once as _process_node, _pretrain_normal_base
            lgr.info("engine='normal_once': pretraining normal-regime marginals once (Phase 1, parent process)")
            nrm_base_state, pretrain_stop_info = _pretrain_normal_base(cfg, graph, diff, nrm_df)
            lgr.info(f"[Phase 1] stop_info={pretrain_stop_info}")
            process_args = [
                (node, cfg_dict, graph, cfg.mdl.obs_u, nrm_df, anm_df, diff, nrm_base_state)
                for node in test_nodes
            ]

        node_results = {}
        if cfg.trn.thr_num == 1:
            lgr.info("Running in single-process mode")
            for args in process_args:
                res = _process_node(*args)
                if res is not None:
                    node_results[res['node']] = res
                    lgr.info(f"Completed node: {res['node']}")
        else:
            lgr.info(f"Running in multi-process mode with {num_processes} workers")
            # requirement 6: explicit spawn context. The caller may have already
            # initialized CUDA, and fork (the default on Linux) is unsafe once CUDA
            # is initialized in the parent -- spawn re-imports this module fresh in
            # each worker instead of forking the parent's (possibly CUDA-init'd)
            # memory. Verified picklable under spawn: CausalGraph is a plain-attribute
            # nn.Module (no parameters/buffers), round-trips through pickle.dumps and
            # through an actual multiprocessing.get_context('spawn') pool.
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=num_processes, mp_context=ctx) as executor:
                futures = [executor.submit(_process_node, *args) for args in process_args]
                completed_count = 0
                for future in as_completed(futures):
                    res = future.result()
                    completed_count += 1
                    label = res['node'] if res is not None else '(skipped, constant)'
                    if res is not None:
                        node_results[res['node']] = res
                    lgr.info(f"Completed {completed_count}/{len(process_args)} nodes: {label}")

        total_time = time.time() - start_time
        lgr.info(f"Execution completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        lgr.info(f"Average time per node: {total_time/num_nodes:.2f} seconds")

        return Result(node_results, cfg=cfg, pretrain_stopping=pretrain_stop_info)
