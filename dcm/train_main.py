"""
train_main.py -- the ORIGINAL per-candidate training path, moved verbatim out of
dcm_rca.py (which used to define _train and _process_node directly).

For each candidate root cause Y, _process_node builds a fresh pair of models via
_create_joint_models (imported from dcm_rca) -- every mechanism in both nrm_mdl
and anm_mdl is freshly initialized, except dcm[Y], which is the literal same
module object shared between nrm_mdl and anm_mdl (the "tied" f_Y). _train then
optimizes tot_loss = nrm_loss + anm_loss over the union of trainable params.

This is the O(N) candidates x O(N) mechanisms-per-candidate = O(N^2) path. See
train_normal_once.py for the pretrain-once alternative. DCM_RCA.run(engine=...)
selects between the two.

Early stopping (cfg.trn.early_stop=True, off by default): each candidate's
_train() call stops on its own once its tvd_diff score stabilizes, same
mechanism/config keys as train_normal_once.py's Phase 2 -- see _train()'s
docstring for why this is safe here even for graphs (e.g. Sink Confounded
Star / snk_conf_grp) that can't use engine='normal_once' at all: this engine
has no cross-candidate state to get out of sync, unlike normal_once's Phase-1
pretrain sharing. When early_stop=False (default), behavior/loop-bound is
byte-identical to before early stopping existed.
"""
import os
import copy
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from functions.models import DCM_FlOW
from functions.true_scm import ObservedDataset
from functions.evaluate import evaluate_cont_loss, plot_marginal_densities

from dcm_rca import _create_joint_models, _compute_scores, _compute_scores_at_epoch, _plot_metric

lgr = logging.getLogger(__name__)


# =============================================================================
# Training (unchanged from train.py's train(), just renamed/relocated)
# =============================================================================

def _train(cfg, nrm_dat, anm_dat, nrm_mdl, anm_mdl, prc=None):
    """Unified training function for both DCM_MLP and DCM_GAN models.

    Early stopping (cfg.trn.early_stop=True, off by default): identical
    mechanism to train_normal_once.py's _train_normal_once -- monitors
    tvd_diff (the wasserstein-add score that drives the ranking, same
    quantity _compute_scores/_compute_scores_at_epoch compute post-hoc) at
    each 5-epoch eval point, stops once cfg.trn.early_stop_patience
    consecutive transitions each have relative change below
    cfg.trn.early_stop_tol, or once cfg.trn.max_epc is reached (whichever
    first). Safe here in a way it ISN'T for engine='normal_once': this
    engine trains normal+anomalous jointly from scratch for every candidate,
    with no Phase-1 pretraining shared across candidates -- so there is
    nothing early-stopping one candidate's loop could get wrong for a
    different candidate (the correctness issue that blocks snk_conf_grp +
    engine='normal_once' does not apply here; early_stop is fine even with
    snk_conf_grp under this engine). Returns (losses, tvd, stop_info) --
    stop_info is {'stopped_epoch':.., 'converged':..}, always populated
    (None/None when early_stop=False, matching train_normal_once.py's
    convention)."""
    device = torch.device(f"cuda:{cfg.exp.gpu_id}" if torch.cuda.is_available() else "cpu")

    early_stop = bool(getattr(cfg.trn, 'early_stop', False))
    max_epoch = cfg.trn.max_epc if early_stop else cfg.trn.num_epc
    tol = cfg.trn.early_stop_tol if early_stop else None
    patience = cfg.trn.early_stop_patience if early_stop else None
    primary_metric = cfg.exp.metric
    tvd_diff_history = []  # only used when early_stop=True
    stop_info = {'stopped_epoch': None, 'converged': None}

    # Detect model type: GAN has 'generators' attribute, BowBackdoor has 'dcm' attribute

    # Single optimizer over the union of nrm_mdl's and anm_mdl's trainable params,
    # deduplicated by identity. Under cfg.trn.upd=='shared' (the only mode this
    # reduced codebase supports -- asserted at DCM_RCA construction), the tied
    # module f_Y (nrm_mdl.dcm[prc] is anm_mdl.dcm[prc]) is the literal same
    # parameter tensors in both models' parameter lists; tracking each parameter
    # in exactly one optimizer means every mechanism, shared or not, gets exactly
    # one Adam step per batch from one moment-estimate history -- fixes
    # BUG(A1-15) (f_Y used to get two independent updates per batch) and, as a
    # side effect, BUG(A1-10) (there's no second optimizer left to be None).
    seen_params = set()
    all_params = []
    for p in list(nrm_mdl.get_dcm_params()) + list(anm_mdl.get_dcm_params()):
        if p.requires_grad and id(p) not in seen_params:
            seen_params.add(id(p))
            all_params.append(p)
    opt = optim.Adam(all_params, lr=cfg.trn.lrn_rat)

    nrm_loader = DataLoader(nrm_dat, batch_size=cfg.trn.bat_siz, shuffle=True)
    anm_loader = DataLoader(anm_dat, batch_size=cfg.trn.bat_siz, shuffle=True)

    print('Data loaders created')

    losses = {'nrm': [], 'anm': []}

    tvd = {}
    for met in ['mmd', 'flow', 'wasserstein', 'baro']:
        tvd[f'nrm_{met}'] = []
        tvd[f'anm_{met}'] = []
    tvd['epochs'] = []

    def _cycle(loader):
        """Infinite iterator that re-shuffles on each pass.
        NOT itertools.cycle: that caches the first pass and replays it verbatim,
        defeating shuffle=True."""
        while True:
            for batch in loader:
                yield batch

    epoch = 0
    while epoch < max_epoch:
        n_nrm, n_anm = len(nrm_loader), len(anm_loader)

        if cfg.trn.steps_per_epoch is not None:
            steps = cfg.trn.steps_per_epoch          # option C
        else:
            steps = max(n_nrm, n_anm)                # option A

        nrm_it, anm_it = _cycle(nrm_loader), _cycle(anm_loader)

        if epoch == 0:
            lgr.info(f"[{prc}] steps/epoch={steps} "
                     f"(nrm={n_nrm} batches over {len(nrm_dat)} samples, "
                     f"anm={n_anm} over {len(anm_dat)}); "
                     f"cycling {'anm' if n_anm < n_nrm else 'nrm'}")

        for _ in range(steps):
            nrm_batch, anm_batch = next(nrm_it), next(anm_it)
            opt.zero_grad()

            nrm_b = {k: nrm_batch[k].to(device) for k in nrm_batch.keys()}
            anm_b = {k: anm_batch[k].to(device) for k in anm_batch.keys()}

            if cfg.trn.upd == 'shared':
                nrm_loss, _ =  nrm_mdl(nrm_b, num_samples=1)
                anm_loss, _ = anm_mdl(anm_b, num_samples=1)
                tot_loss = nrm_loss + anm_loss
                tot_loss.backward()
                opt.step()

            losses['nrm'].append(nrm_loss.item())
            losses['anm'].append(anm_loss.item())



        stop_now = False
        if epoch % 5 == 0:
            with torch.no_grad():
                if cfg.trn.upd == 'shared':
                    assert nrm_mdl.dcm[prc] is anm_mdl.dcm[prc]
                    print(f'nrm_mdl.dcm[prc] is anm_mdl.dcm[prc]', nrm_mdl.dcm[prc] is anm_mdl.dcm[prc])


                gen_sz =  cfg.trn.gen_sam

                nrm_samples = nrm_mdl.generate_samples(batch_size=gen_sz)
                anm_samples = anm_mdl.generate_samples(batch_size=gen_sz)




                # evaluate_cont_loss now scores prc only (not every variable) --
                # matches the only thing tvd/_compute_scores ever read back out.
                eval_loss_nrm = evaluate_cont_loss(nrm_samples, nrm_dat, prc=prc)
                eval_loss_anm = evaluate_cont_loss(anm_samples, anm_dat, prc=prc)

                # collecting normalizing flow loss for prc node for ranking later (DCM_FlOW only)

                nrm_data = {k: v.to(device) for k, v in nrm_dat.data.items()}
                anm_data = {k: v.to(device) for k, v in anm_dat.data.items()}
                _, loss_dict = nrm_mdl(nrm_data, num_samples=1)
                tvd['nrm_flow'].append(loss_dict[prc].item())
                _, loss_dict = anm_mdl(anm_data, num_samples=1)
                tvd['anm_flow'].append(loss_dict[prc].item())

                for met in ['mmd', 'wasserstein', 'baro']:
                    tvd[f'anm_{met}'].append(eval_loss_anm[met])
                    tvd[f'nrm_{met}'].append(eval_loss_nrm[met])
                tvd['epochs'].append(epoch)

                if early_stop:
                    cur_tvd_diff = tvd[f'nrm_{primary_metric}'][-1] + tvd[f'anm_{primary_metric}'][-1]
                    tvd_diff_history.append(cur_tvd_diff)
                    if len(tvd_diff_history) >= patience + 1:
                        recent = tvd_diff_history[-(patience + 1):]
                        rel_changes = [abs(recent[i] - recent[i - 1]) / (abs(recent[i - 1]) + 1e-8)
                                       for i in range(1, len(recent))]
                        lgr.info(f"[{prc}] epoch {epoch} tvd_diff({primary_metric})={cur_tvd_diff:.4f} "
                                  f"rel_changes(last {patience})={[round(r, 4) for r in rel_changes]}")
                        if all(rc < tol for rc in rel_changes):
                            stop_info['stopped_epoch'] = epoch
                            stop_info['converged'] = True
                            stop_now = True
                    if not stop_now and epoch + 5 >= max_epoch:
                        # about to hit the ceiling without ever satisfying the criterion
                        stop_info['stopped_epoch'] = epoch
                        stop_info['converged'] = False

                lgr.info(f"Epoch: {epoch} dst:{cfg.exp.dst.split('/')[0]} Possible RC: {prc}")
                for met in ['mmd', 'wasserstein', 'baro', 'flow']:
                    nrm_v, anm_v = tvd[f'nrm_{met}'][-1], tvd[f'anm_{met}'][-1]
                    lgr.info(f"Nrm {met}[{prc}]: {round(nrm_v, 4)} | Anm {met}[{prc}]: {round(anm_v, 4)} diff: {round(anm_v - nrm_v, 4)}")



                # # Plot and save
                if cfg.exp.plot:
                    plot_dir = os.path.join(cfg.pth.out_dir, "plot")
                    plot_marginal_densities(nrm_samples, nrm_dat, anm_samples, anm_dat, plot_dir, prc)

                    tvd_plot_path = os.path.join(cfg.pth.out_dir, f'tvd_{prc}.png')
                    _plot_metric(cfg, tvd, tvd_plot_path, prc=prc, metric = cfg.exp.metric)



        epoch += 1
        if stop_now:
            break

    if early_stop and stop_info['stopped_epoch'] is None:
        # loop ended (epoch reached max_epoch) without either branch above
        # having set stop_info -- e.g. max_epoch itself isn't a multiple of
        # 5, so the "about to hit the ceiling" check never fired. Record the
        # last eval point actually reached, unconverged.
        stop_info['stopped_epoch'] = tvd['epochs'][-1] if tvd['epochs'] else 0
        stop_info['converged'] = False

    return losses, tvd, stop_info


# =============================================================================
# Per-node worker (requirement 5: returns, writes nothing; requirement 6: must be
# a picklable, module-level callable for spawn -- verified empirically)
# =============================================================================

def _process_node(node, cfg_dict, causal_graph, observe_u, nrm_df, anm_df, diff):
    """Process a single node. Returns a dict describing the outcome, or None if the
    node is a constant and was skipped (the old code's bare `return` here was
    BUG(A1-07) -- an implicit None that broke the caller's tuple-unpacking; the new
    contract makes "skipped" an explicit, well-defined return value instead. This is
    the only place this refactor's data-flow redesign forced a choice on that old
    bug's shape; it does not change which nodes end up scored).

    Writes nothing to disk -- see DCM_RCA.run() / Result.to_csv() (BUG(A1-11) fix).
    """
    cfg = OmegaConf.create(cfg_dict)

    # Setup logging for this process
    logging.basicConfig(level=logging.INFO)
    process_lgr = logging.getLogger(__name__)

    try:
        causal_graph.top_sort()
        print('Causal graph: ', causal_graph.type)
        if causal_graph.type == 'pag':
            print('Pag graph')
            if node in causal_graph.possible_parents.keys():
                print(f'Node {node} has possible parents: {causal_graph.possible_parents[node]}')
                pp = causal_graph.possible_parents[node]
                print('Previous parents: ', causal_graph.dag[node])
                # BUG(A1-14): PAG path mutates causal_graph.dag in place via .extend(pp); under thr_num=1 parent lists accumulate across nodes (workers are isolated only because ProcessPoolExecutor pickles a copy)
                # BUG(A1-14): impact: single-process runs use different graphs than multi-process
                causal_graph.dag[node].extend(pp)
                print('New parents: ', causal_graph.dag[node])
                print('Doing top sort')
                causal_graph.top_sort()


        # # Deep copy graphs for this node
        nrm_grp = copy.deepcopy(causal_graph)

        if cfg.exp.snk_grp == True:  # if true then using all variables as parents for the current node
            nrm_grp = nrm_grp.to_sink_graph(sink_node= node)
            lgr.info(f'Using sink graph (after top sort) for node {node}: {nrm_grp.dag}')
        elif getattr(cfg.exp, "snk_conf_grp", False) == True:
            # "Sink Confounded Star": same directed structure as snk_grp, plus every
            # other node mutually confounded (one shared latent). engine='original'
            # only -- see conf.yaml's snk_conf_grp comment for why 'normal_once'
            # can't share this transform's result across candidates.
            nrm_grp = nrm_grp.to_sink_confounded_graph(sink_node=node)
            lgr.info(f'Using sink-confounded graph (after top sort) for node {node}: '
                      f'dag={nrm_grp.dag} confounders={nrm_grp.confounders}')

        print(f'Testing node: {node}')

        # BUG(A1-??) fix: this "node in diff" (candidate itself is a globally-constant
        # column, already dropped from nrm_df/anm_df by the caller) check used to run
        # AFTER the baro-test block below, which unconditionally does nrm_df[node] --
        # a KeyError, not the intended graceful skip, whenever the candidate under
        # test is itself constant. Never triggered before this exercised engine=
        # 'original' against data with many per-case-constant columns (the causal
        # chamber's Sink Confounded Star run) -- moved here, before any nrm_df[node]
        # access, so the skip actually takes effect before the crash would.
        if cfg.exp.data_type == 'cont' and cfg.exp.normalize == True and node in diff:
            print(f'Node {node} is a constant and will not be tested')
            return None

        # baro test
        print('--------------------------------Baro test for dataset before normalization--------------------------------')
        print('nrm.columns: ', nrm_df.columns)
        print('anm.columns: ', anm_df.columns)
        nrm_med= nrm_df[node].median()
        anm_med= anm_df[node].median()
        print(f'{node} normal median: {round(nrm_med, 3)}')
        print(f'{node} anomalous median: {round(anm_med, 3)}')
        # print(f'shift in median: {round((anm_med-nrm_med)/nrm_med, 3)}')


        if cfg.exp.data_type == 'cont' and cfg.exp.normalize == True:
            if len(diff) > 0:
                print(f'Columns {diff} are constants and will be removed')
                # Drop constant variables to avoid zero-std normalization and keep graph/data aligned.
                for nd in diff:
                    nrm_grp.dag.pop(nd, None)
                    if getattr(nrm_grp, "dim_dict", None) is not None:
                        nrm_grp.dim_dict.pop(nd, None)
                # Remove edges that point to dropped parents (prevents dim_dict KeyError during model init).
                for child, parents in nrm_grp.dag.items():
                    nrm_grp.dag[child] = [p for p in parents if p not in diff]
                # Shrink confounder groups to their non-constant members (drop only
                # the constant members, not the whole group) -- BUG(A1-??) fix: this
                # used to be `if all(v not in diff for v in pair)`, which dropped an
                # ENTIRE group the moment ANY single member was constant, even if
                # >=2 other members remained genuinely confounded. Harmless for the
                # 2-member groups this was written against (shrinking a pair below
                # 2 members always meant dropping it entirely either way), but wrong
                # in general -- verified 2026-08-25: for Experiment 2's real-graph
                # confounder groups (hidden node's children) no group member was
                # ever actually constant in any of the 52 cases, so those published
                # results are unaffected -- but the new Sink Confounded Star graph's
                # 36-member group almost always has >=1 constant member per case, so
                # this bug would have silently degraded it to a plain sink graph on
                # nearly every case. Same >=2-member guard convention already used
                # by to_sink_confounded_graph/to_sink_graph.
                if getattr(nrm_grp, "confounders", None):
                    shrunk = {
                        k: [v for v in members if v not in diff]
                        for k, members in nrm_grp.confounders.items()
                    }
                    nrm_grp.confounders = {k: members for k, members in shrunk.items() if len(members) >= 2}

            if nrm_df.isna().sum().any() or anm_df.isna().sum().any():
                raise ValueError('Nan values found in the data')
            print(f'Normalized data (cont): success')


        print(f'Loading normal dataset from: {cfg.pth.nrm_dat_pth}')
        nrm_dat = ObservedDataset(graph=nrm_grp, samples=nrm_df, observe_U=cfg.mdl.obs_u)
        print(f'Loading anomalous dataset from: {cfg.pth.anm_dat_pth}')
        anm_dat = ObservedDataset(graph=nrm_grp, samples=anm_df, num_smp=cfg.trn.anm_sam, observe_U=cfg.mdl.obs_u)

        assert cfg.exp.mdl == 'dcm_flow', f'Invalid model: {cfg.exp.mdl}'
        cur_model = DCM_FlOW

        # Create models
        nrm_mdl, anm_mdl = _create_joint_models(cfg, cur_model, nrm_grp, node)

        # Train
        losses, tvd, stop_info = _train(cfg, nrm_dat, anm_dat, nrm_mdl, anm_mdl, prc=node)
        process_lgr.info(f'Training completed for node {node}. Total iterations: {len(losses["nrm"])}. '
                          f'stop_info={stop_info}')

        # Check if node is a root cause
        if getattr(cfg.trn, 'early_stop', False):
            # Use the score recorded AT the stopping point -- "the loss
            # stabilized, return with that loss" -- rather than
            # _compute_scores' trailing-last-10-epoch average, which would
            # blend in the (likely still-moving) early eval points whenever a
            # node stops in fewer than 10 eval-worth of epochs. Same
            # convention as train_normal_once.py's _process_node_normal_once.
            rank_metrics, tvd_diff = _compute_scores_at_epoch(cfg, tvd, stop_info['stopped_epoch'])
        else:
            rank_metrics, tvd_diff = _compute_scores(cfg, tvd)
        if tvd_diff is not None:
            process_lgr.info(f'Node {node}: TVD diff = {tvd_diff:.4f}')
        else:
            process_lgr.info(f'Node {node}: insufficient data (fewer than 2 eval epochs)')

        return {
            'node': node,
            'rank_metrics': rank_metrics,
            'tvd_diff': tvd_diff,
            'tvd': tvd,
            'stopped_epoch': stop_info['stopped_epoch'],
            'converged': stop_info['converged'],
        }
    except Exception as e:
        process_lgr.exception(f'Error processing node {node}: {str(e)}')
        return {
            'node': node,
            'error_msg': str(e),
            'rank_metrics': None,
            'tvd_diff': None,
            'tvd': None,
            'stopped_epoch': None,
            'converged': None,
            'error': True,
        }
