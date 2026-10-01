"""
train_normal_once.py -- the pretrain-normal-once training path (efficiency
alternative to train_main.py's original per-candidate path).

Confirmed design (2026-08-14, user sign-off on the asymmetric scheme):

  Phase 1 (once per case, in the parent process, before any candidate is
  tested): train N normal-regime mechanisms *jointly* in one pass --
  _pretrain_normal_base(). Every node is trained PARENTLESS: under sink-graph
  mode (cfg.exp.snk_grp==True) that's exactly what every Z != Y already looks
  like under to_sink_graph(Y), for every Y; under fixed-ADMG mode
  (snk_grp==False, non-PAG) the graph never changes per candidate at all, so
  the real ADMG parent sets are used directly and reused as-is. This is valid
  because in both modes a node Z != Y's mechanism input dimension does not
  depend on which node Y is currently under test -- see EFFICIENCY_PLAN.md
  sec 3 for the argument in full. PAG graphs are excluded (BUG(A1-14): parent
  sets are extended per node, so they're NOT shared across candidates) --
  _pretrain_normal_base raises rather than silently reusing the wrong thing.

  Phase 2 (per candidate Y, unchanged in cost from the original path on the
  anomalous side -- this is the deliberate asymmetry the user confirmed):
    - nrm_mdl: dcm[Z] for every Z != Y is loaded from the Phase-1 pretrained
      base and FROZEN (requires_grad=False). dcm[Y] is freshly initialized
      and trainable -- this is f_Y.
    - anm_mdl: UNCHANGED from train_main.py -- every dcm[Z] for Z != Y is a
      fresh net trained on anomalous data only; dcm[Y] is the same object as
      nrm_mdl.dcm[Y] (tied), so it alone sees both regimes.
  _train_normal_once() needs no logic change from train_main._train() to
  respect the freeze: its parameter-collection loop already filters on
  `p.requires_grad`, so the frozen Phase-1 params are excluded from the
  optimizer automatically. It's kept as an independent copy here (rather than
  imported from train_main) so the two paths can be edited/rolled back
  independently, per the user's "keep both" instruction.

Cost: Phase 1 is one joint training pass over all N nodes (not N separate
optimizations -- DCM_FlOW.forward() already sums every node's NLL into one
scalar loss, so one backward() updates every node's net at once). Phase 2 per
candidate is unchanged from the original path's anomalous-side cost (N-1
fresh anomalous nets + 1 tied f_Y). Net effect: removes the N^2 term on the
normal side only, leaving the anomalous side's N^2 term untouched by design
-- see the conversation/EFFICIENCY_PLAN.md cost table (~N^2 + 2N vs 2N^2).

Epoch count / early stopping (cfg.trn.early_stop=True; off by default, matching
train_main.py's fixed-cfg.trn.num_epc behavior when False): both phases can
stop before cfg.trn.max_epc once their own monitored score stabilizes for
cfg.trn.early_stop_patience consecutive 5-epoch evals within
cfg.trn.early_stop_tol relative change. Phase 1 (_pretrain_normal_base)
monitors the mean, across all N nodes, of the normal-only wasserstein
goodness-of-fit score -- there's no anomalous side in this phase, so no
tvd_diff to monitor the way Phase 2 does, and since all N nodes train jointly
in one backward pass this is necessarily a single whole-model stop decision,
not per-node. Phase 2 (_train_normal_once) monitors each candidate's own
tvd_diff independently, since each candidate already trains independently.
This also answers EFFICIENCY_PLAN.md sec 8 open question 2 (how many epochs
Phase 1 needs relative to Phase 2, since it's amortized over all N
candidates) empirically rather than by picking a fixed number up front.
"""
import os
import copy
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from functions.models import DCM_FlOW
from functions.true_scm import ObservedDataset
from functions.evaluate import evaluate_cont_loss, plot_marginal_densities

from dcm_rca import _compute_scores, _compute_scores_at_epoch, _plot_metric

lgr = logging.getLogger(__name__)


# =============================================================================
# Phase 1: pretrain every node's PARENTLESS normal-regime mechanism once,
# jointly, in the parent process (must run once per case, not once per
# candidate/worker -- see DCM_RCA.run()).
# =============================================================================

def _build_pretrain_graph(cfg, causal_graph, diff):
    """The graph every candidate's Z != Y mechanisms are actually fit against,
    made explicit as a single reusable graph instead of implicitly re-derived
    N times. See module docstring for why this is valid per graph mode."""
    if causal_graph.type == 'pag':
        raise ValueError(
            "engine='normal_once' does not support PAG graphs: parent sets are "
            "extended per candidate node (BUG(A1-14)), so a Z != Y mechanism's "
            "input dimension DOES depend on which Y is under test -- pretraining "
            "once would silently reuse the wrong architecture. Use "
            "engine='original' for PAG graphs."
        )

    pretrain_grp = copy.deepcopy(causal_graph)
    pretrain_grp.top_sort()

    if cfg.exp.snk_grp == True:
        # Every node parentless -- matches what to_sink_graph(Y) gives every
        # Z != Y, for every Y (to_sink_graph also always clears confounders,
        # regardless of which node is sink).
        pretrain_grp.dag = {n: [] for n in pretrain_grp.dag.keys()}
        pretrain_grp.confounders = {}
    # else: fixed ADMG (non-PAG) -- pretrain_grp.dag/confounders are already
    # exactly what every candidate's Z != Y sees; snk_grp==False means
    # _process_node_normal_once never transforms the graph per-candidate
    # either, so this is already the right structure, untouched.

    if len(diff) > 0:
        # Same constant-column drop _process_node applies per candidate --
        # done once here since `diff` doesn't depend on which candidate is
        # under test either.
        for nd in diff:
            pretrain_grp.dag.pop(nd, None)
            if getattr(pretrain_grp, "dim_dict", None) is not None:
                pretrain_grp.dim_dict.pop(nd, None)
        for child, parents in pretrain_grp.dag.items():
            pretrain_grp.dag[child] = [p for p in parents if p not in diff]
        # Shrink (don't drop-whole-group-on-any-hit) -- BUG(A1-??) fix, see
        # train_main.py's identical fix for the full rationale/verification.
        if getattr(pretrain_grp, "confounders", None):
            shrunk = {
                k: [v for v in members if v not in diff]
                for k, members in pretrain_grp.confounders.items()
            }
            pretrain_grp.confounders = {k: members for k, members in shrunk.items() if len(members) >= 2}

    pretrain_grp.top_sort()
    return pretrain_grp


def _pretrain_normal_base(cfg, causal_graph, diff, nrm_df):
    """Train one DCM_FlOW instance -- every node's mechanism fresh, none
    shared/tied (cur_node=None) -- on normal data only. Returns
    (nrm_base_state, stop_info): nrm_base_state is {node: state_dict} with all
    tensors moved to CPU (spawn-picklable, matches DCM_RCA.run()'s existing
    multiprocessing.get_context('spawn') usage for the per-candidate workers);
    stop_info is {'stopped_epoch':.., 'converged':..}.

    Early stopping (cfg.trn.early_stop=True): there is no anomalous side in
    this phase, so there is no tvd_diff (nrm_wasserstein + anm_wasserstein) to
    monitor the way _train_normal_once does. Monitor instead the MEAN, across
    all nodes, of the normal-only wasserstein goodness-of-fit score
    (evaluate_cont_loss(generated, real)['wasserstein'] -- the same quantity
    _train_normal_once computes for its nrm_wasserstein term, just with no
    anm_wasserstein counterpart to add). Because every node's mechanism is
    trained jointly in ONE backward pass here (unlike Phase 2, where each
    candidate trains independently), this is a single whole-model stopping
    decision, not per-node -- there is no way to stop training a subset of
    nodes while continuing others in a joint loss. Same
    cfg.trn.early_stop/max_epc/early_stop_tol/early_stop_patience keys as
    Phase 2, same relative-change-over-patience-consecutive-transitions
    criterion; see _train_normal_once's docstring for the full rationale.
    """
    pretrain_grp = _build_pretrain_graph(cfg, causal_graph, diff)

    device = torch.device(f"cuda:{cfg.exp.gpu_id}" if torch.cuda.is_available() else "cpu")
    lgr.info(f"[pretrain normal base] nodes: {list(pretrain_grp.dag.keys())}")

    nrm_dat = ObservedDataset(graph=pretrain_grp, samples=nrm_df, observe_U=cfg.mdl.obs_u)

    # cur_node needs *some* real node name (DCM_FlOW.__init__ unconditionally
    # does `self.shared_net = self.dcm[cur_node]` at the end, so cur_node=None
    # raises KeyError -- confirmed empirically). Which node doesn't matter:
    # with shared_net left at its default None, `node == cur_node and
    # shared_net != None` is False for every node regardless of cur_node, so
    # every dcm[node] is freshly initialized here; none of them is a
    # shared_net, and .shared_net itself is never read by this function.
    any_node = next(iter(pretrain_grp.dag.keys()))
    base_mdl = DCM_FlOW(pretrain_grp, cur_node=any_node, latent_dim=cfg.mdl.lat_dim,
                         hidden_dim=cfg.mdl.hid_dim, device=device)

    opt = optim.Adam(base_mdl.get_dcm_params(), lr=cfg.trn.lrn_rat)
    loader = DataLoader(nrm_dat, batch_size=cfg.trn.bat_siz, shuffle=True)

    early_stop = bool(getattr(cfg.trn, 'early_stop', False))
    max_epoch = cfg.trn.max_epc if early_stop else cfg.trn.num_epc
    tol = cfg.trn.early_stop_tol if early_stop else None
    patience = cfg.trn.early_stop_patience if early_stop else None
    wasserstein_history = []  # only used when early_stop=True
    stop_info = {'stopped_epoch': None, 'converged': None}

    last_loss = None
    epoch = 0
    stop_now = False
    while epoch < max_epoch:
        for batch in loader:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = base_mdl(batch, num_samples=1)
            loss.backward()
            opt.step()
            last_loss = loss.item()

        stop_now = False
        if epoch % 5 == 0 or epoch == max_epoch - 1:
            lgr.info(f"[pretrain normal base] epoch {epoch} joint nrm loss {last_loss:.4f}")

            if early_stop:
                with torch.no_grad():
                    gen_samples = base_mdl.generate_samples(batch_size=cfg.trn.gen_sam)
                    node_wasserstein = [
                        evaluate_cont_loss(gen_samples, nrm_dat, prc=node)['wasserstein']
                        for node in pretrain_grp.dag.keys()
                    ]
                cur_val = sum(node_wasserstein) / len(node_wasserstein)
                wasserstein_history.append(cur_val)
                if len(wasserstein_history) >= patience + 1:
                    recent = wasserstein_history[-(patience + 1):]
                    rel_changes = [abs(recent[i] - recent[i - 1]) / (abs(recent[i - 1]) + 1e-8)
                                   for i in range(1, len(recent))]
                    lgr.info(f"[pretrain normal base] epoch {epoch} mean nrm wasserstein={cur_val:.4f} "
                              f"rel_changes(last {patience})={[round(r, 4) for r in rel_changes]}")
                    if all(rc < tol for rc in rel_changes):
                        stop_info['stopped_epoch'] = epoch
                        stop_info['converged'] = True
                        stop_now = True
                if not stop_now and epoch + 5 >= max_epoch:
                    stop_info['stopped_epoch'] = epoch
                    stop_info['converged'] = False

        epoch += 1
        if stop_now:
            break

    if early_stop and stop_info['stopped_epoch'] is None:
        stop_info['stopped_epoch'] = epoch - 1
        stop_info['converged'] = False

    nrm_base_state = {
        node: {k: v.detach().cpu().clone() for k, v in base_mdl.dcm[node].state_dict().items()}
        for node in pretrain_grp.dag.keys()
    }
    return nrm_base_state, stop_info


# =============================================================================
# Phase 2: per-candidate model construction. Only the normal side changes
# from _create_joint_models (dcm_rca.py) -- anomalous side is identical.
# =============================================================================

def _create_joint_models_normal_once(cfg, Model, causal_graph, node, nrm_base_state):
    device = torch.device(f"cuda:{cfg.exp.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Fresh net per node, same as _create_joint_models -- including `node`
    # itself for now; its fresh net is overwritten below for every Z != node,
    # and node's own fresh net is kept (that's f_Y).
    nrm_mdl = Model(causal_graph, cur_node=node, latent_dim=cfg.mdl.lat_dim,
                     hidden_dim=cfg.mdl.hid_dim, device=device)

    for z in nrm_mdl.dag.keys():
        if z == node:
            continue
        if z not in nrm_base_state:
            raise KeyError(
                f"No pretrained normal-base weights for node {z!r} -- the "
                f"pretrain graph and this candidate's graph disagree on node set"
            )
        nrm_mdl.dcm[z].load_state_dict(nrm_base_state[z])
        for p in nrm_mdl.dcm[z].parameters():
            p.requires_grad = False

    # DCM_RCA.__init__ already asserts cfg.trn.upd == 'shared' unconditionally
    # for this whole reduced codebase, so (unlike _create_joint_models, which
    # still carries the old if/else for parity with train.py) there's no
    # unshared branch to consider here.
    shared_net = nrm_mdl.get_shared_net()  # nrm_mdl.dcm[node] -- fresh, trainable: f_Y

    # Anomalous side: UNCHANGED from the original path (train_main.py /
    # _create_joint_models). Every Z != node is a fresh net trained on
    # anomalous data only; dcm[node] is shared_net, tied to nrm_mdl's.
    anm_mdl = Model(causal_graph, cur_node=node, latent_dim=cfg.mdl.lat_dim,
                     hidden_dim=cfg.mdl.hid_dim, device=device, shared_net=shared_net)

    return nrm_mdl, anm_mdl


# =============================================================================
# Training loop -- identical logic to train_main._train(). Kept as an
# independent copy (see module docstring) rather than imported, so this path
# can diverge/be edited without touching the original. No change is needed to
# respect the Phase-1 freeze: the parameter-collection loop below already
# filters on p.requires_grad, so frozen nrm_base params are excluded from the
# optimizer automatically.
# =============================================================================

def _train_normal_once(cfg, nrm_dat, anm_dat, nrm_mdl, anm_mdl, prc=None):
    """Unified training function -- identical to train_main._train() except for
    the early-stopping block (guarded by cfg.trn.early_stop, default False --
    when off, behavior/loop-bound is byte-identical to before: bounded by
    cfg.trn.num_epc, no break, returns the same (losses, tvd) as always plus a
    stop_info dict). See that function's docstring/comments for the base
    training-loop rationale (BUG(A1-15), BUG(A1-10) fixes carried over
    unchanged).

    Early stopping (cfg.trn.early_stop=True): monitors tvd_diff -- the
    wasserstein-add score that actually drives the ranking (rank_metrics
    [cfg.exp.metric]['add'], the same quantity _compute_scores/
    _compute_scores_at_epoch compute post-hoc), not the NLL training loss --
    at each 5-epoch eval point. Stops once cfg.trn.early_stop_patience
    consecutive 5-epoch transitions each have relative change below
    cfg.trn.early_stop_tol, or once cfg.trn.max_epc is reached (whichever
    first); a node that never satisfies the criterion trains the full
    max_epc and is returned with converged=False rather than being dropped.
    """
    device = torch.device(f"cuda:{cfg.exp.gpu_id}" if torch.cuda.is_available() else "cpu")

    early_stop = bool(getattr(cfg.trn, 'early_stop', False))
    max_epoch = cfg.trn.max_epc if early_stop else cfg.trn.num_epc
    tol = cfg.trn.early_stop_tol if early_stop else None
    patience = cfg.trn.early_stop_patience if early_stop else None
    primary_metric = cfg.exp.metric
    tvd_diff_history = []  # only used when early_stop=True
    stop_info = {'stopped_epoch': None, 'converged': None}

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

                eval_loss_nrm = evaluate_cont_loss(nrm_samples, nrm_dat, prc=prc)
                eval_loss_anm = evaluate_cont_loss(anm_samples, anm_dat, prc=prc)

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
                                  f"rel_changes(last {patience})={[round(r,4) for r in rel_changes]}")
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
        # having set stop_info -- e.g. max_epoch itself isn't a multiple of 5,
        # so the "about to hit the ceiling" check never fired. Record the last
        # eval point actually reached, unconverged.
        stop_info['stopped_epoch'] = tvd['epochs'][-1] if tvd['epochs'] else 0
        stop_info['converged'] = False

    return losses, tvd, stop_info


# =============================================================================
# Per-node worker -- same contract as train_main._process_node (returns,
# writes nothing; module-level + picklable for spawn), plus the extra
# nrm_base_state argument threaded in from DCM_RCA.run()'s Phase 1.
# =============================================================================

def _process_node_normal_once(node, cfg_dict, causal_graph, observe_u, nrm_df, anm_df, diff, nrm_base_state):
    cfg = OmegaConf.create(cfg_dict)

    logging.basicConfig(level=logging.INFO)
    process_lgr = logging.getLogger(__name__)

    try:
        causal_graph.top_sort()
        print('Causal graph: ', causal_graph.type)
        if causal_graph.type == 'pag':
            # Unreachable in practice: DCM_RCA.run() calls _pretrain_normal_base
            # before dispatching any candidate, and that raises on PAG graphs
            # first. Left in place, unchanged from train_main._process_node, for
            # defensive parity in case this function is ever called directly.
            print('Pag graph')
            if node in causal_graph.possible_parents.keys():
                print(f'Node {node} has possible parents: {causal_graph.possible_parents[node]}')
                pp = causal_graph.possible_parents[node]
                print('Previous parents: ', causal_graph.dag[node])
                causal_graph.dag[node].extend(pp)
                print('New parents: ', causal_graph.dag[node])
                print('Doing top sort')
                causal_graph.top_sort()

        nrm_grp = copy.deepcopy(causal_graph)

        if cfg.exp.snk_grp == True:
            nrm_grp = nrm_grp.to_sink_graph(sink_node=node)
            lgr.info(f'Using sink graph (after top sort) for node {node}: {nrm_grp.dag}')

        print(f'Testing node: {node}')

        print('--------------------------------Baro test for dataset before normalization--------------------------------')
        print('nrm.columns: ', nrm_df.columns)
        print('anm.columns: ', anm_df.columns)
        nrm_med = nrm_df[node].median()
        anm_med = anm_df[node].median()
        print(f'{node} normal median: {round(nrm_med, 3)}')
        print(f'{node} anomalous median: {round(anm_med, 3)}')

        if cfg.exp.data_type == 'cont' and cfg.exp.normalize == True:
            if node in diff:
                print(f'Node {node} is a constant and will not be tested')
                return None

            if len(diff) > 0:
                print(f'Columns {diff} are constants and will be removed')
                for nd in diff:
                    nrm_grp.dag.pop(nd, None)
                    if getattr(nrm_grp, "dim_dict", None) is not None:
                        nrm_grp.dim_dict.pop(nd, None)
                for child, parents in nrm_grp.dag.items():
                    nrm_grp.dag[child] = [p for p in parents if p not in diff]
                # Shrink (don't drop-whole-group-on-any-hit) -- BUG(A1-??) fix, see
                # train_main.py's identical fix for the full rationale/verification.
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

        # Only line that differs from train_main._process_node: reuse the
        # Phase-1 pretrained+frozen normal mechanisms for every Z != node.
        nrm_mdl, anm_mdl = _create_joint_models_normal_once(cfg, cur_model, nrm_grp, node, nrm_base_state)

        losses, tvd, stop_info = _train_normal_once(cfg, nrm_dat, anm_dat, nrm_mdl, anm_mdl, prc=node)
        process_lgr.info(f'Training completed for node {node}. Total iterations: {len(losses["nrm"])}. '
                          f'stop_info={stop_info}')

        if getattr(cfg.trn, 'early_stop', False):
            # Use the score recorded AT the stopping point -- "the loss
            # stabilized, return with that loss" -- rather than
            # _compute_scores' trailing-last-10-epoch average, which would
            # blend in the (likely still-moving) early eval points whenever a
            # node stops in fewer than 10 eval-worth of epochs.
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
