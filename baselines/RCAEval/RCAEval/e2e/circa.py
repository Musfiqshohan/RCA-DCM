from RCAEval.graph_construction.pc import pc_default
from RCAEval.graph_heads.rht import rht
from RCAEval.io.time_series import (
    preprocess,
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    select_useful_cols,
)
from RCAEval.e2e import rca



def _causal_graph_to_adj(causal_graph, column_order):
    """adj encoding rht.py's own adj->graph conversion expects (see that
    function's docstring for the derivation): for every edge X->Y in
    causal_graph (X causally explains an anomaly at Y), set
    adj[idx(X),idx(Y)]=1, adj[idx(Y),idx(X)]=-1 so it survives rht()'s
    internal reversal as the same X->Y edge."""
    import numpy as np
    idx = {name: i for i, name in enumerate(column_order)}
    n = len(column_order)
    adj = np.zeros((n, n), dtype=int)
    for x, y in causal_graph.edges:
        if x in idx and y in idx:
            adj[idx[x], idx[y]] = 1
            adj[idx[y], idx[x]] = -1
    return adj


@rca
def circa(data, inject_time=None, dataset=None, graph=None, **kwargs):
    time_col = data["time"]

    data = preprocess(
        data=data,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # add time again
    data["time"] = time_col

    # graph construction
    pc_input = data.drop(columns=["time"])
    node_names = pc_input.columns.to_list()

    if graph is not None:
        # Use the supplied (already edge-inverted) causal graph instead of
        # PC-algorithm discovery -- graph is ground truth here, not
        # estimated from data.
        adj = _causal_graph_to_adj(graph, node_names)
    else:
        adj = pc_default(pc_input, dataset="ob")
    ranks = rht(adj, inject_time, data)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": data.columns.to_list(),
        "ranks": ranks,
    }
