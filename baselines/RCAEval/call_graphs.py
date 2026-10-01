"""Real microservice call graphs for Sock Shop / Online Boutique, and the
"edge-inverted call graph" transform circa/rcg need (fault-propagation
direction: callee -> caller, matching RCAEval's own convention at
baselines/RCAEval/RCAEval/e2e/RCAWithMissingStructuralKnowledgeCode/
data_simulators/sock_shop.py: `causal_graph = call_graph.reverse()`).

Sock Shop edges are copied verbatim from that same vendored file's
create_sock_shop_graph() (already in this repo's copy of RCAEval, under
its own Apache-2.0 attribution). Online Boutique has no equivalent graph
anywhere in the vendored package -- there is no RCAEval source to copy,
so its edges are the standard, publicly documented Google microservices-demo
service topology (github.com/GoogleCloudPlatform/microservices-demo),
restricted to the 11 services actually present in this project's Online
Boutique node set (confirmed via load_case()).
"""
import networkx as nx


def sockshop_call_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([
        ("front-end", "catalogue"), ("front-end", "user"), ("front-end", "carts"), ("front-end", "orders"),
        ("orders", "user"), ("orders", "carts"), ("orders", "payment"), ("orders", "orders-db"),
        ("orders", "queue-master"), ("queue-master", "rabbitmq"),
        ("shipping", "rabbitmq"),
        ("user", "user-db"), ("user", "session-db"),
        ("catalogue", "catalogue-db"),
        ("carts", "carts-db"),
    ])
    return G


def onlineboutique_call_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([
        ("frontend", "adservice"), ("frontend", "cartservice"), ("frontend", "checkoutservice"),
        ("frontend", "currencyservice"), ("frontend", "productcatalogservice"),
        ("frontend", "recommendationservice"), ("frontend", "shippingservice"),
        ("checkoutservice", "cartservice"), ("checkoutservice", "currencyservice"),
        ("checkoutservice", "emailservice"), ("checkoutservice", "paymentservice"),
        ("checkoutservice", "productcatalogservice"), ("checkoutservice", "shippingservice"),
        ("recommendationservice", "productcatalogservice"),
        ("cartservice", "redis"),
    ])
    return G


CALL_GRAPHS = {"sockshop": sockshop_call_graph, "onlineboutique": onlineboutique_call_graph}


def edge_inverted_causal_graph(dataset: str) -> nx.DiGraph:
    """The graph circa/rcg receive: caller->callee edges reversed to
    callee->caller (a downstream failure's symptom propagates up to its
    callers), i.e. `graph.parents(X)` = X's callees -- the nodes whose
    failure would explain an anomaly seen at X."""
    return CALL_GRAPHS[dataset]().reverse()


def causal_graph_to_circa_adj(causal_graph: nx.DiGraph, column_order):
    """Encode `causal_graph` as the np.ndarray `adj` format RCAEval's
    rht.py expects, over `column_order` (must match the DataFrame columns
    rht() will build `nodes` from, e.g. data.columns minus "time", in order).

    rht.py's own adj->graph conversion (read directly from source):
        adj[a,b]==1 and adj[b,a]==-1  ->  graph.add_edge(nodes[b], nodes[a])
        ... then graph = graph.reverse()
    Net effect: adj[a,b]=1, adj[b,a]=-1 ends up as a FINAL edge nodes[a] ->
    nodes[b] after rht()'s own reversal. So for every desired final edge
    X -> Y (X causally explains Y, per causal_graph), set
    adj[idx(X), idx(Y)] = 1 and adj[idx(Y), idx(X)] = -1.
    """
    import numpy as np
    idx = {name: i for i, name in enumerate(column_order)}
    n = len(column_order)
    adj = np.zeros((n, n), dtype=int)
    for x, y in causal_graph.edges:
        if x in idx and y in idx:
            adj[idx[x], idx[y]] = 1
            adj[idx[y], idx[x]] = -1
    return adj
