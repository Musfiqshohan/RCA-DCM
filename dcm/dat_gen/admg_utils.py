import numpy as np
import pyagrum as gum
from typing import Dict, List, Tuple, Any, Optional
import itertools
import yaml
from pathlib import Path

import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless systems
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import copy

# Font fallback for headless/minimal systems (DejaVu Sans often missing)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'FreeSans', 'DejaVu Sans', 'Arial', 'sans-serif']


def hrd_inv(arg_bn, inv_set, inv_dst='marginal'):
    """
    Hard-intervene: cut all incoming arcs to each intervened var
    and set its CPT to a one-hot at the chosen state.
    """

    bn = gum.BayesNet(arg_bn)
    mrg_dst = {}    # marginal distribution of the variables without any intervention
    for var_name in inv_set:

        if inv_dst == 'uniform':
            k = bn.variable(bn.idFromName(var_name)).domainSize()
            mrg_dst[var_name] = [1/k for i in range(k)]
        elif inv_dst == 'marginal':
            child_id = bn.idFromName(var_name)
            # get marginal distribution of the variable
            ie = gum.LazyPropagation(bn)
            ie.addJointTarget(var_name)
            ie.makeInference()
            p_var = ie.evidenceJointImpact(var_name, [])
            mrg_dst[var_name] = [p_var[i] for i in range(len(p_var))]


    for var_name in inv_set:
        child_id = bn.idFromName(var_name)

        # remove all incoming arcs to the child
        for parent_id in list(bn.parents(child_id)):
            bn.eraseArc(parent_id, child_id)   # both are NodeIds

        # setting the marginal dist
        bn.cpt(child_id).fillWith(mrg_dst[var_name])

    return bn


def cmp_dst(bn1, bn2, inv_set, out):
    trn_bn= hrd_inv(bn1, inv_set, inv_dst='uniform')
    anm_bn= hrd_inv(bn2, inv_set, inv_dst='uniform')

    print(f'Comparing P({out}|do{inv_set})')

    ie1 = gum.LazyPropagation(trn_bn)
    ie2 = gum.LazyPropagation(anm_bn)
    ie1.addJointTarget(set(trn_bn.names()))
    ie2.addJointTarget(set(anm_bn.names()))
    ie1.makeInference()
    ie2.makeInference()
    p1 = ie1.evidenceJointImpact([out], [])
    p2 = ie2.evidenceJointImpact([out], [])

    print(f'P({out}|do{inv_set} in train domain: {p1}')
    print(f'P({out}|do{inv_set} in anomaly domain: {p2}')
    print('dist', np.sum(np.abs(p1.toarray() - p2.toarray())))
    return np.sum(np.abs(p1.toarray() - p2.toarray()))


def create_bayesian_network(variables: Dict[str, Dict[str, Any]], 
                           arcs: List[Tuple[str, str]], 
                           network_name: str = "BN") -> gum.BayesNet:
    """Create a Bayesian network from variable and arc specifications."""
    bn = gum.BayesNet(network_name)
    
    # Add variables
    for var_name, var_props in variables.items():
        bn.add(gum.LabelizedVariable(var_name, var_props['description'], var_props['domain_size']))
    
    # Add arcs
    for parent, child in arcs:
        if parent not in variables or child not in variables:
            raise ValueError(f"Variable not defined: {parent} -> {child}")
        bn.addArc(parent, child)
    
    return bn


def generate_imbalanced_cpt(variable: str, parents: List[str], variables: Dict[str, Dict[str, Any]], 
                           seed: int = None) -> Dict[Tuple, List[float]]:
    """
    Generate imbalanced CPT that ensures each parent has meaningful influence.
    
    Args:
        variable: Name of the variable
        parents: List of parent variable names
        variables: Dictionary of all variables and their properties
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping parent combinations to probability distributions
    """
    if seed is not None:
        np.random.seed(seed)
    
    var_domain = variables[variable]['domain_size']
    cpt = {}
    
    if not parents:
        # Root variable - generate imbalanced distribution
        probs = np.random.dirichlet(np.ones(var_domain) * 0.5)  # Low concentration for imbalance
        cpt[()] = probs.tolist()
    else:
        # Generate parent combinations
        parent_domains = [variables[parent]['domain_size'] for parent in parents]
        parent_combinations = list(np.ndindex(*parent_domains))
        
        for combo in parent_combinations:
            # Create imbalanced distribution that varies with parent values
            # Use parent values to influence the distribution
            base_alpha = 0.3  # Low base concentration for imbalance
            
            # Add parent influence - each parent contributes to imbalance
            alpha = np.ones(var_domain) * base_alpha
            for i, parent_val in enumerate(combo):
                # Each parent value shifts the distribution
                influence = 0.5 + 0.3 * parent_val  # Varies from 0.5 to 0.8
                alpha[i % var_domain] += influence
            
            # Generate imbalanced probabilities
            probs = np.random.dirichlet(alpha)
            cpt[combo] = probs.tolist()
    
    return cpt


def set_random_cpts(bn: gum.BayesNet, variables: Dict[str, Dict[str, Any]], 
                   arcs: List[Tuple[str, str]], seed: int = None) -> gum.BayesNet:
    """
    Set randomly generated imbalanced CPTs for all variables in the network.
    
    Args:
        bn: The Bayesian network
        variables: Dictionary of all variables and their properties
        arcs: List of arcs (parent, child)
        seed: Random seed for reproducibility
    
    Returns:
        The Bayesian network with CPTs set
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Build parent relationships
    parents = {}
    for parent, child in arcs:
        if child not in parents:
            parents[child] = []
        parents[child].append(parent)
    
    # Set CPTs for each variable
    for var_name in variables.keys():
        var_parents = parents.get(var_name, [])
        cpt = generate_imbalanced_cpt(var_name, var_parents, variables, seed)
        
        if not var_parents:
            # Root variable
            bn.cpt(var_name).fillWith(cpt[()])
        else:
            # Conditional variable
            for combo, probs in cpt.items():
                parent_dict = {var_parents[i]: combo[i] for i in range(len(combo))}
                bn.cpt(var_name)[parent_dict] = probs
    
    return bn


def create_graph(graph_name: str, seed: int = 42, return_grp: bool = False):
    """
    Create a Bayesian network from graphs.yaml file.
    
    Args:
        graph_name: Name of the graph to create (e.g., 'bow', 'dbl_bow', 'iv', 'chain', etc.)
        seed: Random seed for CPT generation
        return_grp: If True, returns graph name as 4th element: (bn, cnf_chd, var_cnf, graph_name)
                   If False, returns: (bn, cnf_chd, var_cnf)
    
    Returns:
        If return_grp=False: (bn, cnf_chd, var_cnf)
        If return_grp=True: (bn, cnf_chd, var_cnf, graph_name)
        
    Examples:
        >>> bn, cnf_chd, var_cnf = create_graph('bow', seed=42)
        >>> bn, cnf_chd, var_cnf, grp = create_graph('dbl_bow', seed=42, return_grp=True)
    """
    # Get the path to graphs.yaml
    current_file = Path(__file__).resolve()
    graphs_yaml_path = current_file.parent / 'graphs.yaml'
    
    if not graphs_yaml_path.exists():
        raise FileNotFoundError(f"Graphs YAML file not found at {graphs_yaml_path}")
    
    with open(graphs_yaml_path, 'r') as f:
        graphs_config = yaml.safe_load(f)
    
    if graph_name not in graphs_config['graphs']:
        raise ValueError(f"Graph '{graph_name}' not found in graphs.yaml")
    
    graph_config = graphs_config['graphs'][graph_name]
    
    # Simplified structure - direct access (no variants)
    dag = graph_config['dag']
    confounders = graph_config.get('confounders', {})
    dim_dict = graph_config['dim_dict']
    
    # Convert dag format to arcs format for pyAgrum
    arcs = []
    for child, parents in dag.items():
        for parent in parents:
            arcs.append((parent, child))
    
    # Add arcs from confounders to their children
    for conf_name, children in confounders.items():
        for child in children:
            arcs.append((conf_name, child))
    
    # Build variables dict for pyAgrum
    variables = {}
    # Add observed variables
    for var_name, domain_size in dim_dict.items():
        variables[var_name] = {'description': var_name, 'domain_size': domain_size}
    # Add confounders
    for conf_name in confounders.keys():
        if conf_name not in variables:
            variables[conf_name] = {'description': f'unobserved confounder {conf_name}', 'domain_size': 2}
    
    # Create var_cnf (inverse of confounders)
    var_cnf = {}
    for var_name in dim_dict.keys():
        var_cnf[var_name] = []
    for conf_name, children in confounders.items():
        for child in children:
            if child in var_cnf:
                var_cnf[child].append(conf_name)
    
    cnf_chd = confounders
    
    # Create network structure
    bn = create_bayesian_network(variables, arcs, graph_name.upper())
    
    # Set random imbalanced CPTs
    bn = set_random_cpts(bn, variables, arcs, seed)
    
    if return_grp:
        return bn, cnf_chd, var_cnf, graph_name
    else:
        return bn, cnf_chd, var_cnf



# The following functions are from https://github.com/kenneth-lee-ch/id4ip/blob/main/findMACS.py
def random_bn_with_latents(n: int, arc_ratio: float, max_latents: int = 2, 
                          name: str = "RandomBNWithLatents", seed: int = None, 
                          domain_size: int = 2, save_to_yaml: bool = True):
    """
    Create a random Bayesian network with latent confounders using gum.randomBN.
    
    Args:
        n: Number of observed variables
        arc_ratio: Ratio of arcs to add (0.0 to 1.0)
        max_latents: Maximum number of latent confounders to add
        name: Name for the Bayesian network
        seed: Random seed for reproducibility
        domain_size: Domain size for all variables
        save_to_yaml: If True, save the graph to graphs.yaml in the same format
    
    Returns:
        tuple: (bn, cnf_chd, var_cnf, name) where:
            - bn: The created Bayesian network with latent confounders
            - cnf_chd: Dictionary mapping latent names to their children
            - var_cnf: Dictionary mapping variable names to their latent confounders
            - name: The name of the graph
    """


    name = name+f'n{n}'
    rng = np.random.default_rng(seed)
    
    # Generate graph names for observed variables
    graphnames = [f"V{i}" for i in range(n)]
    
    # Create random BN using pyAgrum's built-in function
    bn = gum.randomBN(n=n, names=graphnames, ratio_arc=arc_ratio, domain_size=domain_size)
    
    # Add latent confounders
    cnf_chd = {}
    var_cnf = {}
    
    # Initialize var_cnf for all observed variables
    for var_name in graphnames:
        var_cnf[var_name] = []
    
    if max_latents > 0:
        # Get all variable names
        var_names = list(bn.names())
        
        # Find all possible variable pairs
        all_pairs = []
        for j in range(len(var_names)):
            for k in range(j + 1, len(var_names)):
                all_pairs.append((var_names[j], var_names[k]))
        
        # Limit number of latents to available pairs
        num_latents = min(max_latents, len(all_pairs))
        
        # Randomly select a subset of pairs
        if num_latents > 0:
            selected_pairs = rng.choice(len(all_pairs), size=num_latents, replace=False)
            
            # Assign latents sequentially to selected pairs
            for i, pair_idx in enumerate(selected_pairs):
                latent_name = f"U{i}"
                var1, var2 = all_pairs[pair_idx]
                
                # Add latent variable
                bn.add(gum.LabelizedVariable(latent_name, f"latent_{i}", domain_size))
                latent_id = bn.idFromName(latent_name)
                
                # Add arcs from latent to both variables
                var1_id = bn.idFromName(var1)
                var2_id = bn.idFromName(var2)
                bn.addArc(latent_id, var1_id)
                bn.addArc(latent_id, var2_id)
                
                # Update tracking dictionaries
                cnf_chd[latent_name] = [var1, var2]
                var_cnf[var1].append(latent_name)
                var_cnf[var2].append(latent_name)
    
    # Generate random CPTs for all variables (including latents)
    for var_name in bn.names():
        gum.initRandom(seed)
        bn.generateCPT(var_name)
    
    # Save graph to YAML in the same format as graphs.yaml
    if save_to_yaml:
        graphs_yaml_path = Path(__file__).parent / 'graphs.yaml'

        observed_vars = [bn.variable(nid).name() for nid in bn.topologicalOrder()]
        observed_vars = [var for var in observed_vars if var in graphnames]
        
        dag = {v: [bn.variable(pid).name() for pid in bn.parents(bn.idFromName(v)) 
                   if bn.variable(pid).name() in observed_vars] for v in observed_vars}
        dim_dict = {v: bn.variable(bn.idFromName(v)).domainSize() for v in observed_vars}
        
        graphs_config = {}
        if graphs_yaml_path.exists():
            with open(graphs_yaml_path, 'r') as f:
                graphs_config = yaml.safe_load(f) or {}
        graphs_config.setdefault('graphs', {})[name] = {
            'dag': dag,
            'confounders': cnf_chd or {},
            'dim_dict': dim_dict
        }
        
        with open(graphs_yaml_path, 'w') as f:
            yaml.dump(graphs_config, f, default_flow_style=False, sort_keys=False)
    
    return bn, cnf_chd, var_cnf, name




def get_ccom(confTochild, latent_conf, sub_graph, visited, cur_node):
    """
        Find the C-component of "cur_node" based on the knowledge of how latents were added previously to a graph

        Arguments:
            confTochild (dict): a dictionary that map each confounder name to all of its children as a string to list mapping. 
            latent_conf (dict): a dictionary that maps each variable name to all of its latent confounder as a string to list mapping
            subgraph (dict.value): a list of all variable names in the bn
            visited (list): a list to keep track of variables that have not been visited given the list of variable names of the graph
            cur_node (str): a string that denotes a variable name in the graph and find the C-component that contains that variable
        Return:
            A list of the variable that form a c-component that contains "cur_node"
    """

    # we are adding each c-component node that we are visiting
    visited += [cur_node]
    nbrs=[]
    # look at for current node to see if has any latent confounder. 
    # latent_conf is a dictionary with {variable : its latent confounder}
    for conf in latent_conf[cur_node]:
        # for each latent confounder :conf
        # we get the children of that confounder and append that to a list
        nbrs+= confTochild[conf]

    # loop through all the children of the latent confounder of the "current node"
    for nbr in nbrs:
        if (nbr in sub_graph) and (nbr not in visited):
            get_ccom(confTochild, latent_conf, sub_graph, visited, nbr)  #visited array is being updated since its being passed by reference

    # we send back the updated visited array after each recursion call
    return visited

def getMACS(bn_with_no_S, ac_component, confTochild, latent_conf):
    """
        find-MACS-on-set for a singleton target Y

        Arg:
        bn_wth_no_S: a bayes net object that does not have S
        ac_component (list): a list to represent AC-component
        c_components: all c_components of G
        Return:
            graph: a outcome-rooted C-tree    
    """
    # copy the graph so that we don't change the original object
    bn = gum.BayesNet(bn_with_no_S)
    # # get the variable names as a list of strings from the graph
    graph_var_names = list(bn.names())
    graph_ids = bn.ids(graph_var_names)
    graph_dict = dict(zip(graph_ids, graph_var_names))
    ancestors_of_target_set = []
    # Get all ancestors with respect to ac_component
    for v in ac_component:
        AnY_id = list(bn.ancestors(v))
        ancestors_of_target_set=  ancestors_of_target_set + AnY_id
    # use set to get the unique members
    all_ancestors_names_wrt_ac_component = [graph_dict[i]for i in ancestors_of_target_set]
    # add back the target sets since the ancestors are not inclusive in pyargum
    all_ancestors_names_wrt_ac_component = all_ancestors_names_wrt_ac_component + ac_component
    variables_to_remove = [i for i in graph_var_names if i not in all_ancestors_names_wrt_ac_component]
    if variables_to_remove:
        # if the list is non-empty, we erase the node from the bn one by one
        for j in variables_to_remove:
            bn.erase(j)
        # recurse on the resulting bn and 
        return getMACS(bn, ac_component, confTochild, latent_conf)
    # get the C-component
    all_c_components = []
    for v in ac_component:
        c_comp_v = get_ccom(confTochild, latent_conf, graph_var_names, [] , v)
        all_c_components = all_c_components + c_comp_v
    variables_to_remove_all_com = [i for i in graph_var_names if i not in all_c_components]
    if variables_to_remove_all_com:
        for j in variables_to_remove_all_com:
            bn.erase(j)
        return getMACS(bn, ac_component, confTochild, latent_conf)
    return bn




def get_RC(trn_bn, anm_bn, cnf_chd, var_cnf):
    prc =[]
    rc =[]

    for nid in trn_bn.topologicalOrder():
        
        Y= trn_bn.variable(nid).name()
        if Y in cnf_chd.keys():
            continue
        print('--Testing if', Y, 'is a Root Cause--')
        Ty = getMACS(trn_bn, [Y] , cnf_chd, var_cnf)
        print(f'Maximal Ancestral Confounded Set of {Y}: {Ty.names()}')

        

        if set(prc+rc).intersection(set(Ty.names())):
            print(f'RC{set(prc+rc)} intersects with {Y}s MACS: {Y} is a possible RC')
            prc.append(Y)
        else:
            print(f'No previousRC in {Y}s MACS')
            par_set = []
            for nbr in Ty.names():

                par= trn_bn.parents(nbr)

                par_set+= list(par)


            par_set = [trn_bn.variable(id).name() for id in set(par_set)]
            par_set = [name for name in par_set if name not in cnf_chd.keys() and name not in Ty.names()]
            print(f'Parents of the MACS:{par_set} will be intervened')

            ret= cmp_dst(trn_bn, anm_bn, par_set, Y)
            if ret > 0.01:
                rc.append(Y)
                print(f'added to rc as: {round(ret, 3)}>{0.01}')
            else:
                print(f'not a rc as: {round(ret, 3)}<={0.01}')

        print(f'Till now Root Causes: {rc}, Possible root causes: {prc}\n')

    return  rc, prc

        

        



def plot_dag_with_latents(
    G: np.ndarray,
    obs_names: List[str],
    num_obs_vars: int,
    num_latent_vars: int,
    graph_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the complete causal DAG with latent variables (hierarchical layout).

    Args:
        G: DAG matrix of shape [num_obs_vars, num_latent_vars + num_obs_vars].
           First num_latent_vars columns: latent->observed.
           Remaining columns: observed->observed (dag_obs[i,j]=1 means j is parent of i).
        obs_names: List of observed variable names (order matches rows/cols of G).
        num_obs_vars: Number of observed variables.
        num_latent_vars: Number of latent variables.
        graph_name: Name for the plot title.
        save_path: Path to save the figure (e.g. baselines/LIT/data/dbl_bow/dag_plot.png).
    """
    # Build adjacency matrix for observed variables
    dag_obs = G[:, num_latent_vars:]
    adj_matrix = np.zeros((num_obs_vars, num_obs_vars))
    for child_idx in range(num_obs_vars):
        for parent_idx in range(num_obs_vars):
            if dag_obs[child_idx, parent_idx] == 1:
                adj_matrix[parent_idx, child_idx] = 1

    latent_to_obs = G[:, :num_latent_vars]

    # Build complete graph
    G_complete = nx.DiGraph()
    for lat_idx in range(num_latent_vars):
        G_complete.add_node(f'L{lat_idx+1}', node_type='latent')
    for obs_idx in range(num_obs_vars):
        name = obs_names[obs_idx] if obs_idx < len(obs_names) else f'X{obs_idx+1}'
        G_complete.add_node(name, node_type='observed')

    for lat_idx in range(num_latent_vars):
        for obs_idx in range(num_obs_vars):
            if latent_to_obs[obs_idx, lat_idx] == 1:
                name = obs_names[obs_idx] if obs_idx < len(obs_names) else f'X{obs_idx+1}'
                G_complete.add_edge(f'L{lat_idx+1}', name)

    for parent_idx in range(num_obs_vars):
        for child_idx in range(num_obs_vars):
            if adj_matrix[parent_idx, child_idx] == 1:
                pn = obs_names[parent_idx] if parent_idx < len(obs_names) else f'X{parent_idx+1}'
                cn = obs_names[child_idx] if child_idx < len(obs_names) else f'X{child_idx+1}'
                G_complete.add_edge(pn, cn)

    latent_nodes = [n for n in G_complete.nodes() if G_complete.nodes[n]['node_type'] == 'latent']
    observed_nodes = [n for n in G_complete.nodes() if G_complete.nodes[n]['node_type'] == 'observed']

    # Hierarchical layout
    pos = {}
    latent_y = 3.0
    latent_x_spacing = 2.0
    latent_total_width = (len(latent_nodes) - 1) * latent_x_spacing if len(latent_nodes) > 1 else 0
    latent_start_x = -latent_total_width / 2
    for i, lat_node in enumerate(sorted(latent_nodes)):
        x_pos = latent_start_x + i * latent_x_spacing if len(latent_nodes) > 1 else 0
        pos[lat_node] = (x_pos, latent_y)

    G_obs_only = G_complete.subgraph(observed_nodes).copy()
    try:
        topo_order_obs = list(nx.topological_sort(G_obs_only))
    except Exception:
        topo_order_obs = sorted(observed_nodes)

    layer_dict = {}
    for node in topo_order_obs:
        obs_preds = [p for p in G_complete.predecessors(node) if p in observed_nodes]
        layer_dict[node] = (max([layer_dict.get(p, 0) for p in obs_preds]) + 1) if obs_preds else 0

    obs_layers = {}
    for node, layer in layer_dict.items():
        obs_layers.setdefault(layer, []).append(node)

    x_spacing = 1.5
    for layer, nodes in sorted(obs_layers.items()):
        y_pos = -layer * 2.0
        total_width = (len(nodes) - 1) * x_spacing
        start_x = -total_width / 2
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (start_x + i * x_spacing, y_pos)

    latent_to_obs_edges = [(u, v) for u, v in G_complete.edges() if u in latent_nodes and v in observed_nodes]
    obs_to_obs_edges = [(u, v) for u, v in G_complete.edges() if u in observed_nodes and v in observed_nodes]

    # Draw
    fig, ax = plt.subplots(figsize=(20, 14))
    nx.draw_networkx_nodes(G_complete, pos, nodelist=latent_nodes,
                          node_color='#FF8C00', node_size=2000, alpha=0.95,
                          node_shape='D', linewidths=3, edgecolors='black', ax=ax)

    obs_in_degrees = {n: sum(1 for p in G_complete.predecessors(n) if p in observed_nodes) for n in observed_nodes}
    max_obs_in = max(obs_in_degrees.values()) if obs_in_degrees else 1
    obs_node_colors = [plt.cm.Blues(0.3 + 0.7 * (obs_in_degrees.get(n, 0) / max(max_obs_in, 1))) for n in observed_nodes]

    nx.draw_networkx_nodes(G_complete, pos, nodelist=observed_nodes,
                          node_color=obs_node_colors, node_size=1600, alpha=0.95,
                          linewidths=3, edgecolors='black', ax=ax)

    nx.draw_networkx_edges(G_complete, pos, edgelist=latent_to_obs_edges,
                          edge_color='#FF6B00', arrows=True, arrowsize=30, width=2.5, alpha=0.8,
                          style='dashed', arrowstyle='->', connectionstyle='arc3,rad=0.15',
                          min_source_margin=18, min_target_margin=18, ax=ax)

    nx.draw_networkx_edges(G_complete, pos, edgelist=obs_to_obs_edges,
                          edge_color='#333333', arrows=True, arrowsize=30, width=2.5, alpha=0.9,
                          style='solid', arrowstyle='->', connectionstyle='arc3,rad=0.1',
                          min_source_margin=18, min_target_margin=18, ax=ax)

    nx.draw_networkx_labels(G_complete, pos, font_size=11, font_weight='bold', font_color='black',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='none', alpha=0.85), ax=ax)

    legend_elements = [
        Patch(facecolor='#FF8C00', edgecolor='black', label='Latent Variables'),
        Patch(facecolor='lightblue', edgecolor='black', label='Observed Variables'),
        Line2D([0], [0], color='#FF6B00', linestyle='--', linewidth=2.5, label='Latent → Observed', alpha=0.8),
        Line2D([0], [0], color='#333333', linestyle='-', linewidth=2.5, label='Observed → Observed', alpha=0.9),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max_obs_in))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Observed In-degree\n(incoming from observed vars)', fontsize=10, fontweight='bold')

    ax.set_title(f'Complete Causal DAG with Latent Variables [{graph_name}]\n'
                 f'({len(latent_nodes)} latent, {len(observed_nodes)} observed, '
                 f'{int(latent_to_obs.sum())} L→O edges, {int(adj_matrix.sum())} O→O edges)',
                 fontsize=15, fontweight='bold', pad=25)
    ax.axis('off')

    def _do_tight_layout():
        plt.tight_layout()

    try:
        _do_tight_layout()
    except ValueError as e:
        if 'fallback to the default font was disabled' in str(e):
            import matplotlib.font_manager as fm
            fm._load_fontmanager(try_read_cache=False)
            _do_tight_layout()
        else:
            raise

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'DAG plot saved to: {save_path}')
    else:
        plt.show()






import itertools
import yaml


def has_directed_cycle(nodes, dag):
    children = {n: [] for n in nodes}
    for child, parents in dag.items():
        for p in parents:
            children[p].append(child)

    state = {n: 0 for n in nodes}  # 0=unvisited, 1=visiting, 2=done

    def dfs(u):
        state[u] = 1
        for v in children[u]:
            if state[v] == 1:
                return True
            if state[v] == 0 and dfs(v):
                return True
        state[u] = 2
        return False

    for n in nodes:
        if state[n] == 0 and dfs(n):
            return True
    return False


def canonicalize_confounders(confounder_pairs):
    confounder_pairs = sorted({tuple(sorted(p)) for p in confounder_pairs})
    return {f"U{i+1}": list(pair) for i, pair in enumerate(confounder_pairs)}


def canonical_signature(nodes, dag, confounders):
    dag_sig = tuple((n, tuple(sorted(dag[n]))) for n in sorted(nodes))
    conf_sig = tuple(sorted(tuple(sorted(v)) for v in confounders.values()))
    return dag_sig, conf_sig


def pag_edge_to_admg_options(edge):
    u, v, etype = edge["u"], edge["v"], edge["type"]

    if etype == "->":
        return [("dir", u, v)]
    elif etype == "<->":
        return [("bi", u, v)]
    elif etype == "o->":
        return [("dir", u, v), ("bi", u, v)]
    elif etype == "o-o":
        return [("dir", u, v), ("dir", v, u), ("bi", u, v)]
    else:
        raise ValueError(f"Unsupported PAG edge type: {etype}")



def top_sort(dag):
    """
    Args: dag: dictionary of the form {node: [parents]}
    Returns: topological order of the graph as a dictionary of the form {node: [parents]}
    """

    # Get all nodes from the DAG (maintains insertion order in Python 3.7+)
    all_nodes = list(dag.keys())
    
    
    top_dag = {}

    cur_dag= copy.deepcopy(dag)
    top_ord = []
    while len(cur_dag) > 0:
        cur_dag= dict(sorted(cur_dag.items(), key= lambda x: len(x[1])))
        root_node, parents = list(cur_dag.items())[0]
        if len(parents) == 0:
            top_ord.append(root_node)
            for nd in cur_dag.keys():
                if root_node in cur_dag[nd]:
                    cur_dag[nd].remove(root_node)
            del cur_dag[root_node]
        else:
            print('Error: graph is not a DAG')
            return False
    
    top_dag = {node:dag[node] for node in top_ord}
    print('Topological order: ', top_ord)
    dag = top_dag
    return dag




def enumerate_admgs_from_pag(pag):
    nodes = pag["nodes"]
    edges = pag["edges"]

    option_lists = [pag_edge_to_admg_options(e) for e in edges]

    all_admgs = []
    seen = set()

    for choices in itertools.product(*option_lists):
        dag = {n: [] for n in nodes}
        conf_pairs = []

        for kind, a, b in choices:
            if kind == "dir":
                dag[b].append(a)   # a -> b
            elif kind == "bi":
                conf_pairs.append((a, b))

        for n in nodes:
            dag[n] = sorted(set(dag[n]))

        if has_directed_cycle(nodes, dag):
            continue

        dag = top_sort(dag)

        confounders = canonicalize_confounders(conf_pairs)
        sig = canonical_signature(nodes, dag, confounders)

        if sig in seen:
            continue
        seen.add(sig)

        admg = {
            "type": "admg",
            "dag": dag,
            "confounders": confounders,
            "dim_dict": {n: 1 for n in nodes},
        }
        all_admgs.append(admg)

    out = {
        "source_graph_type": "PAG",
        "nodes": nodes,
        "num_admgs": len(all_admgs),
        "graphs": {f"admg_{i+1:02d}": g for i, g in enumerate(all_admgs)},
    }   
    return out

