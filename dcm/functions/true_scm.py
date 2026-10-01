import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
import os
import yaml
from pathlib import Path
from functions.utils import get_conf_interval, get_prob_keys
from itertools import product

import torch
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter

from tqdm import tqdm

import copy


import numpy as np
import torch
from torch.utils.data import Dataset
import sys



import torch
import itertools
import logging
lgr = logging.getLogger(__name__)

def load_graph_config(graph_name, grp_yml):
    """Load graph configuration from graphs.yaml file."""
    
    # Get the path to graphs.yaml
    graphs_yaml_path = grp_yml
    

    
    with open(graphs_yaml_path, 'r') as f:
        graphs_config = yaml.load(f, Loader=yaml.FullLoader)
    
    
    if graph_name not in graphs_config['graphs']:
        raise ValueError(f"Graph '{graph_name}' not found in graphs.yaml")
    
    graph_config = graphs_config['graphs'][graph_name]
    
    # Simplified structure - direct access (no variants)
    # Extract components directly from graph config
    dag = graph_config['dag']
    confounders = graph_config.get('confounders', {})
    dim_dict = graph_config['dim_dict']
    reuse = graph_config.get('reuse', [])
    iv_vars = graph_config.get('iv_vars', [])
    type = graph_config.get('type', 'dag')
    possible_parents = graph_config.get('possible_parents', {})
    return dag, confounders, dim_dict, reuse, iv_vars, type, possible_parents


class CausalGraph(nn.Module):
    def __init__(self, name, grp_yml):
        super().__init__()

        # Load graph configuration from YAML
        dag, confounders, dim_dict, reuse, iv_vars, type, possible_parents = load_graph_config(name, grp_yml)

        
        self.graph = name
        self.dag = dag
        self.confounders = confounders
        self.dim_dict = dim_dict
        self.reuse = reuse
        self.iv_vars = iv_vars
        self.type = type
        self.possible_parents = possible_parents
    def to_complete_graph(self):
        """
        Transform the graph into a complete graph.
        
        For each node Vi, adds incoming edges from all nodes Vj where j < i.
        Nodes are used in the order they appear in self.dag.
        
        For each edge in the complete graph, adds a confounder to self.confounders.
        """
        # Get all nodes from the DAG (maintains insertion order in Python 3.7+)
        all_nodes = list(self.dag.keys())
        
        # Create complete DAG: each node gets all previous nodes as parents
        complete_dag = {}
        for i, node in enumerate(all_nodes):
            # All nodes before this one become parents
            complete_dag[node] = all_nodes[:i]
        
        # Create confounders for each edge in the complete graph
        complete_confounders = {}
        confounder_counter = 0

        for child_node in all_nodes:
            for parent_node in complete_dag[child_node]:
                # Create a confounder for this edge
                confounder_name = f'U{confounder_counter}'
                complete_confounders[confounder_name] = [parent_node, child_node]
                confounder_counter += 1

        # Update self.dag and self.confounders
        self.dag = complete_dag
        self.confounders = complete_confounders

        print('Using complete graph')
        
        return self


    def to_sink_confounded_graph(self, sink_node):
        """"Sink Confounded Star" G_Y for candidate Y=sink_node, V_-Y = V\\{Y}:
          directed:    Vi -> Y        for every Vi in V_-Y   (same as to_sink_graph)
          bidirected:  Vi <-> Vj      for every pair Vi != Vj in V_-Y
          absent:      no edges among V_-Y other than the bidirected ones,
                       no bidirected edge touching Y, no edges out of Y
        A complete bidirected clique over V_-Y is exactly what one shared latent
        confounding all of V_-Y looks like once marginalized out -- so this is a
        single (|V_-Y|)-way confounder group, not |V_-Y| choose 2 separate pairs
        (see dcm/functions/models.py's N-way confounder support)."""
        all_nodes = list(self.dag.keys())
        others = [node for node in all_nodes if node != sink_node]

        sink_dag = {node: [] for node in others}
        sink_dag[sink_node] = list(others)

        confounders = {"U_sink": list(others)} if len(others) >= 2 else {}

        self.dag = sink_dag
        self.confounders = confounders
        self.top_sort()

        return self


    def to_sink_graph(self, sink_node):
      
      
        all_nodes = list(self.dag.keys())
        
        
        sink_dag = {node:[] for node in all_nodes if node != sink_node}
        sink_dag[sink_node] = list(sink_dag.keys())


        complete_confounders = {}
        

        self.dag = sink_dag
        self.confounders = complete_confounders
        self.top_sort()

        return self


    def top_sort(self):
        """
        Perform topological sort of the graph.
        """
        # Get all nodes from the DAG (maintains insertion order in Python 3.7+)
        all_nodes = list(self.dag.keys())
        
        
        top_dag = {}

        cur_dag= copy.deepcopy(self.dag)
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
        
        top_dag = {node:self.dag[node] for node in top_ord}
        print('Topological order: ', top_ord)
        self.dag = top_dag
        return True








class ObservedDataset(Dataset):
    """Minimal dataset class for causal inference with DAG structure."""
    
    def __init__(self, samples=None, graph=None, cnd_pmf=None,  alp=0.05, save_loc='./data/bowbackdoor.csv', num_smp= 1e9, observe_U=False):
        self.graph = graph
        self.observe_U = observe_U
        self.data = {}
        self.dag = graph.dag 
        
        # Initialize dim_dict from graph or as empty dict
        if graph.dim_dict is None:
            self.dim_dict = {}
        else:
            self.dim_dict = graph.dim_dict.copy()

        # Load data from samples or CSV
        if samples is not None:
            for node in self.dag.keys():
                self.data[node] = samples[node]
        else:
            data = pd.read_csv(save_loc)

            if num_smp != 1e9:
                data = data.sample(num_smp).reset_index(drop=True)

            for node in self.dag.keys():
                self.data[node] = np.array(data[node])


            print(f'Loaded {len(data)} samples of {data.columns.tolist()}')



        # Data is always continuous now -- any discrete variable is expected to be
        # pre-processed (small noise added) into continuous form before it gets here.
        # evaluate.py reads .discrete via getattr(..., "discrete", []), so this still
        # needs to exist; it's just never populated.
        self.discrete = []

        # Convert to tensors
        for node in self.dag.keys():
            self.data[node] = torch.tensor(self.data[node].to_numpy(dtype=np.float32)).unsqueeze(1)

        self.n_samples = samples.shape[0]

 
        print('Data loaded successfully')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Support both integer indexing and variable name access."""
        if isinstance(idx, str):
            # Access by variable name: return all data for that variable
            return self.data[idx]
        else:
            # Access by integer index: return sample at that index
            return {node: self.data[node][idx] for node in self.dag.keys()}
    

        # 


