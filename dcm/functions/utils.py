
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
from omegaconf import DictConfig
from tqdm import tqdm
import pickle
import threading
import time
import os

from sklearn.manifold import MDS

def _get_unique_id():
    """Generate unique identifier for multithreading"""
    return f"{int(time.time() * 1000)}"
import os


def get_tvd(p1, p2):
    return 0.5 * torch.sum(torch.abs(p1 - p2))

# working
def differentiable_TVD_distance(model, true_pmf, diff_dict={}, num_samples=100):
    

    device = next(model.parameters()).device

    joint = model.differentiable_joint(batch_size=num_samples)  # shape: [num_states_1, num_states_2, ...]



    # Get the number of states for each variable (now supports non-binary)
    num_variables = len(joint.shape)
    state_space = [range(joint.shape[i]) for i in range(num_variables)]  # Create list of state ranges

    # Construct true joint distribution with same shape as the model's joint
    true_joint = torch.zeros_like(joint)
    for state in itertools.product(*state_space):  # Iterate over all variable state combinations
        true_joint[state] = true_pmf[state]  # Extract true probability from dataset


    tvd = get_tvd(joint, true_joint)


    return tvd




def get_gradient_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.norm(norm_type)  # Compute norm of each parameter's gradient
            total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)  # L2 norm
    return total_norm.item()  # Convert to Python scalar


import random


import numpy as np


def get_conf_interval(arr, alpha=0.05, dim=None):
    """
    Confidence intervals for categorical variables (binary or multinomial).
    Returns empirical probs for dim-1 states, uniform bound eps, and n.
    The last state's probability = 1 - sum(others).
    """
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        raise ValueError("arr must be non-empty")

    # infer observed distribution
    states, counts = np.unique(arr, return_counts=True)
    if dim is None:
        dim = len(states)

    emp_probs = counts / n
    emp_dist_full = {s: float(p) for s, p in zip(states, emp_probs)}


    # ensure we include all dim states (even if unseen)
    for s in range(dim):
        if s not in emp_dist_full:
            emp_dist_full[s] = 0.0

    # keep only first dim-1 states
    kept_states = sorted(list(emp_dist_full.keys()))[:dim-1]
    emp_dist = {s: emp_dist_full[s] for s in kept_states}

    # Hoeffding uniform bound with union bound
    eps = np.sqrt(np.log(2 * dim / alpha) / (2 * n))

    return emp_dist, eps, n



def split_interval(emp_mean, eps, k):
    # Generate k points from emp_mean - eps to emp_mean + eps
    return np.linspace(max(0,emp_mean - eps), min(1, emp_mean + eps), k)



def get_prob_keys(vars, dim_dict=None):
    """Generate probability keys for a list of variables"""
    from itertools import product
    
    # Default to binary if dim_dict not provided
    if dim_dict is None:
        dim_dict = {v: 2 for v in vars}
    
    keys = []
    
    # For each variable, generate keys for all its states
    for i, v in enumerate(vars):
        parents = vars[:i]
        
        if not parents:
            # Root variable - generate keys for all states
            for state in range(dim_dict[v]-1):
                keys.append(f'P({v}={state})')
        else:
            # Child variable - generate keys for all states given parent combinations
            # Generate combinations based on each parent's dimension
            parent_combinations = []
            for p in parents:
                parent_combinations.append([f'{p}={j}' for j in range(dim_dict[p])])
            combs = product(*parent_combinations)
            
            # For each parent combination, generate keys for all states of current variable
            for c in combs:
                for state in range(dim_dict[v]-1):
                    keys.append(f'P({v}={state}|{",".join(c)})')

    return keys



def save_ydox(cfg: DictConfig, ydox, emp_mean, epsilon, dist_dict, eps_dict, o, plot_data):


    
    results_data = {
        'emp_mean': emp_mean,
        'epsilon': epsilon,
        'dist_dict': dist_dict,
        'eps_dict': eps_dict,
        'num_samples': cfg.trn.num_sam,
        'observe_U': cfg.mdl.obs_u,
        'explored_dist': o,
        'ball': cfg.opt.bal_typ,
        'small_eps': cfg.opt.sml_eps,
        'fixed_eps': cfg.opt.fix_eps,
        'tvd_eps': cfg.opt.tvd_eps,
        'lambda_int': cfg.opt.lam_int,
        'lambda_obs': cfg.opt.lam_obs,
        'dual_lr': cfg.opt.dul_lrn,
        'graph': cfg.exp.grp,
        'opt': cfg.exp.opt,
        'plot_data': plot_data
    }

    for key in ydox:
        results_data.update({key: ydox[key]})

        
    # Save data with unique filename to avoid conflicts in multithreading
    os.makedirs(f'{cfg.pth.out_dir}', exist_ok=True)
    filename = f'{cfg.pth.out_dir}/stat_dist_{o}_samples_{cfg.trn.num_sam}_{_get_unique_id()}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"Saved final results to {filename}")



def save_cnt_pck(cfg: DictConfig, cnt_pck, o):
    """Save cnt_pck data to a pickle file"""
    cnt_pck_data = {
        'cnt_pck': cnt_pck,
        'explored_dist': o,
        'num_samples': cfg.trn.num_sam,
        'observe_U': cfg.mdl.obs_u,
        'ball': cfg.opt.bal_typ,
        'graph': cfg.exp.grp,
        'opt': cfg.exp.opt
    }
    
    cnt_pck_file = f'{cfg.pth.out_dir}/cnt_pck_{cfg.trn.num_sam}_observe_U_{cfg.mdl.obs_u}_{_get_unique_id()}.pkl'
    with open(cnt_pck_file, 'wb') as f:
        pickle.dump(cnt_pck_data, f)
    print(f"Saved cnt_pck data to {cnt_pck_file}")


def load_cnt_pck(cfg: DictConfig):
    """Load cnt_pck data from pickle file if it exists"""
    cnt_pck_file = f'{cfg.pth.out_dir}/cnt_pck_{cfg.trn.num_sam}_observe_U_{cfg.mdl.obs_u}.pkl'
    
    if os.path.exists(cnt_pck_file) and cfg.opt.bal_typ == 'tvd':
        with open(cnt_pck_file, 'rb') as f:
            cnt_pck_data = pickle.load(f)
        print(f"Loaded cnt_pck data from {cnt_pck_file}, {len(cnt_pck_data['cnt_pck'])}")
        return cnt_pck_data['cnt_pck'], cnt_pck_data['explored_dist']
    else:
        print(f"No existing cnt_pck file found at {cnt_pck_file}")
        return {'dox=0': [[], []], 'dox=1': [[], []]}, 0




def get_joint_from_conditionals(graph, conditional_dist):
    """
    Compute joint distribution from conditional distributions.
    
    Args:
        conditional_dist (dict): Dictionary of conditional distributions
        
    Returns:
        torch.Tensor: Joint probability tensor
    """
    nodes = list(graph.dag.keys())
    joint_dims = [graph.dim_dict[node] for node in nodes]
    joint = torch.zeros(tuple(joint_dims))
    
    # Generate all possible combinations
    combinations = list(itertools.product(*[range(graph.dim_dict[node]) for node in nodes]))
    
    for comb in combinations:
        prob = 1.0
        
        # Process nodes in order
        for node_idx, node in enumerate(nodes):
            node_val = comb[node_idx]
            
            # Get parent values for this node
            prev_nodes = [nodes[i] for i in range(node_idx)]
            prev_vals = [comb[i] for i in range(node_idx)]
            
            # Create key for conditional probability
            if len(prev_nodes) > 0:
                key = f"P({node}={node_val}|{','.join(f'{p}={v}' for p, v in zip(prev_nodes, prev_vals))})"
            else:
                key = f"P({node}={node_val})"
            
            
            # Get conditional probability
            if key in conditional_dist:
                prob *= conditional_dist[key]

            elif node_val == graph.dim_dict[node] - 1:

                sum_prob=0
                for dim in range(graph.dim_dict[node]-1): # last state is 1 - sum of others
                    exs_key = key.replace(f"P({node}={node_val}", f"P({node}={dim}")
                    if exs_key in conditional_dist:
                        sum_prob += conditional_dist[exs_key]

                prob *= 1 - sum_prob

            else:
                # If key not found, assume uniform distribution
                raise ValueError(f'Key not found: {key}')
        
        
        joint[comb] = prob
        
    return joint



import itertools
import re

def get_prob_bin(cond_dict, target, cond=""):
    """Return P(target | cond) or its complement for binary variables."""
    key = f"P({target}|{cond})" if cond else f"P({target})"
    if key in cond_dict:
        return cond_dict[key]
    var, val = re.findall(r'([A-Za-z]\w*)=(\d)', target)[0]
    alt = f"{var}={1-int(val)}"
    alt_key = f"P({alt}|{cond})" if cond else f"P({alt})"
    if alt_key in cond_dict:
        return 1 - cond_dict[alt_key]
    raise KeyError(f"Missing {key}")

def binary_joint_given_I(conditionals, I_vars=['I']):
    """
    Compute P(Y,X | I1,I2,...) for binary variables from conditionals.
    Output format: {'pYX.i1i2...': prob}  → e.g. 'p10.01'
    """
    # Identify all variable names
    vars_found = sorted({v for k in conditionals for v,_ in re.findall(r'([A-Za-z]\w*)=(\d)', k)})
    if not all(v in vars_found for v in ['Y','X']):
        raise ValueError("Must include Y and X in the conditionals")

    p = {}
    for I_vals in itertools.product([0,1], repeat=len(I_vars)):
        I_cond = ",".join(f"{I_vars[i]}={I_vals[i]}" for i in range(len(I_vars)))
        I_key = "".join(str(i) for i in I_vals)
        for X in [0,1]:
            for Y in [0,1]:
                pX = get_prob_bin(conditionals, f"X={X}", I_cond)
                pY = get_prob_bin(conditionals, f"Y={Y}", f"{I_cond},X={X}")
                p[f"p{Y}{X}.{I_key}"] = pY * pX
    return p


import itertools

def check_iv_monotonicity_constraints_from_p(conditionals, I_vars=['I']):
    """
    Check IV monotonicity constraints for multiple binary instruments.

    - Computes pairwise inequalities between all instrument configurations.
    - Also checks the general constraint:
        max_x Σ_y [max_z P(x,y|z)] ≤ 1
    """
    # Compute all joint P(Y,X|I1,...,Ik)
    p = binary_joint_given_I(conditionals, I_vars=I_vars)
    print(f"\nP(Y,X | {','.join(I_vars)}) joint probabilities:")
    for k, v in p.items():
        print(f"  {k}: {v:.4f}")

    configs = list(itertools.product([0,1], repeat=len(I_vars)))
    results = {}

    # ---- 1️⃣ Pairwise monotonicity inequalities ----
    for i1, i2 in itertools.combinations(configs, 2):
        key1 = "".join(str(x) for x in i1)
        key2 = "".join(str(x) for x in i2)
        tag = f"{key1}-{key2}"

        c1 = p[f'p00.{key1}'] + p[f'p10.{key2}']
        c2 = p[f'p01.{key1}'] + p[f'p11.{key2}']
        c3 = p[f'p10.{key1}'] + p[f'p00.{key2}']
        c4 = p[f'p11.{key1}'] + p[f'p01.{key2}']
        all_ok = all([c1 <= 1, c2 <= 1, c3 <= 1, c4 <= 1])

        results[tag] = {
            'c1': round(c1,4),
            'c2': round(c2,4),
            'c3': round(c3,4),
            'c4': round(c4,4),
            'all_satisfied': all_ok
        }

        # print(f"\nMonotonicity between I={key1} and I={key2}:")
        # for c in ['c1','c2','c3','c4']:
        #     print(f"  {c}: {results[tag][c]} {'✓' if results[tag][c] <= 1 else '✗'}")
        # print(f"  ➤ All satisfied: {all_ok}")

    # ---- 2️⃣ Global inequality: max_x Σ_y [max_z P(x,y|z)] ≤ 1 ----
    cfg_keys = ["".join(map(str, t)) for t in configs]
    max_x_terms = []
    for x in [0,1]:
        total = 0
        for y in [0,1]:
            max_over_z = max(p[f'p{y}{x}.{cfg}'] for cfg in cfg_keys)
            total += max_over_z
        max_x_terms.append(total)
    LHS = max(max_x_terms)
    general_ok = LHS <= 1

    # print("\nGlobal constraint (Eq. 7 type):")
    # print(f"  max_x Σ_y [max_z P(x,y|z)] = {LHS:.4f}")
    # print(f"  Constraint satisfied: {'✓' if general_ok else '✗'}")

    # Return all results
    return general_ok


