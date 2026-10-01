import torch
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
from collections import Counter

from tqdm import tqdm

import causalflows
import torch
import itertools


import re
import math
import torch
import torch.nn.functional as F


# GAN Components
class DCM_FlOW(nn.Module):
    def __init__(self, graph, cur_node=None, latent_dim=10, hidden_dim=32, device='cuda', out_dim=2, target_y=1, dox=1, shared_net=None):
        super().__init__()

        self.graph=graph
        self.dag = self.graph.dag

        self.confounders = self.graph.confounders


        self.obs_to_lat = {node: [] for node in self.dag.keys()}
        for cnf in self.confounders.keys():
            group = self.confounders[cnf]
            if len(group) < 2:
                continue
            # N-way confounder group (was hardcoded to exactly 2 via v1,v2=pair
            # -- generalized 2026-08-19 for >2-child hidden-confounder groups,
            # e.g. one unobserved node confounding all of its real children.
            # The rest of this class already sums an arbitrary number of
            # confounder noise draws per node (see forward()/sample()'s
            # `torch.stack(all_nse, dim=0).sum(dim=0)`), so this was the only
            # place actually restricted to pairs.
            for v in group:
                self.obs_to_lat[v].append(cnf)


        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.target_y = target_y
        self.dox = dox
        
        # Set up dimension dictionary - default to out_dim for all nodes if not provided

        if self.graph.dim_dict is None:
            self.dim_dict = {node: out_dim for node in self.dag.keys()}
        else:
            self.dim_dict = self.graph.dim_dict
        
        self.device = device

        # Change from regular dict to ModuleDict to properly register parameters


        print('Creating DCM...')
        self.dcm = nn.ModuleDict()
        for node, parents in self.dag.items():
            in_dim=0
            for p in parents:
                in_dim+=self.dim_dict[p]

            if node == cur_node and shared_net!=None:
                self.dcm[node] = shared_net     # referencing to the net for possible root cause, created in the normal dcm.
            else:
                n_features = self.dim_dict[node]
                n_context = in_dim + self.latent_dim  # context = parents + exogenous noise
                self.dcm[node] =  causalflows.flows.CausalNSF(
                n_features, n_context,
                order=tuple(range(n_features)),
                hidden_features=[128] * 3,
                ).to(self.device)

        
        self.shared_net = self.dcm[cur_node]
    

    def get_shared_net(self):
        return self.shared_net

    def generate_latents(self, batch_size: int, device: torch.device):
        """Generates latent U samples."""

        latents = torch.randn(batch_size, self.latent_dim, requires_grad=True).to(device)

        return latents

    def forward(self, batch_data, num_samples=100):
        """
        Forward pass for any DAG structure.
        
        Args:
            batch_data: Dictionary of observed values for each node
            num_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            data_fit_loss: Total loss for fitting the data
            int_effect: Causal intervention effect
        """
        B = batch_data[list(batch_data.keys())[0]].shape[0]  # Get actual batch size
        device = self.device
        
        # Generate noise for confounders
        conf_noise = {}
        for conf_name, confounded_nodes in self.confounders.items():
            # u = torch.randn(B, num_samples, self.latent_dim, requires_grad=True).to(device)
            u= self.generate_latents(B*num_samples, device)
            u_flat = u.view(B * num_samples, self.latent_dim)
            conf_noise[conf_name] = u_flat
        
        parents_input = {}
        loss_dict = {}
        
        # Process each observed node in the DAG
        for node in self.dag.keys():
            input_onehots = [parents_input[par] for par in self.dag[node]]

            # print(f'parents {self.dag[node]} -> node: {node}  input_onehots-->: {input_onehots.shape}')

            all_nse = [conf_noise[cnf] for cnf in self.obs_to_lat[node]]

            # if len(self.confounders.keys())>0 and node in self.confounders['U']:
            if len(all_nse)>0:
                noise = torch.stack(all_nse, dim=0).sum(dim=0)
                input_onehots.append(noise)
                c= torch.cat(input_onehots, dim=1)
                
                loss = -self.dcm[node](c).log_prob(batch_data[node]).mean()
                loss_dict[node] = loss
            else:
                # Handle non-confounded nodes
                noise = torch.randn(B*num_samples, self.latent_dim, device=device)
                input_onehots.append(noise)
                c= torch.cat(input_onehots, dim=1)
                
                loss = -self.dcm[node](c).log_prob(batch_data[node]).mean()
                loss_dict[node] = loss
            
            # Store real data for future parent references
            obs_expanded = batch_data[node].expand(-1, num_samples).reshape(B, num_samples)
            parents_input[node] = obs_expanded.float()
        
        data_fit_loss = sum(loss_dict.values())


        
        return data_fit_loss, loss_dict


    # adapter for observed backdoor
    def generate_samples(self, batch_size=1000, data_input=None):
        """
        Generate synthetic samples from the trained DCM model for any DAG structure.

        Returns:
            Dictionary with generated samples for each node in the DAG.
        """
        device = self.device
        was_training = self.training  # restore below -- eval() here shouldn't leak into the caller's training loop
        self.eval()  # Set model to evaluation mode

        # Generate noise for confounders
        conf_noise = {}
        for conf_name, confounded_nodes in self.confounders.items():
            u = self.generate_latents(batch_size, device)
            conf_noise[conf_name] = u
        
        # Generate noise for each node
        noise = {}
        for node in self.dag.keys():
            all_nse = [conf_noise[cnf] for cnf in self.obs_to_lat[node]]
            if len(all_nse) > 0:
                # Sum confounder noises if node has confounders
                noise[node] = torch.stack(all_nse, dim=0).sum(dim=0)
            else:
                # Handle non-confounded nodes
                noise[node] = torch.randn(batch_size, self.latent_dim, device=device)

        # Sample nodes in topological order
        samples = {}
        parents_input = {}
        
        for node in self.dag.keys():
            inputs = [parents_input[par] for par in self.dag[node]]
            inputs.append(noise[node])
            c_star = torch.cat(inputs, dim=1)
            samples[node] = self.dcm[node](c_star).sample()  # one sample per batch element
            samples[node] = samples[node].reshape(batch_size, self.dim_dict[node]).float()
            parents_input[node] = samples[node]

        if was_training:
            self.train()  # restore -- caller's training loop resumes right after this call

        return samples


    def get_dcm_params(self):
        return list(self.dcm.parameters())

    












