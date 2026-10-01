# ./algorithms/counterfactual_contribution.py
"""Wrapper function for running the Counterfactual algorithm as described in
"Causal structure-based root cause analysis of outliers" (ICML 2022).

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

from typing import Dict

import networkx as nx
import pandas as pd

from RCAEval.e2e.RCAWithMissingStructuralKnowledgeCode.utils import preprocess_data

from RCAEval.e2e.RCAWithMissingStructuralKnowledgeCode.algorithms.petshop_root_cause_analysis_main.code.counterfactual_attribution import make_counterfactual_attribution_method


def apply_counterfactual_contribution(graph: nx.DiGraph,
                                      target_node: str,
                                      normal_data: pd.DataFrame,
                                      anomaly_data: pd.DataFrame) -> Dict[str, float]:
    """ 
    This is a wrapper around the method in
    "Causal structure-based root cause analysis of outliers" (ICML 2022).
    
    The code for this method can be found in https://github.com/amazon-science/petshop-root-cause-analysis ,
        using code from https://github.com/py-why/dowhy
    """
    normal_metrics = preprocess_data(normal_data)
    abnormal_metrics = preprocess_data(anomaly_data)

    counterfactual_contribution_runner = make_counterfactual_attribution_method()

    causes = counterfactual_contribution_runner(
        graph=graph.reverse(),
        target_node=target_node,
        target_metric="latency",
        target_statistic="Average",
        normal_metrics=normal_metrics,
        abnormal_metrics=abnormal_metrics
    )

    return {x.node: x.score for x in causes}
