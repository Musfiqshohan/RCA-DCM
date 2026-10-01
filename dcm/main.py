# Runs NCM with confidence interval
#
# Thin CLI wrapper around dcm_rca.DCM_RCA (requirement 7: same flags, same CSV
# outputs, so scripts/baseline_snapshot.sh runs unchanged). All the actual pipeline
# logic -- main()/process_node()/train()/check_root_cause() as they used to be --
# now lives in dcm_rca.py, importable without @hydra.main's side effects. This file
# keeps @hydra.main because a CLI entry point is exactly what it's for.
import os
import sys
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path (same pattern as before -- functions.* is imported
# transitively via dcm_rca)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcm_rca import DCM_RCA

lgr = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="conf")
def main(cfg: DictConfig):
    lgr.info("Configuration:")
    lgr.info(OmegaConf.to_yaml(cfg))

    rca = DCM_RCA(cfg=cfg)
    result = rca.run()  # no data/grp args -> falls back to cfg.pth.{nrm,anm}_dat_pth, cfg.exp.grp

    results_file, metrics_file, ranking_file = result.to_csv(cfg.pth.out_dir)

    # No per-node verdict -- ranking (below) is what identifies root causes. Report
    # failed nodes, if any, so a failed run stays visible instead of just quietly
    # missing from the ranking.
    if result.errors:
        lgr.info("\n=== NODE ERRORS ===")
        for node, err in result.errors.items():
            lgr.info(f"Node {node}: {err}")
    lgr.info(f"\nResults saved to: {results_file}")

    # Print ranked root causes
    lgr.info("\n=== RANKED ROOT CAUSES (by TVD difference) ===")
    if not result.scores.empty:
        ranked = result.ranking(metric=cfg.exp.metric, agg="add")
        for rank, (node, tvd_diff) in enumerate(ranked.items(), 1):
            lgr.info(f"  {rank}. {node}: TVD diff = {tvd_diff:.6f}")
    else:
        lgr.info("  No results found")


if __name__ == '__main__':
    main()
