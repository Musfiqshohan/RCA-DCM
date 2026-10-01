"""Ground-truth variables/edges for the light tunnel, "standard" configuration.

Copied verbatim (data only, not code structure) from the `causalchamber` pip
package's own `causalchamber/ground_truth/main.py` (v0.2.1, MIT-licensed,
Copyright (c) 2025 Causal Chamber GmbH) -- `_variables_lt_standard`,
`_edges_lt_standard`, and the lt-standard subset of `_latex_names`. Reproduced
locally rather than importing the package directly because
`import causalchamber` has a live side effect (fetches a dataset directory
listing from S3 and prints a citation banner) on every import -- irrelevant
here since we only need this static table, and undesirable to trigger on
every case in a sweep.

Verified against the installed package (2026-08-19):
    from causalchamber.ground_truth.main import variables, edges
    variables('lt', 'standard') == VARIABLES        # True, 38 names
    edges('lt', 'standard') == EDGES                # True, 57 edges
Also verified: VARIABLES == the non-admin columns of
lt_interventions_standard_v1's raw CSVs, exactly (no diff).
"""

VARIABLES = [
    "red", "green", "blue", "osr_c", "v_c", "current", "pol_1", "pol_2",
    "osr_angle_1", "osr_angle_2", "v_angle_1", "v_angle_2", "angle_1", "angle_2",
    "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3",
    "l_11", "l_12", "l_21", "l_22", "l_31", "l_32",
    "diode_ir_1", "diode_vis_1", "diode_ir_2", "diode_vis_2", "diode_ir_3", "diode_vis_3",
    "t_ir_1", "t_vis_1", "t_ir_2", "t_vis_2", "t_ir_3", "t_vis_3",
]

# (from, to): from -> to, i.e. `from` causally affects `to`.
EDGES = [
    ("red", "ir_1"), ("green", "ir_1"), ("blue", "ir_1"),
    ("red", "ir_2"), ("green", "ir_2"), ("blue", "ir_2"),
    ("red", "ir_3"), ("green", "ir_3"), ("blue", "ir_3"),
    ("red", "vis_1"), ("green", "vis_1"), ("blue", "vis_1"),
    ("red", "vis_2"), ("green", "vis_2"), ("blue", "vis_2"),
    ("red", "vis_3"), ("green", "vis_3"), ("blue", "vis_3"),
    ("red", "current"), ("green", "current"), ("blue", "current"),
    ("pol_1", "ir_3"), ("pol_2", "ir_3"), ("pol_1", "vis_3"), ("pol_2", "vis_3"),
    ("pol_1", "angle_1"), ("pol_2", "angle_2"),
    ("v_angle_1", "angle_1"), ("osr_angle_1", "angle_1"),
    ("v_angle_2", "angle_2"), ("osr_angle_2", "angle_2"),
    ("v_c", "current"), ("osr_c", "current"),
    ("l_11", "ir_1"), ("l_12", "ir_1"), ("l_11", "vis_1"), ("l_12", "vis_1"),
    ("t_ir_1", "ir_1"), ("diode_ir_1", "ir_1"), ("t_vis_1", "vis_1"), ("diode_vis_1", "vis_1"),
    ("l_21", "ir_2"), ("l_22", "ir_2"), ("l_21", "vis_2"), ("l_22", "vis_2"),
    ("t_ir_2", "ir_2"), ("diode_ir_2", "ir_2"), ("t_vis_2", "vis_2"), ("diode_vis_2", "vis_2"),
    ("l_31", "ir_3"), ("l_32", "ir_3"), ("l_31", "vis_3"), ("l_32", "vis_3"),
    ("t_ir_3", "ir_3"), ("diode_ir_3", "ir_3"), ("t_vis_3", "vis_3"), ("diode_vis_3", "vis_3"),
]

# Column name -> LaTeX label, as used by causal_chamber_datasets.yaml's
# root_causes values (e.g. "B", "D^I_1", "R_1").
LATEX_NAMES = {
    "red": "R", "green": "G", "blue": "B", "osr_c": "O_C", "v_c": "R_C",
    "current": "\\tilde{C}", "pol_1": "\\theta_1", "pol_2": "\\theta_2",
    "osr_angle_1": "O_1", "osr_angle_2": "O_2", "v_angle_1": "R_1", "v_angle_2": "R_2",
    "angle_1": "\\tilde{\\theta}_1", "angle_2": "\\tilde{\\theta}_2",
    "ir_1": "\\tilde{I}_1", "vis_1": "\\tilde{V}_1", "ir_2": "\\tilde{I}_2", "vis_2": "\\tilde{V}_2",
    "ir_3": "\\tilde{I}_3", "vis_3": "\\tilde{V}_3",
    "l_11": "L_{11}", "l_12": "L_{12}", "l_21": "L_{21}", "l_22": "L_{22}",
    "l_31": "L_{31}", "l_32": "L_{32}",
    "diode_ir_1": "D^I_1", "diode_vis_1": "D^V_1", "diode_ir_2": "D^I_2", "diode_vis_2": "D^V_2",
    "diode_ir_3": "D^I_3", "diode_vis_3": "D^V_3",
    "t_ir_1": "T^I_1", "t_vis_1": "T^V_1", "t_ir_2": "T^I_2", "t_vis_2": "T^V_2",
    "t_ir_3": "T^I_3", "t_vis_3": "T^V_3",
}
def _normalize_latex(s):
    """causal_chamber_datasets.yaml's root_causes labels are a plainer variant
    of LATEX_NAMES -- no braces, no backslash, spelled-out greek letters kept
    as-is (e.g. "L_11" not "L_{11}", "theta_1" not "\\theta_1"). Strip the
    braces/backslash so both sides compare equal."""
    return s.replace("\\", "").replace("{", "").replace("}", "")


COLUMN_FROM_LATEX = {_normalize_latex(v): k for k, v in LATEX_NAMES.items()}

ADMIN_COLUMNS = {"timestamp", "config", "counter", "flag", "intervention", "camera", "v_board", "v_reg"}

SOURCE_NODES = sorted(set(f for f, _ in EDGES) - set(t for _, t in EDGES))  # 29, no parents
SINK_NODES = sorted(set(t for _, t in EDGES))  # 9, pure measurements


def parents(node):
    return sorted(f for f, t in EDGES if t == node)


def dag():
    """{node: [parents]} for all 38 variables -- ADMG-shape dag dict."""
    return {n: parents(n) for n in VARIABLES}
