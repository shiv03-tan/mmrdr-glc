"""
generate_paper_tables_figures.py  (v2 -- rebuilt for your actual history.csv columns)

Your completed.json turned out to hold no metrics (just mode/epochs_completed/
test_access), and test_access is currently False -- meaning final test-set
scoring hasn't been run yet for these seeds. So everything below is built from
history.csv, at the epoch your own checkpoint rule (argmax selection_score)
would actually pick. These are therefore VALIDATION numbers, not test numbers,
and every table/figure this script produces is labeled that way. Once you run
your test-set evaluation script (whatever produces final test scoring) for all
seeds, re-point this script at that output for the real paper numbers -- ping
me with what that file/format looks like and I'll adapt it in one pass.

Builds:
  Table 1  Main results at best-selection-score epoch, all methods x metrics,
           mean +/- std over available seeds (42/52/62)
  Table 2  Ablation: MTL ASL vs MTL ASL + GLC
  Table 3  Per-lesion AP breakdown, all 7 lesions, at the same best epoch
  Fig  7   Training dynamics: consistency loss / lambda(t) ramp + val curves

NOT built here (need raw per-sample predictions, which history.csv doesn't
have): PR curves, calibration diagram, confusion matrix, macro AUROC,
balanced accuracy, sensitivity@90%-specificity. See the bottom of this file
for the 4 lines to add to your eval code if/when you want those too.

--------------------------------------------------------------------------
HOW TO USE
--------------------------------------------------------------------------
1. Edit CONFIG below if RUNS_ROOT or any folder name is wrong.
2. python generate_paper_tables_figures.py
3. Outputs land in an `outputs` folder next to this script.
--------------------------------------------------------------------------
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG
# =====================================================================

RUNS_ROOT = r"C:\Research\mmrdr-own-study\runs"

# Seed-52 baseline: guessed as the final valid one (most recent). Change if wrong.
METHOD_RUNS = {
    "MTL ASL":       ["uwf_mt_asl_seed42",        "uwf_mt_asl_seed52_resumed_v2",  "uwf_mt_asl_seed62"],
    "MTL ASL + GLC": ["uwf_mt_asl_glc010_seed42", "uwf_mt_asl_glc010_seed52",      "uwf_mt_asl_glc010_seed62"],
}

ABLATION_PAIRS = [("MTL ASL", "MTL ASL + GLC")]

DYNAMICS_PAIR = {
    "MTL ASL":       "uwf_mt_asl_seed42",
    "MTL ASL + GLC": "uwf_mt_asl_glc010_seed42",
}

LESION_ORDER = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
E_GROUP = ["VB_IRMA", "NV", "VH", "RD"]   # 4-label primary rare-lesion endpoint
R_GROUP = ["NV", "VH", "RD"]              # 3-label PDR-aligned / checkpoint endpoint

# Direct-from-history.csv scalar metrics (column name exactly as printed).
SCALAR_METRICS = {
    "Val QWK":            "val_qwk",
    "Val Macro F1 (grade)": "val_macro_f1",
    "Val rare AP (logged, val_rare_ap)": "val_rare_ap",
    "Val all-lesion AP (logged)": "val_all_ap",
    "Selection score":    "selection_score",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# =====================================================================
# END CONFIG
# =====================================================================


def load_best_epoch_row(run_dir):
    """Read history.csv, return the row at argmax(selection_score), or None."""
    csv_path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(csv_path):
        return None, f"[MISSING] {csv_path} not found"
    df = pd.read_csv(csv_path)
    if "selection_score" not in df.columns:
        return None, f"[MISSING COLUMN] 'selection_score' not in {csv_path}"
    if df.empty:
        return None, f"[EMPTY] {csv_path} has no rows"
    best_idx = df["selection_score"].idxmax()
    return df.loc[best_idx], None


def per_run_metrics(row):
    """Compute every metric we want for one run's best-epoch row."""
    out = {}
    for name, col in SCALAR_METRICS.items():
        out[name] = float(row[col]) if col in row and pd.notna(row[col]) else None

    ap_cols_present = all(f"ap_{lesion}" in row.index for lesion in LESION_ORDER)
    if ap_cols_present:
        out["Macro AP (E, 4-label)"] = float(np.mean([row[f"ap_{l}"] for l in E_GROUP]))
        out["AP_R (3-label, recomputed)"] = float(np.mean([row[f"ap_{l}"] for l in R_GROUP]))
        for lesion in LESION_ORDER:
            out[f"ap_{lesion}"] = float(row[f"ap_{lesion}"]) if pd.notna(row[f"ap_{lesion}"]) else None
    else:
        out["Macro AP (E, 4-label)"] = None
        out["AP_R (3-label, recomputed)"] = None
        for lesion in LESION_ORDER:
            out[f"ap_{lesion}"] = None

    out["best_epoch"] = float(row["epoch"]) if "epoch" in row.index else None
    return out


def collect_method_table(method_runs, runs_root):
    all_metric_names = list(SCALAR_METRICS.keys()) + \
        ["Macro AP (E, 4-label)", "AP_R (3-label, recomputed)"] + \
        [f"ap_{l}" for l in LESION_ORDER]

    rows = []
    missing_report = []
    per_run_debug = []
    for method, seed_dirs in method_runs.items():
        per_seed = {name: [] for name in all_metric_names}
        for run_name in seed_dirs:
            run_dir = os.path.join(runs_root, run_name)
            row, err = load_best_epoch_row(run_dir)
            if err:
                missing_report.append(f"  {err}")
                continue
            metrics = per_run_metrics(row)
            per_run_debug.append((method, run_name, metrics["best_epoch"], metrics.get("Val QWK")))
            for name in all_metric_names:
                val = metrics.get(name)
                if val is None:
                    missing_report.append(
                        f"  [MISSING METRIC] '{name}' not available for {run_dir}"
                    )
                else:
                    per_seed[name].append(val)
        row_out = {"Method": method}
        for name, vals in per_seed.items():
            if len(vals) == 0:
                row_out[f"{name}_mean"] = np.nan
                row_out[f"{name}_std"] = np.nan
                row_out[f"{name}_n"] = 0
            else:
                row_out[f"{name}_mean"] = float(np.mean(vals))
                row_out[f"{name}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row_out[f"{name}_n"] = len(vals)
        rows.append(row_out)

    print("\n  Per-run best-epoch check (method, run, best_epoch, val_qwk):")
    for m, r, e, q in per_run_debug:
        q_str = f"{q:.4f}" if q is not None else "--"
        print(f"    {m:18s} {r:30s} epoch={e}  val_qwk={q_str}")

    return pd.DataFrame(rows), missing_report, all_metric_names


def format_mean_std(mean, std, n, decimals=3):
    if pd.isna(mean):
        return "--"
    if n <= 1:
        return f"{mean:.{decimals}f}*"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def df_to_latex_table(df, metric_names, caption, label, pretty_names=None):
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    col_spec = "l" + "c" * len(metric_names)
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    display_names = [pretty_names.get(n, n) if pretty_names else n for n in metric_names]
    lines.append("Method & " + " & ".join(display_names) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        cells = [row["Method"]]
        for name in metric_names:
            mean = row.get(f"{name}_mean", np.nan)
            std = row.get(f"{name}_std", np.nan)
            n = row.get(f"{name}_n", 0)
            cells.append(format_mean_std(mean, std, n))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def build_table1(runs_root, out_dir):
    print("\n=== Table 1: Main results (VALIDATION, at best selection_score epoch) ===")
    df, missing, all_metric_names = collect_method_table(METHOD_RUNS, runs_root)

    headline_metrics = [
        "Val QWK", "Val Macro F1 (grade)", "Macro AP (E, 4-label)",
        "AP_R (3-label, recomputed)", "Val all-lesion AP (logged)", "Selection score",
    ]
    csv_path = os.path.join(out_dir, "table1_main_results_VALIDATION.csv")
    df.to_csv(csv_path, index=False)
    tex = df_to_latex_table(
        df, headline_metrics,
        caption="VALIDATION-set results (test-set scoring has not yet been run for these seeds) at the epoch selected by $S_{val}$. Values are mean $\\pm$ std over available seeds. An asterisk marks a value backed by a single completed seed only. REPLACE WITH TEST-SET NUMBERS BEFORE SUBMISSION.",
        label="tab:main_results_val_interim",
    )
    tex_path = os.path.join(out_dir, "table1_main_results_VALIDATION.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"  wrote {csv_path}")
    print(f"  wrote {tex_path}")
    if missing:
        print(f"  {len(missing)} missing value(s) -- see report below")
    return df, missing


def build_table2(runs_root, df_main, out_dir):
    print("\n=== Table 2: Ablation (VALIDATION, consistency on/off) ===")
    metrics_to_compare = [
        "Val QWK", "Val Macro F1 (grade)", "Macro AP (E, 4-label)",
        "AP_R (3-label, recomputed)", "Val all-lesion AP (logged)",
    ]
    rows = []
    for baseline, glc in ABLATION_PAIRS:
        base_row = df_main[df_main["Method"] == baseline]
        glc_row = df_main[df_main["Method"] == glc]
        if base_row.empty or glc_row.empty:
            print(f"  [SKIPPED] pair ({baseline}, {glc}) not found")
            continue
        base_row = base_row.iloc[0]
        glc_row = glc_row.iloc[0]
        for name in metrics_to_compare:
            b_mean = base_row.get(f"{name}_mean", np.nan)
            g_mean = glc_row.get(f"{name}_mean", np.nan)
            delta = g_mean - b_mean if not (pd.isna(b_mean) or pd.isna(g_mean)) else np.nan
            rows.append({
                "Lesion loss": baseline.replace("MTL ", "").strip(),
                "Metric": name,
                "Without GLC": b_mean,
                "With GLC": g_mean,
                "Delta": delta,
            })
    ab_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "table2_ablation_VALIDATION.csv")
    ab_df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Ablation on VALIDATION data (interim -- replace with test-set numbers): "
             "effect of adding the grade--lesion consistency (GLC) term.}",
             "\\label{tab:ablation_val_interim}",
             "\\begin{tabular}{llccc}", "\\toprule",
             "Lesion loss & Metric & w/o GLC & w/ GLC & $\\Delta$ \\\\", "\\midrule"]
    for _, r in ab_df.iterrows():
        delta_str = f"{r['Delta']:+.3f}" if not pd.isna(r["Delta"]) else "--"
        wo = f"{r['Without GLC']:.3f}" if not pd.isna(r["Without GLC"]) else "--"
        w_ = f"{r['With GLC']:.3f}" if not pd.isna(r["With GLC"]) else "--"
        lines.append(f"{r['Lesion loss']} & {r['Metric']} & {wo} & {w_} & {delta_str} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_path = os.path.join(out_dir, "table2_ablation_VALIDATION.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {tex_path}")
    return ab_df


def build_table3_per_lesion(runs_root, df_main, out_dir):
    print("\n=== Table 3: Per-lesion AP breakdown (VALIDATION, best epoch) ===")
    lesion_metric_names = [f"ap_{l}" for l in LESION_ORDER]
    any_present = any(
        not df_main[f"{m}_mean"].isna().all() for m in lesion_metric_names
    )
    if not any_present:
        print("  [SKIPPED] no per-lesion ap_* columns found in any history.csv")
        return None

    csv_path = os.path.join(out_dir, "table3_per_lesion_VALIDATION.csv")
    cols = ["Method"] + [f"{m}_mean" for m in lesion_metric_names] + [f"{m}_std" for m in lesion_metric_names]
    df_main[cols].to_csv(csv_path, index=False)

    pretty = {f"ap_{l}": l for l in LESION_ORDER}
    tex = df_to_latex_table(
        df_main, lesion_metric_names,
        caption="Per-lesion average precision (AP) on the VALIDATION set at the epoch selected by $S_{val}$, mean $\\pm$ std over available seeds. Replace with test-set numbers before submission.",
        label="tab:per_lesion_val_interim",
        pretty_names=pretty,
    )
    tex_path = os.path.join(out_dir, "table3_per_lesion_VALIDATION.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"  wrote {csv_path}")
    print(f"  wrote {tex_path}")
    return df_main[cols]


def build_training_dynamics(runs_root, out_dir):
    print("\n=== Fig: Training dynamics (lambda ramp + consistency loss) ===")
    baseline_name = list(DYNAMICS_PAIR.keys())[0]
    glc_name = list(DYNAMICS_PAIR.keys())[1]
    glc_run = DYNAMICS_PAIR[glc_name]

    glc_csv = os.path.join(runs_root, glc_run, "history.csv")
    if not os.path.exists(glc_csv):
        print(f"  [SKIPPED] {glc_csv} not found.")
        return

    hist = pd.read_csv(glc_csv)
    if "epoch" not in hist.columns:
        print("  [SKIPPED] no 'epoch' column in history.csv")
        return

    epoch = hist["epoch"]
    cons_loss = hist["train_consistency_loss"] if "train_consistency_loss" in hist.columns else None
    val_qwk = hist["val_qwk"] if "val_qwk" in hist.columns else None
    val_rare_ap = hist["val_rare_ap"] if "val_rare_ap" in hist.columns else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    if cons_loss is not None:
        ax.plot(epoch, cons_loss, color="#5B7CA8", lw=2, label="Raw consistency loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Consistency loss")
    ax.set_title(f"{glc_name}: consistency loss over training")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    if val_qwk is not None:
        ax.plot(epoch, val_qwk, color="#5B7CA8", lw=2, label="Val QWK")
    if val_rare_ap is not None:
        ax.plot(epoch, val_rare_ap, color="#1f7a6b", lw=2, label="Val rare-lesion AP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation score")
    ax.set_title("Validation metrics over training")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "fig_training_dynamics.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isdir(RUNS_ROOT):
        print(f"ERROR: RUNS_ROOT does not exist: {RUNS_ROOT}")
        sys.exit(1)

    df_main, missing = build_table1(RUNS_ROOT, OUT_DIR)
    build_table2(RUNS_ROOT, df_main, OUT_DIR)
    build_table3_per_lesion(RUNS_ROOT, df_main, OUT_DIR)
    build_training_dynamics(RUNS_ROOT, OUT_DIR)

    print("\n" + "=" * 70)
    print(f"Done. Outputs written to: {OUT_DIR}")
    print("\nREMINDER: every number above is from the VALIDATION set, at the")
    print("epoch your own checkpoint rule (argmax selection_score) would pick.")
    print("These are useful for an interim check, but the manuscript's actual")
    print("Table 1 needs TEST-set numbers. Run your test-evaluation script for")
    print("all 6 runs (including glc010_seed62, which has no completed.json")
    print("yet) before finalizing any table for submission.")
    if missing:
        print(f"\n{len(missing)} missing value(s):")
        for line in missing:
            print(line)
    print("=" * 70)


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------
# TO ALSO GET: PR curves, calibration diagram, confusion matrix, macro AUROC,
# balanced accuracy, sensitivity@90%-specificity -- these need raw per-sample
# test predictions, which nothing you currently save provides. Add this to
# your test-evaluation script (wherever it scores the held-out test set):
#
#     np.savez(
#         os.path.join(out_dir, "test_predictions.npz"),
#         grade_true=grade_true,        # (N,)   int, 0..4
#         grade_prob=grade_prob,        # (N, 5) float, softmax probs
#         lesion_true=lesion_true,      # (N, 7) int/bool, order = LESION_ORDER
#         lesion_prob=lesion_prob,      # (N, 7) float, sigmoid probs
#     )
# LESION_ORDER = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
# --------------------------------------------------------------------------
