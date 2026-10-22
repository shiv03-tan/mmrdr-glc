"""MMRDR UWF multitask study -- paired bootstrap significance test, ASL vs ASL+GLC.

Compares your 3-seed baseline arm against your 3-seed GLC arm on the REAL
test set, using the actual files evaluate_test_uwf.py already wrote:
  <run_dir>/test_eval/test_predictions.csv   (per-sample test predictions)
  <run_dir>/best_val_predictions.csv         (used to re-lock Sens@90Spec thresholds)

Method
------
A clustered (by test image) paired bootstrap:
  - All 6 runs share the exact same 2597 test images, so for a given
    bootstrap resample we draw image INDICES once and apply that same
    resample to every run -- this is what makes it "paired".
  - Within an arm (baseline or GLC), the 3 seeds' predictions for the
    resampled images are pooled together before computing the metric, so
    the comparison uses all real per-sample data instead of collapsing to
    3 seed-level numbers.
  - Observed delta = metric(GLC pooled) - metric(baseline pooled), using
    the FULL (non-resampled) pooled data.
  - 95% CI = 2.5/97.5 percentiles of the bootstrap delta distribution.
  - p-value = 2 * min(P(delta_b <= 0), P(delta_b >= 0)), capped at 1 --
    the standard nonparametric bootstrap test for H0: delta = 0.
  - Holm-Bonferroni step-down correction is applied across the set of
    metrics tested (control for the fact that we're testing several).

For Sens@90Spec_R, each run's own 90%-specificity threshold (locked from
that run's best_val_predictions.csv, exactly as in evaluate_test_uwf.py)
is applied once and held fixed across all bootstrap resamples -- only the
sensitivity achieved on the resampled test images varies. This avoids
re-deriving thresholds from test data at every bootstrap iteration, which
would leak test information into the threshold itself.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
python significance_test_uwf.py ^
    --baseline-runs runs\\uwf_mt_asl_seed42 runs\\uwf_mt_asl_seed52_resumed_v2 runs\\uwf_mt_asl_seed62 ^
    --glc-runs      runs\\uwf_mt_asl_glc010_seed42 runs\\uwf_mt_asl_glc010_seed52 runs\\uwf_mt_asl_glc010_seed62 ^
    --n-resamples 10000

Each run directory must already contain test_eval\\test_predictions.csv
(written by evaluate_test_uwf.py) and best_val_predictions.csv.

Output: prints a results table and writes significance_results.json next
to this script.
--------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, cohen_kappa_score,
                              roc_curve)

LESIONS = ('MA', 'HE', 'IH', 'VB_IRMA', 'NV', 'VH', 'RD')
E_IDX = (3, 4, 5, 6)   # VB_IRMA, NV, VH, RD
R_IDX = (4, 5, 6)      # NV, VH, RD


# --------------------------------------------------------------------- IO --

def load_test_predictions(run_dir):
    path = Path(run_dir) / 'test_eval' / 'test_predictions.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run evaluate_test_uwf.py on this run first.")
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    images = [r['image'] for r in rows]
    grade = np.array([int(r['grade']) for r in rows])
    gp = np.array([[float(r[f'g{i}']) for i in range(5)] for r in rows])
    target = np.array([[float(r['target_' + c]) for c in LESIONS] for r in rows])
    p = np.array([[float(r['p_' + c]) for c in LESIONS] for r in rows])
    return images, grade, gp, target, p


def threshold_at_specificity(y_true, y_score, target_spec=0.90):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1 - fpr
    eligible = spec >= target_spec
    if not eligible.any():
        idx = int(np.argmax(spec))
    else:
        idx = int(np.argmax(np.where(eligible, tpr, -1)))
    return float(thresholds[idx])


def load_val_thresholds(run_dir):
    path = Path(run_dir) / 'best_val_predictions.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- needed to lock Sens@90Spec thresholds from validation.")
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    out = {}
    for lesion in ('NV', 'VH', 'RD'):
        y = np.array([float(r['target_' + lesion]) for r in rows])
        p = np.array([float(r['p_' + lesion]) for r in rows])
        out[lesion] = threshold_at_specificity(y, p, 0.90)
    return out


def load_arm(run_dirs):
    """Load + align all seeds of one arm. Returns ref_images, grade, target
    (shared across seeds -- verified identical) and per-seed lists of
    gp/p/pred_pos_R (binarized NV,VH,RD predictions at that run's own
    val-locked threshold)."""
    ref_images = ref_grade = ref_target = None
    gp_list, p_list, predpos_list = [], [], []
    for d in run_dirs:
        images, grade, gp, target, p = load_test_predictions(d)
        if ref_images is None:
            ref_images, ref_grade, ref_target = images, grade, target
        elif images != ref_images:
            order = {im: i for i, im in enumerate(images)}
            idx = [order[im] for im in ref_images]
            grade, gp, target, p = grade[idx], gp[idx], target[idx], p[idx]
            if not (grade == ref_grade).all() or not np.allclose(target, ref_target):
                raise ValueError(f"{d}: test labels don't match other runs after realignment")
        thr = load_val_thresholds(d)
        predpos = np.stack([p[:, R_IDX[i]] >= thr[lesion]
                             for i, lesion in enumerate(('NV', 'VH', 'RD'))], axis=1)
        gp_list.append(gp)
        p_list.append(p)
        predpos_list.append(predpos)
    return ref_images, ref_grade, ref_target, gp_list, p_list, predpos_list


# ---------------------------------------------------------------- metrics --

def pooled_ap(target, p_list, idx, group_idx):
    t = np.tile(target[idx], (len(p_list), 1))
    p = np.concatenate([arr[idx] for arr in p_list], axis=0)
    return float(np.mean([average_precision_score(t[:, j], p[:, j]) for j in group_idx]))


def pooled_qwk(grade, gp_list, idx):
    g_true = np.tile(grade[idx], len(gp_list))
    g_pred = np.concatenate([arr[idx].argmax(1) for arr in gp_list])
    return float(cohen_kappa_score(g_true, g_pred, labels=list(range(5)), weights='quadratic'))


def pooled_macro_f1(grade, gp_list, idx):
    g_true = np.tile(grade[idx], len(gp_list))
    g_pred = np.concatenate([arr[idx].argmax(1) for arr in gp_list])
    return float(f1_score(g_true, g_pred, labels=list(range(5)), average='macro', zero_division=0))


def pooled_sens_r(target, predpos_list, idx):
    y = np.tile(target[idx][:, R_IDX], (len(predpos_list), 1))
    pred = np.concatenate([arr[idx] for arr in predpos_list], axis=0)
    sens = []
    for j in range(3):
        tp = int(((pred[:, j] == 1) & (y[:, j] == 1)).sum())
        fn = int(((pred[:, j] == 0) & (y[:, j] == 1)).sum())
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else float('nan'))
    return float(np.mean(sens))


METRICS = {
    'AP_E':         lambda grade, target, gp, p, predpos, idx: pooled_ap(target, p, idx, E_IDX),
    'AP_R':         lambda grade, target, gp, p, predpos, idx: pooled_ap(target, p, idx, R_IDX),
    'QWK':          lambda grade, target, gp, p, predpos, idx: pooled_qwk(grade, gp, idx),
    'macro_F1':     lambda grade, target, gp, p, predpos, idx: pooled_macro_f1(grade, gp, idx),
    'Sens@90Spec_R': lambda grade, target, gp, p, predpos, idx: pooled_sens_r(target, predpos, idx),
}


def holm_bonferroni(pvals):
    """Returns adjusted p-values via Holm's step-down procedure (no external deps)."""
    names = list(pvals.keys())
    order = sorted(range(len(names)), key=lambda i: pvals[names[i]])
    m = len(names)
    adjusted = {}
    running_max = 0.0
    for rank, i in enumerate(order):
        name = names[i]
        adj = (m - rank) * pvals[name]
        running_max = max(running_max, adj)
        adjusted[name] = float(min(1.0, running_max))
    return adjusted


# --------------------------------------------------------------------- run --

def run_significance_test(baseline_runs, glc_runs, n_resamples, seed):
    base_images, base_grade, base_target, base_gp, base_p, base_predpos = load_arm(baseline_runs)
    glc_images, glc_grade, glc_target, glc_gp, glc_p, glc_predpos = load_arm(glc_runs)

    if base_images != glc_images:
        order = {im: i for i, im in enumerate(base_images)}
        idx_map = [order[im] for im in glc_images]
        if sorted(idx_map) != list(range(len(base_images))):
            raise ValueError("Baseline and GLC arms do not share the same test image set.")
        # realign GLC arm onto baseline's image order
        glc_grade = glc_grade[idx_map]
        glc_target = glc_target[idx_map]
        glc_gp = [a[idx_map] for a in glc_gp]
        glc_p = [a[idx_map] for a in glc_p]
        glc_predpos = [a[idx_map] for a in glc_predpos]
        glc_images = base_images
    if not (base_grade == glc_grade).all() or not np.allclose(base_target, glc_target):
        raise ValueError("Baseline and GLC arms disagree on test labels -- wrong test set?")

    n = len(base_images)
    full_idx = np.arange(n)
    rng = np.random.default_rng(seed)

    observed, boot_deltas, results = {}, {}, {}
    for name, fn in METRICS.items():
        obs_base = fn(base_grade, base_target, base_gp, base_p, base_predpos, full_idx)
        obs_glc = fn(glc_grade, glc_target, glc_gp, glc_p, glc_predpos, full_idx)
        observed[name] = {'baseline': obs_base, 'glc': obs_glc, 'delta': obs_glc - obs_base}
        boot_deltas[name] = np.empty(n_resamples)

    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        for name, fn in METRICS.items():
            d_base = fn(base_grade, base_target, base_gp, base_p, base_predpos, idx)
            d_glc = fn(glc_grade, glc_target, glc_gp, glc_p, glc_predpos, idx)
            boot_deltas[name][b] = d_glc - d_base

    raw_pvals = {}
    for name in METRICS:
        deltas = boot_deltas[name]
        obs = observed[name]['delta']
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        p_le0 = float(np.mean(deltas <= 0))
        p_ge0 = float(np.mean(deltas >= 0))
        p_value = min(1.0, 2 * min(p_le0, p_ge0))
        raw_pvals[name] = p_value
        results[name] = {
            'baseline_mean': observed[name]['baseline'],
            'glc_mean': observed[name]['glc'],
            'observed_delta': obs,
            'ci_95': [float(ci_lo), float(ci_hi)],
            'p_value_raw': p_value,
        }

    adjusted = holm_bonferroni(raw_pvals)
    for name in results:
        results[name]['p_value_holm'] = adjusted[name]
        results[name]['significant_at_0.05_holm'] = adjusted[name] < 0.05

    return results, n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline-runs', type=Path, nargs='+', required=True)
    p.add_argument('--glc-runs', type=Path, nargs='+', required=True)
    p.add_argument('--n-resamples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, default=Path('significance_results.json'))
    args = p.parse_args()

    print(f'Loading {len(args.baseline_runs)} baseline runs + {len(args.glc_runs)} GLC runs...', flush=True)
    results, n = run_significance_test(args.baseline_runs, args.glc_runs, args.n_resamples, args.seed)
    print(f'{n} test images, {args.n_resamples} bootstrap resamples, Holm-Bonferroni across {len(results)} metrics.\n')

    header = f'{"metric":<15}{"baseline":>10}{"GLC":>10}{"delta":>10}{"95% CI":>22}{"p(raw)":>10}{"p(holm)":>10}  sig'
    print(header)
    print('-' * len(header))
    for name, r in results.items():
        ci = f'[{r["ci_95"][0]:+.4f}, {r["ci_95"][1]:+.4f}]'
        sig = 'YES' if r['significant_at_0.05_holm'] else 'no'
        print(f'{name:<15}{r["baseline_mean"]:>10.4f}{r["glc_mean"]:>10.4f}{r["observed_delta"]:>+10.4f}'
              f'{ci:>22}{r["p_value_raw"]:>10.4f}{r["p_value_holm"]:>10.4f}  {sig}')

    args.out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
