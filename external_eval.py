"""Evaluate external-set predictions for baseline vs GLC runs. Uses only labels that exist in the external set.

Thresholds for NV/VH/RD (and all lesions for the thresholded-OR agreement count) are re-derived from each run's
VALIDATION predictions (max sensitivity subject to validation specificity >= 0.90) and applied unchanged.

Example:
  python external_eval.py --out-dir external_results ^
    --run base42=external_preds\\base42_external_predictions.csv,runs\\...\\best_val_predictions.csv ^
    ... (same for base52 base62 glc42 glc52 glc62; labels must start with 'base' or 'glc')
"""
import argparse, csv, math
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

L = ['MA', 'HE', 'IH', 'VB_IRMA', 'NV', 'VH', 'RD']
E = ['VB_IRMA', 'NV', 'VH', 'RD']


def read(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    def col(name):
        v = [r[name].strip() for r in rows]
        return np.array([float(x) if x != '' else np.nan for x in v])
    d = {'image': [r['image'] for r in rows], 'grade': col('grade'),
         'g': np.stack([col(f'g{i}') for i in range(5)], 1),
         'p': np.stack([col(f'p_{c}') for c in L], 1),
         'y': np.stack([col(f'target_{c}') for c in L], 1)}
    return d


def noisy_or(p): return 1 - np.prod(1 - p, 1)


def threshold90(score, y):
    pos, neg = score[y == 1], score[y == 0]
    best = None
    for tau in np.unique(pos):
        sens, spec = (pos >= tau).mean(), (neg < tau).mean()
        if spec >= 0.90 and (best is None or sens > best[1] or (sens == best[1] and tau > best[0])):
            best = (tau, sens)
    return best[0] if best else np.inf


def ap(y, s): return average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan


def run_metrics(ext, val):
    out = {'n': len(ext['image'])}
    gr = ext['grade']; has_grade = not np.isnan(gr).any()
    if has_grade:
        out['kappa_qw'] = cohen_kappa_score(gr.astype(int), ext['g'].argmax(1), labels=list(range(5)), weights='quadratic')
        pdr = (gr == 4).astype(int)
        out['pdr_ap'] = ap(pdr, ext['g'][:, 4])
        out['pdr_auc'] = roc_auc_score(pdr, ext['g'][:, 4]) if 0 < pdr.sum() < len(pdr) else np.nan
    avail = [j for j in range(7) if not np.isnan(ext['y'][:, j]).any()]
    th = [threshold90(val['p'][:, j], val['y'][:, j]) for j in range(7)]
    for j in avail:
        yj = ext['y'][:, j]
        out[f'n_pos_{L[j]}'] = int(yj.sum()); out[f'ap_{L[j]}'] = ap(yj, ext['p'][:, j])
        if L[j] in ('NV', 'VH', 'RD') and 0 < yj.sum() < len(yj):
            out[f'sens_{L[j]}'] = (ext['p'][yj == 1, j] >= th[j]).mean()
            out[f'spec_{L[j]}'] = (ext['p'][yj == 0, j] < th[j]).mean()
    e_av = [L[j] for j in avail if L[j] in E]
    out['E_labels_available'] = '+'.join(e_av)
    if e_av: out['macro_ap_available_E'] = float(np.nanmean([out[f'ap_{c}'] for c in e_av]))
    pg = (1 - ext['g'][:, 0]) >= 0.5
    out['dis_noisyor'] = int(((noisy_or(ext['p']) >= .5) != pg).sum())
    out['dis_maxprob'] = int(((ext['p'].max(1) >= .5) != pg).sum())
    out['dis_thresholdedOR'] = int(((ext['p'] >= np.array(th)).any(1) != pg).sum())
    return out


def pooled_stats(runs, idx):
    g = np.concatenate([r['grade'][idx] for r in runs]); gp = np.concatenate([r['g'][idx] for r in runs])
    st = {}
    if not np.isnan(g).any():
        st['kappa_qw'] = cohen_kappa_score(g.astype(int), gp.argmax(1), labels=list(range(5)), weights='quadratic')
        st['pdr_ap'] = ap((g == 4).astype(int), gp[:, 4])
    aps = []
    for j, c in enumerate(L):
        yj = np.concatenate([r['y'][idx, j] for r in runs])
        if c in E and not np.isnan(yj).any():
            aps.append(ap(yj, np.concatenate([r['p'][idx, j] for r in runs])))
    if aps: st['macro_ap_available_E'] = float(np.nanmean(aps))
    return st


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--run', action='append', required=True, metavar='LABEL=EXT_CSV,VAL_CSV')
    a.add_argument('--out-dir', type=Path, required=True)
    a.add_argument('--boot', type=int, default=2000)
    a.add_argument('--seed', type=int, default=0)
    a = a.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    exts, vals = {}, {}
    for spec in a.run:
        label, rest = spec.split('=', 1); e, v = rest.split(',', 1)
        exts[label], vals[label] = read(e), read(v)
    first = next(iter(exts.values()))['image']
    for k, r in exts.items():
        assert r['image'] == first, f'image order differs in {k}'
    table = {k: run_metrics(exts[k], vals[k]) for k in exts}
    keys = sorted({c for t in table.values() for c in t}, key=lambda s: (s != 'n', s))
    with (a.out_dir / 'per_run_external_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['run', *keys])
        for k, t in table.items(): w.writerow([k, *[t.get(c, '') for c in keys]])
    print(f'per-run metrics -> {a.out_dir / "per_run_external_metrics.csv"}')
    base = [exts[k] for k in sorted(exts) if k.startswith('base')]
    glc = [exts[k] for k in sorted(exts) if k.startswith('glc')]
    if base and glc and len(base) == len(glc):
        rng = np.random.default_rng(a.seed); n = len(first)
        obs_b, obs_g = pooled_stats(base, np.arange(n)), pooled_stats(glc, np.arange(n))
        diffs = {k: [] for k in obs_b}
        for _ in range(a.boot):
            idx = rng.integers(0, n, n)
            sb, sg = pooled_stats(base, idx), pooled_stats(glc, idx)
            for k in diffs: diffs[k].append(sg[k] - sb[k])
        with (a.out_dir / 'pooled_comparison.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['metric', 'baseline', 'glc', 'difference', 'ci_low', 'ci_high', 'boot_p_two_sided'])
            for k in obs_b:
                d = np.array(diffs[k]); lo, hi = np.nanpercentile(d, [2.5, 97.5])
                pv = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                w.writerow([k, f'{obs_b[k]:.4f}', f'{obs_g[k]:.4f}', f'{obs_g[k]-obs_b[k]:+.4f}', f'{lo:+.4f}', f'{hi:+.4f}', f'{pv:.3f}'])
                print(f'{k}: baseline {obs_b[k]:.4f}  GLC {obs_g[k]:.4f}  diff {obs_g[k]-obs_b[k]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
    print('Note: intervals are image-level; the external set is for transfer, not for retuning thresholds or checkpoints.')


if __name__ == '__main__':
    main()
