"""Open UWF set (7 diagnostic groups, no DR grade, no lesion labels): retinal-detachment recognition, DR vs healthy,
and label-free grade-lesion agreement. Baseline vs GLC, paired, with patient-cluster and image-level bootstrap.

Endpoints (fixed before the checkpoints were run on these images):
  PRIMARY   AP of the RD lesion output, diagnosis RD (positive) vs all other diagnoses (negative).
  secondary AUC of the RD output; AUC of 1-P(grade 0) for DR vs Healthy; disagreement rates (noisy-OR, max-prob,
            thresholded OR with MMRDR-validation thresholds). Sensitivity analysis: --subset gradable.
Differences are GLC minus baseline, each arm averaged over its three seeds.

  python external_eval_open.py --manifest open_uwf_manifest.csv --out-dir open_results [--subset gradable] ^
    --run base42=EXT_CSV,VAL_CSV ... (six runs, labels start with base / glc)
"""
import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from external_eval import read, noisy_or, threshold90, L

RD = L.index('RD')


def metrics(r, th, idx, dx):
    d = dx[idx]; p = r['p'][idx]; g = r['g'][idx]
    y = (d == 'RD').astype(int); out = {}
    out['rd_ap'] = average_precision_score(y, p[:, RD]); out['rd_auc'] = roc_auc_score(y, p[:, RD])
    m = np.isin(d, ['DR', 'Healthy']); yd = (d[m] == 'DR').astype(int)
    out['dr_vs_healthy_auc'] = roc_auc_score(yd, 1 - g[m, 0]) if 0 < yd.sum() < len(yd) else np.nan
    pg = (1 - g[:, 0]) >= .5
    out['dis_noisyor'] = ((noisy_or(p) >= .5) != pg).mean()
    out['dis_maxprob'] = ((p.max(1) >= .5) != pg).mean()
    out['dis_thresholdedOR'] = ((p >= th).any(1) != pg).mean()
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--run', action='append', required=True, metavar='LABEL=EXT_CSV,VAL_CSV')
    a.add_argument('--manifest', type=Path, required=True); a.add_argument('--out-dir', type=Path, required=True)
    a.add_argument('--subset', choices=['all', 'gradable'], default='all')
    a.add_argument('--boot', type=int, default=10000); a.add_argument('--seed', type=int, default=0)
    a = a.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    with a.manifest.open(newline='', encoding='utf-8-sig') as f: man = {r['image']: r for r in csv.DictReader(f)}
    runs, th = {}, {}
    for spec in a.run:
        label, rest = spec.split('=', 1); e, v = rest.split(',', 1)
        runs[label] = read(e); val = read(v)
        th[label] = np.array([threshold90(val['p'][:, j], val['y'][:, j]) for j in range(7)])
    first = next(iter(runs.values()))['image']
    for k, r in runs.items(): assert r['image'] == first, f'image order differs in {k}'
    miss = [i for i in first if i not in man]
    if miss: raise SystemExit(f'{len(miss)} predicted images missing from manifest, first {miss[0]}')
    dx = np.array([man[i]['diagnosis'] for i in first]); pat = np.array([man[i]['patient'] for i in first])
    q = np.array([man[i].get('quality', '') for i in first])
    keep = np.ones(len(first), bool) if a.subset == 'all' else (q == '1')
    sel = np.where(keep)[0]; dx_s, pat_s = dx[sel], pat[sel]
    print(f'{len(sel)} images ({a.subset}); RD positives {(dx_s == "RD").sum()}; patients {len(set(pat_s))}')
    sub = {k: {kk: (vv[sel] if isinstance(vv, np.ndarray) else vv) for kk, vv in r.items()} for k, r in runs.items()}
    base = sorted(k for k in runs if k.startswith('base')); glc = sorted(k for k in runs if k.startswith('glc'))
    per = {k: metrics(sub[k], th[k], np.arange(len(sel)), dx_s) for k in runs}
    keys = list(next(iter(per.values())))
    with (a.out_dir / f'per_run_open_{a.subset}.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['run', *keys])
        for k, t in per.items(): w.writerow([k, *[f'{t[c]:.6g}' for c in keys]])
    def arm(names, idx): 
        ms = [metrics(sub[k], th[k], idx, dx_s) for k in names]; return {m: np.nanmean([x[m] for x in ms]) for m in keys}
    n = len(sel); all_idx = np.arange(n); ob, og = arm(base, all_idx), arm(glc, all_idx)
    groups = [np.where(pat_s == p)[0] for p in np.unique(pat_s)]
    def boot(kind):
        rng = np.random.default_rng(a.seed + (1 if kind == 'cluster' else 0)); out = {m: [] for m in keys}
        for _ in range(a.boot):
            idx = rng.integers(0, n, n) if kind == 'image' else np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            d = dx_s[idx]
            if (d == 'RD').sum() < 1 or (d != 'RD').sum() < 1: continue
            b, g = arm(base, idx), arm(glc, idx)
            for m in keys: out[m].append(g[m] - b[m])
        return {m: np.nanpercentile(v, [2.5, 97.5]) for m, v in out.items()}
    ci_i, ci_c = boot('image'), boot('cluster')
    with (a.out_dir / f'paired_open_{a.subset}.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['metric', 'baseline', 'glc', 'difference', 'ci_low_image', 'ci_high_image', 'ci_low_patient', 'ci_high_patient'])
        for m in keys:
            w.writerow([m, f'{ob[m]:.4f}', f'{og[m]:.4f}', f'{og[m]-ob[m]:+.4f}', *[f'{x:+.4f}' for x in ci_i[m]], *[f'{x:+.4f}' for x in ci_c[m]]])
            print(f'{m:20s} base {ob[m]:.4f} GLC {og[m]:.4f} diff {og[m]-ob[m]:+.4f} image [{ci_i[m][0]:+.4f},{ci_i[m][1]:+.4f}] patient [{ci_c[m][0]:+.4f},{ci_c[m][1]:+.4f}]')
    print('Disagreement values are rates (fractions). Differences are conditional on the six fitted models.')


if __name__ == '__main__':
    main()
