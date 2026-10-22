"""Grading and agreement evaluation on a UWF set with three classes (Normal / NPDR / PDR) and no lesion labels.

Five-class predictions are collapsed as: Normal = P(grade 0); NPDR = P(1)+P(2)+P(3); PDR = P(4).
Agreement is label-free: any-lesion rule vs. grade>0 rule, counted three ways (noisy-OR, max-probability,
thresholded OR with the validation-derived 90%-specificity thresholds from each run's own validation predictions).
Rates are disagreements / (images x seeds). Baseline-vs-GLC differences use a paired image-level bootstrap.

Example:
  python external_eval_uwf3.py --manifest peng_manifest.csv --out-dir peng_results ^
    --run base42=preds\\base42_external_predictions.csv,val\\base42_val.csv ... (base52 base62 glc42 glc52 glc62)
  Optional: --split test  (restrict to one split; default all images, none of which was used for training)
"""
import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score)
from external_eval import read, noisy_or, threshold90, L


def collapse(g): return np.stack([g[:, 0], g[:, 1:4].sum(1), g[:, 4]], 1)


def indicators(r, th):
    pg = (1 - r['g'][:, 0]) >= 0.5
    return {'dis_noisyor': ((noisy_or(r['p']) >= .5) != pg).astype(float),
            'dis_maxprob': ((r['p'].max(1) >= .5) != pg).astype(float),
            'dis_thresholdedOR': ((r['p'] >= th).any(1) != pg).astype(float)}


def grading(y, p3):
    pred = p3.argmax(1); out = {}
    out['acc'] = accuracy_score(y, pred); out['bal_acc'] = balanced_accuracy_score(y, pred)
    out['macro_f1'] = f1_score(y, pred, average='macro', labels=[0, 1, 2], zero_division=0)
    out['qwk3'] = cohen_kappa_score(y, pred, labels=[0, 1, 2], weights='quadratic')
    out['auc_pdr_vs_rest'] = roc_auc_score((y == 2).astype(int), p3[:, 2])
    out['auc_anyDR_vs_normal'] = roc_auc_score((y > 0).astype(int), 1 - p3[:, 0])
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--run', action='append', required=True, metavar='LABEL=EXT_CSV,VAL_CSV')
    a.add_argument('--manifest', type=Path, required=True)
    a.add_argument('--out-dir', type=Path, required=True)
    a.add_argument('--split', choices=['train', 'val', 'test'], default=None)
    a.add_argument('--boot', type=int, default=10000)
    a.add_argument('--seed', type=int, default=0)
    a = a.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    with a.manifest.open(newline='', encoding='utf-8-sig') as f:
        man = {r['image']: r for r in csv.DictReader(f)}
    runs = {}
    for spec in a.run:
        label, rest = spec.split('=', 1); e, v = rest.split(',', 1)
        runs[label] = (read(e), read(v))
    first = next(iter(runs.values()))[0]['image']
    for k, (r, _) in runs.items(): assert r['image'] == first, f'image order differs in {k}'
    missing = [i for i in first if i not in man]
    if missing: raise SystemExit(f'{len(missing)} predicted images missing from manifest, first {missing[0]}')
    keep = np.array([a.split is None or man[i]['split'] == a.split for i in first])
    y = np.array([int(man[i]['label3']) for i in first])[keep]; n = int(keep.sum())
    print(f'{n} images' + (f' (split={a.split})' if a.split else ' (all splits)'), 'class counts', np.bincount(y, minlength=3))
    per, ind, p3s = {}, {}, {}
    for k, (r, v) in runs.items():
        th = np.array([threshold90(v['p'][:, j], v['y'][:, j]) for j in range(7)])
        p3 = collapse(r['g'])[keep]; p3s[k] = p3
        ind[k] = {m: x[keep] for m, x in indicators(r, th).items()}
        per[k] = {'n': n, **grading(y, p3), **{m: x.mean() for m, x in ind[k].items()}}
    keys = list(next(iter(per.values())).keys())
    with (a.out_dir / 'per_run_peng_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['run', *keys])
        for k, t in per.items(): w.writerow([k, *[f'{t[c]:.6g}' for c in keys]])
    base = sorted(k for k in runs if k.startswith('base')); glc = sorted(k for k in runs if k.startswith('glc'))
    if base and glc and len(base) == len(glc):
        mets = ['dis_noisyor', 'dis_maxprob', 'dis_thresholdedOR']
        def agree(arm, idx): return {m: np.mean([ind[k][m][idx].mean() for k in arm]) for m in mets}
        def grade_pooled(arm, idx):  # mean over seeds of the per-seed metric
            gs = [grading(y[idx], p3s[k][idx]) for k in arm]
            return {m: np.mean([g[m] for g in gs]) for m in gs[0]}
        allidx = np.arange(n)
        def stats(idx): return ({**grade_pooled(base, idx), **agree(base, idx)}, {**grade_pooled(glc, idx), **agree(glc, idx)})
        ob, og = stats(allidx)
        pk = np.array([man[i].get('patient', '') for i in first])[keep]
        groups = [np.where(pk == p)[0] for p in np.unique(pk)] if (pk != '').all() else None
        def boot(kind):
            rng = np.random.default_rng(a.seed + (1 if kind == 'cluster' else 0)); out = {m: [] for m in ob}
            for _ in range(a.boot):
                if kind == 'image': idx = rng.integers(0, n, n)
                else: idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                if len(np.unique(y[idx])) < 3: continue
                b, g = stats(idx)
                for m in out: out[m].append(g[m] - b[m])
            return {m: np.nanpercentile(v, [2.5, 97.5]) for m, v in out.items()}
        ci_img = boot('image'); ci_cl = boot('cluster') if groups else None
        with (a.out_dir / 'peng_paired_comparison.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['metric', 'baseline_mean_over_seeds', 'glc_mean_over_seeds', 'difference', 'ci_low_image', 'ci_high_image',
                        'ci_low_cluster', 'ci_high_cluster'])
            for m in ob:
                lo, hi = ci_img[m]; cl = ci_cl[m] if ci_cl else (np.nan, np.nan)
                w.writerow([m, f'{ob[m]:.4f}', f'{og[m]:.4f}', f'{og[m]-ob[m]:+.4f}', f'{lo:+.4f}', f'{hi:+.4f}', f'{cl[0]:+.4f}', f'{cl[1]:+.4f}'])
                print(f'{m:22s} base {ob[m]:.4f}  GLC {og[m]:.4f}  diff {og[m]-ob[m]:+.4f}  image [{lo:+.4f}, {hi:+.4f}]'
                      + (f'  cluster [{cl[0]:+.4f}, {cl[1]:+.4f}]' if ci_cl else ''))
        print(f'{len(groups) if groups else 0} filename-derived clusters (cluster interval resamples these; keys are not verified patient IDs)')
    print('Image-level intervals ignore clustering; cluster intervals use filename-derived keys, not verified patient IDs; training variability is shown by the per-run table.')


if __name__ == '__main__':
    main()
