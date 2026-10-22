#!/usr/bin/env python3
"""Full test-set report for baseline vs consistency arms (read-only; never trains).

Inputs are prediction CSVs in the same format as best_val_predictions.csv
(image, grade, prediction, g0..g4, target_<L>, p_<L>).

Example (Windows cmd, one line):
  python test_report.py --B 10000 ^
    --test-a b42.csv b52.csv b62.csv --test-b g42.csv g52.csv g62.csv ^
    --val-a vb42.csv vb52.csv vb62.csv --val-b vg42.csv vg52.csv vg62.csv ^
    --name-a "ASL baseline" --name-b "GLC-Full" --out test_report

Bootstrap: image IDs are resampled once per replicate; every seed's prediction
for a sampled image is kept together and the SAME sample is used for both
arms. Metrics are pooled (computed on all seed rows of the resampled images).
Image-level resampling (no patient IDs): intervals may be optimistic, and they
do not capture variability across training runs.
p-value: two-sided percentile-bootstrap, 2*min(P(d<=0),P(d>=0)); Holm is applied
across all reported pooled metrics (family printed in the output).
"""
import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score as AP, cohen_kappa_score,
                             f1_score, balanced_accuracy_score)

L = ['MA', 'HE', 'IH', 'VB_IRMA', 'NV', 'VH', 'RD']
E = ['VB_IRMA', 'NV', 'VH', 'RD']
R = ['NV', 'VH', 'RD']


def load(files):
    ds = [pd.read_csv(f).set_index('image').sort_index() for f in files]
    ref = ds[0]
    for d in ds:
        assert d.index.equals(ref.index), 'image sets differ'
        assert (d.grade == ref.grade).all(), 'grade labels differ'
        for c in L:
            assert (d['target_' + c] == ref['target_' + c]).all(), 'lesion labels differ'
    return ds


def cross_check(A, B, tag):
    """Pairing audit between the two arms (same images, labels, seed count)."""
    assert len(A) == len(B), f'{tag}: different number of seeds per arm ({len(A)} vs {len(B)})'
    for d in A + B:
        assert d.index.is_unique, f'{tag}: duplicated image IDs'
    ra, rb = A[0], B[0]
    assert ra.index.equals(rb.index), f'{tag}: image IDs differ between arms'
    assert (ra.grade == rb.grade).all(), f'{tag}: grade labels differ between arms'
    for c in L:
        assert (ra['target_' + c] == rb['target_' + c]).all(), f'{tag}: {c} labels differ between arms'
    print(f'cross-arm check passed ({tag}): {len(ra)} unique images, {len(A)} seeds per arm, identical labels')


def pooled_metrics(ds, idx):
    y = np.concatenate([d.grade.values[idx] for d in ds])
    pr = np.concatenate([d.prediction.values[idx] for d in ds])
    T = {c: np.concatenate([d['target_' + c].values[idx] for d in ds]) for c in L}
    P = {c: np.concatenate([d['p_' + c].values[idx] for d in ds]) for c in L}
    m = {c: AP(T[c], P[c]) if T[c].sum() > 0 else np.nan for c in L}
    out = {'AP_E (primary)': np.mean([m[c] for c in E]), 'AP_R': np.mean([m[c] for c in R]),
           'QWK': cohen_kappa_score(y, pr, weights='quadratic'),
           'macroF1': f1_score(y, pr, average='macro'),
           'balancedAcc': balanced_accuracy_score(y, pr)}
    out.update({'AP_' + c: m[c] for c in L})
    return out


def agreement(d, tau=0.5):
    P = np.stack([d['p_' + c].values for c in L], 1)
    A = 1 - np.prod(1 - P, 1)
    Pr = np.stack([d['p_' + c].values for c in R], 1)
    Adv = 1 - np.prod(1 - Pr, 1)
    pdr = 1 - d.g0.values
    p4 = d.g4.values

    def js(a, b, eps=1e-7):
        a = np.clip(a, eps, 1 - eps); b = np.clip(b, eps, 1 - eps); m = (a + b) / 2
        kl = lambda p, q: p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        return 0.5 * kl(a, m) + 0.5 * kl(b, m)

    dis_any = (A >= tau) != (pdr >= tau)
    dis_adv = (Adv >= tau) != (p4 >= tau)
    agg_only = dis_any & (P.max(1) < tau)
    return {'JS_any_mean': js(A, pdr).mean(), 'JS_adv_mean': js(Adv, p4).mean(),
            'dis_any': int(dis_any.sum()), 'dis_any_aggregation_only': int(agg_only.sum()),
            'dis_any_with_positive_lesion': int((dis_any & ~agg_only).sum()),
            'dis_adv': int(dis_adv.sum())}


def threshold_90(val, c, spec=0.90):
    t = val['target_' + c].values; p = val['p_' + c].values
    best = None
    for th in np.unique(p):
        pred = p >= th
        sp = ((~pred) & (t == 0)).sum() / max((t == 0).sum(), 1)
        if sp >= spec:
            se = (pred & (t == 1)).sum() / max((t == 1).sum(), 1)
            if best is None or se > best[1] or (se == best[1] and th > best[0]):
                best = (th, se)
    return best[0] if best else 1.0


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    for k in ('test-a', 'test-b', 'val-a', 'val-b'):
        ap.add_argument('--' + k, nargs='+', required=(k.startswith('test')))
    ap.add_argument('--name-a', default='A'); ap.add_argument('--name-b', default='B')
    ap.add_argument('--B', type=int, default=10000); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='test_report')
    a = ap.parse_args()
    TA, TB = load(a.test_a), load(a.test_b)
    cross_check(TA, TB, 'test')
    N = len(TA[0]); allidx = np.arange(N)
    rep = {}

    # per-seed metrics
    rows = []
    for nm, ds in ((a.name_a, TA), (a.name_b, TB)):
        for i, d in enumerate(ds):
            m = pooled_metrics([d], allidx); m.update(arm=nm, run=i + 1); rows.append(m)
    ps = pd.DataFrame(rows); ps.to_csv(a.out + '_per_seed.csv', index=False)
    summ = ps.groupby('arm').agg(['mean', 'std']).T
    summ.to_csv(a.out + '_per_seed_mean_sd.csv')

    # pooled point estimates + paired image-level bootstrap
    pa, pb = pooled_metrics(TA, allidx), pooled_metrics(TB, allidx)
    keys = list(pa.keys())
    # fixed operating thresholds (from validation) for the paired sensitivity/specificity CIs
    TH = None
    if a.val_a and a.val_b:
        VA0, VB0 = load(a.val_a), load(a.val_b)
        TH = ([{c: threshold_90(v, c) for c in R} for v in VA0], [{c: threshold_90(v, c) for c in R} for v in VB0])
    def se_sp(ds, ths, idx):
        out = {}
        for c in R:
            se, sp = [], []
            for d, th in zip(ds, ths):
                y = d['target_' + c].values[idx]; pr = d['p_' + c].values[idx] >= th[c]
                se.append((pr & (y == 1)).sum() / max((y == 1).sum(), 1))
                sp.append(((~pr) & (y == 0)).sum() / max((y == 0).sum(), 1))
            out[c + ' sensitivity'] = np.mean(se); out[c + ' specificity'] = np.mean(sp)
        return out
    rng = np.random.default_rng(a.seed); D = {k: [] for k in keys}
    PS = {k: [] for k in keys}; OP = {}
    for _ in range(a.B):
        idx = rng.integers(0, N, N)
        if min(TA[0]['target_' + c].values[idx].sum() for c in L) == 0: continue
        ma, mb = pooled_metrics(TA, idx), pooled_metrics(TB, idx)
        for k in keys: D[k].append(mb[k] - ma[k])
        # sensitivity analysis: metric recomputed within each seed, then averaged over seeds
        sa = [pooled_metrics([d], idx) for d in TA]; sb = [pooled_metrics([d], idx) for d in TB]
        for k in keys: PS[k].append(np.nanmean([m[k] for m in sb]) - np.nanmean([m[k] for m in sa]))
        if TH:
            oa, ob = se_sp(TA, TH[0], idx), se_sp(TB, TH[1], idx)
            for k in oa: OP.setdefault(k, []).append(ob[k] - oa[k])
    rows = []
    for k in keys:
        d = np.array(D[k]); lo, hi = np.percentile(d, [2.5, 97.5])
        pv = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        rows.append(dict(metric=k, **{a.name_a: pa[k], a.name_b: pb[k]}, diff=pb[k] - pa[k],
                         ci_lo=lo, ci_hi=hi, p_raw=pv))
    pool = pd.DataFrame(rows); pool['p_holm'] = holm(pool.p_raw.values)
    pool.to_csv(a.out + '_pooled_paired.csv', index=False)

    # estimand sensitivity: mean over seeds of per-seed metrics (individual-model performance)
    rows = []
    for k in keys:
        d = np.array(PS[k]); lo, hi = np.percentile(d, [2.5, 97.5])
        ma_ = np.nanmean([pooled_metrics([x], allidx)[k] for x in TA]); mb_ = np.nanmean([pooled_metrics([x], allidx)[k] for x in TB])
        rows.append(dict(metric=k, **{a.name_a: ma_, a.name_b: mb_}, diff=mb_ - ma_, ci_lo=lo, ci_hi=hi,
                         p_raw=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    psd = pd.DataFrame(rows); psd['p_holm'] = holm(psd.p_raw.values)
    psd.to_csv(a.out + '_perseed_mean_paired.csv', index=False)
    if OP:
        rows = []
        for k, v in OP.items():
            v = np.array(v); lo, hi = np.percentile(v, [2.5, 97.5])
            rows.append(dict(quantity=k, mean_diff_over_seeds=np.mean(v), ci_lo=lo, ci_hi=hi))
        pd.DataFrame(rows).to_csv(a.out + '_operating_points_paired_ci.csv', index=False)

    # agreement on the same (test) predictions
    rows = []
    for nm, ds in ((a.name_a, TA), (a.name_b, TB)):
        for i, d in enumerate(ds):
            r = agreement(d); r.update(arm=nm, run=i + 1); rows.append(r)
    pd.DataFrame(rows).to_csv(a.out + '_agreement.csv', index=False)

    # operating points (thresholds from validation, applied to test)
    if a.val_a and a.val_b:
        VA, VB = load(a.val_a), load(a.val_b); rows = []
        cross_check(VA, VB, 'validation')
        assert len(VA) == len(TA) and len(VB) == len(TB), 'validation/test seed counts differ'
        for nm, vs, ts in ((a.name_a, VA, TA), (a.name_b, VB, TB)):
            for i, (v, t) in enumerate(zip(vs, ts)):
                for c in R:
                    th = threshold_90(v, c); y = t['target_' + c].values; pr = t['p_' + c].values >= th
                    rows.append(dict(arm=nm, run=i + 1, lesion=c, threshold=th,
                                     test_sensitivity=(pr & (y == 1)).sum() / max((y == 1).sum(), 1),
                                     test_specificity=((~pr) & (y == 0)).sum() / max((y == 0).sum(), 1),
                                     n_pos=int((y == 1).sum())))
        pd.DataFrame(rows).to_csv(a.out + '_operating_points.csv', index=False)

    pd.set_option('display.width', 200, 'display.max_columns', 30)
    print('Holm family:', ', '.join(keys), f'(B={a.B} valid replicates={len(D[keys[0]])})')
    print(pool.round(4).to_string(index=False))
    print('\nSensitivity analysis (mean of per-seed metrics, paired image bootstrap):'); print(psd.round(4).to_string(index=False))
    print('\nPer-seed (mean, SD):'); print(summ.round(4).to_string())
    print('\nFiles written with prefix', a.out)


if __name__ == '__main__':
    main()
