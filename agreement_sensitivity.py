#!/usr/bin/env python3
"""Aggregation-sensitivity of grade/lesion disagreement (read-only; CPU; no training).

For each prediction CSV, counts images where the lesion-side indicator and the grade-side
indicator P(grade>0) >= 0.5 fall on opposite sides, under three lesion-side definitions:
  noisyOR   : 1 - prod(1-p_l) >= 0.5                      (the aggregate used in training)
  max       : max_l p_l >= 0.5                            (= binary OR at 0.5)
  thrOR     : any lesion p_l >= its validation-selected threshold (max sensitivity s.t. val spec >= 0.90)
Also prints recognition metrics are NOT recomputed here (unchanged by the definition).
Usage: python agreement_sensitivity.py --pred NAME=file.csv [NAME=file.csv ...] --val NAME=valfile.csv ... --out prefix
(--val gives the validation file of the SAME run used to pick thresholds; omit to skip thrOR.)
"""
import argparse, numpy as np, pandas as pd
L = ['MA', 'HE', 'IH', 'VB_IRMA', 'NV', 'VH', 'RD']

def thr(v, c, spec=0.90):
    t = v['target_' + c].values; p = v['p_' + c].values; best = None
    for th in np.unique(p):
        pred = p >= th; sp = ((~pred) & (t == 0)).sum() / max((t == 0).sum(), 1)
        if sp >= spec:
            se = (pred & (t == 1)).sum() / max((t == 1).sum(), 1)
            if best is None or se > best[1] or (se == best[1] and th > best[0]): best = (th, se)
    return best[0] if best else 1.0

def main():
    a = argparse.ArgumentParser(); a.add_argument('--pred', nargs='+', required=True)
    a.add_argument('--val', nargs='*', default=[]); a.add_argument('--out', default='agreement_sensitivity')
    a = a.parse_args()
    vals = dict(x.split('=', 1) for x in a.val); rows = []
    for item in a.pred:
        name, f = item.split('=', 1); d = pd.read_csv(f)
        P = np.stack([d['p_' + c].values for c in L], 1); pdr = (1 - d.g0.values) >= 0.5
        r = {'run': name, 'n': len(d)}
        for k, side in (('noisyOR', 1 - np.prod(1 - P, 1) >= 0.5), ('max', P.max(1) >= 0.5)):
            r['dis_' + k] = int((side != pdr).sum())
            r['dir_' + k + '_lesion_pos_grade_neg'] = int((side & ~pdr).sum())
            r['dir_' + k + '_lesion_neg_grade_pos'] = int((~side & pdr).sum())
        if name in vals:
            v = pd.read_csv(vals[name]); th = np.array([thr(v, c) for c in L])
            r['dis_thrOR'] = int((((P >= th).any(1)) != pdr).sum())
        rows.append(r)
    out = pd.DataFrame(rows); out.to_csv(a.out + '.csv', index=False); print(out.to_string(index=False))

if __name__ == '__main__': main()
