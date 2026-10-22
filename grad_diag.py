"""Gradient-direction diagnostic for the grade / lesion / consistency objectives.

READ-ONLY: no optimizer step, no weights are changed, the TEST manifest is never read.
It loads saved checkpoints, draws batches from the VALIDATION (default) or TRAINING manifest,
and measures on the shared parameters (encoder + FPN + shared layer; task heads excluded):
  * cosine similarity between pairs of loss gradients (negative = local opposition),
  * gradient norms (magnitude ratio between objectives).
Losses use the same definitions and normalisation as train_uwf_multitask_used.py.
Exploratory analysis. A negative cosine is a local property of one batch at one checkpoint;
it does not by itself show that the consistency term lowers recognition.

Example (Windows, from the folder that contains the training script):
  python grad_diag.py ^
    --train-script train_uwf_multitask_used.py ^
    --train-manifest path\\to\\uwf_train.csv --val-manifest path\\to\\uwf_val.csv ^
    --ckpt baseline42_best=runs\\uwf_mt_asl_seed42\\best.pt ^
    --ckpt glc010_42_best=runs\\uwf_mt_glc010_seed42\\best.pt ^
    --out-dir grad_diag_out
"""
import argparse, csv, importlib.util, json, math, random, sys
from pathlib import Path


def load_train_module(path):
    spec = importlib.util.spec_from_file_location('trainmod', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['trainmod'] = mod
    spec.loader.exec_module(mod)           # main() is guarded by __name__, so nothing runs
    return mod


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-script', type=Path, required=True)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--val-manifest', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'train'], default='val',
                   help='which manifest to draw batches from (test is never used)')
    p.add_argument('--ckpt', action='append', required=True, metavar='LABEL=PATH')
    p.add_argument('--image-root', type=Path, default=Path(r'C:\Datasets\MMRDR\MMRDR-UWF'))
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--n-batches', type=int, default=100, help='batches per sampling mode')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--amp', action='store_true', help='bf16 autocast (default: fp32 for clean gradients)')
    p.add_argument('--resume', action='store_true', help='continue from grad_diag_per_batch.csv already in --out-dir')
    p.add_argument('--out-dir', type=Path, required=True)
    return p.parse_args()


def main():
    a = parse()
    t = load_train_module(a.train_script)
    t.load_training_components.__globals__  # components are loaded lazily below
    Images, Model = t.load_training_components()
    import numpy as np, torch
    from torch.nn import functional as F
    LES, BAD = t.LESIONS, t.BAD_IMAGE
    if not torch.cuda.is_available():
        print('WARNING: CUDA not available; running on CPU (slow).')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr, _ = t.read_manifest(a.train_manifest, 'train')     # hash-checked frozen manifests
    va, _ = t.read_manifest(a.val_manifest, 'val')
    rows = va if a.split == 'val' else tr
    for r in rows:
        if not (a.image_root / r['image']).is_file():
            raise FileNotFoundError(a.image_root / r['image'])
    ds = Images(rows, a.image_root, False)                  # no augmentation: deterministic inputs

    # rare-positive pool: any of VB/IRMA, NV, VH, RD (the evaluation group E)
    rare_idx = [i for i, r in enumerate(rows) if any(r[c] == '1' for c in ('VB_IRMA', 'NV', 'VH', 'RD'))]
    all_idx = list(range(len(rows)))
    print(f'{a.split}: {len(rows)} images, {len(rare_idx)} with >=1 rare lesion in E')
    rng = random.Random(a.seed)

    def batches(pool):
        out = []
        for _ in range(a.n_batches):
            out.append(rng.sample(pool, a.batch_size))
        return out
    plans = {'random': batches(all_idx), 'rare_positive': batches(rare_idx)}   # same batches for every checkpoint

    def js(x, y):
        x, y = x.clamp(1e-6, 1 - 1e-6), y.clamp(1e-6, 1 - 1e-6)
        m = (x + y) / 2
        kl = lambda z: z * (z.log() - m.log()) + (1 - z) * (torch.log1p(-z) - torch.log1p(-m))
        return (kl(x) + kl(y)) / 2

    def losses(g, l, y, lab, mask, ok):
        n = len(y); known = max(float(mask.sum()), 1.0); elig = max(float(ok.sum()), 1.0)
        el = t.lesion_elements(l, lab, 'asl') * mask             # per-element ASL, masked
        gp, lp = g.softmax(1), l.sigmoid()
        a1 = 1 - (1 - lp).prod(1)                                 # any-lesion noisy-OR
        a2 = 1 - (1 - lp[:, 4:7]).prod(1)                         # NV/VH/RD noisy-OR
        return {
            'grade': F.cross_entropy(g, y, reduction='sum') / n,
            'lesion_all': el.sum() / known,
            'lesion_rare': el[:, 3:7].sum() / known,              # VB/IRMA, NV, VH, RD
            'lesion_common': el[:, 0:3].sum() / known,            # MA, HE, IH
            'cons_any': (js(a1, 1 - gp[:, 0]) * ok).sum() / elig, # R1
            'cons_adv': (js(a2, gp[:, 4]) * ok).sum() / elig,     # R2
        }

    PAIRS = [('grade', 'lesion_all'), ('cons_any', 'lesion_all'), ('cons_adv', 'lesion_all'),
             ('cons_any', 'lesion_rare'), ('cons_adv', 'lesion_rare'),
             ('cons_any', 'grade'), ('cons_adv', 'grade'), ('grade', 'lesion_rare')]
    NORMS = ['grade', 'lesion_all', 'lesion_rare', 'cons_any', 'cons_adv']

    a.out_dir.mkdir(parents=True, exist_ok=True)
    per_batch, done = [], set()
    csv_path = a.out_dir / 'grad_diag_per_batch.csv'
    if csv_path.exists() and not a.resume:
        raise SystemExit(f'{csv_path} exists. Use --resume to continue it, or choose a new --out-dir.')
    if csv_path.exists():
        with csv_path.open(newline='') as f:
            for r in csv.DictReader(f):
                row = {k: (v if k in ('ckpt', 'mode') else float(v)) for k, v in r.items()}
                row['batch'] = int(row['batch']); per_batch.append(row)
                done.add((row['ckpt'], row['mode'], row['batch']))
        print(f'Resuming: {len(done)} batches already done')
    col_names = (['ckpt', 'mode', 'batch'] + [f'cos[{u},{v}]' for u, v in PAIRS] + [f'norm[{k}]' for k in NORMS])
    out_f = csv_path.open('a', newline='')
    out_w = csv.DictWriter(out_f, fieldnames=col_names)
    if not done:
        out_w.writeheader(); out_f.flush()
    for spec in a.ckpt:
        label, path = spec.split('=', 1)
        state = torch.load(path, map_location='cpu')
        model = Model(pretrained=False).to(device)
        model.load_state_dict(state['model'])
        model.eval()                                              # dropout off; BN stats are frozen anyway
        shared = [p for n, p in model.named_parameters()
                  if not n.startswith(('grade.', 'lesion.'))]     # encoder + FPN + shared layer
        print(f'[{label}] epoch {state.get("epoch")}  shared params: {sum(p.numel() for p in shared):,}')
        for mode, plan in plans.items():
            for bi, idx in enumerate(plan):
                if (label, mode, bi) in done:
                    continue
                items = [ds[i] for i in idx]
                x = torch.stack([it['image'] for it in items]).to(device)
                y = torch.tensor([it['grade'] for it in items]).to(device)
                lab = torch.stack([it['lesions'] for it in items]).to(device)
                mask = torch.stack([it['mask'] for it in items]).to(device)
                ok = torch.tensor([it['eligible'] for it in items]).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=a.amp):
                    g, l = model(x)
                L = losses(g.float(), l.float(), y, lab, mask, ok)
                grads = {}
                for k, v in L.items():
                    gr = torch.autograd.grad(v, shared, retain_graph=True, allow_unused=True)
                    grads[k] = torch.cat([(q if q is not None else torch.zeros_like(p)).flatten().float()
                                          for q, p in zip(gr, shared)])
                row = {'ckpt': label, 'mode': mode, 'batch': bi}
                for u, v in PAIRS:
                    nu, nv = grads[u].norm(), grads[v].norm()
                    row[f'cos[{u},{v}]'] = float((grads[u] @ grads[v]) / (nu * nv)) if nu > 0 and nv > 0 else float('nan')
                for k in NORMS:
                    row[f'norm[{k}]'] = float(grads[k].norm())
                per_batch.append(row); out_w.writerow(row); out_f.flush()
                del grads, L, g, l, x
                if device.type == 'cuda' and bi % 10 == 9:
                    torch.cuda.empty_cache()
                if (bi + 1) % 20 == 0:
                    print(f'  {label} {mode}: {bi + 1}/{len(plan)} batches', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out_f.close()

    # summary: mean, SD, fraction of batches with negative cosine; norm medians
    summ = []
    for ck in sorted({r['ckpt'] for r in per_batch}):
        for mode in plans:
            R = [r for r in per_batch if r['ckpt'] == ck and r['mode'] == mode]
            for u, v in PAIRS:
                c = np.array([r[f'cos[{u},{v}]'] for r in R], dtype=float); c = c[np.isfinite(c)]
                summ.append({'ckpt': ck, 'mode': mode, 'pair': f'{u} vs {v}', 'n': len(c),
                             'mean_cos': float(c.mean()), 'sd_cos': float(c.std(ddof=1)) if len(c) > 1 else float('nan'),
                             'frac_negative': float((c < 0).mean())})
            for k in NORMS:
                z = np.array([r[f'norm[{k}]'] for r in R])
                summ.append({'ckpt': ck, 'mode': mode, 'pair': f'median norm {k}', 'n': len(z),
                             'mean_cos': float(np.median(z)), 'sd_cos': float('nan'), 'frac_negative': float('nan')})
    with (a.out_dir / 'grad_diag_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    for s in summ:
        print(f"{s['ckpt']:<18}{s['mode']:<14}{s['pair']:<30}n={s['n']:<4}mean={s['mean_cos']:+.4f}"
              f"  sd={s['sd_cos']:.4f}  neg={s['frac_negative']:.2f}")

    # colour plot: per pair, per checkpoint (consistency-bearing pairs only)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    palette = ['#2A6FB0', '#D9622B', '#1E9A72', '#7A4FB5', '#D68A00', '#C0392B']
    cks = sorted({r['ckpt'] for r in per_batch})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mode in zip(axes, plans):
        for j, (u, v) in enumerate(PAIRS):
            for i, ck in enumerate(cks):
                c = [r[f'cos[{u},{v}]'] for r in per_batch if r['ckpt'] == ck and r['mode'] == mode]
                xs = j + (i - (len(cks) - 1) / 2) * 0.18 + np.random.RandomState(0).uniform(-.05, .05, len(c))
                ax.scatter(xs, c, s=10, alpha=.55, color=palette[i % len(palette)], label=ck if j == 0 else None)
                ax.hlines(np.nanmean(c), j + (i - (len(cks) - 1) / 2) * 0.18 - .08, j + (i - (len(cks) - 1) / 2) * 0.18 + .08,
                          color=palette[i % len(palette)], lw=2.5)
        ax.axhline(0, color='#555555', lw=1, ls='--')
        ax.set_xticks(range(len(PAIRS))); ax.set_xticklabels([f'{u}\nvs\n{v}' for u, v in PAIRS], fontsize=7)
        ax.set_title(f'{mode} batches ({a.split} set)'); ax.grid(alpha=.25)
    axes[0].set_ylabel('cosine similarity of shared-parameter gradients'); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out_dir / 'grad_diag.png', dpi=200)
    (a.out_dir / 'grad_diag_config.json').write_text(json.dumps(
        {k: str(v) for k, v in vars(a).items()}, indent=2))
    print('Done. Outputs in', a.out_dir)


if __name__ == '__main__':
    main()
