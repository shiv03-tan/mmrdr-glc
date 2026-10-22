r"""MMRDR UWF multitask study v1 -- CONSISTENCY-TERM ABLATION (+ RESUME).

Identical to the original train_uwf_multitask_used.py (original 5-epoch consistency warmup,
same losses, schedule, checkpoint rule, manifest hash locks) except for ONE new flag:
  --consistency-terms both      original loss (default; reproduces the original behaviour)
  --consistency-terms any       only JS(any-lesion aggregate, P(grade>0))
  --consistency-terms advanced  only JS(NV/VH/RD aggregate, P(grade 4))
Plus --resume (same mechanism as the diagnostic resume script; RNG state is not stored).

USAGE (any-lesion-only ablation, seed 42, matched to glc010_seed42)
python scripts\train_uwf_multitask_ablation_resume.py ^
    --train-manifest splits\uwf_train.csv --val-manifest splits\uwf_val.csv ^
    --image-root C:\Datasets\MMRDR\MMRDR-UWF --mode full --epochs 60 ^
    --batch-size 4 --accumulation 4 --seed 42 --lesion-loss asl ^
    --consistency-weight 0.10 --consistency-terms any ^
    --out-dir runs\uwf_mt_asl_glc010_anyonly_seed42
To resume after an interruption, run the same command with --resume added.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

LESIONS = ('MA', 'HE', 'IH', 'VB_IRMA', 'NV', 'VH', 'RD')
BAD_IMAGE = 'img/tr000215.jpg'
HASHES = {
    'train': 'a88ae180b78e2b59fae104cef5108c488f0a1757cbabccd378165710ec4d1511',
    'val': 'dd5462a6a118bffef516398dc4aea1fd4f06456e761d55e580118a017d04b316',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_manifest(path, split):
    if sha(path) != HASHES[split]:
        raise ValueError(f'{split} manifest differs from the verified frozen file: {path}')
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for r in rows:
        name = r['image'].strip().replace('\\', '/')
        if r['modality'] != 'UWF' or r['split'] != split:
            raise ValueError('Wrong modality or split')
        if not name.startswith('img/tr') or '..' in name or name in seen:
            raise ValueError(f'Invalid or duplicate path: {name}')
        if r['grade'] not in ('0', '1', '2', '3', '4'):
            raise ValueError('Invalid grade')
        if any(r[k] not in ('0', '1') for k in (*LESIONS, 'lr')):
            raise ValueError('Nonbinary lesion/laterality')
        r['image'] = name
        seen.add(name)
    return rows, seen

def audit(train_path, val_path):
    tr, tn = read_manifest(train_path, 'train')
    va, vn = read_manifest(val_path, 'val')
    if tn & vn or len(tr) != 6632 or len(va) != 1175:
        raise ValueError('Split overlap or unexpected count')
    if sum(r['image'] == BAD_IMAGE for r in tr) != 1 or BAD_IMAGE in vn:
        raise ValueError('Unexpected uncertain-label membership')
    report = {'train_n': len(tr), 'val_n': len(va), 'image_overlap': 0,
              'uncertain_record': BAD_IMAGE, 'uncertain_target': 'IH',
              'policy': 'mask IH; exclude this record from consistency; retain grade and six other targets',
              'test_access': False, 'patient_disjointness': 'not independently verifiable'}
    for split, rows in [('train', tr), ('val', va)]:
        report[split + '_grades'] = {str(g): sum(int(r['grade']) == g for r in rows) for g in range(5)}
        report[split + '_positives_available'] = {
            c: sum(int(r[c]) for r in rows if not (r['image'] == BAD_IMAGE and c == 'IH'))
            for c in LESIONS}
    eligible = [r for r in tr if r['image'] != BAD_IMAGE]
    report['training_label_disagreements_excluding_uncertain_record'] = {
        'grade0_any_lesion': sum(int(r['grade']) == 0 and any(int(r[c]) for c in LESIONS) for r in eligible),
        'grade_positive_no_lesion': sum(int(r['grade']) > 0 and not any(int(r[c]) for c in LESIONS) for r in eligible),
        'below4_advanced': sum(int(r['grade']) < 4 and any(int(r[c]) for c in LESIONS[4:]) for r in eligible),
        'grade4_no_advanced': sum(int(r['grade']) == 4 and not any(int(r[c]) for c in LESIONS[4:]) for r in eligible),
    }
    return tr, va, report

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--val-manifest', type=Path, required=True)
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--image-root', type=Path, default=Path(r'C:\Datasets\MMRDR\MMRDR-UWF'))
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lesion-loss', choices=['asl', 'bce'], default='asl')
    p.add_argument('--consistency-weight', type=float, default=0.0)
    p.add_argument('--precision', choices=['auto', 'fp32'], default='auto')
    p.add_argument('--consistency-terms', choices=['both', 'any', 'advanced'], default='both',
                   help="both = original loss; any = any-lesion vs grade>0 term only; advanced = NV/VH/RD vs grade 4 term only")
    p.add_argument('--resume', action='store_true', help='continue from OUT_DIR/last.pt')
    a = p.parse_args()
    if min(a.batch_size, a.accumulation) < 1 or a.epochs <= 5 or a.workers < 0:
        p.error('Positive batch/accumulation, epochs > 5 and workers >= 0 required')
    if not math.isfinite(a.consistency_weight) or a.consistency_weight < 0:
        p.error('Consistency weight must be finite and nonnegative')
    if not a.audit_only and a.out_dir is None:
        p.error('--out-dir is required for smoke/full training')
    return a

# Training definitions are loaded lazily so the CSV audit needs only Python.
def load_training_components():
    global torch, np, nn, F
    import torch
    import numpy as np
    from torch import nn
    from torch.nn import functional as F
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    from torchvision.models import resnet50, ResNet50_Weights

    class Square:
        def __call__(self, im):
            w, h = im.size
            s = max(w, h)
            out = Image.new('RGB', (s, s), (0, 0, 0))
            out.paste(im, ((s-w)//2, (s-h)//2))
            return out

    class Images(torch.utils.data.Dataset):
        def __init__(self, rows, root, training):
            self.rows, self.root = rows, root
            ops = []
            if training:
                ops = [T.RandomHorizontalFlip(0.5),
                       T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR, expand=True, fill=0),
                       T.ColorJitter(brightness=.1, contrast=.1, saturation=.1)]
            self.transform = T.Compose(ops + [Square(),
                T.Resize((512, 512), interpolation=InterpolationMode.BILINEAR), T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225))])

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            with Image.open(self.root / r['image']) as im:
                x = self.transform(im.convert('RGB'))
            mask = torch.ones(7)
            ok = r['image'] != BAD_IMAGE
            if not ok:
                mask[2] = 0
            return {'image': x, 'grade': int(r['grade']),
                    'lesions': torch.tensor([float(r[c]) for c in LESIONS]),
                    'mask': mask, 'eligible': float(ok), 'path': r['image']}

    class Model(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.encoder.fc = nn.Identity()
            self.lateral = nn.ModuleList(nn.Conv2d(c, 256, 1) for c in (256, 512, 1024, 2048))
            self.smooth = nn.ModuleList(nn.Conv2d(256, 256, 3, padding=1) for _ in range(4))
            self.shared = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(.2))
            self.grade = nn.Linear(512, 5)
            self.lesion = nn.Linear(512, 7)

        def train(self, mode=True):
            super().train(mode)
            # Freeze running statistics but keep affine parameters trainable.
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            return self

        def forward(self, x):
            e = self.encoder
            x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
            maps = []
            for layer in (e.layer1, e.layer2, e.layer3, e.layer4):
                x = layer(x)
                maps.append(x)
            pyramid = [l(c) for l, c in zip(self.lateral, maps)]
            for j in (2, 1, 0):
                pyramid[j] = pyramid[j] + F.interpolate(pyramid[j+1], size=pyramid[j].shape[-2:], mode='nearest')
            pooled = [F.adaptive_avg_pool2d(s(p), 1).flatten(1) for s, p in zip(self.smooth, pyramid)]
            h = self.shared(torch.cat(pooled, dim=1))
            return self.grade(h), self.lesion(h)

    # Local classes cannot be pickled by Windows worker spawning.
    # This standalone v1 intentionally requires workers=0.
    return Images, Model

def lesion_elements(logits, targets, kind):
    if kind == 'bce':
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = logits.sigmoid()
    pm = (p - .05).clamp(min=0)
    # gamma_positive=0, gamma_negative=4, detached focusing factor.
    return -targets * F.logsigmoid(logits) - (1-targets) * pm.detach().pow(4) * torch.log1p(-pm.clamp(max=1-1e-6))

def consistency(g, l, terms='both'):
    gp, lp = g.softmax(1), l.sigmoid()
    a = 1 - (1-lp).prod(1)
    b = 1 - (1-lp[:, 4:7]).prod(1)
    def js(x, y):
        x, y = x.clamp(1e-6, 1-1e-6), y.clamp(1e-6, 1-1e-6)
        m = (x+y)/2
        def kl(z):
            return z*(z.log()-m.log()) + (1-z)*(torch.log1p(-z)-torch.log1p(-m))
        return (kl(x)+kl(y))/2
    any_term, adv_term = js(a, 1-gp[:, 0]), js(b, gp[:, 4])
    if terms == 'any':
        return any_term
    if terms == 'advanced':
        return adv_term
    return any_term + adv_term

def lr_factor(progress, epochs):
    if progress < 5:
        return .1 + .9 * progress/5
    t = min(1., (progress-5)/(epochs-5))
    return .01 + .99 * .5 * (1 + math.cos(math.pi*t))

def atomic_save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)

def evaluate(model, loader, device, amp):
    from sklearn.metrics import cohen_kappa_score, f1_score, average_precision_score
    model.eval()
    ys, ls, gps, lps, paths = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            x = batch['image'].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                g, l = model(x)
            g, l = g.float(), l.float()
            if not torch.isfinite(g).all() or not torch.isfinite(l).all():
                raise RuntimeError('Nonfinite validation logits')
            gps.append(g.softmax(1).cpu().numpy()); lps.append(l.sigmoid().cpu().numpy())
            ys.extend(batch['grade'].tolist()); ls.append(batch['lesions'].numpy()); paths.extend(batch['path'])
    gp, lp, y, lab = np.concatenate(gps), np.concatenate(lps), np.array(ys), np.concatenate(ls)
    ap = [float(average_precision_score(lab[:, j], lp[:, j])) if len(np.unique(lab[:, j])) == 2 else None for j in range(7)]
    if any(v is None for v in ap):
        raise RuntimeError('Full validation must contain both classes for each lesion')
    qwk = float(cohen_kappa_score(y, gp.argmax(1), labels=list(range(5)), weights='quadratic'))
    rare = float(np.mean(ap[3:]))
    metrics = {'val_n': len(y), 'val_qwk': qwk,
               'val_macro_f1': float(f1_score(y, gp.argmax(1), labels=list(range(5)), average='macro', zero_division=0)),
               'val_rare_ap': rare, 'val_all_ap': float(np.mean(ap)),
               'selection_score': .5*((qwk+1)/2) + .5*rare}
    metrics.update({'ap_'+c: v for c, v in zip(LESIONS, ap)})
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError('Nonfinite validation metric')
    return metrics, (paths, y, lab, gp, lp)

def save_predictions(path, data):
    paths, y, lab, gp, lp = data
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'grade', 'prediction', *['g'+str(i) for i in range(5)],
                    *['target_'+c for c in LESIONS], *['p_'+c for c in LESIONS]])
        for i, name in enumerate(paths):
            w.writerow([name, int(y[i]), int(gp[i].argmax()), *gp[i].tolist(), *lab[i].tolist(), *lp[i].tolist()])

def load_resume_state(out_dir, args):
    """Torch-free checks + history reload. Returns (history_rows, start_epoch, best_score)."""
    for name in ('last.pt', 'history.csv', 'config.json'):
        if not (out_dir / name).is_file():
            raise FileNotFoundError(f'Cannot resume: {out_dir / name} is missing')
    if (out_dir / 'completed.json').exists():
        raise RuntimeError('This run already has completed.json; nothing to resume')
    saved = json.loads((out_dir / 'config.json').read_text(encoding='utf-8'))
    if saved.get('consistency_terms', 'both') != args.consistency_terms:
        raise ValueError('Resume flag mismatch for --consistency-terms')
    for key in ('seed', 'consistency_weight', 'lesion_loss', 'batch_size', 'accumulation', 'epochs'):
        if saved.get(key) != getattr(args, key):
            raise ValueError(f'Resume flag mismatch for --{key.replace("_", "-")}: run used {saved.get(key)!r}, now {getattr(args, key)!r}')
    if saved.get('train_sha256') != sha(args.train_manifest) or saved.get('val_sha256') != sha(args.val_manifest):
        raise ValueError('Manifests differ from the original run')
    with (out_dir / 'history.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    history = [{k: (int(v) if k in ('val_n', 'epoch', 'successful_updates') else float(v)) for k, v in r.items()} for r in rows]
    if [h['epoch'] for h in history] != list(range(1, len(history) + 1)):
        raise ValueError('history.csv epochs are not 1..N contiguous')
    best = max(h['selection_score'] for h in history)
    return history, len(history) + 1, best

def main():
    args = arguments()
    tr, va, report = audit(args.train_manifest, args.val_manifest)
    print(json.dumps(report, indent=2), flush=True)
    if args.audit_only:
        return
    if args.workers != 0:
        raise ValueError('This Windows-compatible standalone version requires --workers 0')
    for r in tr + va:
        if not (args.image_root / r['image']).is_file():
            raise FileNotFoundError(args.image_root / r['image'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    Images, Model = load_training_components()
    import torchvision
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for study training; audit-only works without CUDA')
    device = torch.device('cuda')
    amp = args.precision == 'auto' and torch.cuda.is_bf16_supported()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    resumed = None
    if args.resume:
        resumed = load_resume_state(args.out_dir, args)
        print(f'RESUME: found {len(resumed[0])} finished epochs; will continue at epoch {resumed[1]}.', flush=True)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(version='uwf-multitask-v1-ablation-'+args.consistency_terms, image_size=512, lesions=LESIONS, mask_policy=report['policy'],
        train_sha256=sha(args.train_manifest), val_sha256=sha(args.val_manifest), script_sha256=sha(__file__),
        torch=str(torch.__version__), torchvision=str(torchvision.__version__), python=sys.version,
        gpu=torch.cuda.get_device_name(0), precision_used='bf16' if amp else 'fp32', test_access=False,
        encoder_lr=1e-5, new_layers_lr=1e-4, weight_decay=.01, bn_running_statistics='frozen',
        initialization='ImageNet V2 ResNet50; new FPN and heads', checkpoint_rule='0.5*((QWK+1)/2)+0.5*rare_AP',
        augmentation='horizontal flip .5; rotation +/-15 expand; jitter .1; square pad; bilinear512; ImageNet norm',
        optimizer='AdamW betas=(.9,.999), eps=1e-8, no decay on biases/norm', schedule='5 epoch LR warmup then cosine',
        consistency_warmup_epochs=5, consistency_terms=args.consistency_terms,
        lesion_coefficient=1, asl_gamma_pos=0, asl_gamma_neg=4, asl_margin=.05,
        selection_notice='smoke outputs are debugging-only' if args.mode=='smoke' else 'validation selected; not test results')
    if not args.resume:
        (args.out_dir/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (args.out_dir/'manifest_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        shutil.copyfile(__file__, args.out_dir/'train_uwf_multitask_ablation_used.py')
    else:
        shutil.copyfile(__file__, args.out_dir/'train_uwf_multitask_ablation_resume_used.py')
    print('Loading pretrained ResNet-50. No fallback to random weights.', flush=True)
    model = Model().to(device)
    groups = {}
    for name, p in model.named_parameters():
        lr = 1e-5 if name.startswith('encoder.') else 1e-4
        decay = .01 if p.ndim > 1 else 0.
        groups.setdefault((lr, decay), []).append(p)
    optimizer = torch.optim.AdamW([{'params': ps, 'lr': lr, 'base_lr': lr, 'weight_decay': wd}
                                  for (lr, wd), ps in groups.items()], betas=(.9,.999), eps=1e-8)
    generator = torch.Generator().manual_seed(args.seed)
    train = torch.utils.data.DataLoader(Images(tr, args.image_root, True), batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    val = torch.utils.data.DataLoader(Images(va, args.image_root, False), batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True)
    print(f'{args.mode.upper()} train={len(tr)} val={len(va)} test=NOT LOADED precision={config["precision_used"]}', flush=True)
    print('Smoke: first 20 shuffled training batches; FULL validation for meaningful AP computation.', flush=True)
    torch.cuda.reset_peak_memory_stats()
    best, start = -math.inf, time.perf_counter()
    history = []
    first_epoch = 1
    if args.resume:
        history, first_epoch, best = resumed
        ckpt = torch.load(args.out_dir/'last.pt', map_location=device, weights_only=False)
        if ckpt['epoch'] != first_epoch - 1:
            raise ValueError(f"last.pt is from epoch {ckpt['epoch']} but history.csv ends at epoch {first_epoch - 1}")
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        del ckpt
        # RNG state is not stored in last.pt: reseed deterministically per resume point.
        reseed = args.seed + 1000 * first_epoch
        random.seed(reseed); np.random.seed(reseed); torch.manual_seed(reseed); torch.cuda.manual_seed_all(reseed)
        generator.manual_seed(reseed)
        start = time.perf_counter() - history[-1]['elapsed_seconds']  # elapsed excludes downtime
        for stale in ('failure.json',):
            if (args.out_dir/stale).exists():
                (args.out_dir/stale).replace(args.out_dir/f'{stale[:-5]}_before_resume_at_epoch{first_epoch}.json')
        log_path = args.out_dir/'resume_log.json'
        log = json.loads(log_path.read_text(encoding='utf-8')) if log_path.exists() else []
        log.append({'resumed_at_epoch': first_epoch, 'unix_time': time.time(), 'rng_state_restored': False,
                    'reseed': reseed, 'script_sha256': sha(__file__), 'torch': str(torch.__version__)})
        log_path.write_text(json.dumps(log, indent=2), encoding='utf-8')
        print(f'RESUMED: model+optimizer restored from epoch {first_epoch - 1}; best selection_score so far {best:.5f}.', flush=True)
    try:
        for epoch in range(first_epoch, (1 if args.mode=='smoke' else args.epochs)+1):
            model.train()
            n_batches = min(20, len(train)) if args.mode=='smoke' else len(train)
            stream = iter(itertools.islice(train, n_batches))
            seen_batches = 0
            sums = [0., 0., 0.]; counts = [0, 0, 0]; updates = 0
            while True:
                window = list(itertools.islice(stream, args.accumulation))
                if not window:
                    break
                n = sum(len(b['grade']) for b in window)
                known = sum(float(b['mask'].sum()) for b in window)
                eligible = sum(float(b['eligible'].sum()) for b in window)
                progress = epoch-1 + seen_batches/n_batches
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * lr_factor(progress, args.epochs)
                weight = args.consistency_weight * min(1., progress/5)  # original 5-epoch warmup (restored)
                optimizer.zero_grad(set_to_none=True)
                for batch in window:
                    x = batch['image'].to(device)
                    y = batch['grade'].to(device)
                    lab, mask, ok = (batch[k].to(device) for k in ('lesions', 'mask', 'eligible'))
                    if not torch.isfinite(x).all():
                        raise RuntimeError('Nonfinite input')
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
                        g, l = model(x)
                    g, l = g.float(), l.float()
                    gl = F.cross_entropy(g, y, reduction='sum')
                    ll = (lesion_elements(l, lab, args.lesion_loss)*mask).sum()
                    cl = (consistency(g, l, args.consistency_terms)*ok).sum() if args.consistency_weight > 0 else g.sum()*0
                    loss = gl/n + ll/known + weight*cl/max(eligible, 1)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    sums = [s+float(v.detach()) for s,v in zip(sums, (gl,ll,cl))]
                    counts = [counts[0]+len(y), counts[1]+int(mask.sum()), counts[2]+int(ok.sum())]
                    seen_batches += 1
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                if updates == 1 or updates % 10 == 0:
                    print(f'epoch={epoch} batch={seen_batches}/{n_batches} grade_loss={sums[0]/counts[0]:.4f} '
                          f'lesion_loss={sums[1]/counts[1]:.4f} grad_norm={float(norm):.3f}', flush=True)
            metrics, data = evaluate(model, val, device, amp)
            metrics.update(epoch=epoch, train_grade_loss=sums[0]/counts[0], train_lesion_loss=sums[1]/counts[1],
                train_consistency_loss=sums[2]/max(counts[2],1), successful_updates=updates,
                elapsed_seconds=time.perf_counter()-start, peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3)
            history.append(metrics)
            with (args.out_dir/'history.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(metrics)); w.writeheader(); w.writerows(history)
            print(json.dumps(metrics), flush=True)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'config': config, 'metrics': metrics}
            if args.mode == 'smoke':
                atomic_save(state, args.out_dir/'smoke_only.pt')
                save_predictions(args.out_dir/'smoke_val_predictions.csv', data)
            else:
                atomic_save(state, args.out_dir/'last.pt')
                if metrics['selection_score'] > best:
                    best = metrics['selection_score']
                    atomic_save(state, args.out_dir/'best.pt')
                    save_predictions(args.out_dir/'best_val_predictions.csv', data)
        (args.out_dir/'completed.json').write_text(json.dumps({'mode':args.mode, 'epochs_completed':len(history), 'test_access':False}), encoding='utf-8')
        print('DONE. Test untouched. '+('SMOKE ONLY; not study results.' if args.mode=='smoke' else 'Full run completed.'), flush=True)
    except Exception as exc:
        (args.out_dir/'failure.json').write_text(json.dumps({'error':repr(exc), 'epoch':epoch, 'test_access':False}), encoding='utf-8')
        raise

if __name__ == '__main__':
    main()
