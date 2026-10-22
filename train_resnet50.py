"""Grade-only ResNet-50 baseline. Default run is a smoke test, NOT a result.

Place beside mmrdr_dataset.py. No test manifest is read. Full mode is explicit.
Training recipe is provisional until smoke testing and protocol review finish.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', required=True, type=Path)
    p.add_argument('--val-manifest', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--data-root', type=Path, default=Path(r'C:\Datasets\MMRDR'))
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.image_size) < 1 or a.workers < 0 or a.lr <= 0:
        p.error('Invalid numeric argument')
    return a


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(path, expected, data_root):
    """Fail closed on labels, paths, duplicates, split and checksum mismatch."""
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    modalities = {r['modality'] for r in rows}
    if len(modalities) != 1 or not modalities <= {'CFP', 'UWF'}:
        raise ValueError('Expected a single CFP or UWF modality')
    modality = next(iter(modalities))
    names = set()
    for r in rows:
        name = r['image'].strip().replace('\\', '/')
        if r['split'] != expected or not name.startswith('img/tr'):
            raise ValueError(f'Wrong split or non-training-pool image: {name}')
        if name in names or '..' in Path(name).parts or Path(name).is_absolute():
            raise ValueError(f'Duplicate or unsafe path: {name}')
        names.add(name)
        if r['grade'] not in {'0', '1', '2', '3', '4'}:
            raise ValueError(f'Invalid grade: {name}')
        for label in ('MA', 'HE', 'IH', 'VB_IRMA', 'NV', 'VH', 'RD', 'lr'):
            if r[label] not in {'0', '1'}:
                raise ValueError(f'Invalid {label}: {name}')
        if not (data_root / ('MMRDR-' + modality) / name).is_file():
            raise FileNotFoundError(name)
    sums = path.parent / 'SHA256SUMS.txt'
    if not sums.is_file():
        raise FileNotFoundError(f'Required frozen hash record: {sums}')
    entries = dict((line.split(maxsplit=1)[1].strip(), line.split()[0].lower())
                   for line in sums.read_text(encoding='utf-8-sig').splitlines() if line.strip())
    if entries.get(path.name) != sha(path):
        raise ValueError(f'Manifest hash differs from frozen record: {path}')
    return modality, names


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def safe_amp_step(model, optimizer, scaler):
    """Unscale once; let GradScaler skip nonfinite element gradients.

    Compute the global norm in float64 to distinguish norm-reduction overflow
    from actual nonfinite gradient elements. Never clip invalid gradients.
    """
    import torch
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        raise RuntimeError('No gradients: model is disconnected from loss')
    finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
    old_scale = scaler.get_scale()
    norm = None
    if finite:
        norm_tensor = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(g.detach().double()) for g in grads
        ]))
        if not torch.isfinite(norm_tensor):
            raise RuntimeError('Nonfinite float64 norm despite finite gradient elements')
        norm = float(norm_tensor)
        coefficient = min(1.0, 1.0 / (norm + 1e-6))
        with torch.no_grad():
            for grad in grads:
                grad.mul_(coefficient)
    # unscale_ recorded inf/nan status; step skips optimizer.step on overflow.
    scaler.step(optimizer)
    scaler.update()
    return finite, old_scale, scaler.get_scale(), norm


def main():
    a = arguments()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    import torchvision
    from sklearn.metrics import cohen_kappa_score, f1_score
    from torch.utils.data import DataLoader
    from torchvision.models import resnet50, ResNet50_Weights
    from mmrdr_dataset import MMRDRDataset, build_transform

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this training run')
    tm, tn = validate_manifest(a.train_manifest, 'train', a.data_root)
    vm, vn = validate_manifest(a.val_manifest, 'val', a.data_root)
    if tm != vm or tn & vn:
        raise ValueError('Modality mismatch or train/validation overlap')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    roots = {m: a.data_root / ('MMRDR-' + m) for m in ('CFP', 'UWF')}
    train = MMRDRDataset(a.train_manifest, roots, build_transform(a.image_size, True))
    val = MMRDRDataset(a.val_manifest, roots, build_transform(a.image_size, False))
    # Strip surrounding CSV path whitespace consistently with validation.
    for ds in (train, val):
        ds.frame['image'] = ds.frame['image'].str.strip()
    loader_args = dict(batch_size=a.batch_size, num_workers=a.workers,
                       pin_memory=True, worker_init_fn=seed_worker)
    train_loader = DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(a.seed), **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(weights='ResNet50_Weights.IMAGENET1K_V2', torch=torch.__version__,
                  torchvision=torchvision.__version__, python=sys.version,
                  gpu=torch.cuda.get_device_name(0), train_sha256=sha(a.train_manifest),
                  val_sha256=sha(a.val_manifest), script_sha256=sha(__file__),
                  loader_sha256=sha(Path(__file__).with_name('mmrdr_dataset.py')),
                  criterion='unweighted cross entropy', checkpoint_selection='validation QWK',
                  test_access=False, amp='float16', deterministic_algorithms=True,
                  overflow_policy='skip invalid gradients, float64 norm, stop at 5 consecutive or 20 per epoch',
                  script_version='amp-fix-v2')
    (a.out_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Loading ImageNet V2 weights (downloads on first use)...', flush=True)
    # Torchvision raises on failed download/load; never silently use random weights.
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats()
    best = -math.inf
    consecutive_overflows = 0
    epochs = 1 if a.mode == 'smoke' else a.epochs
    print(f'{a.mode.upper()}: train={len(train)}, val={len(val)}, test=NOT LOADED', flush=True)
    with (a.out_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        log = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_qwk', 'val_macro_f1', 'val_n', 'peak_allocated_gb', 'successful_updates', 'skipped_updates', 'amp_scale'])
        log.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            total, count = 0., 0
            successful_updates, skipped_updates = 0, 0
            for step, batch in enumerate(train_loader, 1):
                x, y = batch['image'].cuda(non_blocking=True), batch['grade'].cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss).backward()
                updated, old_scale, new_scale, grad_norm = safe_amp_step(model, optimizer, scaler)
                if updated:
                    successful_updates += 1
                    consecutive_overflows = 0
                else:
                    skipped_updates += 1
                    consecutive_overflows += 1
                    event = dict(epoch=epoch, batch=step, old_scale=old_scale,
                                 new_scale=new_scale, paths=batch['path'])
                    with (a.out_dir / 'amp_overflows.jsonl').open('a', encoding='utf-8') as ef:
                        ef.write(json.dumps(event) + '\n')
                    print(f'AMP overflow: skipped batch {step}; scale {old_scale} -> {new_scale}', flush=True)
                    if consecutive_overflows >= 5 or skipped_updates >= 20:
                        raise RuntimeError('Persistent AMP overflows; stop and investigate amp_overflows.jsonl')
                # Mean loss includes all forward passes, including skipped updates.
                total += loss.item() * len(y)
                count += len(y)
                if step == 1 or step % 10 == 0:
                    print(f'epoch={epoch} batch={step} loss={total/count:.4f}', flush=True)
                if a.mode == 'smoke' and step >= 50:
                    break
            model.eval()
            if successful_updates == 0:
                raise RuntimeError('No successful optimizer updates this epoch')
            val_total, targets, predictions, records = 0., [], [], []
            with torch.inference_mode():
                for step, batch in enumerate(val_loader, 1):
                    x, y = batch['image'].cuda(non_blocking=True), batch['grade'].cuda(non_blocking=True)
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = model(x)
                        loss = criterion(logits, y)
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite validation output')
                    probabilities = logits.float().softmax(1).cpu().numpy()
                    truth = y.cpu().tolist()
                    pred = probabilities.argmax(1).tolist()
                    targets.extend(truth)
                    predictions.extend(pred)
                    val_total += loss.item() * len(y)
                    for path, label, prediction, probs in zip(batch['path'], truth, pred, probabilities):
                        records.append([path, label, prediction, *probs.tolist()])
                    if a.mode == 'smoke' and step >= 10:
                        break
            qwk = float(cohen_kappa_score(targets, predictions, labels=list(range(5)), weights='quadratic'))
            metrics = dict(epoch=epoch, train_loss=total/count, val_loss=val_total/len(targets),
                           val_qwk=qwk, val_macro_f1=float(f1_score(targets, predictions, labels=list(range(5)), average='macro', zero_division=0)),
                           val_n=len(targets), peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                           successful_updates=successful_updates, skipped_updates=skipped_updates,
                           amp_scale=scaler.get_scale())
            log.writerow(metrics)
            f.flush()
            print(metrics, flush=True)
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                              scaler=scaler.state_dict(), epoch=epoch, config=config, metrics=metrics)
            if a.mode == 'smoke':
                torch.save(checkpoint, a.out_dir / 'smoke_only.pt')
            elif math.isfinite(qwk) and qwk > best:
                best = qwk
                torch.save(checkpoint, a.out_dir / 'best.pt')
                with (a.out_dir / 'best_val_predictions.csv').open('w', newline='', encoding='utf-8') as pf:
                    writer = csv.writer(pf)
                    writer.writerow(['path', 'grade', 'prediction', 'p0', 'p1', 'p2', 'p3', 'p4'])
                    writer.writerows(records)
    print('DONE. Smoke metrics are debugging-only, not publication results.' if a.mode == 'smoke' else 'DONE. Test set remains untouched.', flush=True)


if __name__ == '__main__':
    main()
