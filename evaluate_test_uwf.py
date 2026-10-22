"""MMRDR UWF multitask study v1 -- TEST-SET evaluation.

Standalone companion to train_uwf_multitask_used.py. Run this exactly ONCE
per checkpoint, only after all model/hyperparameter decisions are frozen --
per your own manuscript's stated protocol ("Test predictions will be
generated after these choices have been fixed").

What it does, per run directory you point it at:
  1. Verifies the test manifest against a locked sha256 (no silent swap of
     the test file).
  2. Loads that run's best.pt checkpoint, rebuilds the exact same
     ResNet-50+FPN architecture used in training.
  3. Runs inference over the full test set (no augmentation, same
     square-pad + resize(512) + ImageNet-normalize pipeline as validation).
  4. Computes every headline metric: QWK, macro F1, balanced accuracy,
     macro AUROC, macro AP over E (4-label) and R (3-label), per-lesion
     AP/AUROC, and sensitivity @ 90% specificity for NV/VH/RD -- using
     decision thresholds LOCKED ON VALIDATION (read from that run's
     best_val_predictions.csv), never re-derived from test itself.
  5. Saves raw per-sample test predictions (same column layout as
     best_val_predictions.csv, so your existing analysis scripts work on
     it unchanged) and a test_metrics.json per run.
  6. Writes test_access.json = {"test_access": true, ...} in a NEW file,
     without touching the original completed.json (keeps provenance
     clean -- you can always tell training-time state from eval-time state).

Does NOT retrain, does NOT touch train/val data, does NOT alter any
existing checkpoint or log file.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
python evaluate_test_uwf.py ^
    --test-manifest splits\\uwf_test.csv ^
    --image-root C:\\Datasets\\MMRDR\\MMRDR-UWF ^
    --runs runs\\uwf_mt_asl_seed42 runs\\uwf_mt_asl_seed52_resumed_v2 runs\\uwf_mt_asl_seed62 ^
           runs\\uwf_mt_asl_glc010_seed42 runs\\uwf_mt_asl_glc010_seed52 runs\\uwf_mt_asl_glc010_seed62

Each run directory must already contain best.pt (written by training) and
best_val_predictions.csv (used here only to lock the 90%-specificity
thresholds -- never to pick anything from test).

Outputs land in <run_dir>\\test_eval\\:
    test_predictions.csv   -- per-sample, same schema as best_val_predictions.csv
    test_metrics.json      -- every headline metric, single JSON
    test_access.json       -- {"test_access": true, "checkpoint": "best.pt", ...}
--------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

LESIONS = ('MA', 'HE', 'IH', 'VB_IRMA', 'NV', 'VH', 'RD')
E_GROUP_IDX = (3, 4, 5, 6)   # VB_IRMA, NV, VH, RD -- matches train script's ap[3:]
R_GROUP_IDX = (4, 5, 6)      # NV, VH, RD -- matches consistency()'s lp[:, 4:7]

# EDIT THIS if your test manifest's sha256 ever legitimately changes (it
# should not -- a mismatch here means you are pointing at the wrong file).
TEST_SHA256 = "9e868693dd2cce6aa6159526768561fb017994a0fe99c9598acaff9cc79d9101"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_test_manifest(path):
    digest = sha(path)
    if digest != TEST_SHA256:
        raise ValueError(
            f"Test manifest hash mismatch.\n  expected: {TEST_SHA256}\n  got:      {digest}\n"
            "This script refuses to run against an unverified test file -- "
            "if you intentionally changed the test manifest, update TEST_SHA256 "
            "at the top of this script after confirming the change is correct."
        )
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for r in rows:
        name = r['image'].strip().replace('\\', '/')
        if r['modality'] != 'UWF' or r['split'] != 'test':
            raise ValueError('Wrong modality or split in test manifest')
        if not name.startswith('img/') or '..' in name or name in seen:
            raise ValueError(f'Invalid or duplicate path: {name}')
        if r['grade'] not in ('0', '1', '2', '3', '4'):
            raise ValueError('Invalid grade')
        if any(r[k] not in ('0', '1') for k in (*LESIONS, 'lr')):
            raise ValueError('Nonbinary lesion/laterality')
        r['image'] = name
        seen.add(name)
    return rows


def load_training_components():
    global torch, nn, F
    import torch
    from torch import nn
    from torch.nn import functional as F
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    from torchvision.models import resnet50

    class Square:
        def __call__(self, im):
            w, h = im.size
            s = max(w, h)
            out = Image.new('RGB', (s, s), (0, 0, 0))
            out.paste(im, ((s - w) // 2, (s - h) // 2))
            return out

    class Images(torch.utils.data.Dataset):
        """No augmentation -- identical to training script's eval-mode transform."""
        def __init__(self, rows, root):
            self.rows, self.root = rows, root
            self.transform = T.Compose([Square(),
                T.Resize((512, 512), interpolation=InterpolationMode.BILINEAR), T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225))])

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            with Image.open(self.root / r['image']) as im:
                x = self.transform(im.convert('RGB'))
            return {'image': x, 'grade': int(r['grade']),
                    'lesions': torch.tensor([float(r[c]) for c in LESIONS]),
                    'path': r['image']}

    class Model(nn.Module):
        """Must match train_uwf_multitask_used.py's Model exactly -- any
        architectural drift here would silently load a checkpoint into the
        wrong-shaped module and either crash or (worse) partially succeed."""
        def __init__(self):
            super().__init__()
            self.encoder = resnet50(weights=None)
            self.encoder.fc = nn.Identity()
            self.lateral = nn.ModuleList(nn.Conv2d(c, 256, 1) for c in (256, 512, 1024, 2048))
            self.smooth = nn.ModuleList(nn.Conv2d(256, 256, 3, padding=1) for _ in range(4))
            self.shared = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(.2))
            self.grade = nn.Linear(512, 5)
            self.lesion = nn.Linear(512, 7)

        def forward(self, x):
            e = self.encoder
            x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
            maps = []
            for layer in (e.layer1, e.layer2, e.layer3, e.layer4):
                x = layer(x)
                maps.append(x)
            pyramid = [l(c) for l, c in zip(self.lateral, maps)]
            for j in (2, 1, 0):
                pyramid[j] = pyramid[j] + F.interpolate(pyramid[j + 1], size=pyramid[j].shape[-2:], mode='nearest')
            pooled = [F.adaptive_avg_pool2d(s(p), 1).flatten(1) for s, p in zip(self.smooth, pyramid)]
            h = self.shared(torch.cat(pooled, dim=1))
            return self.grade(h), self.lesion(h)

    return Images, Model


def threshold_at_specificity(y_true, y_score, target_spec=0.90):
    """Largest score cutoff achieving at least target_spec specificity on
    the given (validation) labels/scores. Returns (threshold, achieved_spec).
    This is computed ONCE on validation and then frozen for test -- never
    recomputed from test data."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1 - fpr
    eligible = spec >= target_spec
    if not eligible.any():
        # fall back to the highest achievable specificity's threshold
        idx = int(np.argmax(spec))
    else:
        # among thresholds meeting the specificity bar, take the one with
        # highest sensitivity (tpr) -- roc_curve already orders by threshold
        idx = int(np.argmax(np.where(eligible, tpr, -1)))
    return float(thresholds[idx]), float(spec[idx])


def load_val_thresholds(run_dir):
    """Read best_val_predictions.csv and lock the 90%-specificity threshold
    for each of NV, VH, RD."""
    val_path = run_dir / 'best_val_predictions.csv'
    if not val_path.exists():
        raise FileNotFoundError(
            f"{val_path} not found -- needed to lock 90%-specificity thresholds on "
            "validation before touching test. Without it this script cannot compute "
            "Sens@90Spec without leaking test information into threshold selection."
        )
    with val_path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    thresholds = {}
    for lesion in ('NV', 'VH', 'RD'):
        y = np.array([float(r['target_' + lesion]) for r in rows])
        p = np.array([float(r['p_' + lesion]) for r in rows])
        thr, achieved = threshold_at_specificity(y, p, target_spec=0.90)
        thresholds[lesion] = {'threshold': thr, 'val_specificity_achieved': achieved}
    return thresholds


def evaluate_run(run_dir, test_rows, image_root, device):
    from sklearn.metrics import (cohen_kappa_score, f1_score, average_precision_score,
                                  roc_auc_score, balanced_accuracy_score)

    Images, Model = load_training_components()
    ckpt_path = run_dir / 'best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Model().to(device)
    model.load_state_dict(state['model'])
    model.eval()

    loader = torch.utils.data.DataLoader(Images(test_rows, image_root), batch_size=8,
                                          shuffle=False, num_workers=0, pin_memory=True)

    ys, ls, gps, lps, paths = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            x = batch['image'].to(device)
            g, l = model(x)
            g, l = g.float(), l.float()
            if not torch.isfinite(g).all() or not torch.isfinite(l).all():
                raise RuntimeError('Nonfinite test logits')
            gps.append(g.softmax(1).cpu().numpy())
            lps.append(l.sigmoid().cpu().numpy())
            ys.extend(batch['grade'].tolist())
            ls.append(batch['lesions'].numpy())
            paths.extend(batch['path'])
    gp, lp, y, lab = np.concatenate(gps), np.concatenate(lps), np.array(ys), np.concatenate(ls)

    ap = [float(average_precision_score(lab[:, j], lp[:, j])) if len(np.unique(lab[:, j])) == 2 else None
          for j in range(7)]
    auroc = [float(roc_auc_score(lab[:, j], lp[:, j])) if len(np.unique(lab[:, j])) == 2 else None
             for j in range(7)]
    if any(v is None for v in ap) or any(v is None for v in auroc):
        raise RuntimeError('Test set must contain both classes for every lesion to score AP/AUROC')

    qwk = float(cohen_kappa_score(y, gp.argmax(1), labels=list(range(5)), weights='quadratic'))
    macro_f1 = float(f1_score(y, gp.argmax(1), labels=list(range(5)), average='macro', zero_division=0))
    bal_acc = float(balanced_accuracy_score(y, gp.argmax(1)))
    ap_e = float(np.mean([ap[i] for i in E_GROUP_IDX]))
    ap_r = float(np.mean([ap[i] for i in R_GROUP_IDX]))
    all_ap = float(np.mean(ap))
    macro_auroc = float(np.mean(auroc))

    thresholds = load_val_thresholds(run_dir)
    sens = {}
    for lesion, idx in zip(('NV', 'VH', 'RD'), R_GROUP_IDX):
        thr = thresholds[lesion]['threshold']
        y_l = lab[:, idx]
        p_l = lp[:, idx]
        pred_pos = p_l >= thr
        tp = int(((pred_pos) & (y_l == 1)).sum())
        fn = int(((~pred_pos) & (y_l == 1)).sum())
        tn = int(((~pred_pos) & (y_l == 0)).sum())
        fp = int(((pred_pos) & (y_l == 0)).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        sens[lesion] = {'sensitivity': sensitivity, 'specificity_achieved_on_test': specificity,
                         'threshold_locked_from_val': thr,
                         'val_specificity_at_lock_time': thresholds[lesion]['val_specificity_achieved']}
    sens_r_90spec = float(np.mean([sens[l]['sensitivity'] for l in ('NV', 'VH', 'RD')]))

    metrics = {
        'test_n': int(len(y)),
        'test_qwk': qwk,
        'test_macro_f1': macro_f1,
        'test_balanced_accuracy': bal_acc,
        'test_macro_auroc': macro_auroc,
        'test_ap_E': ap_e,
        'test_ap_R': ap_r,
        'test_all_ap': all_ap,
        'test_sens_at_90spec_R': sens_r_90spec,
    }
    metrics.update({f'ap_{c}': v for c, v in zip(LESIONS, ap)})
    metrics.update({f'auroc_{c}': v for c, v in zip(LESIONS, auroc)})
    metrics['sens_at_90spec_detail'] = sens

    return metrics, (paths, y, lab, gp, lp)


def save_predictions(path, data):
    paths, y, lab, gp, lp = data
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'grade', 'prediction', *['g' + str(i) for i in range(5)],
                    *['target_' + c for c in LESIONS], *['p_' + c for c in LESIONS]])
        for i, name in enumerate(paths):
            w.writerow([name, int(y[i]), int(gp[i].argmax()), *gp[i].tolist(),
                        *lab[i].tolist(), *lp[i].tolist()])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test-manifest', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    args = p.parse_args()

    print(f'Verifying test manifest against locked hash...', flush=True)
    test_rows = read_test_manifest(args.test_manifest)
    print(f'  OK: {len(test_rows)} test rows, hash verified.', flush=True)

    for r in test_rows:
        if not (args.image_root / r['image']).is_file():
            raise FileNotFoundError(args.image_root / r['image'])

    load_training_components()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required (matches training script requirement)')
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False

    for run_dir in args.runs:
        if not run_dir.is_dir():
            print(f'[SKIP] {run_dir} not found', flush=True)
            continue
        print(f'\n=== Evaluating {run_dir} ===', flush=True)
        out_dir = run_dir / 'test_eval'
        out_dir.mkdir(exist_ok=True)
        if (out_dir / 'test_metrics.json').exists():
            print(f'  [SKIP] {out_dir / "test_metrics.json"} already exists. '
                  'Delete it first if you intentionally want to re-run (e.g. after a '
                  'code fix) -- this script will not silently overwrite a prior test run.', flush=True)
            continue
        start = time.perf_counter()
        metrics, data = evaluate_run(run_dir, test_rows, args.image_root, device)
        elapsed = time.perf_counter() - start
        save_predictions(out_dir / 'test_predictions.csv', data)
        (out_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        (out_dir / 'test_access.json').write_text(json.dumps({
            'test_access': True, 'checkpoint': 'best.pt', 'elapsed_seconds': elapsed,
            'test_manifest_sha256': TEST_SHA256,
        }, indent=2), encoding='utf-8')
        print(f'  wrote {out_dir / "test_predictions.csv"}', flush=True)
        print(f'  wrote {out_dir / "test_metrics.json"}', flush=True)
        print(f'  QWK={metrics["test_qwk"]:.4f}  macro_F1={metrics["test_macro_f1"]:.4f}  '
              f'AP_E={metrics["test_ap_E"]:.4f}  AP_R={metrics["test_ap_R"]:.4f}  '
              f'macro_AUROC={metrics["test_macro_auroc"]:.4f}  '
              f'Sens@90Spec_R={metrics["test_sens_at_90spec_R"]:.4f}  '
              f'({elapsed:.1f}s)', flush=True)

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
