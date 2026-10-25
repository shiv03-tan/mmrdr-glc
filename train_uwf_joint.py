"""MMRDR UWF study v2: joint grade-lesion output layer and foundation-model backbones.

Derived from train_uwf_multitask.py (v1, unchanged and still used for the paper).
Never reads a test manifest. No automatic test evaluation. Never overwrites an
existing output directory.

New options relative to v1
  --backbone   resnet50_fpn (v1 model) | dinov2_l | retfound_mae | retfound_dinov2
  --weights    local checkpoint file, required for the two RETFound backbones
  --head       independent (v1 heads) | joint (640-state layer from joint_head.py)
  --joint-mode none | hard | soft      (constraint on the joint layer)
  --workers    > 0 allowed on Linux

With --backbone resnet50_fpn --head independent --workers 0 the data, model,
loss and schedule match v1; use v1 itself for any rerun reported in the paper.

Joint head loss per accumulation window:
  w_nll * NLL / n_usable + w_asl * ASL(lesion marginals) / known_targets
  (+ w_ce * CE(grade marginal) / n, off by default)
The uncertain IH entry of tr000215 is marginalized in the NLL and masked in ASL.
In hard mode, training images whose labels break R1/R2 are left out of the NLL
only (they still enter ASL); their count is logged each epoch.
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
BACKBONES = {
    # name: (timm architecture or None, input size, weight source)
    'resnet50_fpn': (None, 512, 'torchvision'),
    'dinov2_l': ('vit_large_patch14_dinov2.lvd142m', 518, 'timm'),
    'retfound_mae': ('vit_large_patch16_224', 512, 'file'),
    'retfound_dinov2': ('vit_large_patch14_dinov2', 518, 'file'),
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
              'uncertain_record': BAD_IMAGE, 'uncertain_target': 'IH', 'test_access': False}
    for split, rows in [('train', tr), ('val', va)]:
        report[split + '_grades'] = {str(g): sum(int(r['grade']) == g for r in rows) for g in range(5)}
    return tr, va, report

def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--val-manifest', type=Path, required=True)
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--image-root', type=Path, default=Path('/workspace/data/MMRDR-UWF'))
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--backbone', choices=list(BACKBONES), default='resnet50_fpn')
    p.add_argument('--weights', type=Path, help='checkpoint file for retfound_* backbones')
    p.add_argument('--image-size', type=int, help='default: 512 (patch16/ResNet) or 518 (patch14)')
    p.add_argument('--head', choices=['independent', 'joint'], default='independent')
    p.add_argument('--joint-mode', choices=['none', 'hard', 'soft'], default='soft')
    p.add_argument('--joint-nll-weight', type=float, default=1.0)
    p.add_argument('--joint-asl-weight', type=float, default=1.0)
    p.add_argument('--joint-ce-weight', type=float, default=0.0)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--prefetch', type=int, default=1,
                   help='batches each worker holds in advance; 1 keeps 8 workers inside a 64 MB /dev/shm')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--lesion-loss', choices=['asl', 'bce'], default='asl')
    p.add_argument('--consistency-weight', type=float, default=0.0,
                   help='GLC penalty; independent head only')
    p.add_argument('--precision', choices=['auto', 'fp32'], default='auto')
    p.add_argument('--resume', action='store_true',
                   help='continue an interrupted full run from <out-dir>/last.pt with identical settings')
    a = p.parse_args()
    if min(a.batch_size, a.accumulation) < 1 or a.epochs <= 5 or a.workers < 0:
        p.error('Positive batch/accumulation, epochs > 5 and workers >= 0 required')
    for name in ('consistency_weight', 'joint_nll_weight', 'joint_asl_weight', 'joint_ce_weight'):
        v = getattr(a, name)
        if not math.isfinite(v) or v < 0:
            p.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    if a.head == 'joint' and a.consistency_weight > 0:
        p.error('--consistency-weight applies to the independent head only')
    if a.head == 'joint' and a.joint_nll_weight == 0:
        p.error('--joint-nll-weight 0 would leave the joint layer untrained')
    if BACKBONES[a.backbone][2] == 'file' and not a.audit_only:
        if a.weights is None or not a.weights.is_file():
            p.error(f'{a.backbone} needs --weights pointing to a downloaded checkpoint file')
    if a.prefetch < 1:
        p.error('--prefetch must be >= 1')
    if a.workers > 0 and os.name == 'nt':
        p.error('workers > 0 is not supported on Windows; use --workers 0')
    if a.image_size is None:
        a.image_size = BACKBONES[a.backbone][1]
    if a.resume and a.mode != 'full':
        p.error('--resume applies to --mode full only')
    if not a.audit_only and a.out_dir is None:
        p.error('--out-dir is required for smoke/full training')
    return a

# ---------------------------------------------------------------------------
# Training definitions are loaded lazily so the CSV audit needs only Python.
# ---------------------------------------------------------------------------
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    global asl_on_probs
    from joint_head import JointGradeLesionHead, asl_on_probs

    class Square:
        def __call__(self, im):
            w, h = im.size
            s = max(w, h)
            out = Image.new('RGB', (s, s), (0, 0, 0))
            out.paste(im, ((s-w)//2, (s-h)//2))
            return out

    class Images(torch.utils.data.Dataset):
        def __init__(self, rows, root, training, size):
            self.rows, self.root = rows, root
            ops = []
            if training:
                ops = [T.RandomHorizontalFlip(0.5),
                       T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR, expand=True, fill=0),
                       T.ColorJitter(brightness=.1, contrast=.1, saturation=.1)]
            self.transform = T.Compose(ops + [Square(),
                T.Resize((size, size), interpolation=InterpolationMode.BILINEAR), T.PILToTensor()])
            # uint8 leaves the worker (4x less shared memory); to_input() does ToTensor's /255
            # and the ImageNet normalization on the device, giving the same values.

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

    class ResNetFPN(nn.Module):
        """v1 encoder: ResNet-50 + FPN, pooled pyramid -> 1024 features."""
        out_dim = 1024

        def __init__(self, pretrained=True):
            super().__init__()
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.encoder.fc = nn.Identity()
            self.lateral = nn.ModuleList(nn.Conv2d(c, 256, 1) for c in (256, 512, 1024, 2048))
            self.smooth = nn.ModuleList(nn.Conv2d(256, 256, 3, padding=1) for _ in range(4))

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
            return torch.cat(pooled, dim=1)

    class ViTEncoder(nn.Module):
        """timm ViT; features = [CLS token, mean of patch tokens] after the final norm."""

        def __init__(self, arch, size, source, weights=None, pretrained=True):
            super().__init__()
            import timm
            use_timm_weights = pretrained and source == 'timm'
            self.encoder = timm.create_model(arch, pretrained=use_timm_weights, img_size=size,
                                             num_classes=0, global_pool='')
            self.out_dim = 2 * self.encoder.embed_dim
            self.load_report = {'source': source if pretrained else 'random (test only)'}
            if pretrained and source == 'file':
                self.load_report.update(load_vit_checkpoint(self.encoder, weights))

        def forward(self, x):
            t = self.encoder.forward_features(x)
            k = self.encoder.num_prefix_tokens
            return torch.cat([t[:, 0], t[:, k:].mean(1)], dim=1)

    class Model(nn.Module):
        def __init__(self, backbone, head, joint_mode, weights=None, size=512, pretrained=True):
            super().__init__()
            arch, _, source = BACKBONES[backbone]
            if arch is None:
                self.body = ResNetFPN(pretrained)
            else:
                self.body = ViTEncoder(arch, size, source, weights, pretrained)
            self.shared = nn.Sequential(nn.Linear(self.body.out_dim, 512), nn.ReLU(), nn.Dropout(.2))
            self.head_type = head
            if head == 'independent':
                self.grade = nn.Linear(512, 5)
                self.lesion = nn.Linear(512, 7)
            else:
                self.joint = JointGradeLesionHead(512, mode=joint_mode)

        def train(self, mode=True):
            super().train(mode)
            # v1 behaviour: freeze BN running statistics, keep affine parameters trainable.
            for m in self.body.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            return self

        def forward(self, x):
            h = self.shared(self.body(x))
            if self.head_type == 'independent':
                g, l = self.grade(h).float(), self.lesion(h).float()
                return {'g': g, 'l': l, 'pg': g.softmax(1), 'pl': l.sigmoid()}
            pg, pl, logp = self.joint(h)
            return {'logp': logp, 'pg': pg, 'pl': pl}

        def encoder_parameter(self, name):
            return name.startswith('body.encoder.')

    return Images, Model

def load_vit_checkpoint(model, path):
    """Load a RETFound-style checkpoint into a timm ViT. Fails loudly if blocks are missing."""
    from timm.models.vision_transformer import checkpoint_filter_fn
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('model', 'teacher', 'state_dict', 'student'):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    sd = {}
    for k, v in ckpt.items():
        for prefix in ('module.', 'backbone.', 'encoder.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if k.startswith(('decoder', 'head', 'fc_norm', 'dino_head', 'ibot_head')):
            continue
        k = k.replace('mlp.w12.', 'mlp.fc1.').replace('mlp.w3.', 'mlp.fc2.')
        sd[k] = v
    # a checkpoint with a different patch size belongs to the other RETFound variant
    proj = sd.get('patch_embed.proj.weight')
    want = tuple(model.patch_embed.proj.weight.shape[-2:])
    if proj is None or proj.ndim != 4 or tuple(proj.shape[-2:]) != want:
        raise RuntimeError(f'Checkpoint {path} patch embedding {None if proj is None else tuple(proj.shape)} '
                           f'does not match this backbone (patch {want}); check --backbone')
    sd = checkpoint_filter_fn(sd, model)
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys]
    core = ('blocks.', 'patch_embed.', 'pos_embed', 'cls_token')
    if any(k.startswith(core) for k in missing):
        raise RuntimeError(f'Checkpoint {path} is missing backbone weights: {missing[:10]}')
    extra = [k for k in result.unexpected_keys if k.startswith(core)]
    if extra:
        raise RuntimeError(f'Checkpoint {path} has backbone weights this architecture lacks: {extra[:10]}')
    return {'weights_file': str(path), 'weights_sha256': sha(path),
            'missing_keys': missing, 'unexpected_keys': list(result.unexpected_keys)[:50]}

MEAN, STD = (.485, .456, .406), (.229, .224, .225)

def to_input(x, device):
    """uint8 (B,3,H,W) -> normalized float32 on device; equals ToTensor + Normalize."""
    x = x.to(device, non_blocking=True).float().div_(255)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std

def lesion_elements(logits, targets, kind):
    if kind == 'bce':
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = logits.sigmoid()
    pm = (p - .05).clamp(min=0)
    # gamma_positive=0, gamma_negative=4, detached focusing factor.
    return -targets * F.logsigmoid(logits) - (1-targets) * pm.detach().pow(4) * torch.log1p(-pm.clamp(max=1-1e-6))

def consistency(g, l):
    gp, lp = g.softmax(1), l.sigmoid()
    a = 1 - (1-lp).prod(1)
    b = 1 - (1-lp[:, 4:7]).prod(1)
    def js(x, y):
        x, y = x.clamp(1e-6, 1-1e-6), y.clamp(1e-6, 1-1e-6)
        m = (x+y)/2
        def kl(z):
            return z*(z.log()-m.log()) + (1-z)*(torch.log1p(-z)-torch.log1p(-m))
        return (kl(x)+kl(y))/2
    return js(a, 1-gp[:, 0]) + js(b, gp[:, 4])

def joint_usable(head, y, lab, mask):
    """Rows the joint NLL can use (all rows unless hard mode and labels break a rule)."""
    if head.mode != 'hard':
        return torch.ones(len(y), dtype=torch.bool, device=y.device)
    obs = mask.bool().unsqueeze(1)
    match = (head.L.unsqueeze(0) == lab.unsqueeze(1)) | ~obs
    compat = match.all(2) & (head.state_grade.unsqueeze(0) == y.unsqueeze(1))
    return (compat & head.valid.unsqueeze(0)).any(1)

def rule_violation_rate(gp, lp):
    """Share of images whose argmax grade and 0.5-thresholded lesions break R1 or R2."""
    g = gp.argmax(1)
    les = lp >= .5
    any_l, adv = les.any(1), les[:, 4:7].any(1)
    bad = ((g == 0) != ~any_l) | ((g == 4) != adv)
    return float(bad.float().mean())

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
            x = to_input(batch['image'], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(x)
            gp, lp = out['pg'].float(), out['pl'].float()
            if not torch.isfinite(gp).all() or not torch.isfinite(lp).all():
                raise RuntimeError('Nonfinite validation outputs')
            gps.append(gp.cpu().numpy()); lps.append(lp.cpu().numpy())
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
               'val_rule_violation_at_0.5': rule_violation_rate(torch.from_numpy(gp), torch.from_numpy(lp)),
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

def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

def main():
    args = arguments()
    tr, va, report = audit(args.train_manifest, args.val_manifest)
    print(json.dumps(report, indent=2), flush=True)
    if args.audit_only:
        return
    for r in tr + va:
        if not (args.image_root / r['image']).is_file():
            raise FileNotFoundError(args.image_root / r['image'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    Images, Model = load_training_components()
    import torchvision
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif os.environ.get('MMRDR_ALLOW_CPU') == '1':
        device = torch.device('cpu')  # code tests only, never for study runs
    else:
        raise RuntimeError('CUDA required for study training; audit-only works without CUDA')
    amp = device.type == 'cuda' and args.precision == 'auto' and torch.cuda.is_bf16_supported()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    # ViT attention kernels have no fully deterministic backward; warn instead of failing.
    vit = BACKBONES[args.backbone][0] is not None
    torch.use_deterministic_algorithms(True, warn_only=vit)
    if args.resume:
        if not (args.out_dir/'last.pt').is_file() or not (args.out_dir/'config.json').is_file():
            raise FileNotFoundError(f'--resume needs {args.out_dir}/last.pt and config.json')
        if (args.out_dir/'completed.json').is_file():
            raise RuntimeError(f'{args.out_dir} already completed; nothing to resume')
        old = json.loads((args.out_dir/'config.json').read_text(encoding='utf-8'))
        now = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        keys = ('backbone', 'weights', 'image_size', 'head', 'joint_mode', 'joint_nll_weight', 'joint_asl_weight',
                'joint_ce_weight', 'batch_size', 'accumulation', 'epochs', 'seed', 'encoder_lr', 'head_lr',
                'lesion_loss', 'consistency_weight', 'precision', 'mode')
        diff = {k: (old.get(k), now.get(k)) for k in keys if old.get(k) != now.get(k)}
        if diff:
            raise ValueError(f'--resume settings differ from the original run: {diff}')
        if old.get('train_sha256') != sha(args.train_manifest) or old.get('val_sha256') != sha(args.val_manifest):
            raise ValueError('--resume manifests differ from the original run')
    else:
        args.out_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(version='uwf-joint-v2', lesions=LESIONS,
        train_sha256=sha(args.train_manifest), val_sha256=sha(args.val_manifest), script_sha256=sha(__file__),
        joint_head_sha256=sha(Path(__file__).resolve().parent / 'joint_head.py'),
        torch=str(torch.__version__), torchvision=str(torchvision.__version__), python=sys.version,
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        precision_used='bf16' if amp else 'fp32', test_access=False, deterministic_strict=not vit,
        weight_decay=.01, bn_running_statistics='frozen (ResNet only)',
        checkpoint_rule='0.5*((QWK+1)/2)+0.5*rare_AP',
        augmentation=f'horizontal flip .5; rotation +/-15 expand; jitter .1; square pad; bilinear{args.image_size}; ImageNet norm',
        optimizer='AdamW betas=(.9,.999), eps=1e-8, no decay on biases/norm/scalars', schedule='5 epoch LR warmup then cosine',
        uncertain_label_policy=('IH masked in lesion loss; record excluded from consistency term'
                                if args.head == 'independent' else
                                'IH marginalized in joint NLL and masked in ASL'),
        selection_notice='smoke outputs are debugging-only' if args.mode=='smoke' else 'validation selected; not test results')
    print(f'Building {args.backbone} + {args.head} head'
          + (f' ({args.joint_mode})' if args.head == 'joint' else '') + '. No fallback to random weights.', flush=True)
    model = Model(args.backbone, args.head, args.joint_mode, args.weights, args.image_size).to(device)
    if hasattr(model.body, 'load_report'):
        config['weights_load'] = model.body.load_report
        print(json.dumps(model.body.load_report), flush=True)
    if not args.resume:
        (args.out_dir/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (args.out_dir/'manifest_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        shutil.copyfile(__file__, args.out_dir/'train_uwf_joint_used.py')
        shutil.copyfile(Path(__file__).resolve().parent / 'joint_head.py', args.out_dir/'joint_head_used.py')
    groups = {}
    for name, p in model.named_parameters():
        lr = args.encoder_lr if model.encoder_parameter(name) else args.head_lr
        decay = .01 if p.ndim > 1 else 0.
        groups.setdefault((lr, decay), []).append(p)
    optimizer = torch.optim.AdamW([{'params': ps, 'lr': lr, 'base_lr': lr, 'weight_decay': wd}
                                  for (lr, wd), ps in groups.items()], betas=(.9,.999), eps=1e-8)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == 'cuda',
                     worker_init_fn=seed_worker if args.workers else None,
                     prefetch_factor=args.prefetch if args.workers else None, persistent_workers=False)
    train = torch.utils.data.DataLoader(Images(tr, args.image_root, True, args.image_size),
                                        shuffle=True, generator=generator, **loader_kw)
    val = torch.utils.data.DataLoader(Images(va, args.image_root, False, args.image_size),
                                      shuffle=False, **loader_kw)
    print(f'{args.mode.upper()} train={len(tr)} val={len(va)} test=NOT LOADED precision={config["precision_used"]} '
          f'size={args.image_size} workers={args.workers}', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    best, start = -math.inf, time.perf_counter()
    history = []
    first_epoch = 1
    if args.resume:
        state = torch.load(args.out_dir/'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        first_epoch = int(state['epoch']) + 1
        with (args.out_dir/'history.csv').open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                row = {k: float(v) for k, v in row.items()}
                if row['epoch'] <= state['epoch']:
                    history.append(row)
        if len(history) != state['epoch']:
            raise RuntimeError('history.csv does not match last.pt')
        best = max(h['selection_score'] for h in history)
        start -= history[-1]['elapsed_seconds']
        # fresh, recorded RNG streams for the remaining epochs (not bit-identical to an uninterrupted run)
        rs = args.seed * 1000 + first_epoch
        random.seed(rs); np.random.seed(rs); torch.manual_seed(rs); torch.cuda.manual_seed_all(rs)
        generator.manual_seed(rs)
        note = {'resumed_at_epoch': first_epoch, 'rng_seed_after_resume': rs, 'best_so_far': best,
                'gpu': config['gpu'], 'torch': config['torch'], 'script_sha256': config['script_sha256'],
                'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        (args.out_dir/f'resume_epoch{first_epoch}.json').write_text(json.dumps(note, indent=2), encoding='utf-8')
        print('RESUMING ' + json.dumps(note), flush=True)
        if first_epoch > args.epochs:
            raise RuntimeError('last.pt is already at the final epoch')
    joint = args.head == 'joint'
    epoch = 0
    try:
        for epoch in range(first_epoch, (1 if args.mode=='smoke' else args.epochs)+1):
            model.train()
            n_batches = min(20, len(train)) if args.mode=='smoke' else len(train)
            stream = iter(itertools.islice(train, n_batches))
            seen_batches = 0
            sums = [0., 0., 0.]; counts = [0, 0, 0]; updates = 0; skipped = 0
            while True:
                window = list(itertools.islice(stream, args.accumulation))
                if not window:
                    break
                n = sum(len(b['grade']) for b in window)
                known = sum(float(b['mask'].sum()) for b in window)
                eligible = sum(float(b['eligible'].sum()) for b in window)
                if joint:
                    usable = sum(int(joint_usable(model.joint, b['grade'].to(device), b['lesions'].to(device),
                                                  b['mask'].to(device)).sum()) for b in window)
                progress = epoch-1 + seen_batches/n_batches
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * lr_factor(progress, args.epochs)
                weight = args.consistency_weight * min(1., progress/5)
                optimizer.zero_grad(set_to_none=True)
                for batch in window:
                    x = to_input(batch['image'], device)
                    y = batch['grade'].to(device)
                    lab, mask, ok = (batch[k].to(device) for k in ('lesions', 'mask', 'eligible'))
                    if not torch.isfinite(x).all():
                        raise RuntimeError('Nonfinite input')
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                        out = model(x)
                    if joint:
                        # term 0: joint NLL (grade and lesions together); term 1: ASL on marginals
                        nll, nskip = model.joint.nll(out['logp'], y, lab, mask, reduction='sum')
                        asl = asl_on_probs(out['pl'], lab, mask, reduction='none').sum()
                        ce = F.nll_loss(torch.log(out['pg'].clamp(min=1e-8)), y, reduction='sum')
                        loss = (args.joint_nll_weight*nll/max(usable, 1) + args.joint_asl_weight*asl/known
                                + args.joint_ce_weight*ce/n)
                        terms = (nll, asl, ce)
                        skipped += nskip
                        counts = [counts[0]+len(y)-nskip, counts[1]+int(mask.sum()), counts[2]+len(y)]
                    else:
                        g, l = out['g'], out['l']
                        gl = F.cross_entropy(g, y, reduction='sum')
                        ll = (lesion_elements(l, lab, args.lesion_loss)*mask).sum()
                        cl = (consistency(g, l)*ok).sum() if args.consistency_weight > 0 else g.sum()*0
                        loss = gl/n + ll/known + weight*cl/max(eligible, 1)
                        terms = (gl, ll, cl)
                        counts = [counts[0]+len(y), counts[1]+int(mask.sum()), counts[2]+int(ok.sum())]
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    sums = [s+float(v.detach()) for s, v in zip(sums, terms)]
                    seen_batches += 1
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                if updates == 1 or updates % 10 == 0:
                    names = ('joint_nll', 'asl') if joint else ('grade_loss', 'lesion_loss')
                    print(f'epoch={epoch} batch={seen_batches}/{n_batches} {names[0]}={sums[0]/max(counts[0],1):.4f} '
                          f'{names[1]}={sums[1]/max(counts[1],1):.4f} grad_norm={float(norm):.3f}', flush=True)
            metrics, data = evaluate(model, val, device, amp)
            if joint:
                metrics.update(train_joint_nll=sums[0]/max(counts[0],1), train_asl=sums[1]/max(counts[1],1),
                               train_grade_ce=sums[2]/max(counts[2],1), train_nll_skipped=skipped)
                if args.joint_mode == 'soft':
                    metrics['violation_logit'] = float(model.joint.violation_logit)
            else:
                metrics.update(train_grade_loss=sums[0]/counts[0], train_lesion_loss=sums[1]/counts[1],
                               train_consistency_loss=sums[2]/max(counts[2],1))
            metrics.update(epoch=epoch, successful_updates=updates, elapsed_seconds=time.perf_counter()-start,
                peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3 if device.type == 'cuda' else 0.)
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
