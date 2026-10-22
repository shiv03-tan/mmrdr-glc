"""Qualitative Grad-CAM maps for NV, VH and RD on VALIDATION images only.

Compares the ASL baseline and GLC-Full checkpoints of the same seed on the same randomly sampled
validation positives. No test image is read. The maps are illustrative; the labels are image-level,
so the maps cannot be checked against lesion locations.

Example (Windows):
  python gradcam_val.py ^
    --baseline-ckpt  runs\\baseline_seed42\\best.pt ^
    --glc-ckpt       runs\\glc_full_seed42\\best.pt ^
    --val-predictions runs\\baseline_seed42\\best_val_predictions.csv ^
    --image-root "C:\\Datasets\\MMRDR\\MMRDR-UWF" --out gradcam_val_seed42 --per-lesion 3
"""
import argparse, csv, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet50
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

LESIONS = ['MA', 'HE', 'IH', 'VB_IRMA', 'NV', 'VH', 'RD']
TARGETS = ['NV', 'VH', 'RD']

class Model(nn.Module):                      # identical to the training script
    def __init__(self):
        super().__init__()
        self.encoder = resnet50(weights=None); self.encoder.fc = nn.Identity()
        self.lateral = nn.ModuleList(nn.Conv2d(c, 256, 1) for c in (256, 512, 1024, 2048))
        self.smooth = nn.ModuleList(nn.Conv2d(256, 256, 3, padding=1) for _ in range(4))
        self.shared = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(.2))
        self.grade = nn.Linear(512, 5); self.lesion = nn.Linear(512, 7)
    def forward(self, x, keep=None):
        e = self.encoder
        x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
        maps = []
        for layer in (e.layer1, e.layer2, e.layer3, e.layer4):
            x = layer(x); maps.append(x)
        if keep is not None: keep.append(maps[3])          # layer4 activation for Grad-CAM
        pyramid = [l(c) for l, c in zip(self.lateral, maps)]
        for j in (2, 1, 0):
            pyramid[j] = pyramid[j] + F.interpolate(pyramid[j+1], size=pyramid[j].shape[-2:], mode='nearest')
        pooled = [F.adaptive_avg_pool2d(s(p), 1).flatten(1) for s, p in zip(self.smooth, pyramid)]
        h = self.shared(torch.cat(pooled, dim=1))
        return self.grade(h), self.lesion(h)

class Square:
    def __call__(self, im):
        w, h = im.size; s = max(w, h)
        out = Image.new('RGB', (s, s), (0, 0, 0)); out.paste(im, ((s-w)//2, (s-h)//2)); return out
square_resize = T.Compose([Square(), T.Resize((512, 512), interpolation=InterpolationMode.BILINEAR)])
to_input = T.Compose([T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])

def load(path, device):
    m = Model(); state = torch.load(path, map_location='cpu')
    m.load_state_dict(state['model'] if 'model' in state else state); return m.to(device).eval()

def gradcam(model, x, idx):
    keep = []
    logits_g, logits_l = model(x, keep)
    act = keep[0]; act.retain_grad()
    model.zero_grad(); logits_l[0, idx].backward()
    w = act.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(512, 512), mode='bilinear', align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), torch.sigmoid(logits_l[0, idx]).item()

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--baseline-ckpt', required=True); a.add_argument('--glc-ckpt', required=True)
    a.add_argument('--val-predictions', required=True); a.add_argument('--image-root', required=True)
    a.add_argument('--out', default='gradcam_val'); a.add_argument('--per-lesion', type=int, default=3)
    a.add_argument('--sample-seed', type=int, default=0)
    args = a.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = list(csv.DictReader(open(args.val_predictions, encoding='utf-8-sig')))
    assert all(r['image'].startswith('img/tr') for r in rows), 'only training-manifest (validation) images are allowed'
    rng = random.Random(args.sample_seed)
    picks = []                                           # random positives, not selected by outcome
    for les in TARGETS:
        pos = [r for r in rows if float(r['target_'+les]) == 1.0]
        picks += [(les, r) for r in rng.sample(pos, min(args.per_lesion, len(pos)))]
    models = {'ASL baseline': load(args.baseline_ckpt, device), 'GLC-Full': load(args.glc_ckpt, device)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.0, 2.3*len(picks)))
    log = []
    for i, (les, r) in enumerate(picks):
        im = square_resize(Image.open(Path(args.image_root)/r['image']).convert('RGB'))
        x = to_input(im).unsqueeze(0).to(device).requires_grad_(False)
        axes[i, 0].imshow(im); axes[i, 0].set_ylabel(f'{les}\n{r["image"].split("/")[-1]}\ngrade {r["grade"]}', fontsize=7)
        for j, (name, m) in enumerate(models.items(), start=1):
            cam, p = gradcam(m, x.clone(), LESIONS.index(les))
            axes[i, j].imshow(im); axes[i, j].imshow(cam, cmap='jet', alpha=0.45, vmin=0, vmax=1)
            axes[i, j].set_title(f'{name}: p({les}) = {p:.2f}', fontsize=7)
            log.append((les, r['image'], name, p))
        for k in range(3): axes[i, k].set_xticks([]); axes[i, k].set_yticks([])
    axes[0, 0].set_title('Image', fontsize=7)
    fig.tight_layout(); fig.savefig(out/'gradcam_val.pdf'); fig.savefig(out/'gradcam_val.png', dpi=200)
    with open(out/'gradcam_val_log.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['lesion', 'image', 'model', 'probability']); w.writerows(log)
    print('Saved', out/'gradcam_val.pdf')

if __name__ == '__main__':
    main()
