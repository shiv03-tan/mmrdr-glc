"""Run saved MMRDR-trained checkpoints on an EXTERNAL UWF set. Inference only: no training, no weight update.

External manifest CSV (UTF-8) needs one column `image` (path relative to --image-root).
Optional columns: `grade` (integer 0-4, already mapped to the ICDR scale; leave blank/absent if unknown)
and any of MA, HE, IH, VB_IRMA, NV, VH, RD (0/1; leave blank/absent if the dataset does not label them).
Output CSV has the same columns as the validation prediction files (best_val_predictions.csv);
missing labels are written as empty cells.

Example (Windows, from the folder that contains the training script):
  python external_predict.py ^
    --train-script runs\\uwf_mt_asl_seed52_resumed_v2\\train_uwf_multitask_used.py ^
    --manifest external\\manifest.csv --image-root D:\\external\\images ^
    --ckpt base52=runs\\uwf_mt_asl_seed52_resumed_v2\\best.pt ^
    --ckpt glc52=runs\\uwf_mt_asl_glc010_seed52\\best.pt ^
    --out-dir external_preds
"""
import argparse, csv, importlib.util, sys
from pathlib import Path

LESIONS = ('MA', 'HE', 'IH', 'VB_IRMA', 'NV', 'VH', 'RD')


def load_train_module(path):
    spec = importlib.util.spec_from_file_location('trainmod', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['trainmod'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-script', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--ckpt', action='append', required=True, metavar='LABEL=PATH')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    a = p.parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    t = load_train_module(a.train_script)
    _, Model = t.load_training_components()
    assert tuple(t.LESIONS) == LESIONS, 'lesion order differs from the training script'

    class Square:                              # identical to the training pipeline
        def __call__(self, im):
            w, h = im.size
            s = max(w, h)
            out = Image.new('RGB', (s, s), (0, 0, 0))
            out.paste(im, ((s - w) // 2, (s - h) // 2))
            return out

    tf = T.Compose([Square(), T.Resize((512, 512), interpolation=InterpolationMode.BILINEAR), T.ToTensor(),
                    T.Normalize((.485, .456, .406), (.229, .224, .225))])

    with a.manifest.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'image' not in rows[0]:
        raise SystemExit('manifest needs an `image` column')
    missing = [r['image'] for r in rows if not (a.image_root / r['image']).is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} images not found, first: {missing[0]}')

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            with Image.open(a.image_root / rows[i]['image']) as im:
                return tf(im.convert('RGB'))

    loader = torch.utils.data.DataLoader(DS(), batch_size=a.batch_size, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in a.ckpt:
        label, path = spec.split('=', 1)
        model = Model(pretrained=False).to(device)
        ck = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'], strict=True)
        model.eval()
        gps, lps = [], []
        with torch.inference_mode():
            for x in loader:
                g, l = model(x.to(device))
                gps.append(g.float().softmax(1).cpu()); lps.append(l.float().sigmoid().cpu())
        gp, lp = torch.cat(gps).numpy(), torch.cat(lps).numpy()
        out = a.out_dir / f'{label}_external_predictions.csv'
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['image', 'grade', 'prediction', *[f'g{i}' for i in range(5)],
                        *[f'target_{c}' for c in LESIONS], *[f'p_{c}' for c in LESIONS]])
            for i, r in enumerate(rows):
                w.writerow([r['image'], (r.get('grade') or '').strip(), int(gp[i].argmax()), *gp[i].tolist(),
                            *[(r.get(c) or '').strip() for c in LESIONS], *lp[i].tolist()])
        print('wrote', out, len(rows), 'images')


if __name__ == '__main__':
    main()
