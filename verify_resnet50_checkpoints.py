"""Reproduce all three archived CFP validation runs. Never reads test data.

Run from the study folder, or supply --study-root. Writes a new output folder.
Requires the original mmrdr_dataset_used.py archived under seed42_v2.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

RUNS = {42: 'resnet50_cfp_seed42_v2', 52: 'resnet50_cfp_seed52_full',
        62: 'resnet50_cfp_seed62_full'}
VAL_HASH = '9c7861e7355835259d05e7461f907ea63709a1e8b74b2789209f01ec42e59490'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def compare_predictions(actual, reference, atol=1e-6):
    import numpy as np
    def index(rows):
        result = {}
        for r in rows:
            key = str(r['path']).replace('\\', '/').strip().casefold()
            if key in result:
                raise ValueError('Duplicate prediction path: ' + key)
            result[key] = r
        return result
    a, b = index(actual), index(reference)
    if a.keys() != b.keys():
        raise ValueError('Prediction path sets differ')
    keys = sorted(a)
    if any(int(a[k]['grade']) != int(b[k]['grade']) for k in keys):
        raise ValueError('Reference labels differ')
    pa = np.array([[float(a[k]['p'+str(j)]) for j in range(5)] for k in keys])
    pb = np.array([[float(b[k]['p'+str(j)]) for j in range(5)] for k in keys])
    for probs in (pa, pb):
        if not (np.isfinite(probs).all() and (probs >= 0).all() and
                (probs <= 1).all() and np.allclose(probs.sum(1), 1, atol=1e-6, rtol=0)):
            raise ValueError('Invalid probabilities')
    for rows, probs in ((a, pa), (b, pb)):
        if any(int(rows[k]['prediction']) != int(probs[i].argmax()) for i, k in enumerate(keys)):
            raise ValueError('Prediction differs from probability argmax')
    changed = sum(int(a[k]['prediction']) != int(b[k]['prediction']) for k in keys)
    delta = float(np.max(np.abs(pa-pb)))
    return dict(changed_predictions=changed, max_probability_difference=delta,
                probability_atol=atol, passed=(changed == 0 and delta <= atol))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=Path(r'C:\Research\mmrdr-own-study'))
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()
    root = args.study_root.resolve()
    out = args.out_dir or root / 'runs' / 'cfp_validation_reproduction'
    manifest = root / 'splits' / 'cfp_val.csv'
    if sha(manifest) != VAL_HASH:
        raise ValueError('Frozen validation manifest checksum differs')
    manifest_rows = read_csv(manifest)
    if len(manifest_rows) != 1332 or any(r['split'] != 'val' or r['modality'] != 'CFP'
            or not r['image'].strip().startswith('img/tr') for r in manifest_rows):
        raise ValueError('Wrong validation manifest')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    import torchvision
    from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix
    from torch.utils.data import DataLoader
    from torchvision.models import resnet50
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required to reproduce the archived FP16 inference')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    archive = root / 'runs' / RUNS[42]
    loader_path = archive / 'mmrdr_dataset_used.py'
    trainer_path = archive / 'train_resnet50_used.py'
    configs = {}
    for seed, name in RUNS.items():
        run = root / 'runs' / name
        c = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        if c['seed'] != seed or c['mode'] != 'full' or c['val_sha256'] != VAL_HASH:
            raise ValueError('Unexpected run config: ' + name)
        if c['amp'] != 'float16' or c['checkpoint_selection'] != 'validation QWK':
            raise ValueError('Unexpected inference/selection recipe')
        if sha(loader_path) != c['loader_sha256'] or sha(trainer_path) != c['script_sha256']:
            raise ValueError('Archived source hash mismatch: ' + name)
        if c['torch'] != torch.__version__ or c['torchvision'] != torchvision.__version__:
            raise ValueError('Use the original torch/torchvision versions for reproduction')
        for required in ('best.pt', 'best_val_predictions.csv', 'history.csv'):
            if not (run / required).is_file():
                raise FileNotFoundError(run / required)
        configs[seed] = c
    spec = importlib.util.spec_from_file_location('archived_mmrdr_loader', loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    for seed, name in RUNS.items():
        run, c = root / 'runs' / name, configs[seed]
        print(f'Seed {seed}: loading best.pt; validation only', flush=True)
        # These are the user's own locally trained checkpoints. Restricted loading stays enabled.
        # Torch versions can be serialized as TorchVersion objects inside their configs.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ckpt = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        for k in ('seed', 'val_sha256', 'train_sha256', 'script_sha256', 'loader_sha256',
                  'batch_size', 'image_size', 'amp', 'weights'):
            if ckpt['config'][k] != c[k]:
                raise ValueError('Checkpoint config mismatch: ' + k)
        hist = read_csv(run / 'history.csv')
        if [int(r['epoch']) for r in hist] != list(range(1, c['epochs']+1)):
            raise ValueError('Incomplete or duplicated training epochs')
        selected = max(hist, key=lambda r: float(r['val_qwk']))
        if ckpt['epoch'] != int(selected['epoch']):
            raise ValueError('Checkpoint is not the first maximum-QWK epoch')
        model = resnet50(weights=None, num_classes=5)
        model.load_state_dict(ckpt['model'], strict=True)
        model.cuda().eval()
        roots = {'CFP': Path(c['data_root']) / 'MMRDR-CFP'}
        dataset = module.MMRDRDataset(manifest, roots, module.build_transform(c['image_size'], False))
        dataset.frame['image'] = dataset.frame['image'].str.strip()
        batches = DataLoader(dataset, batch_size=c['batch_size'], shuffle=False,
                             num_workers=0, pin_memory=True)
        records = []
        with torch.inference_mode():
            for step, batch in enumerate(batches, 1):
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(batch['image'].cuda(non_blocking=True))
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite logits')
                probabilities = logits.float().softmax(1).cpu().numpy()
                for path, grade, probs in zip(batch['path'], batch['grade'].tolist(), probabilities):
                    records.append(dict(path=path, grade=grade, prediction=int(probs.argmax()),
                                        **{'p'+str(i): float(probs[i]) for i in range(5)}))
                if step % 100 == 0:
                    print(f'  {len(records)}/1332 images', flush=True)
        comparison = compare_predictions(records, read_csv(run / 'best_val_predictions.csv'))
        y, pred = [r['grade'] for r in records], [r['prediction'] for r in records]
        qwk = float(cohen_kappa_score(y, pred, labels=list(range(5)), weights='quadratic'))
        f1 = float(f1_score(y, pred, labels=list(range(5)), average='macro', zero_division=0))
        metric_match = all(abs(value-float(source[key])) <= 1e-10
                           for source in (selected, ckpt['metrics'])
                           for key, value in [('val_qwk', qwk), ('val_macro_f1', f1)])
        report = dict(seed=seed, epoch=ckpt['epoch'], n=len(y), val_qwk=qwk,
                      val_macro_f1=f1, accuracy=float(accuracy_score(y, pred)),
                      confusion_matrix=confusion_matrix(y, pred, labels=list(range(5))).tolist(),
                      checkpoint_sha256=sha(run/'best.pt'), validation_sha256=sha(manifest),
                      reference_sha256=sha(run/'best_val_predictions.csv'),
                      metrics_match=metric_match, comparison=comparison,
                      passed=comparison['passed'] and metric_match)
        with (out/f'seed{seed}_predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        reports.append(report)
        (out/'report.json').write_text(json.dumps(dict(test_access=False,
            evaluator_sha256=sha(__file__), torch=torch.__version__, torchvision=torchvision.__version__,
            gpu=torch.cuda.get_device_name(0), runs=reports,
            all_passed=len(reports)==3 and all(r['passed'] for r in reports)), indent=2))
        print(json.dumps(report, indent=2), flush=True)
        del model, ckpt
        torch.cuda.empty_cache()
    if not all(r['passed'] for r in reports):
        raise RuntimeError('Reproduction mismatch: retain outputs for diagnosis; do not evaluate test yet')
    print('PASS: all three validation checkpoints reproduced. Test set was not loaded.')
    print('Report:', out/'report.json')


if __name__ == '__main__':
    main()
