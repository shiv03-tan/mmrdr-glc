"""Evaluate the three verified CFP checkpoints on the frozen test partition.
No training, checkpoint selection, threshold fitting, or calibration fitting.
Requires archived source files and successful validation reproduction report.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

RUNS = {42: 'resnet50_cfp_seed42_v2', 52: 'resnet50_cfp_seed52_full', 62: 'resnet50_cfp_seed62_full'}
TEST_HASH = 'c7922148edc96b9430509793c3b862050c9706bb5d2676a8d5ac4f7747a900e7'
CHECKPOINT_HASHES = {
    42: 'c5e9d7d3c05e5237419c2ced5a17533e363705fff9a17b69888b9ae416985908',
    52: '1d663afadce03fd8e8df0416bf131c6b39ec6d78d2d97dbff07aa406c1ef432c',
    62: '36d4e7eeeac7cac0163938de85ac3b181f1776ea8f852cfb4e5b7dfcb4b6af37'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def metrics(y, probabilities):
    import numpy as np
    from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), 5) or not np.isfinite(p).all():
        raise ValueError('Invalid probability shape or values')
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('Invalid probabilities')
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(5)))
    support = cm.sum(1)
    if (support == 0).any():
        raise ValueError('Expected all five grades in this fixed test partition')
    recall = cm.diagonal()/support
    return dict(n=len(y), qwk=float(cohen_kappa_score(y, pred, labels=list(range(5)), weights='quadratic')),
        macro_f1=float(f1_score(y, pred, labels=list(range(5)), average='macro', zero_division=0)),
        accuracy=float((pred == y).mean()), balanced_accuracy=float(recall.mean()),
        per_grade_recall=recall.tolist(), per_grade_support=support.tolist(), confusion_matrix=cm.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=Path(r'C:\Research\mmrdr-own-study'))
    args = parser.parse_args()
    root = args.study_root.resolve()
    out = root/'runs'/'resnet50_cfp_test'
    if out.exists():
        raise FileExistsError(f'Output exists; retain it and inspect before repeating: {out}')
    report_path = root/'runs'/'cfp_validation_reproduction'/'report.json'
    report = json.loads(report_path.read_text())
    if report.get('all_passed') is not True or report.get('test_access') is not False:
        raise ValueError('Successful validation reproduction is required')
    checked = {r['seed']: r for r in report['runs']}
    if len(report['runs']) != 3 or set(checked) != set(RUNS):
        raise ValueError('Expected exactly the three validated seeds')
    archive = root/'runs'/RUNS[42]
    loader_file = archive/'mmrdr_dataset_used.py'
    trainer_file = archive/'train_resnet50_used.py'
    configs = {}
    for seed, folder in RUNS.items():
        run = root/'runs'/folder
        c = json.loads((run/'config.json').read_text())
        if c['seed'] != seed or c['mode'] != 'full' or c['amp'] != 'float16':
            raise ValueError('Unexpected config')
        if not checked[seed]['passed'] or checked[seed]['n'] != 1332:
            raise ValueError('Validation run did not pass')
        for path, value in [(run/'best.pt', CHECKPOINT_HASHES[seed]),
                (loader_file, c['loader_sha256']), (trainer_file, c['script_sha256']),
                (root/'splits'/'cfp_train.csv', c['train_sha256']),
                (root/'splits'/'cfp_val.csv', c['val_sha256'])]:
            if sha(path) != value:
                raise ValueError(f'Checksum mismatch: {path}')
        if checked[seed]['checkpoint_sha256'] != CHECKPOINT_HASHES[seed]:
            raise ValueError('Checkpoint differs from audited report')
        configs[seed] = c
    # Frozen partition is used unchanged, and never combined with development data.
    manifest = root/'splits'/'cfp_test.csv'
    if sha(manifest) != TEST_HASH:
        raise ValueError('Frozen test manifest checksum differs')
    test_rows = rows(manifest)
    names = [r['image'].strip().replace('\\', '/') for r in test_rows]
    if len(test_rows) != 2225 or len(set(names)) != 2225:
        raise ValueError('Wrong test count or duplicate images')
    for r, name in zip(test_rows, names):
        if r['split'] != 'test' or r['modality'] != 'CFP' or not name.startswith('img/ts'):
            raise ValueError('Unexpected split/modality/path')
        if '..' in name.split('/') or r['grade'] not in {'0','1','2','3','4'}:
            raise ValueError('Unsafe path or invalid grade')
    for split in ('train','val'):
        dev = {r['image'].strip().replace('\\', '/') for r in rows(root/'splits'/f'cfp_{split}.csv')}
        if dev.intersection(names):
            raise ValueError('Development/test path overlap')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision.models import resnet50
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the verified FP16 inference recipe')
    for c in configs.values():
        if c['torch'] != torch.__version__ or c['torchvision'] != torchvision.__version__:
            raise ValueError('Use the original torch and torchvision versions')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location('archived_loader', loader_file)
    loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_module)
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(status='running', test_access=True, test_sha256=TEST_HASH,
        validation_report_sha256=sha(report_path), evaluator_sha256=sha(__file__),
        torch=torch.__version__, torchvision=torchvision.__version__, gpu=torch.cuda.get_device_name(0),
        selection='Previously frozen validation-QWK checkpoints; no test selection',
        prediction_rule='Raw softmax argmax; no threshold or calibration fitting', runs=[])
    (out/'evaluate_cfp_test_used.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'summary.json', summary)
    try:
        for seed, folder in RUNS.items():
            c = configs[seed]
            print(f'Seed {seed}: evaluating 2225 test images', flush=True)
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                checkpoint = torch.load(root/'runs'/folder/'best.pt', map_location='cpu', weights_only=True)
            if checkpoint['epoch'] != checked[seed]['epoch']:
                raise ValueError('Wrong checkpoint epoch')
            for key in ('seed','image_size','batch_size','loader_sha256','script_sha256','amp','weights'):
                if checkpoint['config'][key] != c[key]:
                    raise ValueError('Checkpoint/config mismatch: '+key)
            model = resnet50(weights=None, num_classes=5)
            model.load_state_dict(checkpoint['model'], strict=True)
            model.cuda().eval()
            dataset = loader_module.MMRDRDataset(manifest,
                {'CFP': Path(c['data_root'])/'MMRDR-CFP'}, loader_module.build_transform(c['image_size'], False))
            dataset.frame['image'] = dataset.frame['image'].str.strip()
            batches = DataLoader(dataset, batch_size=c['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
            records = []
            with torch.inference_mode():
                for step, batch in enumerate(batches, 1):
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = model(batch['image'].cuda(non_blocking=True))
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite test logits')
                    probs = logits.float().softmax(1).cpu().numpy()
                    for path, grade, prob in zip(batch['path'], batch['grade'].tolist(), probs):
                        records.append(dict(path=path, grade=grade, prediction=int(prob.argmax()),
                            **{'p'+str(j):float(prob[j]) for j in range(5)}))
                    if step % 100 == 0:
                        print(f'  {len(records)}/2225', flush=True)
            if len(records) != 2225:
                raise ValueError('Incomplete test inference')
            score = metrics([r['grade'] for r in records], [[r['p'+str(j)] for j in range(5)] for r in records])
            score.update(seed=seed, epoch=checkpoint['epoch'], checkpoint_sha256=CHECKPOINT_HASHES[seed])
            pred_file = out/f'seed{seed}_test_predictions.csv'
            with pred_file.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            score['predictions_sha256'] = sha(pred_file)
            dump(out/f'seed{seed}_metrics.json', score)
            summary['runs'].append(score)
            dump(out/'summary.json', summary)
            print(json.dumps(score, indent=2), flush=True)
            del checkpoint, model
            torch.cuda.empty_cache()
        summary['across_seed'] = {k:dict(mean=float(np.mean([r[k] for r in summary['runs']])),
            sample_sd=float(np.std([r[k] for r in summary['runs']], ddof=1)))
            for k in ('qwk','macro_f1','accuracy','balanced_accuracy')}
        summary['status'] = 'completed'
        summary['uncertainty_note'] = 'Seed SD describes training variability, not a confidence interval. No patient-level inference.'
        dump(out/'summary.json', summary)
        print('DONE. All three frozen checkpoints evaluated. Upload summary.json.')
    except Exception as exc:
        summary['status'] = 'failed'
        summary['error'] = str(exc)
        dump(out/'summary.json', summary)
        raise


if __name__ == '__main__':
    main()
