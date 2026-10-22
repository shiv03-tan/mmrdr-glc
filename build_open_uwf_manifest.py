"""Manifest for the open UWF dataset (He et al., Scientific Data 2024; figshare 10.6084/m9.figshare.26936446).

Needs, in --root (or its subfolders): the image folder, `Demographics of the participants.xlsx` and `Ground Truth.xlsx`.
Images are found by file name anywhere under --root. Output columns:
  image (relative to --root), diagnosis (DR, AMD, RVO, PM, Uveitis, RD, Healthy), patient, eye, quality (1 gradable, 0 not).
Usage:  python build_open_uwf_manifest.py --root "C:\\...\\Dataset" --out open_uwf_manifest.csv
"""
import argparse
from pathlib import Path
import pandas as pd


def find_xlsx(root, key):
    hits = [p for p in root.rglob('*.xlsx') if key in p.name.lower()]
    if not hits: raise SystemExit(f'no .xlsx with "{key}" in its name under {root}')
    return hits[0]


def col(df, *keys):
    for c in df.columns:
        n = str(c).strip().lower()
        if all(k in n for k in keys): return c
    raise SystemExit(f'column containing {keys} not found; columns are {list(df.columns)}')


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--root', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    a = a.parse_args()
    dem = pd.read_excel(find_xlsx(a.root, 'demographic')); gt = pd.read_excel(find_xlsx(a.root, 'ground truth'))
    idc, pc, dc = col(dem, 'image'), col(dem, 'patient'), col(dem, 'diagnos')
    ec = next((c for c in dem.columns if 'eye' in str(c).lower() and 'id' in str(c).lower()), None)
    out = pd.DataFrame({'name': dem[idc].astype(str).str.strip(), 'diagnosis': dem[dc].astype(str).str.strip(),
                        'patient': dem[pc].astype(str).str.strip(), 'eye': dem[ec].astype(str) if ec else ''})
    q = pd.DataFrame({'name': gt[col(gt, 'image')].astype(str).str.strip(), 'quality': gt[col(gt, 'overall')].astype(int)})
    out = out.merge(q, on='name', how='left')
    files = {}
    for p in a.root.rglob('*'):
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png'): files.setdefault(p.name.lower(), p.relative_to(a.root).as_posix())
    out['image'] = out.name.str.lower().map(files)
    miss = out[out.image.isna()]
    if len(miss): raise SystemExit(f'{len(miss)} listed images not found under {a.root}, first: {list(miss.name[:5])}')
    if out.name.duplicated().any(): raise SystemExit('duplicate image IDs in the demographics sheet')
    out[['image', 'diagnosis', 'patient', 'eye', 'quality']].to_csv(a.out, index=False, encoding='utf-8')
    print(out.diagnosis.value_counts().to_string()); print('images', len(out), '| patients', out.patient.nunique(),
          '| gradable', int(out.quality.sum()), '| missing quality', int(out.quality.isna().sum()))


if __name__ == '__main__':
    main()
