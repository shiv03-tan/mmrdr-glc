"""Build the external manifest for the Peng et al. UWF dataset (Scientific Data 13:777, 2026).

Layout found in the download: <root>/Normal512, <root>/NPDR512, <root>/PDR512 and train.csv, val.csv, test.csv
(columns `Image Name`, `Label`; 0 = Normal, 1 = NPDR, 2 = PDR). Folder names without the 512 suffix and
.xls/.xlsx sheets are accepted too.

Data-quality handling (found by auditing the sheets):
  * identical (name, label) rows repeated inside a split are kept once;
  * a file name listed with two different labels is excluded (and written to the audit file);
  * a file name appearing in more than one split is kept only in the first split (train, val, test order).
The `patient` column is a FILENAME-DERIVED key (first two numbers of the file name). It is not a verified patient
identifier; it is used only for a cluster-bootstrap sensitivity analysis.

Usage:
  python build_peng_manifest.py --root "C:\\...\\diabetic retinopathy" --out peng_manifest.csv
Output columns: image (relative to --root), split, label3, folder, patient. No `grade` column (three classes only).
"""
import argparse, csv
from pathlib import Path
import pandas as pd

FOLDERS = {0: ['Normal512', 'Normal'], 1: ['NPDR512', 'NPDR'], 2: ['PDR512', 'PDR']}
SPLITS = [('train', ['train']), ('val', ['val', 'validate', 'validation']), ('test', ['test'])]


def load_sheet(root, names):
    for n in names:
        for ext in ('.csv', '.xls', '.xlsx'):
            p = root / (n + ext)
            if p.is_file():
                return pd.read_csv(p) if ext == '.csv' else pd.read_excel(p), p.name
    raise SystemExit(f'no {"/".join(names)} .csv/.xls/.xlsx in {root}')


def find_dir(root, lab):
    for f in FOLDERS[lab]:
        if (root / f).is_dir(): return f
    raise SystemExit(f'none of {FOLDERS[lab]} found in {root}')


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a = a.parse_args()
    folder = {lab: find_dir(a.root, lab) for lab in FOLDERS}
    frames, log = [], []
    for split, names in SPLITS:
        df, fname = load_sheet(a.root, names)
        low = {str(c).strip().lower(): c for c in df.columns}
        ic = next((low[k] for k in ('image name', 'image_name', 'imagename', 'image', 'name', 'filename') if k in low), None)
        lc = low.get('label')
        if ic is None or lc is None: raise SystemExit(f'{fname}: need image-name and label columns, found {list(df.columns)}')
        d = pd.DataFrame({'image_name': df[ic].astype(str).str.strip(), 'label3': df[lc].astype(int), 'split': split})
        n0 = len(d); d = d.drop_duplicates(['image_name', 'label3'])
        log.append(f'{split}: {n0} rows in sheet, {n0 - len(d)} repeated rows removed')
        frames.append(d)
    A = pd.concat(frames, ignore_index=True)
    nlab = A.groupby('image_name').label3.nunique()
    conflict = sorted(nlab[nlab > 1].index)
    A = A[~A.image_name.isin(conflict)]
    log.append(f'excluded for conflicting labels ({len(conflict)}): {conflict}')
    before = len(A); A = A.drop_duplicates('image_name', keep='first')
    log.append(f'removed {before - len(A)} names that appear in more than one split')
    A['folder'] = A.label3.map(folder)
    A['image'] = A.folder + '/' + A.image_name
    ok = A.image.map(lambda r: (a.root / r).is_file())
    miss = A[~ok]
    for _, r in miss.iterrows():
        elsewhere = [f for f in folder.values() if (a.root / f / r.image_name).is_file()]
        log.append(f'excluded, file not found in {r.folder}: {r.image_name} (split {r.split}, label {r.label3})'
                   + (f'; a file with this name exists in {elsewhere}' if elsewhere else '; not found in any folder'))
    if len(miss) > 0.02 * len(A):
        raise SystemExit(f'{len(miss)} listed images are missing on disk (more than 2%); check the download. First: {list(miss.image[:5])}')
    A = A[ok]
    k = A.image_name.str.extract(r'^(\d+)_(\d+)_')
    A['patient'] = k[0] + '_' + k[1]
    A[['image', 'split', 'label3', 'folder', 'patient']].to_csv(a.out, index=False, encoding='utf-8')
    tab = A.groupby(['split', 'label3']).size().unstack(fill_value=0)
    log.append('unique images per split and class (0 Normal, 1 NPDR, 2 PDR):\n' + tab.to_string())
    log.append(f'total unique images {len(A)}; filename-derived patient keys {A.patient.nunique()}')
    k2 = A.groupby('patient').split.nunique()
    log.append(f'filename-derived keys present in more than one split: {(k2 > 1).sum()}')
    txt = '\n'.join(log); print(txt)
    (a.out.parent / (a.out.stem + '_audit.txt')).write_text(txt, encoding='utf-8')


if __name__ == '__main__':
    main()
