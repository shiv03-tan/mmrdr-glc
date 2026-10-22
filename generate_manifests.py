"""
generate_manifests.py
======================

Builds the six locked split manifests for the MMRDR grade-lesion consistency
study:

    cfp_train.csv   cfp_val.csv   cfp_test.csv
    uwf_train.csv   uwf_val.csv   uwf_test.csv

Protocol implemented (per the Phase-2 lock):
  - CFP and UWF each use the dataset's OFFICIAL tr/ts partition, identified
    by filename prefix. The ts partition is never touched by this script
    except to be written out unchanged as *_test.csv.
  - The validation set is carved ONLY from the official tr pool, using
    iterative multilabel stratification over [5 one-hot grade columns +
    7 lesion columns] (12 label columns total), NOT the collapsed
    grade x has-rare-lesion indicator used in earlier drafts.
  - One fixed split-generation seed (default 42) is used for both
    modalities. Training seeds 42/52/62 (used later for model training)
    are unrelated to this seed and must all consume the exact same
    manifests produced here.
  - Each manifest row carries: image path, modality, grade, the seven
    lesion labels (named columns, canonical order), laterality (lr),
    which split it belongs to, the split-generation seed, and a SHA-256
    checksum of the source CSV the row was drawn from (for provenance).
  - No patient identifier exists in the released CSVs, so patient-level
    disjointness of the tr-derived validation subset from the official
    test set cannot be verified here even though the official UWF/OCT
    test partition was itself constructed patient-wise upstream. This
    limitation is written into the run log and should be stated in the
    paper as-is -- do not claim patient-level validation separation.

Requires: pandas, numpy, iterative-stratification
    pip install iterative-stratification --break-system-packages

Usage:
    python generate_manifests.py \
        --cfp-csv "C:\\Datasets\\MMRDR\\MMRDR-CFP\\FP.csv" \
        --uwf-csv "C:\\Datasets\\MMRDR\\MMRDR-UWF\\UWF_clean.csv" \
        --out-dir "C:\\Research\\mmrdr-diabetic-retinopathy\\splits" \
        --val-frac 0.15 \
        --seed 42
"""

import argparse
import ast
import hashlib
import os
import sys

import numpy as np
import pandas as pd

LESION_NAMES = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
N_GRADES = 5


def sha256_of_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_lesion(s):
    """Parse the stringified 7-element multi-hot lesion vector. Raises loudly
    on malformed rows rather than silently coercing them, matching the
    corrected-notebook convention."""
    try:
        v = np.array(ast.literal_eval(str(s)), dtype=np.float32)
    except Exception:
        v = np.array(
            [float(x) for x in str(s).strip("[]").split(",") if x.strip()],
            dtype=np.float32,
        )
    if v.shape[0] != 7:
        raise ValueError(f"lesion vector is not 7-dim: {s!r}")
    return (v > 0).astype(np.int64)


def load_and_prepare(csv_path, modality):
    df = pd.read_csv(csv_path)
    required = {"image", "grade", "lesion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    lesion_mat = np.stack(df["lesion"].map(parse_lesion).to_numpy())
    for i, name in enumerate(LESION_NAMES):
        df[name] = lesion_mat[:, i]

    if "lr" not in df.columns:
        df["lr"] = ""

    df["grade"] = df["grade"].astype(int)
    if df["grade"].min() < 0 or df["grade"].max() >= N_GRADES:
        raise ValueError(f"{csv_path}: grade values out of [0,{N_GRADES-1}] range")

    df["modality"] = modality
    df["_basename"] = (
        df["image"].astype(str).map(lambda s: os.path.basename(s.replace("\\", "/")))
    )
    return df


def official_split(df):
    tr = df[df["_basename"].str.startswith("tr")].reset_index(drop=True)
    ts = df[df["_basename"].str.startswith("ts")].reset_index(drop=True)
    if len(tr) == 0 or len(ts) == 0:
        raise RuntimeError(
            "Could not find official tr/ts prefixes in the image column. "
            "Check that this CSV actually encodes the official partition."
        )
    overlap = set(tr["_basename"]) & set(ts["_basename"])
    if overlap:
        raise RuntimeError(f"tr/ts overlap detected on {len(overlap)} filenames")
    return tr, ts


def multilabel_stratified_val_split(tr_df, val_frac, seed):
    """Carve a validation subset out of tr_df using iterative multilabel
    stratification over 5 one-hot grade columns + 7 lesion columns."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as e:
        raise ImportError(
            "iterative-stratification is required for this split. "
            "Install with: pip install iterative-stratification --break-system-packages"
        ) from e

    grade_onehot = pd.get_dummies(tr_df["grade"], prefix="G").reindex(
        columns=[f"G_{g}" for g in range(N_GRADES)], fill_value=0
    )
    label_matrix = pd.concat([grade_onehot, tr_df[LESION_NAMES]], axis=1).to_numpy()

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=seed
    )
    X_dummy = np.zeros((len(tr_df), 1))
    train_idx, val_idx = next(msss.split(X_dummy, label_matrix))

    train_sub = tr_df.iloc[train_idx].reset_index(drop=True)
    val_sub = tr_df.iloc[val_idx].reset_index(drop=True)
    return train_sub, val_sub


def stratification_report(train_sub, val_sub, name):
    print(f"\n[{name}] stratification balance (positive rate, train vs val):")
    print(f"  {'label':10s} {'train':>8s} {'val':>8s} {'delta':>8s}")
    for g in range(N_GRADES):
        tr_rate = (train_sub["grade"] == g).mean()
        va_rate = (val_sub["grade"] == g).mean()
        print(f"  {'grade_'+str(g):10s} {tr_rate:8.4f} {va_rate:8.4f} {va_rate-tr_rate:+8.4f}")
    for lesion in LESION_NAMES:
        tr_rate = train_sub[lesion].mean()
        va_rate = val_sub[lesion].mean()
        print(f"  {lesion:10s} {tr_rate:8.4f} {va_rate:8.4f} {va_rate-tr_rate:+8.4f}")


OUTPUT_COLUMNS = (
    ["image", "modality", "grade"] + LESION_NAMES + ["lr", "split", "split_seed", "source_csv_sha256"]
)


def build_manifest(df, split_name, seed, source_sha):
    out = df.copy()
    out["split"] = split_name
    out["split_seed"] = seed
    out["source_csv_sha256"] = source_sha
    return out[OUTPUT_COLUMNS]


def process_modality(csv_path, modality, val_frac, seed, out_dir):
    print(f"\n{'='*70}\n{modality} -- {csv_path}\n{'='*70}")
    source_sha = sha256_of_file(csv_path)
    print(f"[OK] source checksum (sha256): {source_sha}")

    df = load_and_prepare(csv_path, modality)
    tr_pool, test_df = official_split(df)
    print(f"[OK] official split -> tr pool: {len(tr_pool)} | ts (locked test): {len(test_df)}")

    train_sub, val_sub = multilabel_stratified_val_split(tr_pool, val_frac, seed)
    print(f"[OK] validation carve (seed={seed}, val_frac={val_frac}) "
          f"-> train: {len(train_sub)} | val: {len(val_sub)}")

    assert set(train_sub["_basename"]) & set(val_sub["_basename"]) == set(), "train/val overlap"
    assert set(train_sub["_basename"]) & set(test_df["_basename"]) == set(), "train/test overlap"
    assert set(val_sub["_basename"]) & set(test_df["_basename"]) == set(), "val/test overlap"
    print("[OK] disjointness verified: train / val / test do not overlap")

    stratification_report(train_sub, val_sub, modality)

    prefix = modality.lower()
    manifests = {
        f"{prefix}_train.csv": build_manifest(train_sub, "train", seed, source_sha),
        f"{prefix}_val.csv":   build_manifest(val_sub,   "val",   seed, source_sha),
        f"{prefix}_test.csv":  build_manifest(test_df,   "test",  seed, source_sha),
    }
    os.makedirs(out_dir, exist_ok=True)
    for fname, mdf in manifests.items():
        path = os.path.join(out_dir, fname)
        mdf.to_csv(path, index=False)
        print(f"[OK] wrote {path}  ({len(mdf)} rows)")

    return manifests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfp-csv", required=True, help="Path to MMRDR-CFP FP.csv")
    ap.add_argument("--uwf-csv", required=True, help="Path to MMRDR-UWF UWF_clean.csv")
    ap.add_argument("--out-dir", required=True, help="Directory to write the six manifests")
    ap.add_argument("--val-frac", type=float, default=0.15,
                     help="Fraction of the official tr pool held out for validation")
    ap.add_argument("--seed", type=int, default=42,
                     help="Split-generation seed (independent of model-training seeds)")
    args = ap.parse_args()

    print("MMRDR MANIFEST GENERATION")
    print(f"split-generation seed : {args.seed}")
    print(f"validation fraction   : {args.val_frac}")
    print(f"output directory      : {args.out_dir}")
    print("\nNOTE: no patient identifier is present in the released CSVs.")
    print("      Patient-level disjointness of the tr-derived validation")
    print("      subset from the official test set cannot be verified here.")
    print("      State this explicitly as a limitation in the paper.")

    process_modality(args.cfp_csv, "CFP", args.val_frac, args.seed, args.out_dir)
    process_modality(args.uwf_csv, "UWF", args.val_frac, args.seed, args.out_dir)

    print(f"\n{'='*70}")
    print("DONE. Six manifests written:")
    for prefix in ("cfp", "uwf"):
        for split in ("train", "val", "test"):
            print(f"  {os.path.join(args.out_dir, f'{prefix}_{split}.csv')}")
    print("\nEvery baseline and the proposed model must load exactly these")
    print("files -- do not regenerate splits per-experiment.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
