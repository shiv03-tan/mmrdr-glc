# mmrdr-glc

Code for the study *Does Grade–Lesion Consistency Improve Rare-Lesion Recognition in Ultra-Widefield Diabetic Retinopathy Grading? A Controlled Evaluation*.

A ResNet-50/FPN multitask model predicts a five-class diabetic retinopathy (DR) grade and seven lesion labels (MA, HE, IH, VB/IRMA, NV, VH, RD) on ultra-widefield (UWF) images from the MMRDR dataset. The study compares an asymmetric-loss (ASL) baseline with the same model trained with a grade–lesion consistency term (GLC-Full, λ = 0.10), using three seeds per configuration (42, 52, 62).

This repository contains code only. It does not contain images, annotations, model weights or prediction files.

## Data

| Dataset | Use | Access |
|---|---|---|
| MMRDR (UWF subset) | Training, validation, test | Obtain from the dataset authors under the terms of their release |
| Peng et al., *Scientific Data* 13:777 (2026) | External evaluation (three DR classes, no lesion labels) | Public; see the article's Data Availability section |
| He et al., *Scientific Data* 11:1251 (2024), open UWF set | External evaluation (diagnosis groups, no lesion labels) | Public; see the article's Data Availability section |

Place the data outside this repository. Image and manifest paths are passed as command-line arguments. The default `--image-root` in the training scripts is a Windows path and must be changed for your machine.

## Environment

Python 3.12 with PyTorch, torchvision, NumPy, pandas, scikit-learn, Pillow and Matplotlib. `openpyxl` is needed to read the open UWF spreadsheets. To record the exact versions you use, run `pip freeze > requirements.txt`.

## Scripts

### Partitions and audit
| Script | Purpose |
|---|---|
| `generate_manifests.py` | Builds the fixed train, validation and test partition manifests |
| `count_violations.py` | Counts annotation records that violate the grade–lesion relationships used by the consistency term |

### Training
| Script | Purpose |
|---|---|
| `train_uwf_multitask.py` | Trains the multitask model. `--lesion-loss asl --consistency-weight 0` is the baseline; `--consistency-weight 0.10` is GLC-Full. `--audit-only` checks the manifests without importing PyTorch. The default mode is a short smoke test; use `--mode full` for the 60-epoch recipe. The script never reads a test manifest |
| `train_uwf_multitask_resume.py` | Resumes an interrupted run from its checkpoint |
| `train_uwf_multitask_ablation_resume.py` | Resume script for the any-lesion-only ablation (GLC-Any) |
| `train_uwf_multitask_diagnostic_noramp.py` | Diagnostic run with a larger consistency weight and no warm-up |
| `train_uwf_multitask_diagnostic_noramp_resume.py` | Resume script for the diagnostic run |
| `consistency_loss.py` | Grade–lesion consistency term (Jensen–Shannon divergence between aggregated lesion probabilities and grade probabilities) |
| `mmrdr_dataset.py` | Dataset and preprocessing (square pad, resize to 512, ImageNet normalisation) |

Example (baseline, seed 42):

```
python train_uwf_multitask.py --train-manifest PATH\train.csv --val-manifest PATH\val.csv ^
    --image-root PATH\MMRDR-UWF --mode full --seed 42 ^
    --lesion-loss asl --consistency-weight 0 --out-dir PATH\runs\baseline42
```

### Test-set evaluation and statistics
| Script | Purpose |
|---|---|
| `evaluate_test_uwf.py` | Evaluates trained checkpoints on the test partition |
| `significance_test_uwf.py` | Paired bootstrap comparison of baseline and GLC-Full |
| `agreement_sensitivity.py` | Grade–lesion disagreement under the noisy-OR, maximum-probability and thresholded-OR definitions |
| `compare_val_runs.py` | Validation-set comparison across runs |
| `test_report.py`, `test_report_v3.py` | Test-set reports |
| `generate_paper_tables_figures.py` | Builds the tables and figures from saved results |

### Diagnostics
| Script | Purpose |
|---|---|
| `grad_diag.py` | Gradient diagnostic for the grade, lesion and consistency losses |
| `gradcam_val.py` | Grad-CAM maps on validation images |
| `run_grad_diag_s52.bat` | Windows launcher for the gradient diagnostic on seed 52 |

### External evaluation (no retraining)
The six trained checkpoints are run frozen on two public UWF sets.

| Script | Purpose |
|---|---|
| `build_peng_manifest.py` | Builds a manifest from the Peng et al. release (removes repeated rows and conflicting labels, skips missing files, writes an audit log) |
| `build_open_uwf_manifest.py` | Builds a manifest from the open UWF set (images, diagnosis, patient, eye, quality) |
| `external_predict.py` | Writes per-image predictions for each checkpoint. Needs `--train-script`, a training script that defines `load_training_components()` and `LESIONS` |
| `external_eval.py` | Shared functions (reading predictions, noisy-OR, thresholded OR) |
| `external_eval_uwf3.py` | Peng et al.: three-class grading metrics, disagreement rates, paired bootstrap at image and patient-cluster level |
| `external_eval_open.py` | Open UWF set: retinal-detachment AP and AUC (diagnosis-defined), disagreement rates, bootstrap at image and patient level. `--subset gradable` restricts to images rated gradable |
| `run_peng_external.bat` | Windows launcher for the Peng et al. steps |

### Earlier colour-fundus scripts
`train_resnet50.py`, `evaluate_cfp_test.py` and `verify_resnet50_checkpoints.py` belong to an earlier colour-fundus (CFP) experiment and are not used for the results in the paper.

## Notes on interpretation
- Disagreement rates on the external sets show whether the models keep applying the grade–lesion relationship under distribution shift. They do not show that the relationship is appropriate for other diagnoses.
- The retinal-detachment analysis on the open UWF set uses diagnosis groups as labels. It measures diagnosis-defined discrimination, not lesion recognition.

## Citation
To be added after publication.
