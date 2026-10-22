@echo off
cd /d C:\Research\mmrdr-own-study
python scripts\grad_diag.py --train-script runs\uwf_mt_asl_seed52_resumed_v2\train_uwf_multitask_used.py --train-manifest splits\uwf_train.csv --val-manifest splits\uwf_val.csv --image-root C:\Datasets\MMRDR\MMRDR-UWF --batch-size 4 --ckpt baseline52=runs\uwf_mt_asl_seed52_resumed_v2\best.pt --ckpt glc_best=runs\uwf_mt_asl_glc010_seed52\best.pt --ckpt glc_last=runs\uwf_mt_asl_glc010_seed52\last.pt --out-dir grad_diag_s52
pause
