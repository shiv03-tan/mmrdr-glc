@echo off
REM Peng et al. UWF external test. Paths are already set for your computer. Run from C:\Research\mmrdr-own-study\scripts
set STUDY=C:\Research\mmrdr-own-study
set DATA_ROOT=%STUDY%\data\peng\Ultra-wide-field (SLO) fundus image dataset for intelligent diabetic retinopathy system\diabetic retinopathy
set RUNS=%STUDY%\runs
set OUT=%STUDY%\external_peng
set TRAIN_SCRIPT=%RUNS%\uwf_mt_asl_seed52_resumed_v2\train_uwf_multitask_used.py

REM label = run folder (checkpoint best.pt) and validation-prediction file (used only for thresholds)
set R_base42=%RUNS%\uwf_mt_asl_seed42
set R_base52=%RUNS%\uwf_mt_asl_seed52_resumed_v2
set R_base62=%RUNS%\uwf_mt_asl_seed62
set R_glc42=%RUNS%\uwf_mt_asl_glc010_seed42
set R_glc52=%RUNS%\uwf_mt_asl_glc010_seed52
set R_glc62=%RUNS%\uwf_mt_asl_glc010_seed62
set V_base42=%STUDY%\baseline42_val.csv
set V_base52=%STUDY%\baseline52_val.csv
set V_base62=%STUDY%\baseline62_val.csv
set V_glc42=%STUDY%\glc010_42_val.csv
set V_glc52=%STUDY%\glc010_52_val.csv
set V_glc62=%STUDY%\glc010_62_val.csv

if not exist "%DATA_ROOT%\Normal512" ( echo DATA_ROOT is wrong: no Normal512 folder in "%DATA_ROOT%" & exit /b 1 )
if not exist "%TRAIN_SCRIPT%" ( echo TRAIN_SCRIPT not found: "%TRAIN_SCRIPT%" & echo Run: dir "%RUNS%\uwf_mt_asl_seed52_resumed_v2\*.py" and send me the list & exit /b 1 )
call :chk "%R_base42%" "%V_base42%" || exit /b 1
call :chk "%R_base52%" "%V_base52%" || exit /b 1
call :chk "%R_base62%" "%V_base62%" || exit /b 1
call :chk "%R_glc42%" "%V_glc42%" || exit /b 1
call :chk "%R_glc52%" "%V_glc52%" || exit /b 1
call :chk "%R_glc62%" "%V_glc62%" || exit /b 1
if not exist "%OUT%" mkdir "%OUT%"

pip install pandas scikit-learn >nul
python build_peng_manifest.py --root "%DATA_ROOT%" --out "%OUT%\peng_manifest.csv" || exit /b 1

python external_predict.py --train-script "%TRAIN_SCRIPT%" --manifest "%OUT%\peng_manifest.csv" --image-root "%DATA_ROOT%" ^
  --ckpt base42="%R_base42%\best.pt" --ckpt base52="%R_base52%\best.pt" --ckpt base62="%R_base62%\best.pt" ^
  --ckpt glc42="%R_glc42%\best.pt" --ckpt glc52="%R_glc52%\best.pt" --ckpt glc62="%R_glc62%\best.pt" ^
  --out-dir "%OUT%\preds" --batch-size 16 || exit /b 1

REM PRIMARY: official test split (326 images)
python external_eval_uwf3.py --manifest "%OUT%\peng_manifest.csv" --out-dir "%OUT%\results_test" --split test --boot 10000 ^
  --run base42="%OUT%\preds\base42_external_predictions.csv","%V_base42%" ^
  --run base52="%OUT%\preds\base52_external_predictions.csv","%V_base52%" ^
  --run base62="%OUT%\preds\base62_external_predictions.csv","%V_base62%" ^
  --run glc42="%OUT%\preds\glc42_external_predictions.csv","%V_glc42%" ^
  --run glc52="%OUT%\preds\glc52_external_predictions.csv","%V_glc52%" ^
  --run glc62="%OUT%\preds\glc62_external_predictions.csv","%V_glc62%" || exit /b 1

REM SECONDARY: all 1,606 unique images (none was used to train your models)
python external_eval_uwf3.py --manifest "%OUT%\peng_manifest.csv" --out-dir "%OUT%\results_all" --boot 10000 ^
  --run base42="%OUT%\preds\base42_external_predictions.csv","%V_base42%" ^
  --run base52="%OUT%\preds\base52_external_predictions.csv","%V_base52%" ^
  --run base62="%OUT%\preds\base62_external_predictions.csv","%V_base62%" ^
  --run glc42="%OUT%\preds\glc42_external_predictions.csv","%V_glc42%" ^
  --run glc52="%OUT%\preds\glc52_external_predictions.csv","%V_glc52%" ^
  --run glc62="%OUT%\preds\glc62_external_predictions.csv","%V_glc62%"

echo Done. Send me: %OUT%\peng_manifest_audit.txt and the two csv files in results_test and in results_all.
goto :eof

:chk
if not exist "%~1\best.pt" ( echo missing %~1\best.pt & exit /b 1 )
if not exist "%~2" ( echo missing %~2 & exit /b 1 )
exit /b 0
