@echo off
cd /d "%~dp0"
echo ===================================================
echo   BrainVision: EEG topomaps, CV features, PR models
echo ===================================================
pip install -r requirements.txt --quiet

echo.
echo [1/4] Generating brain topomap images...
python 1_generate_topomaps.py

echo.
echo [2/4] Computer vision feature extraction...
python 2_cv_feature_extraction.py

echo.
echo [3/4] Pattern recognition models...
python 3_pr_classifier_demo.py

echo.
echo [4/4] Exporting data for the website...
python website\build_data.py

echo.
echo Done. Open website\index.html in your browser.
pause
