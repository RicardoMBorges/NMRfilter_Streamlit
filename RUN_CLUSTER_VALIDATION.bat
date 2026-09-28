@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo   NMRfilter v11 - clustering known-positive validation
echo ============================================================
set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" set "CONDA_EXE=conda"
"%CONDA_EXE%" run -n nmrfilter --no-capture-output python tests\test_clustering_semantics.py
if errorlevel 1 (
  echo.
  echo VALIDATION FAILED
  pause
  exit /b 1
)
echo.
echo VALIDATION PASSED
pause
