@echo off
setlocal
cd /d "%~dp0"
set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" (
  echo Could not find conda.exe.
  pause
  exit /b 1
)
echo [1/2] Testing measured-spectrum parser...
"%CONDA_EXE%" run -n nmrfilter --no-capture-output python tests\test_spectrum_io.py
if errorlevel 1 goto :fail
echo [2/2] Testing legacy clustering semantics...
"%CONDA_EXE%" run -n nmrfilter --no-capture-output python tests\test_clustering_semantics.py
if errorlevel 1 goto :fail
echo.
echo ALL INPUT + CLUSTER VALIDATION TESTS PASSED.
pause
exit /b 0
:fail
echo.
echo VALIDATION FAILED.
pause
exit /b 1
