@echo off
setlocal
cd /d "%~dp0"
set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" set "CONDA_EXE=conda"
echo This removes ONLY the Conda environment named nmrfilter.
choice /M "Continue"
if errorlevel 2 exit /b 0
"%CONDA_EXE%" env remove -n nmrfilter -y
pause
