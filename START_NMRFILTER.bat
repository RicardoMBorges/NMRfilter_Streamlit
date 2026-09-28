@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title NMRfilter Streamlit Launcher v7
set "ROOT=%~dp0"
set "APP=%~dp0app.py"
set "LOG=%~dp0nmrfilter_startup.log"
set "PORT=8517"
set "URL=http://127.0.0.1:%PORT%"

>"%LOG%" echo NMRfilter startup log - %DATE% %TIME%
echo ============================================================
echo   NMRfilter Streamlit - Windows launcher v7
echo ============================================================
echo.

echo [1/6] Locating Conda...
set "CONDA_CMD="
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set "CONDA_CMD=C:\ProgramData\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "C:\ProgramData\anaconda3\Scripts\conda.exe" set "CONDA_CMD=C:\ProgramData\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%I"
if not defined CONDA_CMD (
  echo [ERROR] Conda was not found.
  >>"%LOG%" echo [ERROR] Conda was not found.
  goto :fail
)
echo Conda: !CONDA_CMD!
>>"%LOG%" echo Conda: !CONDA_CMD!
"!CONDA_CMD!" --version >>"%LOG%" 2>&1
if errorlevel 1 goto :fail

if not exist "%APP%" (
  echo [ERROR] app.py was not found beside START_NMRFILTER.bat
  >>"%LOG%" echo [ERROR] Missing: %APP%
  goto :fail
)

echo [2/6] Checking the nmrfilter environment...
"!CONDA_CMD!" env list >>"%LOG%" 2>&1
"!CONDA_CMD!" env list 2>nul | findstr /R /C:"^nmrfilter[ ]" >nul
if errorlevel 1 (
  echo Environment not found. Creating nmrfilter with Python 3.11 and Java 17...
  "!CONDA_CMD!" create -n nmrfilter -c conda-forge python=3.11 openjdk=17 pip -y >>"%LOG%" 2>&1
  if errorlevel 1 goto :fail
) else (
  echo Environment nmrfilter already exists.
)

echo [3/6] Installing/updating Python dependencies...
"!CONDA_CMD!" run -n nmrfilter python -m pip install --upgrade pip >>"%LOG%" 2>&1
if errorlevel 1 goto :fail
"!CONDA_CMD!" run -n nmrfilter python -m pip install -r "%ROOT%requirements.txt" >>"%LOG%" 2>&1
if errorlevel 1 goto :fail

echo [4/6] Validating Python, igraph, leidenalg and Streamlit...
"!CONDA_CMD!" run -n nmrfilter python -c "import sys,igraph,leidenalg,streamlit; assert sys.version_info[:2]==(3,11); print(sys.executable); print('igraph',igraph.__version__); print('leidenalg OK'); print('streamlit',streamlit.__version__)" >>"%LOG%" 2>&1
if errorlevel 1 goto :fail
echo Python environment validated.

echo [5/6] Validating Java...
"!CONDA_CMD!" run -n nmrfilter java -version >>"%LOG%" 2>&1
if errorlevel 1 goto :fail
echo Java validated.

echo [6/6] Starting NMRfilter on dedicated port %PORT%...
>>"%LOG%" echo APP=%APP%
>>"%LOG%" echo URL=%URL%

rem Do not reuse another application's server. Refuse to start if our dedicated port is occupied.
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName 127.0.0.1 -Port %PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Port %PORT% is already in use.
  >>"%LOG%" echo [ERROR] Port %PORT% is already in use.
  echo Close the process using %URL% and run this launcher again.
  goto :fail
)

echo Starting exactly: %APP%
echo NMRfilter URL: %URL%
echo Keep this window open while NMRfilter is running.
echo.

rem Open the dedicated NMRfilter URL after a short delay, while Streamlit starts in this window.
start "NMRfilter browser opener" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 40;$i++){try{$r=Invoke-WebRequest -UseBasicParsing -Uri '%URL%/_stcore/health' -TimeoutSec 1;if($r.StatusCode -eq 200){Start-Process '%URL%';exit 0}}catch{};Start-Sleep -Milliseconds 500};exit 1"

"!CONDA_CMD!" run --no-capture-output -n nmrfilter python -m streamlit run "%APP%" --server.address 127.0.0.1 --server.port %PORT% --server.headless true --browser.gatherUsageStats false
set "RC=!ERRORLEVEL!"
>>"%LOG%" echo Streamlit exit code: !RC!
if not "!RC!"=="0" goto :fail
exit /b 0

:fail
echo.
echo ============================================================
echo NMRfilter startup FAILED.
echo See: %LOG%
echo ============================================================
echo.
if exist "%LOG%" (
  echo Last log lines:
  powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 30" 2>nul
)
echo.
pause
exit /b 1
