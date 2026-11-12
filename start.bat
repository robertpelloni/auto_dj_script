@echo off
setlocal
cd /d "%~dp0"

set "PY_CMD=python"
if exist "C:\Python314\python.exe" set "PY_CMD=C:\Python314\python.exe"

if "%~1"=="" (
    echo [*] Launching Auto DJ Command Console GUI...
    "%PY_CMD%" auto_dj.py --gui
) else (
    "%PY_CMD%" auto_dj.py %*
)
endlocal
