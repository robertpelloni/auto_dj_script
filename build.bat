@echo off
setlocal
cd /d "%~dp0"

set "PY_CMD=python"
if exist "C:\Python314\python.exe" set "PY_CMD=C:\Python314\python.exe"

echo [1/3] Compiling Python modules...
"%PY_CMD%" -m compileall -q autodj config.py auto_dj.py
if errorlevel 1 exit /b 1

echo [2/3] Verifying CLI entrypoint...
"%PY_CMD%" auto_dj.py --version
if errorlevel 1 exit /b 1

echo [3/3] Running test suite...
set "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
"%PY_CMD%" -m pytest -q
if errorlevel 1 exit /b 1

<nul set /p =success> .build_success
echo [*] Build and verification succeeded.
endlocal
