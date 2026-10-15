@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist "E:\anaconda3\envs\animedev\python.exe" (
  echo [ERROR] animedev interpreter not found:
  echo         E:\anaconda3\envs\animedev\python.exe
  pause
  exit /b 1
)
echo HomeEase starting with the animedev environment...
echo.
"E:\anaconda3\envs\animedev\python.exe" run.py %*
pause
