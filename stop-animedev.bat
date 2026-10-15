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
echo Stopping HomeEase (models, streams and database will be released)...
"E:\anaconda3\envs\animedev\python.exe" run.py --stop
timeout /t 3 /nobreak >nul
echo Done.
pause
