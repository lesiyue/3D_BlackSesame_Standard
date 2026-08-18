@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo ========================================
echo 3D 標注閉環工具 - 啟動中
echo ========================================

where python >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 Python，請先雙擊 install.bat
    pause
    exit /b 1
)

python -X utf8 server.py

pause
