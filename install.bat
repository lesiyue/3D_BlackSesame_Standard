@echo off
chcp 65001 >nul
setlocal

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo ========================================
echo 3D 標注閉環工具 - 依賴安裝
echo ========================================

where python >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 Python，請先安裝 Python 3.9+
    echo 下載地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

python -X utf8 --version

echo.
echo [1/4] 安裝 fastapi / uvicorn / pyyaml ...
python -X utf8 -m pip install --quiet fastapi uvicorn pyyaml

echo [2/4] 安裝 numpy ...
python -X utf8 -m pip install --quiet numpy

echo [3/4] 安裝 open3d ...
python -X utf8 -m pip install --quiet open3d

echo [4/4] 驗證安裝 ...
python -X utf8 -c "import fastapi, uvicorn, yaml, numpy, open3d; print('所有依賴已就緒')"

if errorlevel 1 (
    echo [錯誤] 依賴安裝失敗，請檢查網絡或手動執行 pip install
    pause
    exit /b 1
)

echo.
echo ========================================
echo 安裝完成！雙擊 start.bat 啟動服務
echo ========================================
pause
