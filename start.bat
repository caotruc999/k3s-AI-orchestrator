@echo off
REM Bam dup vao file nay de khoi dong du an: minikube -> server AI.
REM Yeu cau: Docker Desktop da mo va da chay xong (doi het icon xoay) truoc khi bam file nay.

cd /d "%~dp0"

echo ==============================================
echo  Buoc 1/2: Khoi dong minikube (cluster K3s gia lap)...
echo ==============================================
minikube start
if errorlevel 1 (
    echo.
    echo [LOI] minikube start that bai. Kiem tra lai Docker Desktop da mo chua.
    pause
    exit /b 1
)

echo.
echo ==============================================
echo  Buoc 2/2: Khoi dong server AI (che do REAL)...
echo ==============================================
call venv\Scripts\activate.bat
python modules\ai_service.py

pause
