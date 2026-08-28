@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title VoxCPM-2 Pytorch Turkce Versiyon - UI Kurulum

echo.
echo ================================================================
echo VoxCPM-2 Pytorch Turkce Versiyon - UI
echo Otomatik Kurulum
echo ================================================================
echo.
echo Python 3.11 yoksa otomatik kurulacak.
echo .venv, FFmpeg, PyTorch, VoxCPM ve model dosyalari hazirlanacak.
echo.
echo Ilk kurulum internet hizina gore uzun surebilir.
echo Bu pencereyi kapatmayin.
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"

if errorlevel 1 (
    echo.
    echo ================================================================
    echo KURULUM BASARISIZ
    echo ================================================================
    echo.
    echo Ayrinti icin logs klasorundeki en yeni install_*.log dosyasina bakin.
    echo.
    pause
    exit /b 1
)

echo.
echo Kurulum tamamlandi. UI yeni pencerede aciliyor.
timeout /t 3 /nobreak >nul
exit /b 0
