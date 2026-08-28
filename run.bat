@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title VoxCPM-2 Pytorch Turkce Versiyon - UI

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "FFMPEG_BIN=%~dp0runtime\ffmpeg\bin"

if not exist "%VENV_PYTHON%" (
    echo.
    echo Kurulum bulunamadi.
    echo Once install.bat dosyasini calistirin.
    echo.
    pause
    exit /b 1
)

if not exist "%FFMPEG_BIN%\ffmpeg.exe" (
    echo.
    echo FFmpeg bulunamadi.
    echo install.bat dosyasini tekrar calistirin.
    echo.
    pause
    exit /b 1
)

set "PATH=%FFMPEG_BIN%;%PATH%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_TELEMETRY=1"

echo.
echo ================================================================
echo VoxCPM-2 Pytorch Turkce Versiyon - UI
echo ================================================================
echo.
echo Tarayici otomatik acilacak.
echo 7861 doluysa 7862-7870 arasinda ilk bos port secilecek.
echo UI acikken bu terminal penceresini kapatmayin.
echo.

"%VENV_PYTHON%" "%~dp0voxcpm2_pytorch_webui.py"

if errorlevel 1 (
    echo.
    echo UI bir hata ile kapandi.
    echo Yukaridaki hata mesajini kontrol edin.
    echo.
    pause
)

endlocal
