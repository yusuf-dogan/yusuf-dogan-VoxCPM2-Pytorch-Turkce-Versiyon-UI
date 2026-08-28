param()

# Windows PowerShell 5.1 Türkçe karakter uyumluluğu
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root "runtime"
$Downloads = Join-Path $Runtime "downloads"
$LocalPythonDir = Join-Path $Runtime "python311"
$LocalPythonExe = Join-Path $LocalPythonDir "python.exe"
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$FfmpegDir = Join-Path $Runtime "ffmpeg"
$FfmpegBin = Join-Path $FfmpegDir "bin"
$Logs = Join-Path $Root "logs"

$PythonVersion = "3.11.9"
$PythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

$FfmpegVersion = "7.1.1"
$FfmpegUrl = "https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-full_build-shared.zip"
$FfmpegSha256 = "9F28727E8B472A04C1D2E520AAA425DCA82721B995139B35710091130EA6E699"

New-Item -ItemType Directory -Force -Path $Runtime, $Downloads, $Logs | Out-Null

$Transcript = Join-Path $Logs ("install_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
try { Start-Transcript -Path $Transcript -Force | Out-Null } catch {}

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "====================================================================" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "====================================================================" -ForegroundColor Cyan
}

function Download-File([string]$Url, [string]$Destination) {
    if (Test-Path $Destination) {
        Write-Host "Dosya zaten mevcut: $Destination"
        return
    }

    Write-Host "İndiriliyor: $Url"
    $Curl = Get-Command "curl.exe" -ErrorAction SilentlyContinue

    if ($Curl) {
        & $Curl.Source -L --fail --retry 5 --retry-delay 3 -o $Destination $Url
        if ($LASTEXITCODE -ne 0) {
            throw "İndirme başarısız: $Url"
        }
    } else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination
    }
}

function Test-Python311([string]$Exe, [string[]]$PrefixArgs) {
    try {
        $Code = "import sys,struct; print(sys.executable); print(f'{sys.version_info.major}.{sys.version_info.minor}'); print(struct.calcsize('P')*8)"
        $Result = & $Exe @PrefixArgs "-c" $Code 2>$null
        if ($LASTEXITCODE -ne 0 -or $Result.Count -lt 3) {
            return $null
        }

        if ($Result[1].Trim() -eq "3.11" -and $Result[2].Trim() -eq "64") {
            return $Result[0].Trim()
        }
    } catch {}

    return $null
}

function Find-Python311 {
    $Py = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($Py) {
        $Found = Test-Python311 $Py.Source @("-3.11")
        if ($Found) { return $Found }
    }

    $Python = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($Python) {
        $Found = Test-Python311 $Python.Source @()
        if ($Found) { return $Found }
    }

    if (Test-Path $LocalPythonExe) {
        $Found = Test-Python311 $LocalPythonExe @()
        if ($Found) { return $Found }
    }

    return $null
}

function Invoke-VenvPython([string[]]$Arguments) {
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python komutu başarısız: $($Arguments -join ' ')"
    }
}

Write-Host ""
Write-Host "VoxCPM-2 Pytorch Türkçe Versiyon - UI" -ForegroundColor Green
Write-Host "Otomatik Windows Kurulumu"
Write-Host ""
Write-Host "İlk kurulum internet bağlantısı gerektirir."
Write-Host "Model ve bağımlılık indirmeleri birkaç GB olabilir."
Write-Host ""

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Yalnız 64-bit Windows desteklenmektedir."
}

# ------------------------------------------------------------------
# 1/7 Python
# ------------------------------------------------------------------
Write-Step "1/7 - Python 3.11 hazırlanıyor"

$BasePython = Find-Python311

if ($BasePython) {
    Write-Host "Uygun mevcut Python 3.11 bulundu: $BasePython"
} else {
    Write-Host "Uygun Python 3.11 bulunamadı. Python $PythonVersion otomatik kurulacak."

    $Installer = Join-Path $Downloads "python-$PythonVersion-amd64.exe"
    Download-File $PythonUrl $Installer

    $Signature = Get-AuthenticodeSignature $Installer
    if ($Signature.Status -ne "Valid") {
        throw "Python kurulum dosyasının Authenticode imzası doğrulanamadı."
    }

    $Args = @(
        "/quiet",
        "InstallAllUsers=0",
        "TargetDir=`"$LocalPythonDir`"",
        "Include_pip=1",
        "Include_launcher=0",
        "Include_test=0",
        "Include_doc=0",
        "Include_tcltk=0",
        "AssociateFiles=0",
        "Shortcuts=0",
        "PrependPath=0",
        "CompileAll=0"
    )

    $P = Start-Process -FilePath $Installer -ArgumentList $Args -Wait -PassThru
    if ($P.ExitCode -ne 0) {
        throw "Python kurulumu başarısız. Exit code: $($P.ExitCode)"
    }

    if (-not (Test-Path $LocalPythonExe)) {
        throw "Yerel Python kurulumu bulunamadı: $LocalPythonExe"
    }

    $BasePython = $LocalPythonExe
}

$BasePythonVersion = & $BasePython -c "import sys; print('.'.join(map(str,sys.version_info[:3])))"
Write-Host "Kullanılacak Python: $BasePython ($BasePythonVersion)"

# ------------------------------------------------------------------
# 2/7 FFmpeg Shared
# ------------------------------------------------------------------
Write-Step "2/7 - FFmpeg Shared $FfmpegVersion hazırlanıyor"

$FfmpegExe = Join-Path $FfmpegBin "ffmpeg.exe"

if (-not (Test-Path $FfmpegExe)) {
    $FfmpegZip = Join-Path $Downloads "ffmpeg-$FfmpegVersion-full_build-shared.zip"
    Download-File $FfmpegUrl $FfmpegZip

    $Hash = (Get-FileHash $FfmpegZip -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($Hash -ne $FfmpegSha256) {
        Remove-Item $FfmpegZip -Force -ErrorAction SilentlyContinue
        throw "FFmpeg SHA256 doğrulaması başarısız. Dosya silindi; install.bat dosyasını tekrar çalıştırın."
    }

    $Temp = Join-Path $Runtime "ffmpeg_extract"
    Remove-Item $Temp -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $Temp | Out-Null

    Expand-Archive -Path $FfmpegZip -DestinationPath $Temp -Force
    $Inner = Get-ChildItem $Temp -Directory | Select-Object -First 1

    if (-not $Inner) {
        throw "FFmpeg ZIP içeriği beklenen biçimde değil."
    }

    Remove-Item $FfmpegDir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $FfmpegDir | Out-Null

    Get-ChildItem $Inner.FullName -Force | ForEach-Object {
        Move-Item $_.FullName $FfmpegDir -Force
    }

    Remove-Item $Temp -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $FfmpegExe)) {
    throw "FFmpeg hazırlanamadı: $FfmpegExe"
}

$env:PATH = "$FfmpegBin;$env:PATH"
Write-Host "FFmpeg hazır: $FfmpegExe"

# ------------------------------------------------------------------
# 3/7 venv
# ------------------------------------------------------------------
Write-Step "3/7 - Python sanal ortamı (.venv) hazırlanıyor"

if (Test-Path $VenvPython) {
    $VenvVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($VenvVersion -ne "3.11") {
        Write-Host "Mevcut .venv Python 3.11 değil; yeniden oluşturuluyor."
        Remove-Item $VenvDir -Recurse -Force
    }
}

if (-not (Test-Path $VenvPython)) {
    & $BasePython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw ".venv oluşturulamadı."
    }
}

Write-Host ".venv hazır: $VenvDir"

# ------------------------------------------------------------------
# 4/7 Packages
# ------------------------------------------------------------------
Write-Step "4/7 - PyTorch CUDA 12.8 ve Python paketleri kuruluyor"

Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Invoke-VenvPython @(
    "-m", "pip", "install",
    "-r", (Join-Path $Root "requirements-torch-cu128.txt")
)

Invoke-VenvPython @(
    "-m", "pip", "install",
    "--upgrade-strategy", "only-if-needed",
    "-r", (Join-Path $Root "requirements.txt")
)

Invoke-VenvPython @("-m", "pip", "check")

# ------------------------------------------------------------------
# Windows TorchCodec - FFmpeg DLL düzeltmesi
# ------------------------------------------------------------------
Write-Host ""
Write-Host "TorchCodec için FFmpeg DLL'leri hazırlanıyor..."

$SitePackages = & $VenvPython -c "import site; print(site.getsitepackages()[0])"

if ($LASTEXITCODE -ne 0) {
    throw "Python site-packages klasörü bulunamadı."
}

$TorchCodecDir = Join-Path $SitePackages "torchcodec"

if (-not (Test-Path $TorchCodecDir)) {
    throw "TorchCodec klasörü bulunamadı: $TorchCodecDir"
}

$FfmpegDlls = Get-ChildItem `
    -Path $FfmpegBin `
    -Filter "*.dll" `
    -File

if (-not $FfmpegDlls) {
    throw "FFmpeg DLL dosyaları bulunamadı: $FfmpegBin"
}

foreach ($Dll in $FfmpegDlls) {
    Copy-Item `
        -Path $Dll.FullName `
        -Destination $TorchCodecDir `
        -Force
}

Write-Host "$($FfmpegDlls.Count) FFmpeg DLL dosyası TorchCodec klasörüne kopyalandı."

# ------------------------------------------------------------------
# 5/7 Model
# ------------------------------------------------------------------
Write-Step "5/7 - VoxCPM2 model dosyaları indiriliyor"

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

Invoke-VenvPython @((Join-Path $Root "scripts\download_model.py"))

# ------------------------------------------------------------------
# 6/7 Verify
# ------------------------------------------------------------------
Write-Step "6/7 - Kurulum doğrulanıyor"

Invoke-VenvPython @((Join-Path $Root "scripts\verify_install.py"))

# ------------------------------------------------------------------
# 7/7 Finish
# ------------------------------------------------------------------
Write-Step "7/7 - Kurulum tamamlandı"

$Marker = Join-Path $Root ".install_complete"
@"
VoxCPM-2 Pytorch Türkçe Versiyon - UI
Installed: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Base Python: $BasePythonVersion
Torch: 2.8.0+cu128
VoxCPM: 2.0.3
TorchCodec: 0.7.0
FFmpeg Shared: 7.1.1
"@ | Set-Content -Path $Marker -Encoding UTF8

Write-Host ""
Write-Host "KURULUM BAŞARILI." -ForegroundColor Green
Write-Host "UI birkaç saniye içinde yeni pencerede başlatılacak."
Write-Host "Sonraki kullanımlarda yalnız run.bat dosyasını çalıştırın."
Write-Host "Kurulum günlüğü: $Transcript"
Write-Host ""

try { Stop-Transcript | Out-Null } catch {}

Start-Process -FilePath (Join-Path $Root "run.bat")
