<#
.SYNOPSIS
    Đóng gói VieNeu Voice Clone thành app Windows chạy độc lập (CPU/ONNX).

.DESCRIPTION
    Script tạo một venv sạch, cài đúng phần phụ thuộc torch-free của app, rồi
    chạy PyInstaller theo packaging/vieneu_clone.spec.

    Kết quả: packaging\dist\VieNeuVoiceClone\  (kèm VieNeuVoiceClone.exe)
    Kèm -Zip: packaging\dist\VieNeuVoiceClone-win64.zip

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -Zip

.NOTES
    Yêu cầu: Windows x64, Python 3.10–3.13 trong PATH.
    Bản build phải chạy TRÊN Windows — PyInstaller không cross-compile.
#>
param(
    [switch]$Zip,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$PackagingDir = $PSScriptRoot
$ProjectRoot  = Split-Path -Parent $PackagingDir
$VenvDir      = Join-Path $PackagingDir ".venv-build"
$DistDir      = Join-Path $PackagingDir "dist"
$BuildDir     = Join-Path $PackagingDir "build"

Write-Host "=== VieNeu Voice Clone — build Windows ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"

if ($Clean) {
    Write-Host ">> Dọn build cũ..." -ForegroundColor Yellow
    foreach ($d in @($DistDir, $BuildDir, $VenvDir)) {
        if (Test-Path $d) { Remove-Item -Recurse -Force $d }
    }
}

# --- Python ---
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Không tìm thấy Python trong PATH. Cài Python 3.12 từ https://www.python.org/downloads/ rồi chạy lại."
}
& python --version

# --- venv build ---
if (-not (Test-Path $VenvDir)) {
    Write-Host ">> Tạo venv build tại $VenvDir ..." -ForegroundColor Yellow
    & python -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host ">> Cài phụ thuộc (torch-free, CPU/ONNX)..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip wheel
# Cài chính project ở chế độ editable => kéo đúng dependency core trong pyproject.
& $VenvPython -m pip install -e "$ProjectRoot"
& $VenvPython -m pip install pyinstaller==6.22.3

# Chốt lại: nếu môi trường lỡ có torch thì loại ra, để bundle không phình ~2 GB.
& $VenvPython -m pip uninstall -y torch torchaudio 2>$null | Out-Null

# --- PyInstaller ---
Write-Host ">> Chạy PyInstaller..." -ForegroundColor Yellow
Push-Location $PackagingDir
try {
    & $VenvPython -m PyInstaller "vieneu_clone.spec" --noconfirm --clean
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller thất bại (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}

$AppDir = Join-Path $DistDir "VieNeuVoiceClone"
$Exe    = Join-Path $AppDir "VieNeuVoiceClone.exe"
if (-not (Test-Path $Exe)) { throw "Không thấy $Exe — build không thành công." }

# Kèm hướng dẫn ngắn cạnh exe cho người dùng cuối.
$readmeSrc = Join-Path $PackagingDir "DOC_NGUOI_DUNG.txt"
if (Test-Path $readmeSrc) { Copy-Item $readmeSrc (Join-Path $AppDir "DOC_TRUOC_KHI_CHAY.txt") -Force }

$size = [math]::Round(((Get-ChildItem $AppDir -Recurse | Measure-Object Length -Sum).Sum / 1GB), 2)
Write-Host ""
Write-Host "✅ Build xong: $Exe" -ForegroundColor Green
Write-Host "   Kích thước thư mục: $size GB"

if ($Zip) {
    $ZipPath = Join-Path $DistDir "VieNeuVoiceClone-win64.zip"
    Write-Host ">> Nén $ZipPath ..." -ForegroundColor Yellow
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Compress-Archive -Path $AppDir -DestinationPath $ZipPath
    Write-Host "✅ Zip: $ZipPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "Chạy thử:  $Exe" -ForegroundColor Cyan
Write-Host "Lần đầu app tải model (~vài trăm MB) về %LOCALAPPDATA%\VieNeuVoiceClone\models."
