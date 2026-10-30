# Windows setup: isolated virtual environment in .venv + pinned dependencies.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Prefer Python 3.11 via the py launcher; fall back to whatever `python` is.
$py = $null
foreach ($candidate in @("py -3.11", "py -3.12", "py -3.10", "python")) {
    try {
        $ver = Invoke-Expression "$candidate -c `"import sys; print('%d.%d' % sys.version_info[:2])`"" 2>$null
        if ($ver -match '^3\.(10|11|12)$') { $py = $candidate; break }
    } catch { }
}
if (-not $py) { throw "Python 3.10-3.12 not found. Install Python 3.11 from https://www.python.org/downloads/" }
Write-Host "Using $py"

if (-not (Test-Path .venv)) { Invoke-Expression "$py -m venv .venv" }
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install --retries 10 --timeout 120 -r requirements.txt

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example - add your GEMINI_API_KEY to enable the AI steps (optional)."
}

$tess = (Get-Command tesseract -ErrorAction SilentlyContinue) -or (Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe")
if (-not $tess) {
    Write-Warning "Tesseract OCR not found. Install it with:  winget install UB-Mannheim.TesseractOCR"
}

& $venvPy scripts\check_setup.py
Write-Host "`nActivate the environment with:  .\.venv\Scripts\Activate.ps1"
