Param(
    [string]$PythonExe = "py"
)

$ErrorActionPreference = "Stop"

# Go to the scripts directory
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir
Write-Host "[INFO] Using project directory: $projectDir"

# Creates a virtual environment
$venvPythonPath = Join-Path $projectDir ".venv\Scripts\python.exe"

if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] Creating virtual environment..."
    & $PythonExe -m venv ".venv"
} else {
    Write-Host "[INFO] Virtual environment folder exists, checking python..."
}

if (-not (Test-Path $venvPythonPath)) {
    Write-Host "[WARN] .venv exists but python.exe not found inside. Recreating venv..."
    Remove-Item -Recurse -Force ".venv"
    & $PythonExe -m venv ".venv"
}

$venvPython = $venvPythonPath
$venvPip    = Join-Path $projectDir ".venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Could not find venv python at $venvPython"
    exit 1
}

Write-Host "[INFO] Using venv Python at: $venvPython"

# Install Python packages
Write-Host "[INFO] Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "[INFO] Installing required packages (opencv-contrib-python, numpy)..."
& $venvPip install opencv-contrib-python numpy

# Run training script
Write-Host "[INFO] Running training script..."
& $venvPython ".\src\train_recognizer.py"

# Run webcam recognition
Write-Host "[INFO] Starting webcam recognition (press 'q' in the window to quit)..."
& $venvPython ".\src\webcam_recognition.py"
