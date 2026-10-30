# run_app.ps1 - Safely stop existing instances and start Streamlit on pinned port 8501
$ErrorActionPreference = "Continue"

Write-Host "Checking for existing instances on ports 8501 and 8502..." -ForegroundColor Cyan

# Stop processes listening on 8501 or 8502
$ports = @(8501, 8502)
foreach ($p in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue
    if ($conns) {
        $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $pids) {
            if ($procId -and $procId -ne $PID) {
                Write-Host "Stopping process $procId holding port $p..." -ForegroundColor Yellow
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Start-Sleep -Seconds 1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$pythonPath = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Error "Virtual environment Python not found at: $pythonPath"
    exit 1
}

Write-Host "Starting SkinDiseaseAI app on http://localhost:8501 ..." -ForegroundColor Green
& $pythonPath -m streamlit run app.py --server.port 8501
