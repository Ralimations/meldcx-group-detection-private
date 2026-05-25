param(
    [string]$BackendPort = "8765",
    [string]$FrontendPort = "5173"
)

$ErrorActionPreference = "Stop"

# Set environment for Python cache
$cacheRoot = Join-Path $PSScriptRoot ".cache\pycache"
$env:PYTHONPYCACHEPREFIX = $cacheRoot
if (-not (Test-Path $env:PYTHONPYCACHEPREFIX)) {
    New-Item -Path $env:PYTHONPYCACHEPREFIX -ItemType Directory | Out-Null
}

Write-Host "--- Starting Group Detection Dashboard ---" -ForegroundColor Cyan
Write-Host "Backend:  http://localhost:$BackendPort"
Write-Host "Frontend: http://localhost:$FrontendPort"
Write-Host "Press Ctrl+C to stop both processes.`n" -ForegroundColor Yellow

$jobs = @()

# Start Backend
Write-Host "[Launcher] Starting Backend..." -ForegroundColor Gray
$backendProc = Start-Process python -ArgumentList "dashboard_server.py" -NoNewWindow -PassThru

# Start Frontend
Write-Host "[Launcher] Starting Frontend..." -ForegroundColor Gray
$frontendProc = Start-Process cmd -ArgumentList "/c npm run dev" -WorkingDirectory "dashboard" -NoNewWindow -PassThru

try {
    # Loop while both processes are running
    while (-not $backendProc.HasExited -and -not $frontendProc.HasExited) {
        Start-Sleep -Milliseconds 500
    }
    
    if ($backendProc.HasExited) {
        Write-Host "`n[Launcher] Backend process has exited unexpectedly." -ForegroundColor Red
    }
    if ($frontendProc.HasExited) {
        Write-Host "`n[Launcher] Frontend process has exited unexpectedly." -ForegroundColor Red
    }
}
catch {
    # This block handles external interrupts like Ctrl+C
    Write-Host "`n[Launcher] Interrupt received." -ForegroundColor Yellow
}
finally {
    Write-Host "[Launcher] Stopping all processes..." -ForegroundColor Yellow
    
    # Use taskkill /T to ensure process trees are terminated (kills children like vite)
    if ($null -ne $backendProc -and -not $backendProc.HasExited) {
        taskkill /F /T /PID $backendProc.Id 2>$null | Out-Null
    }
    if ($null -ne $frontendProc -and -not $frontendProc.HasExited) {
        taskkill /F /T /PID $frontendProc.Id 2>$null | Out-Null
    }
    
    Write-Host "[Launcher] Dashboard stopped." -ForegroundColor Cyan
}
