param(
    [Parameter(Position = 0)]
    [ValidateSet("detect", "undistort", "init-source", "copy-source", "dashboard", "help")]
    [string]$Mode = "detect",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"

function Initialize-RepoEnvironment {
    $cacheRoot = Join-Path $PSScriptRoot ".cache\pycache"
    $env:PYTHONPYCACHEPREFIX = $cacheRoot
    if (-not (Test-Path $env:PYTHONPYCACHEPREFIX)) {
        New-Item -Path $env:PYTHONPYCACHEPREFIX -ItemType Directory | Out-Null
    }
}

Initialize-RepoEnvironment

switch ($Mode) {
    "help" {
        Write-Host ""
        Write-Host "Usage:" -ForegroundColor Cyan
        Write-Host "  .\run.ps1 detect [args]"
        Write-Host "  .\run.ps1 undistort --source <source>"
        Write-Host "  .\run.ps1 init-source --source <source>"
        Write-Host "  .\run.ps1 copy-source --from-source <source-a> --to-source <source-b>"
        Write-Host "  .\run.ps1 dashboard"
        Write-Host "  .\run.ps1 help"
        Write-Host ""
        Write-Host "Modes:" -ForegroundColor Cyan
        Write-Host "  detect       Run the main detection pipeline."
        Write-Host "  undistort    Launch the undistortion tuner for a source."
        Write-Host "  init-source  Create required DB-backed config rows for a source."
        Write-Host "  copy-source  Copy runtime, ROI, and undistortion from one source to another."
        Write-Host "  dashboard    Start the dashboard backend and frontend."
        Write-Host "  help         Show this help text."
        Write-Host ""
        Write-Host "Undistort Workflow:" -ForegroundColor Cyan
        Write-Host "  1. Initialize the source once:"
        Write-Host "     .\run.ps1 init-source --source media/test15.mp4"
        Write-Host "  2. Launch the tuner:"
        Write-Host "     .\run.ps1 undistort --source media/test15.mp4"
        Write-Host "     Optional model override:"
        Write-Host "     .\run.ps1 undistort --source media/test15.mp4 --model standard"
        Write-Host "  3. In the tuner:"
        Write-Host "     - adjust FX, FY, K1-K4, and Balance"
        Write-Host "     - press 's' to save"
        Write-Host "     - press 'r' to reset to saved values"
        Write-Host "     - press '0' for a neutral reset"
        Write-Host "     - press 'q' or ESC to exit without saving"
        Write-Host "  4. Run detection again:"
        Write-Host "     .\run.ps1 detect --source media/test15.mp4"
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Cyan
        Write-Host "  .\run.ps1 detect"
        Write-Host "  .\run.ps1 detect --source media/test9.mp4"
        Write-Host "  .\run.ps1 init-source --source media/test1.mp4"
        Write-Host "  .\run.ps1 copy-source --from-source media/test9.mp4 --to-source media/test10.mp4"
        Write-Host "  .\run.ps1 undistort --source media/test1.mp4"
        Write-Host "  .\run.ps1 undistort --source media/test15.mp4 --model standard"
        Write-Host "  .\run.ps1 dashboard"
        Write-Host "  .\run.ps1 help"
        Write-Host ""
    }

    "detect" {
        Write-Host "Running detection pipeline with settings from config.py..." -ForegroundColor Cyan
        & python .\run_openvino_yolo.py @Args
    }

    "undistort" {
        Write-Host "Running live undistortion tuner..." -ForegroundColor Cyan
        & python .\scripts\configuration\image_undistorter.py @Args
    }

    "init-source" {
        Write-Host "Initializing source config rows..." -ForegroundColor Cyan
        & python .\scripts\configuration\init_source_config.py @Args
    }

    "copy-source" {
        Write-Host "Copying source config rows..." -ForegroundColor Cyan
        & python .\scripts\configuration\copy_source_config.py @Args
    }

    "dashboard" {
        $cacheRoot = $env:PYTHONPYCACHEPREFIX

        Write-Host "Starting Dashboard Backend (Flask) in a new window..." -ForegroundColor Cyan
        Start-Process powershell -ArgumentList "-Command", "`$env:PYTHONPYCACHEPREFIX='$cacheRoot'; python dashboard_server.py"

        Write-Host "Starting Dashboard Frontend (Vite) in a new window..." -ForegroundColor Cyan
        Start-Process powershell -ArgumentList "-Command", "cd dashboard; npm run dev"

        Write-Host ""
        Write-Host "All components are starting up." -ForegroundColor Green
        Write-Host "- Backend URL:  http://localhost:8765"
        Write-Host "- Frontend URL: http://localhost:5173"
        Write-Host "- Logs are available in the newly opened windows."
        Write-Host ""
    }
}
