# MLSharp-3D-Maker Test Script (PowerShell)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running MLSharp Unit Tests" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path "test_app.py")) {
    Write-Host "[WARNING] Test file not found" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host "Running tests..." -ForegroundColor Green

$process = Start-Process -FilePath "python_env\python.exe" -ArgumentList "test_app.py" -NoNewWindow -Wait -PassThru

if ($process.ExitCode -eq 0) {
    Write-Host "All tests passed!" -ForegroundColor Green
} else {
    Write-Host "Some tests failed!" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
exit $process.ExitCode
