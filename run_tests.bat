@echo off
chcp 65001 >nul
echo ========================================
echo Running MLSharp Unit Tests
echo ========================================
echo.

if not exist "test_app.py" (
    echo [WARNING] Test file test_app.py not found
    echo.
    echo No unit tests configured for this project.
    echo.
    echo Skipped tests.
    echo.
    echo ========================================
    echo Tests skipped (no test file found)
    echo ========================================
    pause
    exit /b 0
)

python_env\python.exe test_app.py

echo.
echo ========================================
if %ERRORLEVEL% EQU 0 (
    echo All tests passed!
) else (
    echo Some tests failed!
)
echo ========================================

pause