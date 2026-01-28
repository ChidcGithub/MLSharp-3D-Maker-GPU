@echo off
chcp 65001 >nul
echo ========================================
echo Running MLSharp Unit Tests
echo ========================================
echo.

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