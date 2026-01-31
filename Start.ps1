# MLSharp-3D-Maker - 统一启动脚本
$ErrorActionPreference = "Continue"

Clear-Host

Write-Host "==============================================================================================" -ForegroundColor Cyan
Write-Host "      ___           ___       ___           ___           ___           ___           ___     " -ForegroundColor Cyan
Write-Host "     /\__\         /\__\     /\  \         /\__\         /\  \         /\  \         /\  \    " -ForegroundColor Cyan
Write-Host "    /::|  |       /:/  /    /::\  \       /:/  /        /::\  \       /::\  \       /::\  \   " -ForegroundColor Cyan
Write-Host "   /:|:|  |      /:/  /    /:/\ \  \     /:/__/        /:/\:\  \     /:/\:\  \     /:/\:\  \  " -ForegroundColor Cyan
Write-Host "  /:/|:|__|__   /:/  /    _\:\~\ \  \   /::\  \ ___   /::\~\:\  \   /::\~\:\  \   /::\~\:\  \ " -ForegroundColor Cyan
Write-Host " /:/ |::::\__\ /:/__/    /\ \:\ \ \__\ /:/\:\  /\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\" -ForegroundColor Cyan
Write-Host " \/__/~~/:/  / \:\  \    \:\ \:\ \/__/ \/__\:\/:/  / \/__\:\/:/  / \/_|::\/:/  / \/__\:\/:/  /" -ForegroundColor Cyan
Write-Host "       /:/  /   \:\  \    \:\ \:\__\        \::/  /       \::/  /     |:|::/  /       \::/  / " -ForegroundColor Cyan
Write-Host "      /:/  /     \:\  \    \:\/:/  /        /:/  /        /:/  /      |:|\/__/         \/__/  " -ForegroundColor Cyan
Write-Host "     /:/  /       \:\__\    \::/  /        /:/  /        /:/  /       |:|  |                  " -ForegroundColor Cyan
Write-Host "     \/__/         \/__/     \/__/         \/__/         \/__/         \|__|                  " -ForegroundColor Cyan
Write-Host "==============================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "                                        3D 模型生成工具" -ForegroundColor Green
Write-Host ""
Write-Host "==============================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[信息] 正在初始化..." -ForegroundColor Yellow
Write-Host ""

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# 检查 Python 环境
$pythonPath = "$scriptPath\python_env\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "[错误] 未找到 Python 环境!" -ForegroundColor Red
    Write-Host ""
    Write-Host "==============================================================================================" -ForegroundColor Red
    Write-Host "错误详情" -ForegroundColor Red
    Write-Host "==============================================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Python 环境不存在: $pythonPath" -ForegroundColor White
    Write-Host ""
    Write-Host "可能的原因:" -ForegroundColor Yellow
    Write-Host "  1. Python 环境未正确安装" -ForegroundColor White
    Write-Host "  2. 文件夹结构被破坏" -ForegroundColor White
    Write-Host "  3. 文件被误删或移动" -ForegroundColor White
    Write-Host ""
    Write-Host "解决方案:" -ForegroundColor Green
    Write-Host "  1. 重新下载完整的程序包" -ForegroundColor White
    Write-Host "  2. 检查 python_env 文件夹是否存在" -ForegroundColor White
    Write-Host "  3. 联系技术支持" -ForegroundColor White
    Write-Host ""
    Write-Host "==============================================================================================" -ForegroundColor Red
    Write-Host ""
    Read-Host "按 Enter 键退出"
    exit 1
}

Write-Host "[成功] Python 环境已找到" -ForegroundColor Green
Write-Host ""

# 检查 app.py
$appPath = "$scriptPath\app.py"
if (-not (Test-Path $appPath)) {
    Write-Host "[错误] 未找到主程序文件!" -ForegroundColor Red
    Write-Host ""
    Write-Host "==============================================================================================" -ForegroundColor Red
    Write-Host "错误详情" -ForegroundColor Red
    Write-Host "==============================================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "主程序文件不存在: $appPath" -ForegroundColor White
    Write-Host ""
    Write-Host "解决方案:" -ForegroundColor Green
    Write-Host "  1. 确保 app.py 文件在同一目录下" -ForegroundColor White
    Write-Host "  2. 重新下载程序" -ForegroundColor White
    Write-Host ""
    Write-Host "==============================================================================================" -ForegroundColor Red
    Write-Host ""
    Read-Host "按 Enter 键退出"
    exit 1
}

Write-Host "[成功] 主程序文件已找到" -ForegroundColor Green
Write-Host ""

# 检查模型文件
$modelPath = "$scriptPath\model_assets\sharp_2572gikvuh.pt"
if (-not (Test-Path $modelPath)) {
    Write-Host "[警告] 未找到模型文件!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "==============================================================================================" -ForegroundColor Yellow
    Write-Host "警告" -ForegroundColor Yellow
    Write-Host "==============================================================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "模型文件不存在: $modelPath" -ForegroundColor White
    Write-Host ""
    Write-Host "程序可能无法正常运行,请确保模型文件已正确下载" -ForegroundColor White
    Write-Host ""
    Write-Host "==============================================================================================" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "[信息] 正在启动程序..." -ForegroundColor Yellow
Write-Host ""
Write-Host "==============================================================================================" -ForegroundColor Cyan
Write-Host "系统信息" -ForegroundColor Cyan
Write-Host "==============================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "支持模式:" -ForegroundColor Yellow
Write-Host "NVIDIA GPU (CUDA)" -ForegroundColor Green
Write-Host "AMD GPU (ROCm)" -ForegroundColor Green
Write-Host "Intel GPU (CPU 回退)" -ForegroundColor Green
Write-Host "CPU 模式" -ForegroundColor Green
Write-Host ""
Write-Host "==============================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[提示] 浏览器将自动打开..." -ForegroundColor Cyan
Write-Host "[提示] 按 Ctrl+C 可停止服务" -ForegroundColor Cyan
Write-Host ""
Write-Host "==============================================================================================" -ForegroundColor Cyan
Write-Host ""

& $pythonPath "app.py" --enable-auto-tune --config config/config.yaml

# 错误处理
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "==============================================================================================" -ForegroundColor Red
    Write-Host "启动失败!" -ForegroundColor Red
    Write-Host "==============================================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "错误代码: $LASTEXITCODE" -ForegroundColor White
    Write-Host ""
    Write-Host "可能的原因:" -ForegroundColor Yellow
    Write-Host "  1. Python 环境未正确安装" -ForegroundColor White
    Write-Host "  2. 依赖库缺失或不兼容" -ForegroundColor White
    Write-Host "  3. 模型文件不存在或已损坏" -ForegroundColor White
    Write-Host "  4. 端口 8000 被占用" -ForegroundColor White
    Write-Host "  5. 系统资源不足(内存/显存)" -ForegroundColor White
    Write-Host "  6. 显卡驱动问题" -ForegroundColor White
    Write-Host ""
    Write-Host "解决方案:" -ForegroundColor Green
    Write-Host "  1. 查看上面的详细错误信息" -ForegroundColor White
    Write-Host "  2. 关闭其他占用端口 8000 的程序" -ForegroundColor White
    Write-Host "  3. 检查显卡驱动是否正确安装" -ForegroundColor White
    Write-Host "  4. 重启计算机" -ForegroundColor White
    Write-Host "  5. 查看日志文件 logs/ 中的详细信息" -ForegroundColor White
    Write-Host ""
    Write-Host "==============================================================================================" -ForegroundColor Red
    Write-Host ""
    Read-Host "按 Enter 键退出"
}