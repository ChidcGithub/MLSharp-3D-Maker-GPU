```
          _____                    _____            _____                    _____                    _____                    _____                    _____          
         /\    \                  /\    \          /\    \                  /\    \                  /\    \                  /\    \                  /\    \         
        /::\____\                /::\____\        /::\    \                /::\____\                /::\    \                /::\    \                /::\    \        
       /::::|   |               /:::/    /       /::::\    \              /:::/    /               /::::\    \              /::::\    \              /::::\    \       
      /:::::|   |              /:::/    /       /::::::\    \            /:::/    /               /::::::\    \            /::::::\    \            /::::::\    \      
     /::::::|   |             /:::/    /       /:::/\:::\    \          /:::/    /               /:::/\:::\    \          /:::/\:::\    \          /:::/\:::\    \     
    /:::/|::|   |            /:::/    /       /:::/__\:::\    \        /:::/____/               /:::/__\:::\    \        /:::/__\:::\    \        /:::/__\:::\    \    
   /:::/ |::|   |           /:::/    /        \:::\   \:::\    \      /::::\    \              /::::\   \:::\    \      /::::\   \:::\    \      /::::\   \:::\    \   
  /:::/  |::|___|______    /:::/    /       ___\:::\   \:::\    \    /::::::\    \   _____    /::::::\   \:::\    \    /::::::\   \:::\    \    /::::::\   \:::\    \  
 /:::/   |::::::::\    \  /:::/    /       /\   \:::\   \:::\    \  /:::/\:::\    \ /\    \  /:::/\:::\   \:::\    \  /:::/\:::\   \:::\____\  /:::/\:::\   \:::\____\ 
/:::/    |:::::::::\____\/:::/____/       /::\   \:::\   \:::\____\/:::/  \:::\    /::\____\/:::/  \:::\   \:::\____\/:::/  \:::\   \:::|    |/:::/  \:::\   \:::|    |
\::/    / ~~~~~/:::/    /\:::\    \       \:::\   \:::\   \::/    /\::/    \:::\  /:::/    /\::/    \:::\  /:::/    /\::/   |::::\  /:::|____|\::/    \:::\  /:::|____|
 \/____/      /:::/    /  \:::\    \       \:::\   \:::\   \/____/  \/____/ \:::\/:::/    /  \/____/ \:::\/:::/    /  \/____|:::::\/:::/    /  \/_____/\:::\/:::/    /  
             /:::/    /    \:::\    \       \:::\   \:::\    \               \::::::/    /            \::::::/    /         |:::::::::/    /            \::::::/    /   
            /:::/    /      \:::\    \       \:::\   \:::\____\               \::::/    /              \::::/    /          |::|\::::/    /              \::::/    /    
           /:::/    /        \:::\    \       \:::\  /:::/    /               /:::/    /               /:::/    /           |::| \::/____/                \::/____/     
          /:::/    /          \:::\    \       \:::\/:::/    /               /:::/    /               /:::/    /            |::|  ~|                       ~~           
         /:::/    /            \:::\    \       \::::::/    /               /:::/    /               /:::/    /             |::|   |                                   
        /:::/    /              \:::\____\       \::::/    /               /:::/    /               /:::/    /              \::|   |                                   
        \::/    /                \::/    /        \::/    /                \::/    /                \::/    /                \:|   |                                   
         \/____/                  \/____/          \/____/                  \/____/                  \/____/                  \|___|                                    
```                                                                                                                                                                     

# MLSharp 3D Maker

---

## Tip: 此分支为此项目能够在搭载骁龙芯片的平台上运行提供基础。
### 后续计划添加更多芯片及NPU推理SDK支持

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Windows|Linux-lightgrey.svg)
![API](https://img.shields.io/badge/API-RESTful-blueviolet.svg)
![stars](https://img.shields.io/github/stars/chidcGithub/MLSharp-3D-Maker-GPU)
</div>

# 使用说明

## 项目概述

MLSharp-3D-Maker 是一个基于 Apple ml-sharp 模型的 3D 高斯泼溅（3D Gaussian Splatting）生成工具，可以从单张照片生成高质量的 3D 模型。


## 项目结构及更新

```
MLSharp-3D-Maker-GPU-by-Chidc/
├── app.py                        # 主应用程序（重构版本）⭐
├── config/                       # 配置文件目录（推荐使用）
│   ├── config.yaml                   # YAML 格式配置文件
│   └── config.json                   # JSON 格式配置文件
├── gpu_utils.py                  # GPU 工具模块
├── logger.py                     # 日志模块
├── metrics.py                    # 监控指标模块 ⭐
├── optimistic.md                 # 性能优化方案文档 ⭐
├── Start.bat                     # Windows 启动脚本
├── Start.ps1                     # PowerShell 启动脚本
├── model_assets/                 # 模型文件和资源
│   ├── sharp_2572gikvuh.pt      # ml-sharp 模型权重
│   ├── inputs/                   # 输入示例
│   └── outputs/                  # 输出示例
├── python_env/                   # Python 环境
├── logs/                         # 日志文件夹
├── tmp/                          # 临时文件和备份
│   └── 1.28/                     # 2026-01-28 备份
└──  temp_workspace/               # 临时工作目录
```

<details>
<summary><b>点击展开查看最新更新详情</b></summary>

### 最新更新（2026-01-31）

**Snapdragon GPU 适配 v9.1**
- **Adreno GPU 检测** - 自动检测 Snapdragon/Adreno 系列 GPU
- **Qualcomm 模式** - 新增 `--mode qualcomm` 启动模式
- **ONNX Runtime 支持** - 添加 ONNX Runtime + DirectML 加速方案
- **智能回退** - 检测到 Snapdragon GPU 时自动使用 CPU 模式
- **平台支持** - Windows/Android 平台识别
- **文档更新** - 添加 Snapdragon GPU 支持说明和限制

</details>

---

## 快速开始

### 推荐启动方式

#### 智能运行（推荐新手）⭐：
```bash
双击运行 Start.ps1
```

**功能特点：**
- **自动检测**: GPU 类型（NVIDIA/AMD/Intel/Snapdragon）、环境配置、依赖库
- **智能推荐**: 根据显卡自动推荐最佳启动脚本
- **全面诊断**: 100+ 错误处理，智能识别问题
- **解决方案**: 每个错误都提供详细的解决建议
- **日志记录**: 所有运行日志保存在 logs/ 文件夹
- **彩色输出**: 清晰的视觉反馈，易于阅读

#### 使用命令行参数（高级用户）：
```bash
# 自动检测模式（默认）
python app.py

# 强制使用 GPU 模式
python app.py --mode gpu

# 强制使用 CPU 模式
python app.py --mode cpu

# Snapdragon GPU 模式（检测后使用 CPU 或 ONNX Runtime）
python app.py --mode qualcomm

# 自定义端口
python app.py --port 8080

# 不自动打开浏览器
python app.py --no-browser
```

### 访问地址

启动后访问：http://127.0.0.1:8000

---

## 依赖安装

### 基础依赖

```bash
pip install -r requirements.txt
```

### Snapdragon GPU 加速（可选）

如果使用 Snapdragon GPU（如 Snapdragon X Elite），安装 ONNX Runtime GPU 版本：

```bash
# Windows
pip install onnxruntime-gpu

# Linux/Mac
pip install onnxruntime
```

### 验证安装

```bash
# 检查 ONNX Runtime 是否安装
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

如果输出包含 `DmlExecutionProvider`，说明 DirectML 支持已启用，可以使用 Snapdragon GPU 加速。

---

## 命令行参数

<details>
<summary><b>点击展开查看命令行参数详情</b></summary>

### 基本参数

| 参数                     | 简写   | 类型     | 默认值            | 说明                     |
|------------------------|------|--------|----------------|------------------------|
| `--mode`               | `-m` | string | `auto`         | 启动模式                   |
| `--port`               | `-p` | int    | `8000`         | Web 服务端口               |
| `--host`               |      | string | `127.0.0.1`    | Web 服务主机地址             |
| `--input-size`         |      | int[]  | `[1536, 1536]` | 输入图像尺寸 [宽度, 高度]        |
| `--no-browser`         |      | flag   | false          | 不自动打开浏览器               |
| `--no-amp`             |      | flag   | false          | 禁用混合精度推理（AMP）          |
| `--no-cudnn-benchmark` |      | flag   | false          | 禁用 cuDNN Benchmark     |
| `--config`             | `-c` | string | -              | 配置文件路径（支持 YAML 和 JSON） |
| `--enable-cache`       |      | flag   | true           | 启用推理缓存（默认：启用）          |
| `--no-cache`           |      | flag   | false          | 禁用推理缓存                 |
| `--cache-size`         |      | int    | `100`          | 缓存最大条目数                |
| `--clear-cache`        |      | flag   | false          | 启动时清空缓存                |
| `--enable-auto-tune`   |      | flag   | false          | 启用性能自动调优               |
| `--redis-url`          |      | string | -              | Redis 连接 URL（分布式缓存）    |
| `--enable-webhook`     |      | flag   | false          | 启用 Webhook 异步通知        |

### 启动模式 (--mode)

| 模式         | 说明                                |
|------------|-----------------------------------|
| `auto`     | 自动检测并选择最佳模式（默认）                   |
| `gpu`      | 强制使用 GPU 模式（自动检测厂商）               |
| `cpu`      | 强制使用 CPU 模式                       |
| `nvidia`   | 强制使用 NVIDIA GPU 模式                |
| `amd`      | 强制使用 AMD GPU 模式（ROCm）             |
| `qualcomm` | 强制使用 Snapdragon GPU 模式（检测后使用 CPU） |

### 输入尺寸 (--input-size)

设置推理时使用的输入图像尺寸。默认为 1536x1536，这是模型训练时使用的尺寸。

**使用示例：**
```bash
# 使用默认尺寸 1536x1536
python app.py

# 使用自定义尺寸 1024x1024
python app.py --input-size 1024 1024

# 使用 768x768 快速测试
python app.py --input-size 768 768
```

**约束条件：**
- 输入尺寸必须能被 **64 整除**（模型编码器使用基于补丁的分割）
- **宽度和高度必须相等**（模型使用正方形输入）
- **最大支持尺寸为 1536x1536**（SPN 编码器在更大尺寸下会出现补丁分割错误）
- 如果提供的尺寸不符合要求，程序会自动调整到最接近的有效尺寸

**自动调整示例：**
```bash
# 1000x1000 → 自动调整为 1024x1024
python app.py --input-size 1000 1000

# 1200x800 → 自动调整为 1200x1200（保持正方形）
python app.py --input-size 1200 800
```

**推荐尺寸：**
| 尺寸 | 用途 | 显存需求 | 输出质量 |
|------|------|---------|---------|
| 512x512 | 快速测试 | 低 | 基础 |
| 768x768 | 平衡模式 | 中等 | 良好 |
| 1024x1024 | 标准模式 | 中等 | 优秀 |
| 1536x1536 | 高质量（默认/最大） | 高 | 最佳 |

**注意：** 最大支持尺寸为 1536x1536，超过此尺寸会导致 SPN 编码器出现补丁分割错误。

**注意事项：**
- 较大的输入尺寸会提高模型输出质量，但需要更多的显存和计算时间
- 较小的输入尺寸可以加快推理速度，降低显存占用，但可能降低输出质量
- 推荐范围：512x512 到 1536x1536
- **最大支持尺寸为 1536x1536**，超过此尺寸会导致补丁分割错误
- 如果显存不足，建议使用较小的尺寸
- 如果使用非标准尺寸，程序会自动调整并显示警告信息

### 使用示例

```bash
# 基本使用
python app.py
python app.py --mode gpu
python app.py --mode cpu

# 指定 GPU 厂商
python app.py --mode nvidia
python app.py --mode amd

# 自定义端口和主机
python app.py --port 8080
python app.py --host 0.0.0.0 --port 8000

# 自定义输入尺寸
python app.py --input-size 1024 1024
python app.py --input-size 768 768

# 禁用优化选项（调试用）
python app.py --no-browser
python app.py --no-amp
python app.py --no-cudnn-benchmark

# 启用梯度检查点（减少显存占用）
python app.py --gradient-checkpointing

# 缓存管理（默认开启）
python app.py                           # 默认启用缓存
python app.py --no-cache               # 禁用缓存
python app.py --cache-size 200         # 设置缓存大小为 200
python app.py --clear-cache            # 启动时清空缓存

# 性能自动调优（高级功能）
python app.py --enable-auto-tune       # 启动时自动测试并选择最优优化配置

# 组合使用
python app.py --mode nvidia --port 8080 --no-browser --input-size 1024 1024
python app.py --gradient-checkpointing --input-size 1536 1536
python app.py --cache-size 200 --mode gpu
python app.py --clear-cache --mode gpu

# 使用配置文件
python app.py --config config.yaml
python app.py --config config.json
python app.py -c config.yaml

# 配置文件 + 命令行参数（命令行参数优先）
python app.py --config config.yaml --port 8080 --input-size 1024 1024
```

### 获取帮助

```bash
python app.py --help
python app.py -h
```

</details>

---

## GPU 支持情况

<details>
<summary><b>点击展开查看 GPU 支持详情</b></summary>

### NVIDIA GPU
| 架构      | 显卡系列         | 计算能力    | 支持状态      | 优化               |
|---------|--------------|---------|-----------|------------------|
| Ampere  | RTX 30/40 系列 | 8.0+    | 完全支持      | AMP, TF32, cuDNN |
| Turing  | RTX 20 系列    | 7.5     | 完全支持      | AMP, cuDNN       |
| Pascal  | GTX 10/16 系列 | 6.1     | 完全支持      | AMP, cuDNN       |
| Maxwell | GTX 9xx 系列   | 5.2     | 支持        | AMP              |
| Kepler  | GTX 7xx 系列   | 3.0-3.7 | ⚠️ 老旧 GPU | 基础               |
| Fermi   | GTX 6xx 系列   | 2.1     | ❌ 不推荐     | -                |

### AMD GPU
| 架构     | 显卡系列          | ROCm 支持 | 支持状态    |
|--------|---------------|---------|---------|
| RDNA 2 | RX 6000 系列    | 完全支持    | 完全支持    |
| RDNA 1 | RX 5000 系列    | 完全支持    | 完全支持    |
| GCN 5  | Vega 系列       | 完全支持    | 支持      |
| GCN 4  | RX 400/500 系列 | ⚠️      | ⚠️ 部分支持 |
| GCN 3  | RX 300 系列     | ❌       | ❌ 不支持   |

### Intel GPU
| 架构      | 显卡系列   | 支持状态        |
|---------|--------|-------------|
| Xe      | Arc 系列 | ⚠️ 仅 CPU 模式 |
| Iris Xe | 集成显卡   | ⚠️ 仅 CPU 模式 |
| UHD     | 集成显卡   | ⚠️ 仅 CPU 模式 |

### Qualcomm/Snapdragon GPU
| 架构     | 显卡系列       | 支持状态      | 说明                             |
|--------|------------|-----------|--------------------------------|
| Adreno | 600/700 系列 | ⚠️ CPU 模式 | 检测到 Snapdragon GPU，使用 CPU 模式运行 |
| Adreno | 500 系列及以下  | ⚠️ CPU 模式 | 检测到 Snapdragon GPU，使用 CPU 模式运行 |

**Snapdragon GPU 加速方案**:

**方案 1：ONNX Runtime + DirectML（推荐 Windows）**
```bash
# 安装 ONNX Runtime GPU 版本（包含 DirectML 支持）
pip install onnxruntime-gpu
```
- 支持 Snapdragon X Elite/8cx 等 Windows on ARM 设备
- 需要将模型转换为 ONNX 格式
- 使用 DirectML 执行提供者进行 GPU 加速

**方案 2：Android 设备**
- 使用 SNPE (Snapdragon Neural Processing Engine) SDK
- 使用 QNN (Qualcomm Neural Network) SDK

**方案 3：PyTorch（仅 CPU）**
- 当前默认使用 PyTorch CPU 模式
- 无需额外配置，兼容性最好

**注意**: PyTorch 原生不支持 Adreno GPU，必须通过 ONNX Runtime 才能实现 GPU 加速。
- **Windows**: 已集成 ONNX Runtime 检测，安装 `onnxruntime-gpu` 后可启用 DirectML 加速
- **检测**: 系统会自动检测 Snapdragon/Adreno GPU 并显示相关信息
- **模型转换**: 使用 ONNX Runtime 需要将 PyTorch 模型转换为 ONNX 格式（待实现）

</details>

---

## 日志系统

<details>
<summary><b>点击展开查看日志系统详情</b></summary>

### 日志特性

MLSharp 使用 Loguru 作为日志系统，提供专业的日志管理功能：

- **结构化日志**: 包含时间戳、日志级别、来源信息
- **彩色输出**: 控制台彩色显示，易于区分不同级别
- **文件日志**: 自动保存到 `logs/` 目录
- **日志轮转**: 自动轮转和压缩日志文件（10MB 轮转，保留7天）
- **错误追踪**: 完整的错误堆栈追踪和诊断信息
- **多级别**: DEBUG, INFO, WARNING, ERROR, CRITICAL

### 日志文件

日志文件保存在 `logs/` 目录：
- 文件命名：`mlsharp_YYYYMMDD.log`
- 压缩文件：`mlsharp_YYYYMMDD.log.zip`
- 保留时间：7天

### 日志级别

| 级别       | 用途   | 示例         |
|----------|------|------------|
| DEBUG    | 调试信息 | 变量值、函数调用   |
| INFO     | 一般信息 | 启动信息、处理进度  |
| WARNING  | 警告信息 | 性能警告、兼容性问题 |
| ERROR    | 错误信息 | 处理失败、异常    |
| CRITICAL | 严重错误 | 系统崩溃、致命错误  |

### 日志输出示例

```
2026-01-28 20:00:00 | INFO     | MLSharp:run:10 - 服务启动
2026-01-28 20:00:01 | SUCCESS  | MLSharp:load_model:50 - 模型加载完成
2026-01-28 20:00:02 | WARNING  | MLSharp:detect_gpu:30 - 显存不足 4GB
2026-01-28 20:00:03 | ERROR    | MLSharp:predict:100 | 处理失败: 显存溢出
```

### 查看日志

```bash
# 查看今天的日志
type logs\mlsharp_20260128.log

# 查看所有日志文件
dir logs\

# 查看错误日志
findstr /C:"ERROR" logs\mlsharp_*.log
```
</details>

---

## 配置文件使用

<details>
<summary><b>点击展开查看配置文件使用详情</b></summary>

### 配置文件格式

支持 YAML 和 JSON 两种格式的配置文件。

**默认配置文件**: 如果不指定 `--config` 参数，系统会自动使用项目根目录下的 `config.yaml` 作为默认配置文件。

#### YAML 格式 (config.yaml)

```yaml
# MLSharp-3D-Maker 配置文件
# 支持的格式: YAML

# 服务配置
server:
  host: "127.0.0.1"        # 服务主机地址
  port: 8000               # 服务端口

# 启动模式
mode: "auto"               # 启动模式: auto, gpu, cpu, nvidia, amd

# 浏览器配置
browser:
  auto_open: true          # 自动打开浏览器

# GPU 优化配置
gpu:
  enable_amp: true         # 启用混合精度推理 (AMP)
  enable_cudnn_benchmark: true  # 启用 cuDNN Benchmark
  enable_tf32: true        # 启用 TensorFloat32

# 日志配置
logging:
  level: "INFO"            # 日志级别: DEBUG, INFO, WARNING, ERROR
  console: true            # 控制台输出
  file: false              # 文件输出

# 模型配置
model:
  checkpoint: "model_assets/sharp_2572gikvuh.pt"  # 模型权重路径
  temp_dir: "temp_workspace"                     # 临时工作目录

# 推理配置
inference:
  input_size: [1536, 1536]  # 输入图像尺寸 [宽度, 高度] (默认: 1536x1536)

# 优化配置
optimization:
  gradient_checkpointing: false  # 启用梯度检查点（减少显存占用，但会略微降低推理速度）
  checkpoint_segments: 3         # 梯度检查点分段数（暂未使用）

# 缓存配置
cache:
  enabled: true                  # 启用推理缓存（默认：启用）
  size: 100                      # 缓存最大条目数（默认：100）

# Redis 缓存配置
redis:
  enabled: false                 # 启用 Redis 缓存（默认：禁用）
  url: "redis://localhost:6379/0"  # Redis 连接 URL
  prefix: "mlsharp"              # 缓存键前缀

# Webhook 配置
webhook:
  enabled: false                 # 启用 Webhook 通知（默认：禁用）
  task_completed: ""             # 任务完成通知 URL
  task_failed: ""                # 任务失败通知 URL

# 监控配置
monitoring:
  enabled: true            # 启用监控
  enable_gpu: true         # 启用 GPU 监控
  metrics_path: "/metrics" # Prometheus 指标端点路径

# 性能配置
performance:
  max_workers: 4           # 最大工作线程数
  max_concurrency: 10      # 最大并发数
  timeout_keep_alive: 30   # 保持连接超时(秒)
  max_requests: 1000       # 最大请求数

# 性能调优缓存（自动生成，无需手动配置）
performance_cache:
  last_test: null          # 上次测试时间（ISO 8601 格式）
  best_config: null        # 最优配置
  gpu: null                # GPU 信息
```

#### JSON 格式 (config.json)

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000
  },
  "mode": "auto",
  "browser": {
    "auto_open": true
  },
  "gpu": {
    "enable_amp": true,
    "enable_cudnn_benchmark": true,
    "enable_tf32": true
  },
  "logging": {
    "level": "INFO",
    "console": true,
    "file": false
  },
  "model": {
    "checkpoint": "model_assets/sharp_2572gikvuh.pt",
    "temp_dir": "temp_workspace"
  },
  "inference": {
    "input_size": [1536, 1536]
  },
  "optimization": {
    "gradient_checkpointing": false,
    "checkpoint_segments": 3
  },
  "cache": {
    "enabled": true,
    "size": 100
  },
  "redis": {
    "enabled": false,
    "url": "redis://localhost:6379/0",
    "prefix": "mlsharp"
  },
  "webhook": {
    "enabled": false,
    "task_completed": "",
    "task_failed": ""
  },
  "monitoring": {
    "enabled": true,
    "enable_gpu": true,
    "metrics_path": "/metrics"
  },
  "performance": {
    "max_workers": 4,
    "max_concurrency": 10,
    "timeout_keep_alive": 30,
    "max_requests": 1000
  }
}
```

### 使用配置文件

**基本使用：**
```bash
# 使用 YAML 配置文件
python app.py --config config.yaml

# 使用 JSON 配置文件
python app.py --config config.json

# 简写
python app.py -c config.yaml

# 推荐：使用 config 文件夹管理配置文件
python app.py --config config/performance.yaml
python app.py --config config/settings.json
```

**配置文件 + 命令行参数：**
```bash
# 命令行参数会覆盖配置文件中的对应设置
python app.py --config config.yaml --port 8080 --mode gpu
```

**配置文件自动创建/更新**：
```bash
# 如果配置文件不存在，会自动创建并包含默认配置
# 如果配置文件已存在，仅更新性能调优缓存，其他配置保持不变
python app.py --enable-auto-tune --config config/auto_tune.json
```

### 参数优先级

命令行参数 > 配置文件 > 默认值

例如：
```bash
# config.yaml 中设置 port: 8000
# 命令行参数指定 --port 8080
# 最终使用 8080
python app.py --config config.yaml --port 8080
```

### 配置项说明

| 配置项                                   | 说明                 | 可选值                         |
|---------------------------------------|--------------------|-----------------------------|
| `server.host`                         | 服务主机地址             | IP 地址                       |
| `server.port`                         | 服务端口               | 1-65535                     |
| `mode`                                | 启动模式               | auto, gpu, cpu, nvidia, amd |
| `browser.auto_open`                   | 自动打开浏览器            | true, false                 |
| `gpu.enable_amp`                      | 启用混合精度推理           | true, false                 |
| `gpu.enable_cudnn_benchmark`          | 启用 cuDNN Benchmark | true, false                 |
| `gpu.enable_tf32`                     | 启用 TensorFloat32   | true, false                 |
| `logging.level`                       | 日志级别               | DEBUG, INFO, WARNING, ERROR |
| `logging.console`                     | 控制台输出              | true, false                 |
| `logging.file`                        | 文件输出               | true, false                 |
| `model.checkpoint`                    | 模型权重路径             | 文件路径                        |
| `model.temp_dir`                      | 临时工作目录             | 目录路径                        |
| `inference.input_size`                | 输入图像尺寸             | [宽度, 高度]，默认 [1536, 1536]    |
| `monitoring.enabled`                  | 启用监控               | true, false                 |
| `monitoring.enable_gpu`               | 启用 GPU 监控          | true, false                 |
| `monitoring.metrics_path`             | Prometheus 指标端点路径  | 路径字符串                       |
| `optimization.gradient_checkpointing` | 启用梯度检查点            | true, false                 |
| `optimization.checkpoint_segments`    | 梯度检查点分段数           | 正整数                         |
| `performance.max_workers`             | 最大工作线程数            | 正整数                         |
| `performance.max_concurrency`         | 最大并发数              | 正整数                         |
| `performance.timeout_keep_alive`      | 保持连接超时(秒)          | 正整数                         |
| `performance.max_requests`            | 最大请求数              | 正整数                         |
| `auto_tune.enabled`                   | 启用性能自动调优           | true, false                 |
| `auto_tune.test_size`                 | 测试图像尺寸             | [宽度, 高度]                  |
| `auto_tune.warmup_runs`               | 预热运行次数             | 正整数                         |
| `auto_tune.test_runs`                 | 测试运行次数             | 正整数                         |
| `performance_cache.last_test`         | 上次测试时间             | ISO 8601 时间戳（自动生成）     |
| `performance_cache.best_config`       | 最优配置               | 配置字典（自动生成）            |
| `performance_cache.gpu`               | GPU 信息               | GPU 信息（自动生成）             |

</details>

---

## 性能自动调优

<details>
<summary><b>点击展开查看自动调优功能详情</b></summary>

### MLSharp 提供了智能性能自动调优功能，可以自动测试并选择最优的优化配置。

### 调优特性

- **智能基准测试**: 自动测试多种优化配置组合
- **最优配置选择**: 根据测试结果自动选择最佳配置
- **显卡适配**: 根据显卡能力自动过滤不适用的配置
- **快速测试**: 使用小尺寸快速完成测试（约10秒）
- **详细日志**: 输出完整的测试过程和结果
- **性能提升**: 相对于无优化配置提升 30-50%
- **结果缓存**: 自动保存测试结果到配置文件，7天内有效
- **智能跳过**: 检测到有效缓存时自动跳过测试，加快启动速度

### 测试配置

自动调优器会测试以下配置组合：

| 配置          | 描述                  | 适用场景              |
|-------------|---------------------|-------------------|
| 基准配置        | 无任何优化               | 所有显卡              |
| 仅 AMP       | 仅启用混合精度             | 计算能力 ≥ 5.3        |
| 仅 cuDNN     | 仅启用 cuDNN Benchmark | NVIDIA，计算能力 ≥ 6.0 |
| 仅 TF32      | 仅启用 TensorFloat32   | NVIDIA，计算能力 ≥ 8.0 |
| AMP + cuDNN | 混合精度 + cuDNN        | NVIDIA，计算能力 ≥ 6.0 |
| AMP + TF32  | 混合精度 + TF32         | NVIDIA，计算能力 ≥ 8.0 |
| 全部优化        | 启用所有优化              | 高端 NVIDIA GPU     |

### 启用自动调优

```bash
# 启用性能自动调优（使用默认配置文件 config.yaml）
python app.py --enable-auto-tune

# 组合使用
python app.py --enable-auto-tune --mode gpu --input-size 1024 1024

# 指定配置文件（结果将保存到该文件）
python app.py --enable-auto-tune --config config.yaml

# 使用 config 文件夹保存配置（推荐）
python app.py --enable-auto-tune --config config/performance.yaml

# 如果配置文件不存在，会自动创建并包含默认配置
python app.py --enable-auto-tune --config config/auto_tune.json
```

**注意**: 如果不指定 `--config` 参数，系统会自动使用项目根目录下的 `config.yaml` 作为默认配置文件。

### 缓存机制

自动调优结果会自动保存到配置文件中，避免重复测试：

- **缓存有效期**: 7 天
- **缓存条件**: GPU 型号、厂商、计算能力必须匹配
- **自动跳过**: 检测到有效缓存时自动跳过测试
- **自动应用**: 直接使用缓存的最优配置
- **自动创建/更新**: 配置文件不存在时自动创建（包含默认配置），存在时仅更新性能调优缓存
- **目录支持**: 自动创建配置目录（如 config 文件夹）

**日志输出示例（使用缓存时）**:
```
[INFO] 发现有效的性能调优缓存（3 天前）
============================================================
[INFO] 使用缓存的性能配置
============================================================
配置名称: 全部优化
描述: 启用所有优化
```

**日志输出示例（创建配置文件时）**:
```
[INFO] 配置文件不存在，自动创建新配置文件: config.yaml
[SUCCESS] 性能调优结果已添加到配置文件: config.yaml
```

**日志输出示例（更新现有配置文件时）**:
```
[INFO] 配置文件已存在，更新性能调优缓存: config.yaml
[SUCCESS] 性能调优结果已更新到配置文件: config.yaml
```

**配置文件处理说明**:
- 配置文件存在时：仅更新 `performance_cache` 字段，其他配置保持不变
- 配置文件不存在时：创建新配置文件，包含完整的默认配置

### 配置文件格式

调优结果会保存在配置文件的 `performance_cache` 字段中：

```yaml
# config.yaml
performance_cache:
  last_test: "2026-01-31T12:00:00+00:00"
  best_config:
    name: "全部优化"
    amp: true
    cudnn_benchmark: true
    tf32: true
    description: "启用所有优化"
  gpu:
    name: "NVIDIA GeForce RTX 4090"
    vendor: "NVIDIA"
    compute_capability: 89
```

### 调优过程

1. **缓存检查**: 检查配置文件中是否有有效的调优缓存（7天内）
2. **命中缓存**: 如果缓存有效且 GPU 匹配，直接使用缓存结果
3. **基准测试**: 如果缓存无效或过期，执行完整的测试
4. **预热阶段**: 运行 2 次预热，稳定性能
5. **测试阶段**: 对每个配置运行 3 次测试
6. **结果统计**: 计算平均推理时间和吞吐量
7. **最优选择**: 选择最快的配置并应用
8. **保存缓存**: 将最优配置保存到配置文件

### 调优输出示例

```
============================================================
[INFO] 性能自动调优
============================================================

正在测试不同优化配置...

测试配置: 基准配置
  描述: 无任何优化
  运行 1/3: 2.543 秒
  运行 2/3: 2.512 秒
  运行 3/3: 2.528 秒
  平均推理时间: 2.528 秒

测试配置: 仅 AMP
  描述: 仅启用混合精度推理
  运行 1/3: 1.892 秒
  运行 2/3: 1.876 秒
  运行 3/3: 1.884 秒
  平均推理时间: 1.884 秒

测试配置: 全部优化
  描述: 启用所有优化
  运行 1/3: 1.245 秒
  运行 2/3: 1.238 秒
  运行 3/3: 1.241 秒
  平均推理时间: 1.241 秒

============================================================
[INFO] 调优结果
============================================================
[SUCCESS] 最优配置: 全部优化
[INFO]   描述: 启用所有优化
[INFO]   平均推理时间: 1.241 秒
[INFO]   吞吐量: 0.81 FPS

[SUCCESS] 性能自动调优完成！
[INFO] 已应用最优配置
```

### 最佳实践

1. **首次运行**: 建议在首次运行时启用自动调优
2. **硬件变更**: 更换显卡后重新运行自动调优
3. **驱动更新**: 显卡驱动更新后重新测试
4. **定期调优**: 建议每月运行一次自动调优
5. **缓存管理**: 系统会自动缓存调优结果 7 天，无需手动管理
6. **配置文件**: 推荐使用 `config/` 文件夹管理配置文件，如 `config/performance.yaml`
7. **自动创建/更新**: 配置文件不存在时自动创建（包含默认配置），存在时仅更新性能调优缓存
8. **清除缓存**: 如需强制重新测试，删除配置文件中的 `performance_cache` 字段或使用新的配置文件
</details>

---

## 性能优化建议

<details>
<summary><b>点击展开查看性能优化建议</b></summary>

### GPU 模式优化
1. **使用合适的图片尺寸**
   - 推荐: 512x512 - 1024x1024
   - 避免超过 2048x2048

2. **启用所有优化**
   - AMP（混合精度）已默认启用
   - cuDNN Benchmark 已默认启用
   - TF32 已默认启用（Ampere 架构）

3. **显存不足时启用梯度检查点**
   - 使用 `--gradient-checkpointing` 参数
   - 可减少 30-50% 显存占用
   - 速度略微降低 10-20%（可接受）

4. **关闭其他 GPU 占用程序**
   - 关闭浏览器硬件加速
   - 关闭其他 AI 应用
   - 关闭游戏或图形密集型应用

### CPU 模式优化
1. **使用更小的图片**
   - 推荐: 512x512 或更小

2. **减少并发数**
   - 修改配置中的 `max_workers`
   - 推荐值: CPU 核心数 / 2

3. **使用更快的启动脚本**
   - `Start_CPU_Fast.bat` - 快速模式

### 系统级优化
1. **增加虚拟内存**
   - 设置为物理内存的 1.5-2 倍

2. **使用 SSD**
   - 模型加载和 I/O 操作更快

3. **关闭不必要的后台程序**
- 释放更多系统资源

</details>

---

## 推理缓存

<details>
<summary><b>点击展开查看推理缓存详情</b></summary>

## MLSharp 提供了智能推理缓存功能，可以显著提升重复场景的处理速度。

### 缓存特性

- **智能哈希**: 基于图像内容和焦距生成唯一的缓存键
- **LRU 淘汰**: 最近最少使用算法自动淘汰旧缓存
- **统计监控**: 实时缓存命中率、命中/未命中次数统计
- **线程安全**: 使用锁机制保证多线程安全
- **内存管理**: 可配置的缓存大小限制

### 启用缓存

缓存功能默认启用，可通过命令行参数或配置文件控制：

```bash
# 命令行参数
python app.py                           # 默认启用缓存
python app.py --no-cache               # 禁用缓存
python app.py --cache-size 200         # 设置缓存大小为 200
```

```yaml
# config.yaml
cache:
  enabled: true      # 启用缓存（默认：true）
  size: 100          # 缓存最大条目数（默认：100）
```

### API 端点

#### 获取缓存统计

```bash
curl http://127.0.0.1:8000/v1/cache
```

**返回示例**:
```json
{
  "enabled": true,
  "size": 45,
  "max_size": 100,
  "hits": 120,
  "misses": 30,
  "hit_rate": 80.0
}
```

#### 清空缓存

```bash
curl -X POST http://127.0.0.1:8000/v1/cache/clear
```

**返回示例**:
```json
{
  "status": "success",
  "message": "缓存已清空"
}
```

### 性能提升

缓存功能可以显著提升处理速度，特别是在重复场景中：

| 缓存命中率 | 速度提升 | 适用场景   |
|-------|------|--------|
| 30%   | 30%  | 少量重复图片 |
| 50%   | 50%  | 中等重复场景 |
| 80%   | 80%  | 大量重复图片 |

### 最佳实践

1. **适当调整缓存大小**: 根据内存和实际需求调整缓存大小
2. **监控缓存命中率**: 定期检查缓存命中率，评估缓存效果
3. **定期清空缓存**: 如果内存紧张，可以定期清空缓存
4. **禁用缓存场景**: 处理完全不同的图片时，可以禁用缓存

</details>

---

## Redis 分布式缓存

<details>
<summary><b>点击展开查看 Redis 缓存详情</b></summary>

## MLSharp 支持 Redis 分布式缓存，用于多实例部署和持久化缓存。

### Redis 缓存特性

- **分布式缓存**: 支持多实例共享缓存
- **持久化**: 缓存数据持久化到 Redis
- **TTL 支持**: 自动过期机制
- **混合使用**: 可与本地缓存同时使用
- **高性能**: 基于 Redis 内存数据库

### 启用 Redis 缓存

```bash
# 使用 Redis 缓存
python app.py --redis-url redis://localhost:6379/0

# 使用 Redis 缓存 + Webhook
python app.py --redis-url redis://localhost:6379/0 --enable-webhook
```

### 配置文件

```yaml
# config.yaml
redis:
  enabled: true
  url: "redis://localhost:6379/0"
  prefix: "mlsharp"
```

### 性能对比

| 缓存类型 | 命中速度 | 分布式支持 | 持久化 | 适用场景 |
|---------|---------|----------|--------|---------|
| 本地缓存 | 最快 | ❌ | ❌ | 单实例部署 |
| Redis 缓存 | 快 | ✅ | ✅ | 多实例部署 |

### 最佳实践

1. **生产环境推荐**: 使用 Redis 缓存以支持多实例部署
2. **本地开发**: 使用本地缓存，无需 Redis 服务
3. **混合使用**: Redis 用于持久化，本地缓存用于加速
4. **监控 Redis**: 定期检查 Redis 连接状态和内存使用

</details>

---

## Webhook 异步通知

<details>
<summary><b>点击展开查看 Webhook 支持详情</b></summary>

## MLSharp 支持 Webhook 异步通知，可用于任务状态跟踪和集成第三方服务。

### Webhook 事件

| 事件类型 | 说明 | 触发时机 |
|---------|------|---------|
| task_completed | 任务完成 | 3D 模型生成成功 |
| task_failed | 任务失败 | 处理过程中发生错误 |

### 启用 Webhook

```bash
# 启用 Webhook
python app.py --enable-webhook
```

### Webhook API

#### 获取 Webhook 列表

```bash
curl http://127.0.0.1:8000/v1/webhooks
```

**响应**:
```json
{
  "enabled": true,
  "webhooks": {
    "task_completed": "https://example.com/webhook/completed",
    "task_failed": "https://example.com/webhook/failed"
  }
}
```

#### 注册 Webhook

```bash
curl -X POST "http://127.0.0.1:8000/v1/webhooks" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "task_completed",
    "url": "https://example.com/webhook/completed"
  }'
```

**响应**:
```json
{
  "status": "success",
  "message": "Webhook 已注册: task_completed -> https://example.com/webhook/completed"
}
```

#### 注销 Webhook

```bash
curl -X DELETE "http://127.0.0.1:8000/v1/webhooks/task_completed"
```

**响应**:
```json
{
  "status": "success",
  "message": "Webhook 已注销: task_completed"
}
```

### Webhook Payload

#### task_completed

```json
{
  "event": "task_completed",
  "task_id": "abc123",
  "status": "success",
  "url": "/files/abc123/output.ply",
  "processing_time": 15.5,
  "timestamp": 1706659200.0
}
```

#### task_failed

```json
{
  "event": "task_failed",
  "task_id": "abc123",
  "status": "error",
  "error": "显存不足",
  "timestamp": 1706659200.0
}
```

### HTTP Headers

每个 Webhook 请求包含以下 HTTP 头：

| Header | 说明 |
|--------|------|
| Content-Type | application/json |
| X-Webhook-Event | 事件类型 |
| X-Webhook-Timestamp | 时间戳 |

### 最佳实践

1. **验证签名**: 生产环境应验证 Webhook 签名
2. **幂等处理**: 确保重复 Webhook 不会导致问题
3. **超时处理**: 设置合理的超时时间
4. **错误重试**: 实现指数退避重试机制

</details>

---

## 监控指标

<details>
<summary><b>点击展开查看监控指标详情</b></summary>

## MLSharp 提供了完整的 Prometheus 兼容监控指标，可用于性能监控和问题诊断。

### 启用监控

监控功能默认启用，可通过配置文件控制：

```yaml
# config.yaml
monitoring:
  enabled: true             # 启用监控
  enable_gpu: true          # 启用 GPU 监控
  metrics_path: "/metrics"  # Prometheus 指标端点路径
```

### 访问指标

启动服务后，可以通过以下方式访问监控指标：

```bash
# 访问 Prometheus 指标端点
curl http://127.0.0.1:8000/metrics
```

### 监控指标说明

#### HTTP 请求指标

| 指标名称                            | 类型        | 说明          |
|---------------------------------|-----------|-------------|
| `http_requests_total`           | Counter   | HTTP 请求总数   |
| `http_request_duration_seconds` | Histogram | HTTP 请求响应时间 |

**标签**:
- `method`: HTTP 方法（GET, POST 等）
- `endpoint`: 端点路径
- `status`: HTTP 状态码

#### 预测请求指标

| 指标名称                             | 类型        | 说明      |
|----------------------------------|-----------|---------|
| `predict_requests_total`         | Counter   | 预测请求总数  |
| `predict_duration_seconds`       | Histogram | 预测请求总耗时 |
| `predict_stage_duration_seconds` | Histogram | 预测各阶段耗时 |

**标签**:
- `status`: 请求状态（success/error）
- `stage`: 阶段名称（image_load, inference, ply_save, total）

#### GPU 监控指标

| 指标名称                      | 类型    | 说明            |
|---------------------------|-------|---------------|
| `gpu_memory_used_mb`      | Gauge | GPU 内存使用量（MB） |
| `gpu_utilization_percent` | Gauge | GPU 利用率百分比    |
| `gpu_info`                | Gauge | GPU 信息        |

**标签**:
- `device_id`: 设备 ID
- `name`: GPU 名称
- `vendor`: 厂商名称

#### 系统指标

| 指标名称              | 类型    | 说明      |
|-------------------|-------|---------|
| `active_tasks`    | Gauge | 当前活跃任务数 |
| `app_info`        | Info  | 应用信息    |
| `input_size_info` | Gauge | 输入图像尺寸  |

### Prometheus 集成

#### 安装 Prometheus

```bash
# 下载 Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.47.0/prometheus-2.47.0.linux-amd64.tar.gz
tar xvfz prometheus-2.47.0.linux-amd64.tar.gz
cd prometheus-2.47.0.linux-amd64

# 创建配置文件
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mlsharp'
    static_configs:
      - targets: ['localhost:8000']
EOF

# 启动 Prometheus
./prometheus
```

访问 Prometheus UI: http://localhost:9090

#### 使用 Grafana 可视化

1. 安装 Grafana
2. 添加 Prometheus 数据源
3. 创建仪表板

**推荐仪表板配置**:

- HTTP 请求速率: `rate(http_requests_total[5m])`
- 预测请求速率: `rate(predict_requests_total[5m])`
- 平均响应时间: `rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])`
- GPU 内存使用: `gpu_memory_used_mb`
- GPU 利用率: `gpu_utilization_percent`
- 活跃任务数: `active_tasks`

### 性能监控示例

#### 查看请求速率

```bash
# 查看最近 5 分钟的请求速率
curl 'http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])'
```

#### 查看平均响应时间

```bash
# 查看最近 5 分钟的平均响应时间
curl 'http://localhost:9090/api/v1/query?query=rate(http_request_duration_seconds_sum[5m])%20%2F%20rate(http_request_duration_seconds_count[5m])'
```

#### 查看 GPU 使用情况

```bash
# 查看 GPU 内存使用
curl 'http://localhost:9090/api/v1/query?query=gpu_memory_used_mb'

# 查看 GPU 利用率
curl 'http://localhost:9090/api/v1/query?query=gpu_utilization_percent'
```

### 监控最佳实践

1. **设置告警规则**
   - 请求错误率超过 5%
   - 平均响应时间超过 60 秒
   - GPU 内存使用超过 90%
   - GPU 利用率超过 95%

2. **定期检查指标**
   - 每天查看请求量和响应时间趋势
   - 监控 GPU 资源使用情况
   - 分析错误日志和失败请求

3. **性能优化**
   - 根据响应时间调整输入尺寸
   - 根据 GPU 使用情况优化并发数
   - 根据错误率优化模型配置
   - 显存不足时启用梯度检查点（--gradient-checkpointing）

</details>

---

## API 文档

<details>
<summary><b>点击展开查看 API 文档详情</b></summary>

## MLSharp 提供了完整的 REST API，支持从单张图片生成 3D 模型。

### 访问地址

启动服务后，可以通过以下方式访问 API 文档：

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **OpenAPI JSON**: http://127.0.0.1:8000/openapi.json

### API 版本控制

所有 API 端点都使用版本控制，当前版本为 `v1`。

| 版本  | 基础路径      | 状态     |
|-----|----------|--------|
| v1  | `/v1`    | 当前版本   |
| v2  | `/v2`    | 计划中    |

**向后兼容性**: v1 API 将继续维护和更新。

### 认证方式

当前版本无需认证，未来版本将支持 API Key 和 JWT Token 认证。

### 响应格式

所有 API 响应使用 JSON 格式。

#### 成功响应

```json
{
  "status": "success",
  "url": "http://127.0.0.1:8000/files/abc123/output.ply",
  "processing_time": 15.5,
  "task_id": "abc123"
}
```

#### 错误响应

```json
{
  "error": "ValidationError",
  "message": "请求参数验证失败",
  "status_code": 422,
  "path": "/v1/predict",
  "timestamp": "2026-01-31T12:00:00Z"
}
```

### API 端点

#### 1. 预测接口

**端点**: `POST /v1/predict`

**描述**: 从单张图片生成 3D 模型

**请求**:
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Body**:
  - `file`: 图片文件（JPG 格式，推荐尺寸: 512x512 - 1024x1024）

**响应模型**:
```json
{
  "status": "string",
  "url": "string",
  "processing_time": "float",
  "task_id": "string"
}
```

**示例**:
```bash
curl -X POST "http://127.0.0.1:8000/v1/predict" \
  -F "file=@input.jpg"
```

**Python 示例**:
```python
import requests

with open('input.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8000/v1/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"3D 模型 URL: {result['url']}")
```

#### 2. 健康检查

**端点**: `GET /v1/health`

**描述**: 检查服务是否正常运行以及 GPU 状态

**响应模型**:
```json
{
  "status": "string",
  "gpu_available": "boolean",
  "gpu_vendor": "string",
  "gpu_name": "string"
}
```

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/health"
```

**响应**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_vendor": "NVIDIA",
  "gpu_name": "NVIDIA GeForce RTX 4090"
}
```

#### 3. 系统统计

**端点**: `GET /v1/stats`

**描述**: 获取系统统计信息

**响应模型**:
```json
{
  "gpu": {
    "available": "boolean",
    "vendor": "string",
    "name": "string",
    "count": "integer",
    "memory_mb": "float"
  }
}
```

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/stats"
```

**响应**:
```json
{
  "gpu": {
    "available": true,
    "vendor": "NVIDIA",
    "name": "NVIDIA GeForce RTX 4090",
    "count": 1,
    "memory_mb": 2048.5
  }
}
```

#### 4. 缓存统计

**端点**: `GET /v1/cache`

**描述**: 获取缓存统计信息

**响应模型**:
```json
{
  "enabled": "boolean",
  "size": "integer",
  "max_size": "integer",
  "hits": "integer",
  "misses": "integer",
  "hit_rate": "float"
}
```

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/cache"
```

**响应**:
```json
{
  "enabled": true,
  "size": 45,
  "max_size": 100,
  "hits": 120,
  "misses": 30,
  "hit_rate": 80.0
}
```

#### 5. 清空缓存

**端点**: `POST /v1/cache/clear`

**描述**: 清空所有缓存条目

**响应模型**:
```json
{
  "status": "string",
  "message": "string"
}
```

**示例**:
```bash
curl -X POST "http://127.0.0.1:8000/v1/cache/clear"
```

**响应**:
```json
{
  "status": "success",
  "message": "缓存已清空"
}
```

#### 6. Prometheus 指标

**端点**: `GET /metrics`

**描述**: 获取 Prometheus 格式的监控指标

**响应格式**: text/plain

**示例**:
```bash
curl "http://127.0.0.1:8000/metrics"
```

#### 7. 获取 Webhook 列表

**端点**: `GET /v1/webhooks`

**描述**: 获取所有已注册的 Webhook

**响应模型**:
```json
{
  "enabled": "boolean",
  "webhooks": {
    "event_type": "string"
  }
}
```

**示例**:
```bash
curl "http://127.0.0.1:8000/v1/webhooks"
```

**响应**:
```json
{
  "enabled": true,
  "webhooks": {
    "task_completed": "https://example.com/webhook/completed",
    "task_failed": "https://example.com/webhook/failed"
  }
}
```

#### 8. 注册 Webhook

**端点**: `POST /v1/webhooks`

**描述**: 注册一个新的 Webhook

**请求体**:
```json
{
  "event_type": "string",
  "url": "string"
}
```

**响应模型**:
```json
{
  "status": "string",
  "message": "string"
}
```

**示例**:
```bash
curl -X POST "http://127.0.0.1:8000/v1/webhooks" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "task_completed",
    "url": "https://example.com/webhook/completed"
  }'
```

**响应**:
```json
{
  "status": "success",
  "message": "Webhook 已注册: task_completed -> https://example.com/webhook/completed"
}
```

#### 9. 注销 Webhook

**端点**: `DELETE /v1/webhooks/{event_type}`

**描述**: 注销指定事件类型的 Webhook

**路径参数**:
- `event_type`: 事件类型

**响应模型**:
```json
{
  "status": "string",
  "message": "string"
}
```

**示例**:
```bash
curl -X DELETE "http://127.0.0.1:8000/v1/webhooks/task_completed"
```

**响应**:
```json
{
  "status": "success",
  "message": "Webhook 已注销: task_completed"
}
```

### 错误处理

API 使用标准 HTTP 状态码表示请求状态：

| 状态码  | 说明                  |
|------|---------------------|
| 200  | 成功                  |
| 400  | 请求参数错误              |
| 404  | 资源不存在               |
| 422  | 请求参数验证失败（Pydantic） |
| 500  | 服务器内部错误             |

### 完整 Python 客户端示例

```python
import requests
import json

class MLSharpClient:
    """MLSharp 3D Maker API 客户端"""
    
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/v1"
    
    def predict(self, image_path):
        """从图片生成 3D 模型"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.api_base}/predict",
                files={'file': f}
            )
            response.raise_for_status()
            return response.json()
    
    def health(self):
        """健康检查"""
        response = requests.get(f"{self.api_base}/health")
        response.raise_for_status()
        return response.json()
    
    def stats(self):
        """获取系统统计"""
        response = requests.get(f"{self.api_base}/stats")
        response.raise_for_status()
        return response.json()
    
    def cache_stats(self):
        """获取缓存统计"""
        response = requests.get(f"{self.api_base}/cache")
        response.raise_for_status()
        return response.json()
    
    def clear_cache(self):
        """清空缓存"""
        response = requests.post(f"{self.api_base}/cache/clear")
        response.raise_for_status()
        return response.json()
    
    def list_webhooks(self):
        """获取 Webhook 列表"""
        response = requests.get(f"{self.api_base}/webhooks")
        response.raise_for_status()
        return response.json()
    
    def register_webhook(self, event_type: str, url: str):
        """注册 Webhook"""
        response = requests.post(
            f"{self.api_base}/webhooks",
            json={"event_type": event_type, "url": url}
        )
        response.raise_for_status()
        return response.json()
    
    def unregister_webhook(self, event_type: str):
        """注销 Webhook"""
        response = requests.delete(f"{self.api_base}/webhooks/{event_type}")
        response.raise_for_status()
        return response.json()

# 使用示例
if __name__ == "__main__":
    client = MLSharpClient()
    
    # 健康检查
    health = client.health()
    print(f"服务状态: {health['status']}")
    print(f"GPU: {health['gpu_name']}")
    
    # 生成 3D 模型
    result = client.predict("input.jpg")
    print(f"任务 ID: {result['task_id']}")
    print(f"处理时间: {result['processing_time']:.2f} 秒")
    print(f"下载 URL: {result['url']}")
```

### 最佳实践

1. **错误处理**: 始终检查响应状态码和错误消息
2. **重试机制**: 对网络错误实现指数退避重试
3. **超时设置**: 为所有请求设置合理的超时时间
4. **缓存利用**: 利用缓存 API 避免重复计算
5. **健康检查**: 定期调用健康检查接口监控服务状态
6. **日志记录**: 记录所有 API 调用和响应时间

</details>

---

## 代码架构

<details>
<summary><b>点击展开查看代码架构详情</b></summary>

### 核心类

#### 1. 配置类
- **AppConfig**: 应用配置管理
- **GPUConfig**: GPU 配置和状态
- **CLIArgs**: 命令行参数解析

#### 2. 工具类
- **Logger**: 统一日志输出

#### 3. 管理器类
- **GPUManager**: GPU 检测、初始化和优化配置
- **ModelManager**: 模型加载和推理管理
- **MetricsManager**: 监控指标收集和管理

#### 4. 应用主类
- **MLSharpApp**: 应用主入口和生命周期管理

### 代码质量改进

| 方面    | 改进                        |
|-------|---------------------------|
| 代码行数  | 减少 33.84%（1965 → ~1300 行） |
| 类型提示  | 完整覆盖                      |
| 文档字符串 | 所有类和方法                    |
| 代码复用  | 消除重复                      |
| 可测试性  | 组件独立                      |
| 可维护性  | 显著提升                      |

### 性能对比

| 指标   | 重构前     | 重构后     | 变化    |
|------|---------|---------|-------|
| 启动时间 | ~15-20秒 | ~5-10秒  | 减少50% |
| 首次推理 | ~30-40秒 | ~30-40秒 | 无变化   |
| 后续推理 | ~15-20秒 | ~15-20秒 | 无变化   |
| 内存占用 | ~2-4GB  | ~2-4GB  | 无变化   |

</details>

---

## 当前已知问题

<details>
<summary><b>点击展开查看当前已知问题</b></summary>

### 问题 1: CUDA 不可用（Intel 集显 + NVIDIA 独显）
**症状**: 系统检测到 NVIDIA 显卡但提示 CUDA 不可用
**原因**: PyTorch 可能未编译 CUDA 支持或驱动未正确安装
**解决方案**:
```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，重新安装带 CUDA 的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 问题 2: ProcessPoolExecutor 内存占用较高
**症状**: 多个并发请求时内存占用增长较快
**原因**: 进程池会为每个进程创建独立的内存空间
**解决方案**:
- 减少进程池大小：`max_workers=2`
- 或回退到线程池：改用 `ThreadPoolExecutor`

### 问题 3: 日志文件可能过大
**症状**: logs/ 目录占用大量磁盘空间
**原因**: loguru 默认不限制日志文件大小
**解决方案**:
- 定期清理旧日志文件
- 或在配置中启用日志压缩

</details>

---

## 故障排除

<details>
<summary><b>点击展开查看故障排除详情</b></summary>

### 问题 1: 启动失败
**症状**: 双击启动脚本后闪退或报错

**解决方案**:
1. 检查 Python 环境是否完整
2. 查看日志文件 `logs/` 中的错误信息
3. 使用命令行参数查看详细错误：`python app.py --no-browser`
4. 检查项目路径是否存在中文

### 问题 2: GPU 检测不到
**症状**: 提示使用 CPU 模式，但实际有 GPU

**解决方案**:
1. NVIDIA 用户检查显卡驱动和 CUDA
2. AMD 用户检查 ROCm 驱动
3. 检查显卡是否被其他程序占用
4. 使用命令行参数强制指定：`python app.py --mode nvidia`

### 问题 3: GPU 厂商检测错误
**症状**: NVIDIA GPU 被误识别为 AMD 或 Intel

**解决方案**:
1. 使用命令行参数强制指定模式：`python app.py --mode nvidia`
2. 手动选择对应的启动脚本

### 问题 4: 内存不足
**症状**: 提示显存不足或程序崩溃

**解决方案**:
1. 使用较小的输入图片（建议 < 1024x1024）
2. 关闭其他占用显存的程序
3. 使用 CPU 模式：`python app.py --mode cpu`
4. 禁用混合精度：`python app.py --no-amp`
5. 启用梯度检查点：`python app.py --gradient-checkpointing`（减少 30-50% 显存）

### 问题 5: 推理速度慢
**症状**: 推理时间过长

**可能原因**:
- 使用 CPU 模式
- 老旧 GPU
- 显存不足
- 图片过大
- 缓存未启用

**解决方案**:
1. 使用 GPU 模式（如果可用）
2. 使用更快的启动脚本
3. 缩小输入图片尺寸
4. 升级硬件
5. 启用缓存：`python app.py --enable-cache`（默认已启用）
6. 增加缓存大小：`python app.py --cache-size 200`

### 问题 6: 缓存占用内存过多
**症状**: 程序运行时间过长后内存占用持续增长

**解决方案**:
1. 减小缓存大小：`python app.py --cache-size 50`
2. 禁用缓存：`python app.py --no-cache`
3. 定期清空缓存：调用 `POST /v1/cache/clear` API
4. 重启服务

### 问题 7: 缓存未生效
**症状**: 重复处理相同图片时速度没有提升

**可能原因**:
- 缓存被禁用
- 图片内容或焦距略有不同
- 缓存已满并被淘汰

**解决方案**:
1. 检查缓存是否启用：访问 `GET /v1/cache` 查看 `enabled` 字段
2. 确保使用完全相同的图片和焦距
3. 增加缓存大小：`python app.py --cache-size 200`
4. 查看缓存命中率：访问 `GET /v1/cache` 查看 `hit_rate`

### 问题 8: 端口被占用
**症状**: 启动时报错端口已被使用

**解决方案**:
1. 使用其他端口：`python app.py --port 8080`
2. 关闭占用 8000 端口的程序
3. 使用命令查找并关闭占用端口的进程

</details>

---

## 版本历史

<details>
<summary><b>点击展开查看版本历史</b></summary>

暂无

</details>

---

## 技术栈

- **后端框架**: FastAPI + Uvicorn
- **深度学习**: PyTorch + Apple ml-sharp 模型
- **3D 渲染**: 3D Gaussian Splatting
- **GPU 加速**: CUDA (NVIDIA) / ROCm (AMD)
- **CPU 优化**: OpenMP / MKL
- **日志系统**: Loguru
- **监控指标**: Prometheus + Prometheus Client
- **架构设计**: 面向对象 + 管理器模式

---

## 许可证
本项目基于 Apple ml-sharp 模型，请遵守相关开源协议。

---

## 未来改进

### 已完成
- 单元测试: 为每个类添加单元测试
- 配置文件: 支持从配置文件加载配置
- 日志系统: 使用专业的日志库（如 loguru）
- 异步优化: 进一步优化异步处理

<details>
<summary><b>点击展开查看未来改进计划</b></summary>

### 待改进
#### 高优先级
1. **认证授权** - 添加用户认证
   - API Key 认证
   - JWT Token 支持
   - 速率限制

#### 中优先级
1. **任务队列** - 异步任务处理
   - Redis 队列支持
   - 任务状态追踪
   - 批量处理支持

2. **批量处理 API** - 批量图片处理
   - 多文件上传
   - 批量预测
   - 结果打包下载

#### 低优先级
1. **国际化** - 多语言支持
   - i18n 支持
   - 中英文界面
   - 可扩展语言包
   - Google 系统优化

2. **插件系统** - 可扩展架构
   - 自定义插件
   - 模型插件
   - 后处理插件

3. **批处理 API** - 批量图片处理
   - 多文件上传
   - 批量预测
   - 结果打包下载
</details>

---

## 贡献

欢迎提交 **Issue** 和 **Pull Request！**


## 联系方式

- 项目主页: [https://github.com/ChidcGithub/MLSharp-3D-Maker-GPU](https://github.com/ChidcGithub/MLSharp-3D-Maker-GPU)
- 问题反馈: [Issues](https://github.com/ChidcGithub/MLSharp-3D-Maker-GPU/issues)

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐️ Star！**

Modded with ❤️ by Chidc with CPU-Mode-Provider GemosDoDo
README.md Verison Code **2601312022**
</div>
