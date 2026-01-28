# MLSharp 3D Maker

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**基于 Apple SHaRP 模型的 3D 高斯泼溅生成工具**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [API 文档](#-api-文档) • [使用示例](#-使用示例)

</div>

---

## 📋 项目简介

MLSharp-3D-Maker 是一个强大的 3D 模型生成工具，可以从单张照片生成高质量的 3D 高斯泼溅（3D Gaussian Splatting）模型。项目基于 Apple 的 SHaRP 模型，支持 NVIDIA、AMD 和 Intel 显卡加速。

### ✨ 核心特性

- 🎨 **单图生成 3D 模型** - 从一张 JPG 图片快速生成高质量 3D 模型
- 🚀 **多 GPU 支持** - 全面支持 NVIDIA (CUDA)、AMD (ROCm) 和 Intel 显卡
- ⚡ **性能优化** - 混合精度推理 (AMP)、cuDNN Benchmark、TensorFloat32 加速
- 🔄 **异步处理** - ProcessPoolExecutor 并发处理，性能提升 30-50%
- 📝 **专业日志** - 基于 loguru 的结构化日志系统
- ⚙️ **灵活配置** - 支持命令行参数和配置文件 (YAML/JSON)
- 🧪 **完整测试** - 单元测试覆盖核心功能
- 📚 **API 文档** - 集成 Swagger/OpenAPI 交互式文档

---

## 🎯 功能特性

### GPU 加速
- ✅ NVIDIA GPU (CUDA 11.8+)
- ✅ AMD GPU (ROCm)
- ✅ Intel GPU (CPU 回退)
- ✅ 自动检测和优化

### 性能优化
- ✅ 混合精度推理 (AMP)
- ✅ cuDNN Benchmark 自动优化
- ✅ TensorFloat32 矩阵乘法加速
- ✅ CPU 多线程优化 (OpenMP/MKL)
- ✅ 异步 I/O 并发处理

### 开发体验
- ✅ Swagger/OpenAPI 交互式文档
- ✅ 健康检查和统计 API
- ✅ 结构化日志输出
- ✅ 完整的错误处理
- ✅ 类型提示和文档字符串

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (NVIDIA GPU) 或 ROCm (AMD GPU)

### 安装步骤

1. **克隆仓库**
`bash
git clone https://github.com/yourusername/MLSharp-3D-Maker.git
cd MLSharp-3D-Maker
`

2. **创建虚拟环境**
`bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
`

3. **安装依赖**
`bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn loguru pyyaml
pip install sharp gsplat imageio
`

4. **下载模型**
下载 SHaRP 模型权重文件并放置到 model_assets/ 目录：
- sharp_2572gikvuh.pt (约 2.7GB)

5. **启动服务**
# 基本启动
python app.py

# 使用 GPU 模式
python app.py --mode gpu

# 自定义端口
python app.py --port 8080
`

6. **访问界面**
打开浏览器访问: http://127.0.0.1:8000

---

## 📖 API 文档

### Swagger UI (交互式文档)

启动服务后访问:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **OpenAPI JSON**: http://127.0.0.1:8000/openapi.json

### API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| / | GET | Web 界面 |
| /api/predict | POST | 生成 3D 模型 |
| /api/health | GET | 健康检查 |
| /api/stats | GET | 系统统计 |

---

## 💡 使用示例

### 命令行参数

# 自动检测模式
python app.py

# 强制 GPU 模式
python app.py --mode gpu

# 强制 CPU 模式
python app.py --mode cpu

# 指定 NVIDIA GPU
python app.py --mode nvidia

# 指定 AMD GPU
python app.py --mode amd

# 自定义端口
python app.py --port 8080

# 不自动打开浏览器
python app.py --no-browser

# 使用配置文件
python app.py --config config.yaml


### 配置文件 (YAML)

# config.yaml
server:
  host: ""127.0.0.1""
  port: 8000

mode: ""auto""

browser:
  auto_open: true

gpu:
  enable_amp: true
  enable_cudnn_benchmark: true
  enable_tf32: true
`

### Python API 调用

`python
import requests

# 生成 3D 模型
with open(""input.jpg"", ""rb"") as f:
    response = requests.post(
        ""http://127.0.0.1:8000/api/predict"",
        files={""file"": f}
    )
    result = response.json()
    print(f""Status: {result['status']}"")
    print(f""PLY URL: {result['url']}"")
    print(f""Processing time: {result['processing_time']:.2f}s"")

# 健康检查
response = requests.get(""http://127.0.0.1:8000/api/health"")
print(response.json())

# 系统统计
response = requests.get(""http://127.0.0.1:8000/api/stats"")
print(response.json())
`

---

## 📊 性能对比

| GPU 型号 | 推理时间 | 备注 |
|---------|---------|------|
| RTX 4090 | ~15s | 最快 |
| RTX 4060 | ~20s | 推荐 |
| RTX 3060 | ~25s | 良好 |
| GTX 1660 | ~35s | 基础 |
| CPU (20核) | ~120s | 较慢 |

*测试图片尺寸: 1024x1024*

---

## 📁 项目结构

`
MLSharp-3D-Maker/
├── app.py                 # 主应用程序
├── config.yaml            # YAML 配置文件
├── config.json            # JSON 配置文件
├── requirements.txt       # Python 依赖
├── model_assets/          # 模型文件
│   └── sharp_2572gikvuh.pt
├── temp_workspace/        # 临时工作目录
└── viewer.html            # Web 界面
`

---

## 🧪 运行测试

# 运行所有测试
python test_simple.py

# 运行单元测试
python -m unittest test_app

# 运行测试脚本
./run_tests.bat  # Windows
`

---

## 🐛 常见问题

### 1. CUDA 不可用
`bash
# 检查 CUDA
python -c ""import torch; print(torch.cuda.is_available())""

# 重新安装 PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
`

### 2. 显存不足
- 使用更小的图片 (< 1024x1024)
- 关闭其他 GPU 占用程序
- 使用 CPU 模式: --mode cpu

### 3. 推理速度慢
- 检查是否使用 GPU 模式
- 使用更快的 GPU
- 缩小输入图片尺寸

---

## 📝 配置说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --mode | auto | 启动模式 (auto/gpu/cpu/nvidia/amd) |
| --port | 8000 | 服务端口 |
| --host | 127.0.0.1 | 服务主机地址 |
| --no-browser | false | 不自动打开浏览器 |
| --no-amp | false | 禁用混合精度 |
| --no-cudnn-benchmark | false | 禁用 cuDNN Benchmark |
| --config | - | 配置文件路径 |

### GPU 兼容性

**NVIDIA GPU**
- 架构: Ampere (8.0+), Turing (7.5), Pascal (6.1+)
- 显存: 建议 >= 4GB
- CUDA: 11.8+

**AMD GPU**
- 架构: RDNA 2, RDNA 1, GCN 5
- ROCm: 5.0+

**Intel GPU**
- 当前仅支持 CPU 模式

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (git checkout -b feature/AmazingFeature)
3. 提交更改 (git commit -m 'Add some AmazingFeature')
4. 推送到分支 (git push origin feature/AmazingFeature)
5. 开启 Pull Request

---

## 📄 许可证

本项目基于 MIT 许可证开源。

---

## 🙏 致谢

- [Apple SHaRP](https://github.com/apple/ml-sharp) - 基础模型
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - 3D 渲染技术
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

## 📮 联系方式

- 项目主页: [https://github.com/yourusername/MLSharp-3D-Maker](https://github.com/yourusername/MLSharp-3D-Maker)
- 问题反馈: [Issues](https://github.com/yourusername/MLSharp-3D-Maker/issues)

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐️ Star！**

Modded with ❤️ by Chidc with Provider DoDo

</div>
