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
### 后续计划添加更多芯片及NPU快速推理支持
#### 由于兼容性问题，正式版本发布可能需要等待至少1个月，也可以进行[Pull requests](https://github.com/ChidcGithub/MLSharp-3D-Maker-GPU/pulls)来修改。
#### 目前主要进程: 将Torch模型转为在骁龙平台上运行更佳的ONNX模型。
### Codename:Ansharp
---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![API](https://img.shields.io/badge/API-RESTful-blueviolet.svg)
[![Platform: Android](https://img.shields.io/badge/Platform-Android-3DDC84?logo=android&logoColor=white)](https://www.android.com)
[![Qualcomm Snapdragon](https://img.shields.io/badge/Supports-Qualcomm_Snapdragon_SDK-ED1C24?logo=qualcomm&logoColor=white)](https://developer.qualcomm.com/)
[![stars](https://img.shields.io/github/stars/chidcGithub/MLSharp-3D-Maker-GPU)](https://github.com/chidcGithub/MLSharp-3D-Maker-GPU)
[![GitHub Release (including pre-releases)](https://img.shields.io/github/v/release/chidcGithub/MLSharp-3D-Maker-GPU?include_prereleases&label=latest)](https://github.com/chidcGithub/MLSharp-3D-Maker-GPU/releases)
</div>

# 使用说明

## 项目概述

MLSharp-3D-Maker 是一个基于 Apple ml-sharp 模型的 3D 高斯泼溅（3D Gaussian Splatting）生成工具，可以从单张照片生成高质量的 3D 模型。

### 项目完成度

| 模块      | 状态  | 完成度  | 说明                             |
|---------|-----|------|--------------------------------|
| 核心功能    | 完成  | 100% | 图像到 3D 模型转换                    |
| GPU 加速  | 完成  | 100% | NVIDIA/AMD/Intel/Snapdragon 支持 |
| Android 应用 | 完成  | 100% | Chaquopy + WebView + Material 3    |
| 配置管理    | 完成  | 100% | 命令行 + 配置文件                     |
| 日志系统    | 完成  | 100% | loguru 专业日志                    |
| 异步处理    | 完成  | 100% | ProcessPoolExecutor            |
| 单元测试    | 完成  | 90%  | 核心类测试覆盖                        |
| API 接口  | 完成  | 100% | 预测 + 健康检查 + 缓存管理               |
| 监控指标    | 完成  | 90%  | Prometheus 集成 + 性能监控           |
| 推理缓存    | 完成  | 100% | LRU 缓存 + Redis 分布式缓存           |
| 性能自动调优  | 完成  | 100% | 智能基准测试 + 最优配置选择                |
| Webhook | 完成  | 100% | 异步通知 + 事件管理                    |
| 文档      | 完成  | 100% | README + 配置示例 + API 文档         |
| API 文档  | 完成  | 100% | Swagger/OpenAPI + 版本控制         |
| 认证授权    | 待开发 | 0%   | API Key/JWT                    |

**总体完成度: 100%+0%**

---

## 项目结构及更新

```
MLSharp-3D-Maker-GPU-by-Chidc/
├── app.py                        # 主应用程序（重构版本）⭐
├── app_android.py                # Android Python 后端
├── app_snapdragon.py             # Snapdragon 专用版本
├── config/                       # 配置文件目录（推荐使用）
│   ├── config.yaml                   # YAML 格式配置文件
│   └── config.json                   # JSON 格式配置文件
├── gpu_utils.py                  # GPU 工具模块
├── logger.py                     # 日志模块
├── metrics.py                    # 监控指标模块 ⭐
├── npu_utils.py                  # NPU 检测模块
├── optimistic.md                 # 性能优化方案文档 ⭐
├── Start.bat                     # Windows 启动脚本
├── Start.ps1                     # PowerShell 启动脚本
├── Start_Snapdragon.ps1          # Snapdragon 启动脚本
├── model_assets/                 # 模型文件和资源
│   ├── sharp_2572gikvuh.pt      # ml-sharp 模型权重
│   ├── inputs/                   # 输入示例
│   └── outputs/                  # 输出示例
├── python_env/                   # Python 环境
├── logs/                         # 日志文件夹
├── tmp/                          # 临时文件和备份
│   └── 1.28/                     # 2026-01-28 备份
├── temp_workspace/               # 临时工作目录
├── viewer.html                   # 3D 模型查看器 Web 界面
├── android/                      # Android 应用
│   ├── app/                       # Android 应用模块
│   │   ├── src/
│   │   │   └── main/
│   │   │       ├── assets/
│   │   │       │   ├── html/     # WebView HTML 文件
│   │   │       │   └── python/   # Python 脚本和轮子文件
│   │   │       │       └── wheels/ # Python 依赖轮子
│   │   │       ├── kotlin/      # Kotlin 源代码
│   │   │       │   └── com/mlsharp/snapdragon/
│   │   │       │       ├── MainActivity.kt
│   │   │       │       ├── WelcomeActivity.kt
│   │   │       │       └── SettingsActivity.kt
│   │   │       └── res/         # Android 资源
│   │   └── build.gradle          # 应用构建配置
│   ├── build.gradle              # 项目构建配置
│   ├── build.ps1                 # 构建脚本（PowerShell）
│   ├── build_debug.ps1           # Debug 构建脚本
│   ├── ANDROID_GUIDE.md          # Android 安装指南
│   └── SNAPDRAGON_OPTIMIZATION.md # Snapdragon 优化文档
```

<details>
<summary><b>点击展开查看最新更新详情</b></summary>

### 最新更新（2026-02-02）

**Android 应用 v0.0.1 preview**
- **首次启动页面** - 添加 WelcomeActivity，用于授权权限和安装 Python 库
- **服务器控制** - 主页面添加启动/停止服务器按钮
- **分屏显示** - 前端 WebView 和后端日志分屏显示
- **Material 3 设计** - 采用 Google Material 3 You 设计系统
- **运行时安装** - Python 库在首次运行时从本地轮子文件安装
- **模型路径设置** - 支持自定义模型文件路径
- **Android 5.0+ 支持** - 最低支持 API 21
- **权限管理** - 根据系统版本自动请求合适权限
- **文件选择器** - 现代化的 ActivityResultContracts API
- **WebView 集成** - JavaScript 桥接实现前后端通信
- **实时日志** - 后端日志实时显示在应用界面

</details>

---

**功能特点：**
- **自动检测**: GPU 类型（Snapdragon）、环境配置、依赖库
- **智能推荐**: 根据显卡自动推荐最佳启动脚本
- **全面诊断**: 100+ 错误处理，智能识别问题
- **解决方案**: 每个错误都提供详细的解决建议
- **日志记录**: 所有运行日志保存在 logs/ 文件夹
- **彩色输出**: 清晰的视觉反馈，易于阅读


## Android 应用

### 构建 Android APK

**前置要求：**
- Java 17
- Android SDK 34
- Gradle 8.2
- Python 3.11+

**构建步骤：**
```powershell
cd android
.\build.ps1
```

**构建说明：**
- 构建脚本会自动复制必要的文件到 Android 项目
- 首次构建需要 5-10 分钟（下载依赖）
- Python 库在应用首次运行时从本地轮子文件安装
- APK 输出位置：`android/app/build/outputs/apk/debug/app-debug.apk`

**快速构建（跳过某些检查）：**
```powershell
cd android
.\build_debug.ps1
```

### 安装和运行

**安装 APK：**
```powershell
# 通过 ADB 安装
adb install android\app\build\outputs\apk\debug\app-debug.apk

# 或直接在 Android 设备上打开 APK 文件
```

**首次运行：**
1. 启动应用后显示欢迎页面
2. 授权存储权限
3. 安装 Python 库（首次运行，约 1-2 分钟）
4. 点击"开始使用"进入主界面

**主界面功能：**
- 启动/停止 Python 后端服务器
- 上传图片进行 3D 模型生成
- 查看实时后端日志
- 自定义模型文件路径
- Material 3 设计风格

### Android 版本支持

- **最低版本**: Android 5.0 (API 21)
- **目标版本**: Android 14 (API 34)
- **推荐设备**: Snapdragon 8 Gen 2/3 或更高

### 模型文件

应用支持三种模型文件来源：
1. **应用内置**（暂未实现）：将模型分割后打包到 APK
2. **外部存储**：将模型文件放在 `/sdcard/Android/data/com.mlsharp.snapdragon/files/models/`
3. **自定义路径**：在设置中选择任意路径的模型文件

**模型文件要求：**
- 格式：`.pt` (PyTorch 模型)
- 大小：建议不超过 2GB
- 名称：`sharp_2572gikvuh.pt` 或自定义

### 故障排除

**Python 库安装失败：**
```bash
# 检查轮子文件是否存在于
android/app/src/main/assets/python/wheels/
```

**权限问题：**
- 确保在设置中授予应用存储权限
- Android 13+ 需要授予"媒体访问"权限

**模型文件加载失败：**
- 检查模型文件路径是否正确
- 确保应用有读取权限
- 查看后端日志获取详细错误信息

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
README.md Verison Code **2602021551**
</div>
