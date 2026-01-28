# -*- coding: utf-8 -*-
"""
MLSharp-3D-Maker - 统一版本
支持 NVIDIA/AMD/Intel GPU 和 CPU,自动检测并优化
"""
import sys
import os
import subprocess
import platform
import traceback
import argparse
import shutil
import uuid
import threading
import webbrowser
import time
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import json
import yaml
from loguru import logger

# 设置输出编码为 UTF-8(Windows)
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    sys.stdin = codecs.getreader('utf-8')(sys.stdin.detach())

# ================= 配置类 =================
@dataclass
class AppConfig:
    """应用配置"""
    base_dir: str
    python_env: str
    assets_dir: str
    checkpoint: str
    temp_dir: str
    
    @classmethod
    def from_current_dir(cls) -> 'AppConfig':
        """从当前目录创建配置"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return cls(
            base_dir=base_dir,
            python_env=os.path.join(base_dir, "python_env"),
            assets_dir=os.path.join(base_dir, "model_assets"),
            checkpoint=os.path.join(base_dir, "model_assets", "sharp_2572gikvuh.pt"),
            temp_dir=os.path.join(base_dir, "temp_workspace")
        )


@dataclass
class GPUConfig:
    """GPU 配置"""
    available: bool = False
    vendor: str = "Unknown"
    name: str = "N/A"
    cuda_version: Optional[str] = None
    count: int = 0
    compute_capability: int = 0
    supports_tf32: bool = False
    supports_bf16: bool = False
    use_amp: bool = False
    use_cudnn_benchmark: bool = False
    use_tf32: bool = False
    is_rocm: bool = False


@dataclass
class CLIArgs:
    """命令行参数"""
    mode: str = 'auto'
    port: int = 8000
    host: str = '127.0.0.1'
    no_browser: bool = False
    no_amp: bool = False
    no_cudnn_benchmark: bool = False
    config_file: Optional[str] = None


# ================= 配置文件加载 =================
def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    从配置文件加载配置
    
    支持 YAML 和 JSON 格式
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif file_ext == '.json':
                return json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_ext}")
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")


def merge_config_with_args(config: Dict[str, Any], args: CLIArgs) -> Dict[str, Any]:
    """
    合并配置文件和命令行参数
    
    命令行参数优先级高于配置文件
    
    Args:
        config: 配置文件字典
        args: 命令行参数
        
    Returns:
        合并后的配置字典
    """
    # 服务器配置
    if args.host != '127.0.0.1':
        config.setdefault('server', {})['host'] = args.host
    if args.port != 8000:
        config.setdefault('server', {})['port'] = args.port
    
    # 启动模式
    if args.mode != 'auto':
        config['mode'] = args.mode
    
    # 浏览器配置
    if args.no_browser:
        config.setdefault('browser', {})['auto_open'] = False
    
    # GPU 配置
    if args.no_amp:
        config.setdefault('gpu', {})['enable_amp'] = False
    if args.no_cudnn_benchmark:
        config.setdefault('gpu', {})['enable_cudnn_benchmark'] = False
    
    return config


def config_to_cli_args(config: Dict[str, Any]) -> CLIArgs:
    """
    将配置字典转换为 CLIArgs
    
    Args:
        config: 配置字典
        
    Returns:
        CLIArgs 对象
    """
    server = config.get('server', {})
    browser = config.get('browser', {})
    gpu = config.get('gpu', {})
    
    return CLIArgs(
        mode=config.get('mode', 'auto'),
        port=server.get('port', 8000),
        host=server.get('host', '127.0.0.1'),
        no_browser=not browser.get('auto_open', True),
        no_amp=not gpu.get('enable_amp', True),
        no_cudnn_benchmark=not gpu.get('enable_cudnn_benchmark', True),
        config_file=None
    )


# ================= 日志工具 =================
class Logger:
    """日志工具类 - 基于 loguru"""
    
    def __init__(self):
        """初始化日志系统"""
        # 移除默认的 handler
        logger.remove()
        
        # 添加控制台 handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # 添加文件 handler（可选）
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"mlsharp_{time.strftime('%Y%m%d')}.log")
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    @staticmethod
    def section(title: str, char: str = '=', length: int = 60):
        """打印分隔线"""
        print(f"\n{char * length}")
        print(f"[INFO] {title}")
        print(f"{char * length}\n")
    
    @staticmethod
    def error(error_msg: str, solution: Optional[str] = None):
        """打印错误信息和解决方案"""
        logger.error(error_msg)
        if solution:
            logger.info(f"解决方案: {solution}")
        logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
    
    @staticmethod
    def success(msg: str):
        """打印成功信息"""
        logger.success(msg)
    
    @staticmethod
    def warning(msg: str):
        """打印警告信息"""
        logger.warning(msg)
    
    @staticmethod
    def info(msg: str):
        """打印信息"""
        logger.info(msg)
    
    @staticmethod
    def debug(msg: str):
        """打印调试信息"""
        logger.debug(msg)
    
    @staticmethod
    def exception(msg: str):
        """打印异常信息"""
        logger.exception(msg)


# ================= 命令行参数解析 =================
def parse_command_args() -> Tuple[CLIArgs, Optional[Dict[str, Any]]]:
    """解析命令行参数
    
    Returns:
        (CLIArgs, 配置文件字典或None)
    """
    parser = argparse.ArgumentParser(
        description='MLSharp 3D模型生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
启动模式说明:
  auto     自动检测并选择最佳模式（默认）
  gpu      强制使用 GPU 模式（自动检测厂商）
  cpu      强制使用 CPU 模式
  nvidia   强制使用 NVIDIA GPU 模式
  amd      强制使用 AMD GPU 模式（ROCm）

配置文件:
  支持 YAML 和 JSON 格式
  配置文件优先级低于命令行参数

示例:
  python app.py                    # 自动检测模式
  python app.py --mode gpu         # 强制 GPU 模式
  python app.py --config config.yaml  # 使用配置文件
  python app.py --port 8080        # 使用 8080 端口
        """
    )
    
    parser.add_argument('--mode', '-m', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu', 'nvidia', 'amd'],
                        help='启动模式：auto(自动), gpu(GPU), cpu(CPU), nvidia(NVIDIA), amd(AMD)')
    parser.add_argument('--port', '-p', type=int, default=8000,
                        help='Web 服务端口（默认：8000）')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Web 服务主机地址（默认：127.0.0.1）')
    parser.add_argument('--no-browser', action='store_true',
                        help='不自动打开浏览器')
    parser.add_argument('--no-amp', action='store_true',
                        help='禁用混合精度推理（AMP）')
    parser.add_argument('--no-cudnn-benchmark', action='store_true',
                        help='禁用 cuDNN Benchmark')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='配置文件路径（支持 YAML 和 JSON）')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config_dict = None
    if args.config:
        try:
            config_dict = load_config_file(args.config)
            # 合并配置文件和命令行参数
            config_dict = merge_config_with_args(config_dict, args)
            # 转换为 CLIArgs
            cli_args = config_to_cli_args(config_dict)
            cli_args.config_file = args.config
        except Exception as e:
            print(f"[ERROR] 加载配置文件失败: {e}")
            print("[INFO] 使用默认配置和命令行参数")
            cli_args = CLIArgs(
                mode=args.mode,
                port=args.port,
                host=args.host,
                no_browser=args.no_browser,
                no_amp=args.no_amp,
                no_cudnn_benchmark=args.no_cudnn_benchmark,
                config_file=None
            )
    else:
        cli_args = CLIArgs(
            mode=args.mode,
            port=args.port,
            host=args.host,
            no_browser=args.no_browser,
            no_amp=args.no_amp,
            no_cudnn_benchmark=args.no_cudnn_benchmark,
            config_file=None
        )
    
    return cli_args, config_dict


# ================= 导入 FastAPI 相关模块 =================
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ================= GPU 管理器 =================
class GPUManager:
    """GPU 管理器"""
    
    def __init__(self, config: GPUConfig, args: CLIArgs):
        self.config = config
        self.args = args
        self.device = torch.device("cpu")
    
    @staticmethod
    def detect_gpu_vendor_wmi() -> str:
        """通过 WMI 检测显卡厂商"""
        try:
            # 首先尝试使用 PowerShell Get-CimInstance(Windows 11 推荐)
            result = subprocess.run(
                ['powershell', '-Command', 
                 'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'],
                capture_output=True, text=True, encoding='utf-8', errors='ignore'
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                # 优先检测 NVIDIA/AMD 独立显卡，避免被 Intel 集显干扰
                nvidia_found = False
                amd_found = False
                intel_found = False
                
                for line in lines:
                    name = line.strip().lower()
                    if 'nvidia' in name or 'geforce' in name or 'quadro' in name or 'tesla' in name or 'rtx' in name or 'gtx' in name:
                        nvidia_found = True
                    elif 'amd' in name or 'radeon' in name or 'rx' in name:
                        amd_found = True
                    elif 'intel' in name or 'iris' in name or 'uhd' in name or 'arc' in name:
                        intel_found = True
                
                # 返回优先级最高的厂商
                if nvidia_found:
                    return 'NVIDIA'
                elif amd_found:
                    return 'AMD'
                elif intel_found:
                    return 'Intel'
            else:
                # 回退到 wmic 命令(Windows 10 及更早版本)
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True, text=True, encoding='utf-8', errors='ignore'
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]
                    nvidia_found = False
                    amd_found = False
                    intel_found = False
                    
                    for line in lines:
                        name = line.strip().lower()
                        if 'nvidia' in name or 'geforce' in name or 'quadro' in name or 'tesla' in name or 'rtx' in name or 'gtx' in name:
                            nvidia_found = True
                        elif 'amd' in name or 'radeon' in name or 'rx' in name:
                            amd_found = True
                        elif 'intel' in name or 'iris' in name or 'uhd' in name or 'arc' in name:
                            intel_found = True
                    
                    if nvidia_found:
                        return 'NVIDIA'
                    elif amd_found:
                        return 'AMD'
                    elif intel_found:
                        return 'Intel'
        except Exception as e:
            Logger.warning(f"WMI 检测失败: {e}")
        return 'Unknown'
    
    @staticmethod
    def check_rocm_available() -> bool:
        """检查 ROCm 是否可用"""
        try:
            if torch.cuda.is_available():
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    return True
                device_name = torch.cuda.get_device_name(0).lower()
                if 'amd' in device_name or 'radeon' in device_name:
                    return True
            return False
        except Exception as e:
            Logger.warning(f"ROCm 检测失败: {e}")
            return False
    
    def initialize(self) -> torch.device:
        """初始化 GPU 设备"""
        Logger.section("GPU 初始化")
        
        if self.args.mode != 'auto':
            Logger.info(f"用户指定启动模式: {self.args.mode.upper()}")
        else:
            Logger.info("自动检测模式")
        
        force_mode = self.args.mode
        if force_mode == 'cpu':
            Logger.info("强制使用 CPU 模式")
        
        try:
            if torch.cuda.is_available() and force_mode != 'cpu':
                self.config.available = True
                self.config.name = torch.cuda.get_device_name(0)
                self.config.cuda_version = torch.version.cuda
                self.config.count = torch.cuda.device_count()
                
                self.config.is_rocm = self.check_rocm_available()
                system_vendor = self.detect_gpu_vendor_wmi()
                
                # 优先根据 GPU 名称判断厂商
                gpu_name_lower = self.config.name.lower()
                
                # 判断 GPU 类型
                if self.config.is_rocm:
                    self.config.vendor = "AMD"
                    Logger.success(f"检测到 AMD GPU: {self.config.name}")
                    Logger.info("   ROCm 支持: 是")
                elif 'nvidia' in gpu_name_lower or 'geforce' in gpu_name_lower or 'quadro' in gpu_name_lower or 'tesla' in gpu_name_lower or 'rtx' in gpu_name_lower or 'gtx' in gpu_name_lower:
                    self.config.vendor = "NVIDIA"
                    Logger.success(f"检测到 NVIDIA GPU: {self.config.name}")
                elif 'amd' in gpu_name_lower or 'radeon' in gpu_name_lower or 'rx' in gpu_name_lower:
                    self.config.vendor = "AMD"
                    Logger.success(f"检测到 AMD GPU: {self.config.name}")
                elif 'intel' in gpu_name_lower or 'iris' in gpu_name_lower or 'uhd' in gpu_name_lower or 'arc' in gpu_name_lower:
                    self.config.vendor = "Intel"
                    Logger.success(f"检测到 Intel GPU: {self.config.name}")
                else:
                    # 如果 GPU 名称无法判断，使用系统检测结果
                    if system_vendor == 'NVIDIA':
                        self.config.vendor = "NVIDIA"
                        Logger.success(f"检测到 NVIDIA GPU: {self.config.name}")
                    elif system_vendor == 'AMD':
                        self.config.vendor = "AMD"
                        Logger.success(f"检测到 AMD GPU: {self.config.name}")
                    elif system_vendor == 'Intel':
                        self.config.vendor = "Intel"
                        Logger.success(f"检测到 Intel GPU: {self.config.name}")
                    else:
                        self.config.vendor = "Unknown"
                        Logger.warning(f"检测到未知 GPU: {self.config.name}")
                
                Logger.info(f"   CUDA/ROCm 版本: {self.config.cuda_version}")
                Logger.info(f"   GPU 数量: {self.config.count}")
                
                # 强制模式处理
                if force_mode == 'nvidia':
                    if self.config.vendor != "NVIDIA":
                        Logger.warning(f"强制使用 NVIDIA 模式，但检测到 {self.config.vendor} GPU")
                    self.config.vendor = "NVIDIA"
                    Logger.info("已强制设置为 NVIDIA 模式")
                elif force_mode == 'amd':
                    if self.config.vendor != "AMD":
                        Logger.warning(f"强制使用 AMD 模式，但检测到 {self.config.vendor} GPU")
                    self.config.vendor = "AMD"
                    Logger.info("已强制设置为 AMD 模式")
                
                # 获取显卡属性
                props = torch.cuda.get_device_properties(0)
                self.config.compute_capability = props.major * 10 + props.minor
                self.config.supports_tf32 = props.major >= 8
                self.config.supports_bf16 = props.major >= 8
                
                Logger.info(f"   计算能力: {props.major}.{props.minor}")
                Logger.info(f"   显存: {props.total_memory / 1024**3:.2f} GB")
                
                if props.total_memory < 4 * 1024**3:
                    Logger.warning("   警告: 显存不足 4GB,可能导致性能问题")
                
                # 配置优化
                self._configure_optimizations(props)
                
                self.device = torch.device("cuda")
            else:
                self._setup_cpu_mode()
        
        except Exception as e:
            Logger.error(f"设备初始化失败: {e}")
            self.device = torch.device("cpu")
            self.config.available = False
        
        # CPU 优化设置
        if not self.config.available:
            torch.set_num_threads(os.cpu_count())
            os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
            os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
            Logger.success(f"CPU 优化已启用({os.cpu_count()} 核心)")
        
        return self.device
    
    def _configure_optimizations(self, props):
        """配置 GPU 优化选项"""
        Logger.info("\n根据显卡能力配置优化:")
        
        # cuDNN Benchmark
        if self.config.vendor == "NVIDIA" and self.config.compute_capability >= 60 and not self.args.no_cudnn_benchmark:
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                self.config.use_cudnn_benchmark = True
                Logger.success("  cuDNN Benchmark: 已启用")
            except Exception as e:
                Logger.warning(f"  cuDNN Benchmark: 启用失败 ({e})")
        else:
            if self.config.vendor != "NVIDIA":
                Logger.info("  cuDNN Benchmark: 不适用(非 NVIDIA GPU)")
            else:
                Logger.warning("  cuDNN Benchmark: 已禁用(显卡计算能力不足)")
        
        # TensorFloat32
        if self.config.vendor == "NVIDIA" and self.config.supports_tf32:
            try:
                torch.set_float32_matmul_precision('high')
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.config.use_tf32 = True
                Logger.success("  TensorFloat32: 已启用")
            except Exception as e:
                Logger.warning(f"  TensorFloat32: 启用失败 ({e})")
        else:
            if self.config.vendor != "NVIDIA":
                Logger.info("  TensorFloat32: 不适用(非 NVIDIA GPU)")
            else:
                Logger.warning("  TensorFloat32: 已禁用(显卡不支持)")
        
        # 混合精度
        if self.config.compute_capability >= 53 and not self.args.no_amp:
            self.config.use_amp = True
            Logger.success("  混合精度推理 (AMP): 已启用")
        else:
            Logger.warning("  混合精度推理 (AMP): 已禁用(显卡计算能力不足)")
    
    def _setup_cpu_mode(self):
        """设置 CPU 模式"""
        system_vendor = self.detect_gpu_vendor_wmi()
        self.config.vendor = system_vendor
        self.device = torch.device("cpu")
        
        Logger.warning("使用 CPU 模式")
        Logger.info("   原因: CUDA/ROCm 不可用")
        
        if system_vendor == "AMD":
            Logger.info("   检测到 AMD 显卡,但 PyTorch 未编译 ROCm 支持")
            Logger.info("   解决方案: 安装 ROCm 版本的 PyTorch")
        elif system_vendor == "NVIDIA":
            Logger.info("   检测到 NVIDIA 显卡,但 CUDA 不可用")
            Logger.info("   请检查:")
            Logger.info("     1. 是否安装 NVIDIA 显卡驱动")
            Logger.info("     2. 显卡是否支持 CUDA")
            Logger.info("     3. PyTorch 是否编译了 CUDA 支持")
        elif system_vendor == "Intel":
            Logger.info("   检测到 Intel 显卡")
            Logger.info("   Intel GPU 暂不支持 GPU 加速")
        else:
            Logger.info("   未检测到支持的 GPU")


# ================= 模型管理器 =================
class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: AppConfig, gpu_config: GPUConfig, device: torch.device):
        self.config = config
        self.gpu_config = gpu_config
        self.device = device
        self.predictor = None
    
    def load_model(self):
        """加载模型"""
        Logger.section("模型加载")
        Logger.info(f"模型文件: {self.config.checkpoint}")
        
        # 检查模型文件
        if not os.path.exists(self.config.checkpoint):
            Logger.error(
                "模型文件不存在!",
                f"请确保模型文件位于: {self.config.checkpoint}\n"
                "下载地址: 请查看项目 README 或联系开发者"
            )
            sys.exit(1)
        
        # 检查文件大小
        model_size = os.path.getsize(self.config.checkpoint) / (1024 * 1024)
        Logger.info(f"模型文件大小: {model_size:.2f} MB")
        
        if model_size < 100:
            Logger.warning(
                "模型文件大小异常(太小),可能已损坏或不完整\n"
                "建议: 重新下载模型文件"
            )
        
        try:
            from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
            from sharp.utils import io
            from sharp.utils.gaussians import Gaussians3D, SceneMetaData, save_ply, unproject_gaussians
            
            Logger.info("正在创建预测器...")
            self.predictor = create_predictor(PredictorParams())
            
            Logger.info("正在加载模型权重...")
            state_dict = torch.load(self.config.checkpoint, weights_only=True, map_location=self.device)
            
            Logger.info("正在加载权重到预测器...")
            self.predictor.load_state_dict(state_dict)
            self.predictor.eval()
            self.predictor.to(self.device)
            
            Logger.success("模型加载完成!")
            Logger.info(f"设备: {self.device}")
            
            if self.gpu_config.available:
                memory_mb = torch.cuda.memory_allocated(self.device) / 1024**2
                Logger.info(f"显存占用: {memory_mb:.2f} MB")
            
        except ImportError as e:
            Logger.error(
                f"Sharp 模块导入失败: {e}",
                "可能的原因:\n"
                "1. Sharp 库未安装\n"
                "2. 模型文件路径错误\n"
                "3. Python 环境配置不正确\n\n"
                "解决方案:\n"
                "- 检查 model_assets/ 文件夹是否存在\n"
                "- 重新安装依赖: pip install -r requirements.txt\n"
                "- 确保使用正确的 Python 环境"
            )
            sys.exit(1)
        except Exception as e:
            Logger.error(
                f"模型加载失败: {e}",
                "请检查:\n"
                "1. 模型文件是否完整\n"
                "2. PyTorch 版本是否兼容\n"
                "3. 是否有足够的内存/显存\n"
                "4. Python 环境是否正确配置"
            )
            sys.exit(1)
    
    @torch.no_grad()
    def predict(self, image: np.ndarray, f_px: float) -> Any:
        """从图像预测3D高斯"""
        import torch.nn.functional as F
        
        internal_shape = (1536, 1536)
        height, width = image.shape[:2]
        
        # 预处理
        if self.gpu_config.use_amp and self.gpu_config.available and self.gpu_config.vendor == "NVIDIA":
            try:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    image_pt = torch.from_numpy(image.copy()).half().to(self.device, non_blocking=True).permute(2, 0, 1) / 255.0
                    disparity_factor = torch.tensor([f_px / width], dtype=torch.float32, device=self.device)
                    
                    image_resized_pt = F.interpolate(
                        image_pt[None],
                        size=(internal_shape[1], internal_shape[0]),
                        mode="bilinear",
                        align_corners=False,
                        antialias=False,
                    )
                    
                    gaussians_ndc = self.predictor(image_resized_pt, disparity_factor)
            except Exception as e:
                Logger.warning(f"混合精度推理失败,回退到 FP32: {e}")
                image_pt = torch.from_numpy(image.copy()).float().to(self.device, non_blocking=True).permute(2, 0, 1) / 255.0
                disparity_factor = torch.tensor([f_px / width], dtype=torch.float32, device=self.device)
                
                image_resized_pt = F.interpolate(
                    image_pt[None],
                    size=(internal_shape[1], internal_shape[0]),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
                
                gaussians_ndc = self.predictor(image_resized_pt, disparity_factor)
        else:
            image = np.ascontiguousarray(image.copy())
            image_pt = torch.from_numpy(image).float().to(self.device, non_blocking=True).permute(2, 0, 1) / 255.0
            disparity_factor = torch.tensor([f_px / width], dtype=torch.float32, device=self.device)
            
            image_resized_pt = F.interpolate(
                image_pt[None],
                size=(internal_shape[1], internal_shape[0]),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            
            gaussians_ndc = self.predictor(image_resized_pt, disparity_factor)
        
        # 后处理
        intrinsics = torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height
        
        from sharp.utils.gaussians import unproject_gaussians
        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4, device=self.device), intrinsics_resized, internal_shape
        )
        
        return gaussians


# ================= 应用主类 =================
class MLSharpApp:
    """MLSharp 应用主类"""
    
    def __init__(self):
        self.args, self.config_dict = parse_command_args()
        self.app_config = AppConfig.from_current_dir()
        self.gpu_config = GPUConfig()
        
        # 清理临时目录
        if os.path.exists(self.app_config.temp_dir):
            try:
                shutil.rmtree(self.app_config.temp_dir)
            except:
                pass
        os.makedirs(self.app_config.temp_dir, exist_ok=True)
        
        # 初始化 GPU
        import torch
        self.gpu_manager = GPUManager(self.gpu_config, self.args)
        self.device = self.gpu_manager.initialize()
        
        # 加载模型
        self.model_manager = ModelManager(self.app_config, self.gpu_config, self.device)
        self.model_manager.load_model()
        
        # 创建 FastAPI 应用
        self.app = self._create_app()
        # 使用 ProcessPoolExecutor 替代 ThreadPoolExecutor 以避免 GIL 限制
        from concurrent.futures import ProcessPoolExecutor
        self.executor = ProcessPoolExecutor(max_workers=min(4, os.cpu_count()))
    
    def _create_app(self):
        """创建 FastAPI 应用"""
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI(
            title="MLSharp 3D Maker API",
            description="基于 Apple SHaRP 模型的 3D 高斯泼溅生成工具",
            version="7.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        app.mount("/files", StaticFiles(directory=self.app_config.temp_dir), name="files")
        
        @app.get("/", tags=["UI"])
        async def read_index():
            """访问 Web 界面"""
            return FileResponse(os.path.join(self.app_config.base_dir, "viewer.html"))
        
        @app.post("/api/predict", tags=["Prediction"])
        async def predict(file: UploadFile = File(..., description="上传的图片文件 (JPG格式)")):
            """从单张图片生成 3D 模型
            
            上传一张 JPG 图片，系统将使用 SHaRP 模型生成 3D 高斯泼溅模型。
            
            - **file**: JPG 格式的图片文件（推荐尺寸: 512x512 - 1024x1024）
            
            返回:
                - status: 请求状态
                - url: 生成的 PLY 文件下载地址
                - processing_time: 处理时间（秒）
            """
            return await self._handle_predict(file)
        
        @app.get("/api/health", tags=["System"])
        async def health_check():
            """健康检查端点
            
            检查服务是否正常运行以及 GPU 状态。
            
            返回:
                - status: 服务状态 (healthy/unhealthy)
                - gpu_available: GPU 是否可用
                - gpu_vendor: GPU 厂商 (NVIDIA/AMD/Intel)
                - gpu_name: GPU 型号名称
            """
            return {
                "status": "healthy",
                "gpu_available": self.gpu_config.available,
                "gpu_vendor": self.gpu_config.vendor,
                "gpu_name": self.gpu_config.name
            }
        
        @app.get("/api/stats", tags=["System"])
        async def get_stats():
            """获取系统统计信息
            
            返回当前系统的 GPU 使用情况和性能指标。
            
            返回:
                - gpu.available: GPU 是否可用
                - gpu.vendor: GPU 厂商
                - gpu.name: GPU 型号
                - gpu.count: GPU 数量
                - gpu.memory_mb: 当前 GPU 内存使用量（MB）
            """
            stats = {
                "gpu": {
                    "available": self.gpu_config.available,
                    "vendor": self.gpu_config.vendor,
                    "name": self.gpu_config.name,
                    "count": self.gpu_config.count,
                    "memory_mb": 0
                }
            }
            if self.gpu_config.available:
                import torch
                try:
                    stats["gpu"]["memory_mb"] = torch.cuda.memory_allocated(self.device) / 1024**2
                except:
                    pass
            return stats
        
        return app
    
    async def _handle_predict(self, file: UploadFile):
        """处理预测请求 - 异步优化版本"""
        from sharp.utils import io
        from sharp.utils.gaussians import save_ply
        
        task_id = str(uuid.uuid4())[:8]
        try:
            start_time = time.time()
            task_dir = os.path.join(self.app_config.temp_dir, task_id)
            output_dir = os.path.join(task_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存上传的文件 - 使用 asyncio.to_thread
            file_path = os.path.join(task_dir, "input.jpg")
            await asyncio.to_thread(self._save_file, file, file_path)
            
            Logger.info(f"[Task {task_id}] 已保存文件: {file_path}")
            
            # 加载图像 - 使用 asyncio.to_thread
            load_start = time.time()
            image, _, f_px = await asyncio.to_thread(io.load_rgb, Path(file_path))
            height, width = image.shape[:2]
            Logger.info(f"[Task {task_id}] 图像信息: {width}x{height}, 焦距: {f_px} (加载耗时: {time.time()-load_start:.2f}s)")
            
            # 检查图片尺寸
            if width > 4096 or height > 4096:
                Logger.warning(f"[Task {task_id}] 图片尺寸过大 ({width}x{height}),可能导致性能问题")
            
            # 预测 - GPU 推理在单独线程中执行
            Logger.info(f"[Task {task_id}] 开始推理...")
            inference_start = time.time()
            gaussians = await asyncio.to_thread(self.model_manager.predict, image, f_px)
            if self.gpu_config.available:
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            Logger.info(f"[Task {task_id}] 推理完成,耗时: {inference_time:.2f}秒")
            
            # 保存 PLY - 使用 asyncio.to_thread
            output_ply_path = os.path.join(output_dir, "output.ply")
            save_start = time.time()
            await asyncio.to_thread(save_ply, gaussians, f_px, (height, width), output_ply_path)
            Logger.info(f"[Task {task_id}] PLY保存完成,耗时: {time.time()-save_start:.2f}s")
            
            # 重命名 - 异步文件操作
            final_ply = os.path.join(task_dir, "output.ply")
            await asyncio.to_thread(os.rename, output_ply_path, final_ply)
            
            elapsed_time = time.time() - start_time
            Logger.info(f"[Task {task_id}] 处理完成,总耗时: {elapsed_time:.2f}秒")
            
            download_url = f"/files/{task_id}/output.ply"
            return {"status": "success", "url": download_url, "processing_time": elapsed_time}
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                Logger.error(f"[Task {task_id}] 显存不足: {e}")
                return JSONResponse({
                    "status": "error",
                    "message": "显存不足,请使用较小的图片",
                    "solution": "建议使用小于 1024x1024 的图片,或关闭其他占用显存的程序"
                }, status_code=507)
            raise
        except Exception as e:
            Logger.error(f"[Task {task_id}] 处理失败: {e}")
            return JSONResponse({
                "status": "error",
                "message": f"处理失败: {str(e)}",
                "solution": "请尝试重新启动程序或使用较小的图片"
            }, status_code=500)
    
    def _save_file(self, upload_file, file_path):
        """保存上传的文件"""
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    
    def print_startup_banner(self):
        """打印启动横幅"""
        print("\n" + "=" * 60)
        print(" " * 20 + "MLSharp")
        print(" " * 12 + "3D 模型生成工具 - 自动检测 GPU")
        print("=" * 60)
        print()
        print("支持模式:")
        print("  ✓ NVIDIA GPU (CUDA)")
        print("  ✓ AMD GPU (ROCm)")
        print("  ✓ Intel GPU (CPU 回退)")
        print("  ✓ CPU 模式")
        print()
        print("=" * 60)
        print()
    
    def print_system_info(self):
        """打印系统信息"""
        Logger.section("系统信息")
        Logger.info(f"操作系统: {platform.system()} {platform.release()}")
        Logger.info(f"Python 版本: {sys.version.split()[0]}")
        Logger.info(f"工作目录: {self.app_config.base_dir}")
    
    def print_service_info(self):
        """打印服务信息"""
        Logger.section("Web 服务")
        
        if self.gpu_config.available:
            Logger.success("GPU 加速已启用")
            Logger.info(f"GPU 厂商: {self.gpu_config.vendor}")
            Logger.info(f"GPU 型号: {self.gpu_config.name}")
            if self.gpu_config.vendor == "NVIDIA":
                Logger.info(f"计算能力: {self.gpu_config.compute_capability}")
                Logger.info(f"[优化] 混合精度推理: {'已启用' if self.gpu_config.use_amp else '已禁用'}")
                Logger.info(f"[优化] cuDNN Benchmark: {'已启用' if self.gpu_config.use_cudnn_benchmark else '已禁用'}")
                Logger.info(f"[优化] TensorFloat32: {'已启用' if self.gpu_config.use_tf32 else '已禁用'}")
            elif self.gpu_config.vendor == "AMD":
                Logger.info("使用 ROCm 加速")
        else:
            Logger.warning("使用 CPU 模式")
            Logger.info(f"CPU 核心数: {os.cpu_count()}")
            Logger.info("[优化] 多线程优化: 已启用")
        
        print()
        service_url = f"http://{self.args.host}:{self.args.port}"
        Logger.success(f"服务地址: {service_url}")
        if not self.args.no_browser:
            Logger.info("浏览器将自动打开...")
        Logger.info("按 Ctrl+C 停止服务")
        print()
    
    def open_browser(self):
        """打开浏览器"""
        service_url = f"http://{self.args.host}:{self.args.port}"
        time.sleep(2)
        try:
            webbrowser.open(service_url)
        except Exception as e:
            Logger.warning(f"无法自动打开浏览器: {e}")
            Logger.info(f"请手动访问: {service_url}")
    
    def run(self):
        """运行应用"""
        import uvicorn
        
        # 打印启动信息
        self.print_startup_banner()
        self.print_system_info()
        self.print_service_info()
        
        # 启动浏览器
        if not self.args.no_browser:
            threading.Thread(target=self.open_browser, daemon=True).start()
        
        # 启动服务
        try:
            uvicorn.run(
                self.app,
                host=self.args.host,
                port=self.args.port,
                log_level="warning",
                limit_concurrency=10,
                limit_max_requests=1000,
                timeout_keep_alive=30,
                workers=1,
            )
        except KeyboardInterrupt:
            print("\n")
            Logger.section("服务已停止")
            Logger.info("感谢使用 MLSharp!")
        except Exception as e:
            Logger.error(f"服务启动失败: {e}")
            sys.exit(1)


# ================= 主程序入口 =================
if __name__ == "__main__":
    # 初始化日志系统
    log_system = Logger()
    
    app = MLSharpApp()
    app.run()
