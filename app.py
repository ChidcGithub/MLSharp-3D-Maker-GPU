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
from metrics import init_metrics, get_metrics_manager

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
    input_size: Tuple[int, int] = (1536, 1536)
    gradient_checkpointing: bool = False
    checkpoint_segments: int = 3
    enable_cache: bool = True
    cache_size: int = 100
    clear_cache: bool = False
    enable_auto_tune: bool = False


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


def validate_input_size(width: int, height: int) -> Tuple[int, int]:
    """
    验证并调整输入尺寸以符合模型要求
    
    SHaRP 模型的编码器使用基于补丁的分割，要求：
    - 尺寸必须能被 64 整除（补丁大小）
    - 宽度和高度必须相等
    - 最大尺寸限制为 1536（SPN 编码器在更大尺寸下会出现补丁分割问题）
    
    Args:
        width: 输入宽度
        height: 输入高度
        
    Returns:
        调整后的 (width, height)
    """
    # 检查宽高是否相等
    if width != height:
        Logger.warning(f"输入尺寸宽度和高度不相等 ({width}x{height})，模型使用正方形输入")
        size = max(width, height)
        width = height = size
        Logger.info(f"已调整为 {width}x{height}")
    
    # 限制最大尺寸为 1536（SPN 编码器在更大尺寸下会出现补丁分割问题）
    max_size = 1536
    if width > max_size or height > max_size:
        Logger.warning(f"输入尺寸 {width}x{height} 超过最大支持尺寸 {max_size}x{max_size}")
        Logger.warning(f"SPN 编码器在更大尺寸下会出现补丁分割错误")
        Logger.info(f"已调整为 {max_size}x{max_size}")
        width = height = max_size
    
    # 检查是否能被 64 整除
    if width % 64 != 0 or height % 64 != 0:
        Logger.warning(f"输入尺寸 {width}x{height} 不能被 64 整除")
        # 向上取整到最近的 64 倍数
        width = ((width + 63) // 64) * 64
        height = ((height + 63) // 64) * 64
        Logger.info(f"已调整为 {width}x{height}")
    
    # 再次检查调整后的尺寸是否超过最大值
    if width > max_size or height > max_size:
        Logger.warning(f"调整后的尺寸 {width}x{height} 仍然超过最大支持尺寸")
        Logger.info(f"已调整为 {max_size}x{max_size}")
        width = height = max_size
    
    return width, height


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
    
    # 推理配置
    if args.input_size != (1536, 1536):
        config.setdefault('inference', {})['input_size'] = list(args.input_size)
    
    # 优化配置
    if args.gradient_checkpointing:
        config.setdefault('optimization', {})['gradient_checkpointing'] = True
    if args.checkpoint_segments != 3:
        config.setdefault('optimization', {})['checkpoint_segments'] = args.checkpoint_segments
    
    # 缓存配置
    if args.enable_cache:
        config.setdefault('cache', {})['enabled'] = True
    if args.no_cache:
        config.setdefault('cache', {})['enabled'] = False
    if args.cache_size != 100:
        config.setdefault('cache', {})['size'] = args.cache_size
    
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
    inference = config.get('inference', {})
    optimization = config.get('optimization', {})
    cache = config.get('cache', {})
    
    input_size = tuple(inference.get('input_size', [1536, 1536]))
    
    return CLIArgs(
        mode=config.get('mode', 'auto'),
        port=server.get('port', 8000),
        host=server.get('host', '127.0.0.1'),
        no_browser=not browser.get('auto_open', True),
        no_amp=not gpu.get('enable_amp', True),
        no_cudnn_benchmark=not gpu.get('enable_cudnn_benchmark', True),
        config_file=None,
        input_size=input_size,
        gradient_checkpointing=optimization.get('gradient_checkpointing', False),
        checkpoint_segments=optimization.get('checkpoint_segments', 3),
        enable_cache=cache.get('enabled', True),
        cache_size=cache.get('size', 100)
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
    parser.add_argument('--input-size', type=int, nargs=2, default=[1536, 1536],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='输入图像尺寸（默认：1536 1536）')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='启用梯度检查点（减少显存占用，但会略微降低推理速度）')
    parser.add_argument('--checkpoint-segments', type=int, default=3,
                        help='梯度检查点分段数 (默认: 3)')
    parser.add_argument('--enable-cache', action='store_true', default=True,
                        help='启用推理缓存（默认：启用）')
    parser.add_argument('--no-cache', action='store_true',
                        help='禁用推理缓存')
    parser.add_argument('--cache-size', type=int, default=100,
                        help='缓存最大条目数（默认：100）')
    parser.add_argument('--clear-cache', action='store_true',
                        help='启动时清空缓存')
    parser.add_argument('--enable-auto-tune', action='store_true',
                        help='启用性能自动调优（启动时自动测试并选择最优配置）')
    
    args = parser.parse_args()
    
    # 处理缓存参数
    enable_cache = args.enable_cache and not args.no_cache
    
    # 转换 input_size 为元组
    input_size = tuple(args.input_size)
    
    # 验证输入尺寸
    validated_width, validated_height = validate_input_size(*input_size)
    if validated_width != input_size[0] or validated_height != input_size[1]:
        Logger.info(f"输入尺寸已从 {input_size[0]}x{input_size[1]} 调整为 {validated_width}x{validated_height}")
    input_size = (validated_width, validated_height)
    
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
                config_file=None,
                input_size=input_size,
                gradient_checkpointing=args.gradient_checkpointing,
                checkpoint_segments=args.checkpoint_segments,
                enable_cache=enable_cache,
                cache_size=args.cache_size,
                clear_cache=args.clear_cache,
                enable_auto_tune=args.enable_auto_tune
            )
    else:
        cli_args = CLIArgs(
            mode=args.mode,
            port=args.port,
            host=args.host,
            no_browser=args.no_browser,
            no_amp=args.no_amp,
            no_cudnn_benchmark=args.no_cudnn_benchmark,
            config_file=None,
            input_size=input_size,
            gradient_checkpointing=args.gradient_checkpointing,
            checkpoint_segments=args.checkpoint_segments,
            enable_cache=enable_cache,
            cache_size=args.cache_size,
            clear_cache=args.clear_cache,
            enable_auto_tune=args.enable_auto_tune
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
                
                # 运行自动调优（如果启用）
                self.run_auto_tune()
                
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
    
    def run_auto_tune(self):
        """
        运行性能自动调优
        
        自动测试不同的优化配置组合，选择最优配置
        """
        if not self.args.enable_auto_tune:
            return
        
        try:
            tuner = PerformanceAutoTuner(self.config, self.device)
            best_config = tuner.benchmark_optimizations()
            
            if best_config:
                Logger.success("性能自动调优完成！")
                Logger.info("已应用最优配置")
            else:
                Logger.warning("性能自动调优失败，使用默认配置")
        except Exception as e:
            Logger.warning(f"性能自动调优失败: {e}")
            Logger.info("使用默认配置")
    
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


# ================= 缓存管理器 =================
class CacheManager:
    """推理缓存管理器"""
    
    def __init__(self, enabled: bool = True, max_size: int = 100):
        """
        初始化缓存管理器
        
        Args:
            enabled: 是否启用缓存
            max_size: 最大缓存条目数
        """
        self.enabled = enabled
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.cache_order: list = []  # 用于 LRU 淘汰
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
    
    def _get_cache_key(self, image: np.ndarray, f_px: float) -> str:
        """
        计算缓存键
        
        Args:
            image: 输入图像
            f_px: 焦距
            
        Returns:
            缓存键（基于图像哈希和焦距）
        """
        import hashlib
        
        # 计算图像哈希（使用 MD5）
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        
        # 组合哈希和焦距
        cache_key = f"{image_hash}_{f_px:.6f}"
        
        return cache_key
    
    def get(self, image: np.ndarray, f_px: float) -> Optional[Any]:
        """
        从缓存获取结果
        
        Args:
            image: 输入图像
            f_px: 焦距
            
        Returns:
            缓存的高斯结果，如果未命中则返回 None
        """
        if not self.enabled:
            return None
        
        with self.lock:
            cache_key = self._get_cache_key(image, f_px)
            
            if cache_key in self.cache:
                # 缓存命中
                self.hits += 1
                result = self.cache[cache_key]
                
                # 更新 LRU 顺序
                self.cache_order.remove(cache_key)
                self.cache_order.append(cache_key)
                
                hit_rate = self.hits / (self.hits + self.misses) * 100
                Logger.debug(f"缓存命中: 命中率 {hit_rate:.1f}% ({self.hits}/{self.hits + self.misses})")
                
                return result
            else:
                # 缓存未命中
                self.misses += 1
                return None
    
    def set(self, image: np.ndarray, f_px: float, result: Any):
        """
        将结果存入缓存
        
        Args:
            image: 输入图像
            f_px: 焦距
            result: 预测结果
        """
        if not self.enabled:
            return
        
        with self.lock:
            cache_key = self._get_cache_key(image, f_px)
            
            # 如果缓存已满，淘汰最旧的条目
            if len(self.cache) >= self.max_size:
                oldest_key = self.cache_order.pop(0)
                del self.cache[oldest_key]
                Logger.debug(f"缓存已满，淘汰最旧条目: {oldest_key}")
            
            # 存入缓存
            self.cache[cache_key] = result
            self.cache_order.append(cache_key)
            Logger.debug(f"缓存已添加: {cache_key}")
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.cache_order.clear()
            self.hits = 0
            self.misses = 0
            Logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "enabled": self.enabled,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }
    
    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        if stats["enabled"]:
            Logger.section("缓存统计")
            Logger.info(f"缓存状态: {'已启用' if stats['enabled'] else '已禁用'}")
            Logger.info(f"缓存大小: {stats['size']}/{stats['max_size']}")
            Logger.info(f"命中次数: {stats['hits']}")
            Logger.info(f"未命中次数: {stats['misses']}")
            Logger.info(f"命中率: {stats['hit_rate']:.1f}%")


# ================= 性能自动调优器 =================
class PerformanceAutoTuner:
    """性能自动调优器"""
    
    def __init__(self, gpu_config: GPUConfig, device: torch.device):
        """
        初始化性能自动调优器
        
        Args:
            gpu_config: GPU 配置
            device: 设备
        """
        self.gpu_config = gpu_config
        self.device = device
        self.optimization_results = {}
    
    def benchmark_optimizations(self) -> Dict[str, Any]:
        """
        基准测试各种优化配置，选择最优配置
        
        Returns:
            最优配置字典
        """
        Logger.section("性能自动调优")
        Logger.info("正在测试不同优化配置...")
        
        # 测试配置列表
        test_configs = [
            {
                'name': '基准配置',
                'amp': False,
                'cudnn_benchmark': False,
                'tf32': False,
                'description': '无任何优化'
            },
            {
                'name': '仅 AMP',
                'amp': True,
                'cudnn_benchmark': False,
                'tf32': False,
                'description': '仅启用混合精度推理'
            },
            {
                'name': '仅 cuDNN Benchmark',
                'amp': False,
                'cudnn_benchmark': True,
                'tf32': False,
                'description': '仅启用 cuDNN 自动调优'
            },
            {
                'name': '仅 TF32',
                'amp': False,
                'cudnn_benchmark': False,
                'tf32': True,
                'description': '仅启用 TensorFloat32'
            },
            {
                'name': 'AMP + cuDNN Benchmark',
                'amp': True,
                'cudnn_benchmark': True,
                'tf32': False,
                'description': 'AMP 和 cuDNN 自动调优'
            },
            {
                'name': 'AMP + TF32',
                'amp': True,
                'cudnn_benchmark': False,
                'tf32': True,
                'description': 'AMP 和 TensorFloat32'
            },
            {
                'name': '全部优化',
                'amp': True,
                'cudnn_benchmark': True,
                'tf32': True,
                'description': '启用所有优化'
            }
        ]
        
        # 根据显卡能力过滤不适用的配置
        if self.gpu_config.vendor != "NVIDIA":
            test_configs = [cfg for cfg in test_configs if not (cfg['cudnn_benchmark'] or cfg['tf32'])]
            Logger.info("非 NVIDIA GPU，仅测试 AMP 优化")
        elif self.gpu_config.compute_capability < 60:
            test_configs = [cfg for cfg in test_configs if not cfg['cudnn_benchmark']]
            Logger.info("显卡计算能力 < 6.0，跳过 cuDNN Benchmark")
        elif self.gpu_config.compute_capability < 80:
            test_configs = [cfg for cfg in test_configs if not cfg['tf32']]
            Logger.info("显卡不支持 TF32，跳过 TF32 测试")
        
        # 执行基准测试
        results = []
        for config in test_configs:
            try:
                Logger.info(f"\n测试配置: {config['name']}")
                Logger.info(f"  描述: {config['description']}")
                
                # 应用配置
                self._apply_config(config)
                
                # 运行基准测试
                avg_time = self._run_benchmark()
                
                Logger.info(f"  平均推理时间: {avg_time:.3f} 秒")
                
                results.append({
                    'config': config,
                    'avg_time': avg_time,
                    'throughput': 1.0 / avg_time if avg_time > 0 else 0
                })
                
            except Exception as e:
                Logger.warning(f"  测试失败: {e}")
                continue
        
        # 选择最优配置
        if results:
            best_result = min(results, key=lambda x: x['avg_time'])
            Logger.section("调优结果")
            Logger.success(f"最优配置: {best_result['config']['name']}")
            Logger.info(f"  描述: {best_result['config']['description']}")
            Logger.info(f"  平均推理时间: {best_result['avg_time']:.3f} 秒")
            Logger.info(f"  吞吐量: {best_result['throughput']:.2f} FPS")
            
            # 应用最优配置
            self._apply_config(best_result['config'])
            
            self.optimization_results = {
                'best_config': best_result['config'],
                'all_results': results
            }
            
            return best_result['config']
        else:
            Logger.warning("所有配置测试失败，使用默认配置")
            return test_configs[0] if test_configs else {}
    
    def _apply_config(self, config: Dict[str, Any]):
        """
        应用优化配置
        
        Args:
            config: 配置字典
        """
        # 混合精度
        if config.get('amp', False):
            if self.gpu_config.compute_capability >= 53:
                self.gpu_config.use_amp = True
            else:
                self.gpu_config.use_amp = False
        
        # cuDNN Benchmark
        if config.get('cudnn_benchmark', False):
            if self.gpu_config.vendor == "NVIDIA" and self.gpu_config.compute_capability >= 60:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                self.gpu_config.use_cudnn_benchmark = True
            else:
                torch.backends.cudnn.benchmark = False
                self.gpu_config.use_cudnn_benchmark = False
        else:
            torch.backends.cudnn.benchmark = False
            self.gpu_config.use_cudnn_benchmark = False
        
        # TensorFloat32
        if config.get('tf32', False):
            if self.gpu_config.vendor == "NVIDIA" and self.gpu_config.supports_tf32:
                torch.set_float32_matmul_precision('high')
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.gpu_config.use_tf32 = True
            else:
                self.gpu_config.use_tf32 = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.gpu_config.use_tf32 = False
    
    def _run_benchmark(self, warmup_runs: int = 2, test_runs: int = 3) -> float:
        """
        运行基准测试
        
        Args:
            warmup_runs: 预热运行次数
            test_runs: 测试运行次数
            
        Returns:
            平均推理时间（秒）
        """
        import time
        
        # 创建测试输入
        test_size = (512, 512)  # 使用较小尺寸进行快速测试
        dummy_input = torch.randn(3, *test_size, device=self.device)
        dummy_disparity = torch.tensor([1.0], device=self.device)
        
        # 预热
        for _ in range(warmup_runs):
            try:
                with torch.no_grad():
                    _ = self._dummy_forward(dummy_input, dummy_disparity)
            except:
                pass
        
        if self.gpu_config.available:
            torch.cuda.synchronize()
        
        # 测试
        times = []
        for i in range(test_runs):
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    _ = self._dummy_forward(dummy_input, dummy_disparity)
                
                if self.gpu_config.available:
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                Logger.info(f"  运行 {i+1}/{test_runs}: {elapsed:.3f} 秒")
                
            except Exception as e:
                Logger.warning(f"  运行 {i+1}/{test_runs} 失败: {e}")
                continue
        
        # 计算平均时间
        if times:
            avg_time = sum(times) / len(times)
            return avg_time
        else:
            return float('inf')
    
    def _dummy_forward(self, x: torch.Tensor, disparity: torch.Tensor):
        """
        模拟前向传播（用于基准测试）
        
        Args:
            x: 输入张量
            disparity: 视差张量
            
        Returns:
            模拟输出
        """
        # 简单的卷积操作模拟推理
        import torch.nn.functional as F
        conv1 = torch.nn.Conv2d(3, 64, 3, padding=1).to(self.device)
        conv2 = torch.nn.Conv2d(64, 128, 3, padding=1).to(self.device)
        
        out = F.relu(conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(conv2(out))
        
        return out


# ================= 模型管理器 =================
class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: AppConfig, gpu_config: GPUConfig, device: torch.device, input_size: Tuple[int, int] = (1536, 1536), gradient_checkpointing: bool = False, enable_cache: bool = True, cache_size: int = 100):
        self.config = config
        self.gpu_config = gpu_config
        self.device = device
        self.predictor = None
        self.input_size = input_size
        self.gradient_checkpointing = gradient_checkpointing
        self.cache_manager = CacheManager(enabled=enable_cache, max_size=cache_size)
    
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
            
            # 应用梯度检查点
            if self.gradient_checkpointing and self.gpu_config.available:
                Logger.info("正在应用梯度检查点...")
                self._apply_gradient_checkpointing()
                Logger.success("梯度检查点已启用（显存占用将减少，但推理速度可能略微降低）")
            
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
    
    def _apply_gradient_checkpointing(self):
        """
        应用梯度检查点到模型
        
        梯度检查点通过重新计算中间激活值来减少显存占用，
        但会略微增加计算时间。
        """
        try:
            from torch.utils.checkpoint import checkpoint
            
            # 获取预测器的主要模块
            if hasattr(self.predictor, 'monodepth_model'):
                # 包装 monodepth 模型
                original_forward = self.predictor.monodepth_model.forward
                
                def checkpointed_forward(x):
                    return checkpoint(original_forward, x, use_reentrant=False)
                
                self.predictor.monodepth_model.forward = checkpointed_forward
                Logger.info("  已应用梯度检查点到 monodepth 模型")
            
            if hasattr(self.predictor, 'decoder'):
                # 包装 decoder
                original_forward = self.predictor.decoder.forward
                
                def checkpointed_forward(x):
                    return checkpoint(original_forward, x, use_reentrant=False)
                
                self.predictor.decoder.forward = checkpointed_forward
                Logger.info("  已应用梯度检查点到 decoder")
            
        except Exception as e:
            Logger.warning(f"应用梯度检查点失败: {e}")
            Logger.info("  梯度检查点未启用，将使用正常推理模式")
    
    @torch.no_grad()
    def predict(self, image: np.ndarray, f_px: float) -> Any:
        """从图像预测3D高斯（带缓存支持）"""
        import torch.nn.functional as F
        
        # 检查缓存
        cached_result = self.cache_manager.get(image, f_px)
        if cached_result is not None:
            return cached_result
        
        internal_shape = self.input_size
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
        
        # 存入缓存
        self.cache_manager.set(image, f_px, gaussians)
        
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
        self.model_manager = ModelManager(
            self.app_config, 
            self.gpu_config, 
            self.device, 
            self.args.input_size,
            self.args.gradient_checkpointing,
            self.args.enable_cache,
            self.args.cache_size
        )
        self.model_manager.load_model()
        
        # 清空缓存（如果指定）
        if self.args.clear_cache:
            Logger.info("正在清空缓存...")
            self.model_manager.cache_manager.clear()
            Logger.success("缓存已清空")
        
        # 初始化监控指标
        self.metrics_manager = init_metrics(enable_gpu=self.gpu_config.available)
        if self.gpu_config.available:
            self.metrics_manager.set_gpu_info(0, self.gpu_config.name, self.gpu_config.vendor)
        self.metrics_manager.set_input_size(*self.args.input_size)
        
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
        
        @app.get("/api/cache", tags=["System"])
        async def get_cache_stats():
            """获取缓存统计信息
            
            返回当前缓存的使用情况和性能指标。
            
            返回:
                - enabled: 缓存是否启用
                - size: 当前缓存条目数
                - max_size: 最大缓存条目数
                - hits: 缓存命中次数
                - misses: 缓存未命中次数
                - hit_rate: 缓存命中率（百分比）
            """
            return self.model_manager.cache_manager.get_stats()
        
        @app.post("/api/cache/clear", tags=["System"])
        async def clear_cache():
            """清空缓存
            
            清空所有缓存条目并重置统计信息。
            
            返回:
                - status: 操作状态
                - message: 操作消息
            """
            self.model_manager.cache_manager.clear()
            return {"status": "success", "message": "缓存已清空"}
        
        @app.get("/metrics", tags=["Monitoring"])
        async def metrics():
            """Prometheus 指标端点
            
            返回 Prometheus 格式的监控指标数据。
            
            包括：
                - HTTP 请求计数和响应时间
                - 预测请求计数和响应时间
                - GPU 内存使用量和利用率
                - 活跃任务数
                - 应用信息
            """
            from fastapi.responses import Response
            return Response(
                content=self.metrics_manager.get_metrics(),
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
        
        # 添加监控中间件
        @app.middleware("http")
        async def monitoring_middleware(request, call_next):
            """监控中间件 - 记录所有 HTTP 请求"""
            import time
            start_time = time.time()
            
            # 增加活跃任务计数
            if request.url.path == "/api/predict":
                self.metrics_manager.set_active_tasks(
                    self.metrics_manager.active_tasks._value.get() + 1 if self.metrics_manager.active_tasks._value else 1
                )
            
            try:
                response = await call_next(request)
                
                # 记录请求指标
                duration = time.time() - start_time
                self.metrics_manager.record_http_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code,
                    duration=duration
                )
                
                return response
            finally:
                # 减少活跃任务计数
                if request.url.path == "/api/predict":
                    current_tasks = self.metrics_manager.active_tasks._value.get() if self.metrics_manager.active_tasks._value else 1
                    self.metrics_manager.set_active_tasks(max(0, current_tasks - 1))
        
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
            load_time = time.time() - load_start
            Logger.info(f"[Task {task_id}] 图像信息: {width}x{height}, 焦距: {f_px} (加载耗时: {load_time:.2f}s)")
            self.metrics_manager.record_predict_stage("image_load", load_time)
            
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
            self.metrics_manager.record_predict_stage("inference", inference_time)
            
            # 保存 PLY - 使用 asyncio.to_thread
            output_ply_path = os.path.join(output_dir, "output.ply")
            save_start = time.time()
            await asyncio.to_thread(save_ply, gaussians, f_px, (height, width), output_ply_path)
            save_time = time.time() - save_start
            Logger.info(f"[Task {task_id}] PLY保存完成,耗时: {save_time:.2f}s")
            self.metrics_manager.record_predict_stage("ply_save", save_time)
            
            # 重命名 - 异步文件操作
            final_ply = os.path.join(task_dir, "output.ply")
            await asyncio.to_thread(os.rename, output_ply_path, final_ply)
            
            elapsed_time = time.time() - start_time
            Logger.info(f"[Task {task_id}] 处理完成,总耗时: {elapsed_time:.2f}秒")
            
            # 记录预测指标
            self.metrics_manager.record_predict_request("success", elapsed_time)
            self.metrics_manager.record_predict_stage("total", elapsed_time)
            
            download_url = f"/files/{task_id}/output.ply"
            return {"status": "success", "url": download_url, "processing_time": elapsed_time}
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                Logger.error(f"[Task {task_id}] 显存不足: {e}")
                elapsed_time = time.time() - start_time
                self.metrics_manager.record_predict_request("error", elapsed_time)
                return JSONResponse({
                    "status": "error",
                    "message": "显存不足,请使用较小的图片",
                    "solution": "建议使用小于 1024x1024 的图片,或关闭其他占用显存的程序"
                }, status_code=507)
            raise
        except Exception as e:
            Logger.error(f"[Task {task_id}] 处理失败: {e}")
            elapsed_time = time.time() - start_time
            self.metrics_manager.record_predict_request("error", elapsed_time)
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
        
        Logger.info(f"输入尺寸: {self.args.input_size[0]}x{self.args.input_size[1]}")
        
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
        
        # 缓存信息
        if self.args.enable_cache:
            Logger.success(f"[缓存] 推理缓存: 已启用（最大 {self.args.cache_size} 条）")
        else:
            Logger.info(f"[缓存] 推理缓存: 已禁用")
        
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
