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
from pydantic import BaseModel, Field

import numpy as np
import torch
import json
import yaml
from loguru import logger
from metrics import init_metrics, get_metrics_manager
from i18n import i18n

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
    # 内存回收配置
    enable_auto_gc: bool = True           # 启用自动垃圾回收
    auto_gc_interval: int = 30            # 自动检查间隔（秒）
    auto_gc_threshold: float = 85.0       # 触发清理的阈值（百分比）
    enable_smart_reclaim: bool = True     # 启用智能内存回收


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
    no_cache: bool = False  # 禁用缓存
    cache_size: int = 100
    clear_cache: bool = False
    enable_auto_tune: bool = False
    redis_url: Optional[str] = None
    enable_webhook: bool = False
    app_config: Optional[AppConfig] = None  # 应用配置（用于性能自动调优）
    # GPU 内存回收参数
    enable_auto_gc: bool = True           # 启用自动垃圾回收
    auto_gc_interval: int = 30            # 自动检查间隔（秒）
    auto_gc_threshold: float = 85.0       # 触发清理的阈值（百分比）
    enable_smart_reclaim: bool = True     # 启用智能内存回收
    # 语言设置
    language: str = 'zh'                  # 语言设置 ('zh' 或 'en')


# ================= 配置文件加载 =================
def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    从配置文件加载配置

    支持 YAML 和 JSON 格式

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件路径不安全
    """
    # 路径安全验证
    config_path_real = os.path.realpath(config_path)

    # 检查文件扩展名是否合法
    file_ext = os.path.splitext(config_path_real)[1].lower()
    if file_ext not in ['.yaml', '.yml', '.json']:
        raise ValueError(i18n.t('config_format_unsupported').format(file_ext))

    # 检查文件是否存在
    if not os.path.exists(config_path_real):
        raise FileNotFoundError(i18n.t('config_not_found').format(config_path))

    # 安全检查：确保配置文件路径在合理范围内
    # 允许当前工作目录、用户主目录、应用目录等
    cwd_real = os.path.realpath(os.getcwd())
    allowed_paths = [cwd_real]

    # 添加用户主目录（如果有）
    home_dir = os.path.expanduser('~')
    if home_dir and os.path.exists(home_dir):
        allowed_paths.append(os.path.realpath(home_dir))

    # 添加常见的配置目录
    for allowed in allowed_paths:
        if config_path_real.startswith(allowed):
            break
    else:
        # 如果不在允许的路径内，发出警告但仍允许加载（因为用户可能有意指定外部配置）
        import warnings
        warnings.warn(i18n.t('config_path_unexpected').format(config_path_real))

    try:
        with open(config_path_real, 'r', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif file_ext == '.json':
                return json.load(f)
            else:
                raise ValueError(i18n.t('config_format_unsupported').format(file_ext))
    except Exception as e:
        raise RuntimeError(i18n.t('config_load_failed').format(e))


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
        print(i18n.t('print_size_mismatch').format(width, height))
        size = max(width, height)
        width = height = size
        print(i18n.t('print_adjusted').format(width, height))
    
    # 限制最大尺寸为 1536（SPN 编码器在更大尺寸下会出现补丁分割问题）
    max_size = 1536
    if width > max_size or height > max_size:
        print(i18n.t('print_exceeds_max').format(width, height, max_size, max_size))
        print(i18n.t('print_patch_error'))
        print(i18n.t('print_adjusted').format(max_size, max_size))
        width = height = max_size
    
    # 检查是否能被 64 整除
    if width % 64 != 0 or height % 64 != 0:
        print(i18n.t('print_not_divisible').format(width, height))
        # 向上取整到最近的 64 倍数
        width = ((width + 63) // 64) * 64
        height = ((height + 63) // 64) * 64
        print(i18n.t('print_adjusted').format(width, height))
    
    # 再次检查调整后的尺寸是否超过最大值
    if width > max_size or height > max_size:
        print(i18n.t('print_still_exceeds').format(width, height))
        print(i18n.t('print_adjusted').format(max_size, max_size))
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
    
    # GPU 内存回收配置
    if not args.enable_auto_gc:
        config.setdefault('gpu', {})['enable_auto_gc'] = False
    if args.auto_gc_interval != 30:
        config.setdefault('gpu', {})['auto_gc_interval'] = args.auto_gc_interval
    if args.auto_gc_threshold != 85.0:
        config.setdefault('gpu', {})['auto_gc_threshold'] = args.auto_gc_threshold
    if not args.enable_smart_reclaim:
        config.setdefault('gpu', {})['enable_smart_reclaim'] = False
    
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
        cache_size=cache.get('size', 100),
        enable_auto_gc=gpu.get('enable_auto_gc', True),
        auto_gc_interval=gpu.get('auto_gc_interval', 30),
        auto_gc_threshold=gpu.get('auto_gc_threshold', 85.0),
        enable_smart_reclaim=gpu.get('enable_smart_reclaim', True)
    )


# ================= 日志工具 =================
class Logger:
    """日志工具类 - 基于 loguru"""
    
    def __init__(self):
        """初始化日志系统"""
        # 移除默认的 handler
        logger.remove()
        
        # 添加控制台 handler - 改进格式，增加更多上下文信息
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
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
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="14 days",  # 增加保留时间
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    @staticmethod
    def section(title_key: str, char: str = '=', length: int = 60):
        """打印分隔线
        Args:
            title_key: 翻译键或直接文本
        """
        title = i18n.t(title_key) if hasattr(i18n, 't') and i18n.t(title_key) != title_key else title_key
        print(f"\n{char * length}")
        print(f"[INFO] {title}")
        print(f"{char * length}\n")
    
    @staticmethod
    def styled_section(title: str, style: str = 'default'):
        """打印样式化的分隔线"""
        styles = {
            'default': ('=', '-', '─'),
            'highlight': ('*', '•', '◆'),
            'warning': ('!', '!', '⚠'),
            'success': ('+', '+', '✓'),
            'error': ('×', '×', '✗'),
            'info': ('→', '→', 'ℹ')
        }
        
        if style not in styles:
            style = 'default'
        
        border, separator, icon = styles[style]
        logger.opt(colors=True).info(f"<yellow>{border * 20}</yellow> <cyan>{icon} {title}</cyan> <yellow>{border * 20}</yellow>")
    
    @staticmethod
    def progress_info(current: int, total: int, message: str = None):
        """显示进度信息"""
        if message is None:
            message = i18n.t('progress_processing')
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 20
        filled_length = int(bar_length * current // total) if total > 0 else 0
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        logger.info(f"{message}: |{bar}| {percentage:.1f}% ({current}/{total})")

    @staticmethod
    def error(error_msg: str, solution: Optional[str] = None, exc_info: Optional[Exception] = None):
        """打印错误信息和解决方案

        Args:
            error_msg: 错误消息
            solution: 可选的解决方案提示
            exc_info: 可选的异常对象，如果提供则打印堆栈信息
        """
        logger.opt(colors=True).error(f"<red>❌ {i18n.t('log_error')}:</red> {error_msg}")
        if solution:
            logger.opt(colors=True).info(f"<green>💡 {i18n.t('log_solution')}:</green> {solution}")
        # 只在有活跃异常或显式传入异常时打印堆栈
        if exc_info is not None:
            logger.opt(colors=True).debug(f"<magenta>📋 {i18n.t('log_detail_error')}:</magenta>\n{traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__)}")
        elif sys.exc_info()[0] is not None:
            logger.opt(colors=True).debug(f"<magenta>📋 {i18n.t('log_detail_error')}:</magenta>\n{traceback.format_exc()}")

    @staticmethod
    def success(msg: str):
        """打印成功信息"""
        logger.opt(colors=True).success(f"<green>✓</green> {msg}")

    @staticmethod
    def warning(msg: str):
        """打印警告信息"""
        logger.opt(colors=True).warning(f"<yellow>⚠</yellow> {msg}")

    @staticmethod
    def info(msg: str):
        """打印信息"""
        logger.opt(colors=True).info(f"{msg}")

    @staticmethod
    def debug(msg: str):
        """打印调试信息"""
        logger.opt(colors=True).debug(f"<cyan>🔍</cyan> {msg}")

    @staticmethod
    def exception(msg: str, exc_info: Optional[Exception] = None):
        """打印异常信息

        Args:
            msg: 异常消息
            exc_info: 可选的异常对象
        """
        logger.opt(colors=True).error(f"<red>💥 {i18n.t('log_exception')}:</red> {msg}")
        # 只在有活跃异常或显式传入异常时打印堆栈
        if exc_info is not None:
            logger.opt(colors=True).debug(f"<magenta>📋 {i18n.t('log_detail_exception')}:</magenta>\n{traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__)}")
        elif sys.exc_info()[0] is not None:
            logger.opt(colors=True).debug(f"<magenta>📋 {i18n.t('log_detail_exception')}:</magenta>\n{traceback.format_exc()}")

    @staticmethod
    def critical(msg: str, solution: Optional[str] = None, exc_info: Optional[Exception] = None):
        """打印严重错误信息

        Args:
            msg: 严重错误消息
            solution: 可选的紧急解决方案
            exc_info: 可选的异常对象
        """
        logger.opt(colors=True).critical(f"<red>🔥 {i18n.t('log_critical')}:</red> {msg}")
        if solution:
            logger.opt(colors=True).info(f"<green>🚨 {i18n.t('log_emergency_solution')}:</green> {solution}")
        # 只在有活跃异常或显式传入异常时打印堆栈
        if exc_info is not None:
            logger.opt(colors=True).debug(f"<magenta>📋 {i18n.t('log_detail_error')}:</magenta>\n{traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__)}")
        elif sys.exc_info()[0] is not None:
            logger.opt(colors=True).debug(f"<magenta>📋 {i18n.t('log_detail_error')}:</magenta>\n{traceback.format_exc()}")
    
    @staticmethod
    def performance(msg: str):
        """打印性能相关信息"""
        logger.opt(colors=True).info(f"<blue>⏱️ {i18n.t('log_performance')}:</blue> {msg}")
    
    @staticmethod
    def gpu_info(msg: str):
        """打印GPU相关信息"""
        logger.opt(colors=True).info(f"<cyan>🎮 {i18n.t('log_gpu')}:</cyan> {msg}")
    
    @staticmethod
    def cache_info(msg: str):
        """打印缓存相关信息"""
        logger.opt(colors=True).info(f"<purple>📦 {i18n.t('log_cache')}:</purple> {msg}")

    @staticmethod
    def plain_success(msg: str):
        """打印不带样式的成功信息（用于特殊场景）"""
        logger.success(msg)

    @staticmethod
    def plain_warning(msg: str):
        """打印不带样式的警告信息（用于特殊场景）"""
        logger.warning(msg)

    @staticmethod
    def plain_info(msg: str):
        """打印不带样式的信息（用于特殊场景）"""
        logger.info(msg)

    @staticmethod
    def plain_debug(msg: str):
        """打印不带样式的调试信息（用于特殊场景）"""
        logger.debug(msg)

    @staticmethod
    def plain_exception(msg: str):
        """打印不带样式的异常信息（用于特殊场景）"""
        logger.exception(msg)


# ================= 命令行参数解析 =================
def parse_command_args() -> Tuple[CLIArgs, Optional[Dict[str, Any]]]:
    """解析命令行参数
    
    Returns:
        (CLIArgs, 配置文件字典或None)
    """
    # 先解析语言参数，以便帮助信息可以使用正确的语言
    import sys
    lang = 'zh'  # 默认语言
    for i, arg in enumerate(sys.argv):
        if arg in ('--lang', '--language'):
            if i + 1 < len(sys.argv) and sys.argv[i + 1] in ('zh', 'en'):
                lang = sys.argv[i + 1]
            break
    
    # 设置语言
    i18n.set_language(lang)
    
    parser = argparse.ArgumentParser(
        description=i18n.t('cli_description'),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=i18n.t('cli_epilog') if hasattr(i18n, 'translations') and 'cli_epilog' in i18n.translations.get(lang, {}) else """
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
                        help=i18n.t('arg_mode_help'))
    parser.add_argument('--port', '-p', type=int, default=8000,
                        help=i18n.t('arg_port_help'))
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help=i18n.t('arg_host_help'))
    parser.add_argument('--no-browser', action='store_true',
                        help=i18n.t('arg_no_browser_help'))
    parser.add_argument('--no-amp', action='store_true',
                        help=i18n.t('arg_no_amp_help'))
    parser.add_argument('--no-cudnn-benchmark', action='store_true',
                        help=i18n.t('arg_no_cudnn_help'))
    parser.add_argument('--config', '-c', type=str, default=None,
                        help=i18n.t('arg_config_help'))
    parser.add_argument('--input-size', type=int, nargs=2, default=[1536, 1536],
                        metavar=('WIDTH', 'HEIGHT'),
                        help=i18n.t('arg_input_size_help'))
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help=i18n.t('arg_gradient_help'))
    parser.add_argument('--checkpoint-segments', type=int, default=3,
                        help=i18n.t('arg_segments_help'))
    parser.add_argument('--enable-cache', action='store_true', default=True,
                        help=i18n.t('arg_cache_help'))
    parser.add_argument('--no-cache', action='store_true',
                        help=i18n.t('arg_no_cache_help'))
    parser.add_argument('--cache-size', type=int, default=100,
                        help=i18n.t('arg_cache_size_help'))
    parser.add_argument('--clear-cache', action='store_true',
                        help=i18n.t('arg_clear_cache_help'))
    parser.add_argument('--enable-auto-tune', action='store_true',
                        help=i18n.t('arg_auto_tune_help'))
    parser.add_argument('--redis-url', type=str, default=None,
                        help=i18n.t('arg_redis_help'))
    parser.add_argument('--enable-webhook', action='store_true',
                        help=i18n.t('arg_webhook_help'))
    
    # GPU 内存回收参数
    parser.add_argument('--enable-auto-gc', action='store_true', default=True,
                        help=i18n.t('arg_auto_gc_help'))
    parser.add_argument('--no-auto-gc', action='store_true',
                        help=i18n.t('arg_no_auto_gc_help'))
    parser.add_argument('--auto-gc-interval', type=int, default=30,
                        help=i18n.t('arg_gc_interval_help'))
    parser.add_argument('--auto-gc-threshold', type=float, default=85.0,
                        help=i18n.t('arg_gc_threshold_help'))
    parser.add_argument('--enable-smart-reclaim', action='store_true', default=True,
                        help=i18n.t('arg_smart_reclaim_help'))
    parser.add_argument('--no-smart-reclaim', action='store_true',
                        help=i18n.t('arg_no_smart_reclaim_help'))
    
    # 语言设置
    parser.add_argument('--lang', '--language', type=str, default='zh',
                        choices=['zh', 'en'],
                        help=i18n.t('arg_lang_help'))
    
    args = parser.parse_args()
    
    # 确保语言设置正确（以防通过参数解析改变）
    i18n.set_language(args.lang)
    
    # 处理缓存参数
    enable_cache = args.enable_cache and not args.no_cache
    
    # 转换 input_size 为元组
    input_size = tuple(args.input_size)
    
    # 验证输入尺寸
    validated_width, validated_height = validate_input_size(*input_size)
    if validated_width != input_size[0] or validated_height != input_size[1]:
        Logger.info(i18n.t('input_size_adjusted').format(input_size[0], input_size[1], validated_width, validated_height))
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
            cli_args.language = args.lang  # 设置语言
        except Exception as e:
            print(f"[ERROR] {i18n.t('load_config_failed', e)}")
            print(f"[INFO] {i18n.t('use_default_config_and_args')}")
            # 处理内存回收参数
            enable_auto_gc = args.enable_auto_gc and not args.no_auto_gc
            enable_smart_reclaim = args.enable_smart_reclaim and not args.no_smart_reclaim
            
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
                enable_auto_tune=args.enable_auto_tune,
                enable_auto_gc=enable_auto_gc,
                auto_gc_interval=args.auto_gc_interval,
                auto_gc_threshold=args.auto_gc_threshold,
                enable_smart_reclaim=enable_smart_reclaim,
                language=args.lang  # 设置语言
            )
    else:
        # 处理内存回收参数
        enable_auto_gc = args.enable_auto_gc and not args.no_auto_gc
        enable_smart_reclaim = args.enable_smart_reclaim and not args.no_smart_reclaim
        
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
            enable_auto_tune=args.enable_auto_tune,
            enable_auto_gc=enable_auto_gc,
            auto_gc_interval=args.auto_gc_interval,
            auto_gc_threshold=args.auto_gc_threshold,
            enable_smart_reclaim=enable_smart_reclaim,
            language=args.lang  # 设置语言
        )
    
    # 设置 app_config（性能自动调优需要）
    app_config = AppConfig.from_current_dir()
    cli_args.app_config = app_config
    
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
        self.app_config = args.app_config if hasattr(args, 'app_config') else None
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
            Logger.warning(i18n.t('wmi_detection_failed').format(e))
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
            Logger.warning(i18n.t('rocm_detection_failed').format(e))
            return False
    
    def initialize(self) -> torch.device:
        """初始化 GPU 设备"""
        Logger.section("gpu_init_title")
        
        if self.args.mode != 'auto':
            Logger.info(i18n.t('user_specified_mode_info').format(self.args.mode.upper()))
        else:
            Logger.info(i18n.t('auto_detect_mode'))
        
        force_mode = self.args.mode
        if force_mode == 'cpu':
            Logger.info(i18n.t('force_cpu_mode'))
        
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
                    Logger.success(i18n.t('amd_gpu_detected').format(self.config.name))
                    Logger.info("   ROCm: Yes")
                elif 'nvidia' in gpu_name_lower or 'geforce' in gpu_name_lower or 'quadro' in gpu_name_lower or 'tesla' in gpu_name_lower or 'rtx' in gpu_name_lower or 'gtx' in gpu_name_lower:
                    self.config.vendor = "NVIDIA"
                    Logger.success(i18n.t('nvidia_gpu_detected').format(self.config.name))
                elif 'amd' in gpu_name_lower or 'radeon' in gpu_name_lower or 'rx' in gpu_name_lower:
                    self.config.vendor = "AMD"
                    Logger.success(i18n.t('amd_gpu_detected').format(self.config.name))
                elif 'intel' in gpu_name_lower or 'iris' in gpu_name_lower or 'uhd' in gpu_name_lower or 'arc' in gpu_name_lower:
                    self.config.vendor = "Intel"
                    Logger.success(i18n.t('intel_gpu_detected').format(self.config.name))
                else:
                    # 如果 GPU 名称无法判断，使用系统检测结果
                    if system_vendor == 'NVIDIA':
                        self.config.vendor = "NVIDIA"
                        Logger.success(i18n.t('nvidia_gpu_detected').format(self.config.name))
                    elif system_vendor == 'AMD':
                        self.config.vendor = "AMD"
                        Logger.success(i18n.t('amd_gpu_detected').format(self.config.name))
                    elif system_vendor == 'Intel':
                        self.config.vendor = "Intel"
                        Logger.success(i18n.t('intel_gpu_detected').format(self.config.name))
                    else:
                        self.config.vendor = "Unknown"
                        Logger.warning(i18n.t('unknown_gpu_detected').format(self.config.name))
                
                Logger.info(i18n.t('cuda_version_info').format(self.config.cuda_version))
                Logger.info(i18n.t('gpu_count_info').format(self.config.count))
                
                # 强制模式处理
                if force_mode == 'nvidia':
                    if self.config.vendor != "NVIDIA":
                        Logger.warning(i18n.t('force_nvidia_mode').format(self.config.vendor))
                    self.config.vendor = "NVIDIA"
                    Logger.info(i18n.t('set_nvidia_mode'))
                elif force_mode == 'amd':
                    if self.config.vendor != "AMD":
                        Logger.warning(i18n.t('force_amd_mode').format(self.config.vendor))
                    self.config.vendor = "AMD"
                    Logger.info(i18n.t('set_amd_mode'))
                
                # 获取显卡属性
                props = torch.cuda.get_device_properties(0)
                self.config.compute_capability = props.major * 10 + props.minor
                self.config.supports_tf32 = props.major >= 8
                self.config.supports_bf16 = props.major >= 8
                
                Logger.info(i18n.t('compute_capability_info_full').format(f"{props.major}.{props.minor}"))
                Logger.info(i18n.t('gpu_memory_info_full').format(props.total_memory / 1024**3))
                
                if props.total_memory < 4 * 1024**3:
                    Logger.warning(i18n.t('low_vram_warning'))
                
                # 配置优化
                self._configure_optimizations(props)
                
                # 运行自动调优（如果启用）
                self.run_auto_tune()
                
                self.device = torch.device("cuda")
            else:
                self._setup_cpu_mode()
        
        except Exception as e:
            Logger.error(i18n.t('gpu_init_failed').format(e))
            self.device = torch.device("cpu")
            self.config.available = False
        
        # CPU 优化设置
        if not self.config.available:
            torch.set_num_threads(os.cpu_count())
            os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
            os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
            Logger.success(i18n.t('cpu_optimization_enabled').format(os.cpu_count()))
        
        # 启动 GPU 自动内存监控
        if self.config.available and self.args.enable_auto_gc:
            Logger.info(i18n.t('gpu_memory_management'))
            Logger.success(i18n.t('auto_gc_enabled'))
            Logger.info(i18n.t('gc_interval_info').format(self.args.auto_gc_interval))
            Logger.info(i18n.t('gc_threshold_info').format(self.args.auto_gc_threshold))
            self.start_auto_monitor(
                interval_seconds=self.args.auto_gc_interval,
                threshold_percent=self.args.auto_gc_threshold
            )
        elif self.config.available:
            Logger.info(i18n.t('gpu_memory_management'))
            Logger.info(i18n.t('auto_gc_disabled'))
        
        return self.device
    
    def _configure_optimizations(self, props):
        """配置 GPU 优化选项"""
        Logger.info(i18n.t('configure_optimizations'))
        
        # cuDNN Benchmark
        if self.config.vendor == "NVIDIA" and self.config.compute_capability >= 60 and not self.args.no_cudnn_benchmark:
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                self.config.use_cudnn_benchmark = True
                Logger.success(i18n.t('cudnn_enabled_status'))
            except Exception as e:
                Logger.warning(i18n.t('cudnn_enable_failed').format(e))
        else:
            if self.config.vendor != "NVIDIA":
                Logger.info(i18n.t('cudnn_not_applicable'))
            else:
                Logger.warning(i18n.t('cudnn_disabled_capability'))
        
        # TensorFloat32
        if self.config.vendor == "NVIDIA" and self.config.supports_tf32:
            try:
                torch.set_float32_matmul_precision('high')
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.config.use_tf32 = True
                Logger.success(i18n.t('tf32_enabled_status'))
            except Exception as e:
                Logger.warning(i18n.t('tf32_enable_failed').format(e))
        else:
            if self.config.vendor != "NVIDIA":
                Logger.info(i18n.t('tf32_not_applicable'))
            else:
                Logger.warning(i18n.t('tf32_disabled_support'))
        
        # 混合精度
        if self.config.compute_capability >= 53 and not self.args.no_amp:
            self.config.use_amp = True
            Logger.success(i18n.t('amp_enabled_status'))
        else:
            Logger.warning(i18n.t('amp_disabled_capability'))
    
    def run_auto_tune(self):
        """
        运行性能自动调优
        
        自动测试不同的优化配置组合，选择最优配置
        """
        if not self.args.enable_auto_tune:
            return
        
        try:
            # 如果没有指定配置文件，使用默认的 config.yaml
            config_file_path = self.args.config_file
            if not config_file_path:
                config_file_path = os.path.join(self.app_config.base_dir, 'config.yaml')
                Logger.info(i18n.t('using_default_config_file').format(config_file_path))
            
            tuner = PerformanceAutoTuner(self.config, self.device, config_file_path=config_file_path)
            best_config = tuner.benchmark_optimizations()
            
            if best_config:
                Logger.success(i18n.t('tuning_complete'))
                Logger.info(i18n.t('config_applied'))
            else:
                Logger.warning(i18n.t('tuning_failed_default'))
        except Exception as e:
            Logger.warning(i18n.t('tuning_failed').format(e))
            Logger.info(i18n.t('use_default_config'))
    
    def _setup_cpu_mode(self):
        """设置 CPU 模式"""
        system_vendor = self.detect_gpu_vendor_wmi()
        self.config.vendor = system_vendor
        self.device = torch.device("cpu")

        Logger.warning(i18n.t('cpu_mode'))
        Logger.info(i18n.t('cuda_unavailable_cpu_mode'))

        if system_vendor == "AMD":
            Logger.info(i18n.t('amd_no_rocm'))
        elif system_vendor == "NVIDIA":
            Logger.info(i18n.t('nvidia_no_cuda'))
        elif system_vendor == "Intel":
            Logger.info(i18n.t('intel_gpu'))
        else:
            Logger.info(i18n.t('no_supported_gpu'))
    
    def get_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """
        获取 GPU 内存使用信息
        
        Args:
            device_id: GPU 设备 ID
            
        Returns:
            包含内存使用信息的字典:
            - total_mb: 总显存 (MB)
            - used_mb: 已用显存 (MB)
            - free_mb: 可用显存 (MB)
            - used_percent: 使用百分比
        """
        if not torch.cuda.is_available():
            return {
                'total_mb': 0,
                'used_mb': 0,
                'free_mb': 0,
                'used_percent': 0
            }
        
        try:
            props = torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            reserved_memory = torch.cuda.memory_reserved(device_id)
            free_memory = total_memory - reserved_memory
            
            return {
                'total_mb': total_memory / 1024**2,
                'used_mb': allocated_memory / 1024**2,
                'free_mb': free_memory / 1024**2,
                'used_percent': (allocated_memory / total_memory) * 100
            }
        except Exception as e:
            Logger.warning(i18n.t('gpu_memory_query_failed').format(e))
            return {
                'total_mb': 0,
                'used_mb': 0,
                'free_mb': 0,
                'used_percent': 0
            }
    
    def clear_cache(self, device_id: int = 0) -> bool:
        """
        清理 GPU 缓存
        
        释放 PyTorch 预留的但未使用的显存
        
        Args:
            device_id: GPU 设备 ID
            
        Returns:
            是否成功清理
        """
        if not torch.cuda.is_available():
            Logger.debug(i18n.t('gpu_unavailable_skip_cache'))
            return False
        
        try:
            before_info = self.get_memory_info(device_id)
            
            # 清空缓存
            torch.cuda.empty_cache()
            
            after_info = self.get_memory_info(device_id)
            freed_mb = before_info['used_mb'] - after_info['used_mb']
            
            Logger.debug(i18n.t('gpu_cache_cleared').format(device_id))
            if freed_mb > 1:
                Logger.info(i18n.t('vram_freed_info').format(freed_mb))
            
            return True
        except Exception as e:
            Logger.warning(i18n.t('gpu_cache_clear_failed').format(e))
            return False
    
    def force_gc(self, device_id: int = 0) -> bool:
        """
        强制执行 GPU 垃圾回收
        
        包括: 清理缓存 + 释放未使用张量 + 同步 GPU
        
        Args:
            device_id: GPU 设备 ID
            
        Returns:
            是否成功回收
        """
        if not torch.cuda.is_available():
            Logger.debug(i18n.t('gpu_unavailable_skip_gc'))
            return False
        
        try:
            before_info = self.get_memory_info(device_id)
            
            # 1. 清理 PyTorch 缓存
            torch.cuda.empty_cache()
            
            # 2. 强制同步 GPU（确保所有计算完成）
            torch.cuda.synchronize()
            
            # 3. 释放未使用的显存
            import gc
            gc.collect()
            
            # 4. 再次清理缓存
            torch.cuda.empty_cache()
            
            after_info = self.get_memory_info(device_id)
            freed_mb = before_info['used_mb'] - after_info['used_mb']
            
            Logger.debug(i18n.t('gpu_gc_complete').format(device_id))
            if freed_mb > 1:
                Logger.info(i18n.t('vram_recovered_info').format(freed_mb))
            
            return True
        except Exception as e:
            Logger.warning(i18n.t('gpu_gc_failed_msg').format(e))
            return False
    
    def smart_reclaim(self, threshold_percent: float = 85.0, device_id: int = 0) -> bool:
        """
        智能内存回收
        
        当显存使用率超过阈值时自动清理
        
        Args:
            threshold_percent: 触发清理的阈值（百分比）
            device_id: GPU 设备 ID
            
        Returns:
            是否执行了清理
        """
        if not torch.cuda.is_available():
            return False
        
        try:
            mem_info = self.get_memory_info(device_id)
            
            if mem_info['used_percent'] >= threshold_percent:
                Logger.warning(i18n.t('gpu_vram_high').format(mem_info['used_percent']))
                Logger.info(i18n.t('vram_total_info').format(mem_info['total_mb']))
                Logger.info(i18n.t('vram_used_info').format(mem_info['used_mb']))
                Logger.info(i18n.t('vram_free_info').format(mem_info['free_mb']))
                Logger.info(i18n.t('smart_recovery'))
                
                return self.force_gc(device_id)
            
            return False
        except Exception as e:
            Logger.warning(i18n.t('smart_recovery_failed').format(e))
            return False
    
    def start_auto_monitor(self, interval_seconds: int = 30, threshold_percent: float = 85.0):
        """
        启动自动内存监控线程
        
        定期检查显存使用率并自动清理
        
        Args:
            interval_seconds: 检查间隔（秒）
            threshold_percent: 触发清理的阈值（百分比）
        """
        if not torch.cuda.is_available():
            Logger.info(i18n.t('gpu_unavailable_monitor'))
            return
        
        if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
            Logger.warning(i18n.t('monitoring_already_running'))
            return

        import threading
        import time

        # 使用 threading.Event 替代布尔标志，确保线程安全
        if not hasattr(self, '_monitor_active'):
            self._monitor_active = threading.Event()
        self._monitor_active.set()  # 设置事件标志
        self._monitor_interval = interval_seconds
        self._monitor_threshold = threshold_percent

        def monitor_loop():
            """监控循环"""
            Logger.info(i18n.t('gpu_monitoring_started').format(interval_seconds, threshold_percent))

            while self._monitor_active.is_set():  # 使用 is_set() 检查事件状态
                try:
                    time.sleep(interval_seconds)

                    if not self._monitor_active.is_set():
                        break

                    self.smart_reclaim(threshold_percent)

                except Exception as e:
                    Logger.warning(i18n.t('monitor_exception').format(e))

            Logger.info(i18n.t('gpu_monitor_stopped_msg'))

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_auto_monitor(self):
        """停止自动内存监控"""
        if hasattr(self, '_monitor_active'):
            self._monitor_active.clear()  # 清除事件标志

        if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
    
    def __del__(self):
        """析构函数，确保监控线程停止"""
        self.stop_auto_monitor()


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
                Logger.debug(i18n.t('cache_hit_debug').format(hit_rate, self.hits, self.hits + self.misses))
                
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
                Logger.debug(i18n.t('cache_evict').format(oldest_key))
            
            # 存入缓存
            self.cache[cache_key] = result
            self.cache_order.append(cache_key)
            Logger.debug(i18n.t('cache_added').format(cache_key))
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.cache_order.clear()
            self.hits = 0
            self.misses = 0
            Logger.info(i18n.t('cache_cleared_msg'))
    
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
            Logger.section("cache_stats_title")
            Logger.info(i18n.t('cache_enabled_status') if stats['enabled'] else i18n.t('cache_disabled_status'))
            Logger.info(i18n.t('cache_size_info_full').format(stats['size'], stats['max_size']))
            Logger.info(i18n.t('cache_hits_info').format(stats['hits']))
            Logger.info(i18n.t('cache_misses_info').format(stats['misses']))
            Logger.info(i18n.t('cache_hit_rate_info').format(stats['hit_rate']))

# ================= Redis 缓存管理器 =================
class RedisCacheManager:
    """Redis 缓存管理器 - 用于分布式缓存"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "mlsharp"):
        """
        初始化 Redis 缓存管理器

        Args:
            redis_url: Redis 连接 URL
            prefix: 缓存键前缀
            trust_redis: 是否信任 Redis 数据源（启用 pickle 反序列化）
                         警告：仅在 Redis 服务器受信任时启用，否则可能导致安全风险
        """
        self.redis_url = redis_url
        self.prefix = prefix
        self.trust_redis = True  # 默认信任，因为通常是本地 Redis
        self.redis_client = None
        self.enabled = False
        self._init_redis()

    def _init_redis(self):
        """初始化 Redis 客户端"""
        try:
            import redis
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            # 测试连接
            self.redis_client.ping()
            self.enabled = True
            Logger.info(i18n.t('redis_connected_info').format(self.redis_url))
        except ImportError:
            Logger.warning(i18n.t('redis_not_installed_msg'))
            Logger.info(i18n.t('redis_install_cmd_msg'))
        except Exception as e:
            Logger.warning(i18n.t('redis_connection_failed_msg').format(e))
            Logger.info(i18n.t('redis_unavailable_msg'))

    def _get_cache_key(self, image: np.ndarray, f_px: float) -> str:
        """计算缓存键"""
        import hashlib
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        return f"{self.prefix}:result:{image_hash}_{f_px:.6f}"

    def get(self, image: np.ndarray, f_px: float) -> Optional[Any]:
        """从 Redis 获取缓存结果

        安全警告：使用 pickle 反序列化数据，仅在 Redis 服务器受信任时使用
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            cache_key = self._get_cache_key(image, f_px)
            data = self.redis_client.get(cache_key)

            if data:
                # 安全检查：仅在信任 Redis 时使用 pickle 反序列化
                if not self.trust_redis:
                    Logger.warning(i18n.t('redis_data_untrusted_msg').format(cache_key))
                    return None
                # 反序列化 - 使用受限的 Unpickler 提高安全性
                import pickle
                import io

                class RestrictedUnpickler(pickle.Unpickler):
                    """受限的 Unpickler，只允许反序列化已知的安全类型"""
                    safe_classes = {}

                    def find_class(self, module, name):
                        # 允许 numpy 和 torch 相关的类型
                        if module.startswith('numpy') or module.startswith('torch'):
                            return super().find_class(module, name)
                        # 允许基本类型
                        if module == 'builtins' and name in ('dict', 'list', 'tuple', 'set', 'frozenset', 'str', 'int', 'float', 'bool', 'bytes', 'NoneType'):
                            return super().find_class(module, name)
                        # 其他类型需要显式允许
                        if (module, name) in self.safe_classes:
                            return super().find_class(module, name)
                        raise pickle.UnpicklingError(i18n.t('unsafe_deserialize').format(module, name))

                result = RestrictedUnpickler(io.BytesIO(data)).load()
                Logger.debug(i18n.t('redis_cache_hit').format(cache_key))
                return result
            else:
                Logger.debug(i18n.t('redis_cache_miss').format(cache_key))
                return None
        except pickle.UnpicklingError as e:
            Logger.warning(i18n.t('redis_deserialize_rejected_msg').format(e))
            return None
        except Exception as e:
            Logger.error(i18n.t('redis_get_failed').format(e))
            return None
    
    def set(self, image: np.ndarray, f_px: float, result: Any, ttl: int = 3600):
        """
        将结果存入 Redis 缓存
        
        Args:
            image: 输入图像
            f_px: 焦距
            result: 预测结果
            ttl: 过期时间（秒），默认 1 小时
        """
        if not self.enabled or not self.redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(image, f_px)
            # 序列化
            import pickle
            data = pickle.dumps(result)
            
            # 存入 Redis
            self.redis_client.setex(cache_key, ttl, data)
            Logger.debug(i18n.t('redis_cache_added').format(cache_key, ttl))
        except Exception as e:
            Logger.error(i18n.t('redis_set_failed').format(e))
    
    def clear(self):
        """清空 Redis 缓存"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            # 获取所有匹配前缀的键
            keys = self.redis_client.keys(f"{self.prefix}:*")
            if keys:
                self.redis_client.delete(*keys)
                Logger.info(i18n.t('redis_cleared_info').format(len(keys)))
            else:
                Logger.info(i18n.t('redis_cache_empty'))
        except Exception as e:
            Logger.error(i18n.t('redis_clear_failed').format(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 Redis 缓存统计信息"""
        if not self.enabled or not self.redis_client:
            return {
                "enabled": False,
                "type": "local"
            }
        
        try:
            keys = self.redis_client.keys(f"{self.prefix}:*")
            return {
                "enabled": True,
                "type": "redis",
                "size": len(keys),
                "url": self.redis_url
            }
        except Exception as e:
            Logger.error(i18n.t('redis_stats_failed').format(e))
            return {
                "enabled": False,
                "type": "local",
                "error": str(e)
            }
    
    def __del__(self):
        """析构函数，确保 Redis 连接关闭"""
        if self.redis_client:
            try:
                self.redis_client.close()
                Logger.debug(i18n.t('redis_conn_closed'))
            except Exception as e:
                Logger.debug(i18n.t('redis_conn_close_failed').format(e))

# ================= Webhook 管理器 =================
class WebhookManager:
    """Webhook 通知管理器"""
    
    def __init__(self, enabled: bool = False):
        """
        初始化 Webhook 管理器
        
        Args:
            enabled: 是否启用 Webhook
        """
        self.enabled = enabled
        self.webhooks: Dict[str, str] = {}  # event_type -> url
        self._init_httpx()
    
    def _init_httpx(self):
        """初始化 HTTP 客户端"""
        try:
            import httpx
            self.http_client = httpx.AsyncClient(timeout=30.0)
            Logger.info(i18n.t('webhook_client_init_msg'))
        except ImportError:
            Logger.warning(i18n.t('httpx_not_installed'))
            Logger.info(i18n.t('httpx_install_cmd'))
            self.http_client = None
    
    def register_webhook(self, event_type: str, url: str):
        """
        注册 Webhook
        
        Args:
            event_type: 事件类型（task_completed, task_failed, etc.）
            url: Webhook URL
        """
        if not self.enabled:
            Logger.warning(i18n.t('webhook_disabled_msg'))
            return
        
        self.webhooks[event_type] = url
        Logger.info(i18n.t('webhook_registered_info').format(event_type, url))
    
    def unregister_webhook(self, event_type: str):
        """
        注销 Webhook
        
        Args:
            event_type: 事件类型
        """
        if event_type in self.webhooks:
            del self.webhooks[event_type]
            Logger.info(i18n.t('webhook_unregistered_info').format(event_type))
    
    async def send_webhook(self, event_type: str, payload: Dict[str, Any]):
        """
        发送 Webhook 通知
        
        Args:
            event_type: 事件类型
            payload: 通知数据
        """
        if not self.enabled or event_type not in self.webhooks:
            return
        
        url = self.webhooks[event_type]
        
        if not self.http_client:
            Logger.error(i18n.t('http_client_not_init_msg'))
            return
        
        try:
            response = await self.http_client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Event": event_type,
                    "X-Webhook-Timestamp": str(time.time())
                }
            )
            
            if response.status_code == 200:
                Logger.info(i18n.t('webhook_sent_info').format(event_type, url))
            else:
                Logger.warning(i18n.t('webhook_send_failed_msg').format(event_type, url, response.status_code))
        except Exception as e:
            Logger.error(i18n.t('webhook_send_exception_msg').format(event_type, url, e))
    
    async def notify_task_completed(self, task_id: str, url: str, processing_time: float):
        """通知任务完成"""
        await self.send_webhook("task_completed", {
            "event": "task_completed",
            "task_id": task_id,
            "status": "success",
            "url": url,
            "processing_time": processing_time,
            "timestamp": time.time()
        })
    
    async def notify_task_failed(self, task_id: str, error: str):
        """通知任务失败"""
        await self.send_webhook("task_failed", {
            "event": "task_failed",
            "task_id": task_id,
            "status": "error",
            "error": error,
            "timestamp": time.time()
        })
    
    async def close(self):
        """关闭 HTTP 客户端"""
        if self.http_client:
            await self.http_client.aclose()
            Logger.info(i18n.t('webhook_client_closed_msg'))


# ================= 性能自动调优器 =================
class PerformanceAutoTuner:
    """性能自动调优器"""
    
    def __init__(self, gpu_config: GPUConfig, device: torch.device, config_file_path: str = None):
        """
        初始化性能自动调优器
        
        Args:
            gpu_config: GPU 配置
            device: 设备
            config_file_path: 配置文件路径
        """
        self.gpu_config = gpu_config
        self.device = device
        self.optimization_results = {}
        self.config_file_path = config_file_path
        self.cache_ttl_days = 7  # 缓存有效期（天）
    
    def _load_cached_results(self) -> Optional[Dict[str, Any]]:
        """
        加载缓存的调优结果
        
        Returns:
            缓存的结果，如果过期或不存在则返回 None
        """
        if not self.config_file_path or not os.path.exists(self.config_file_path):
            return None
        
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                if self.config_file_path.endswith('.yaml') or self.config_file_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # 检查是否有缓存的调优结果
            cache = config_data.get('performance_cache', {})
            if not cache:
                return None
            
            # 检查是否过期
            last_test = cache.get('last_test')
            if last_test:
                from datetime import datetime, timezone
                last_test_time = datetime.fromisoformat(last_test)
                now = datetime.now(timezone.utc)
                days_diff = (now - last_test_time).days
                
                if days_diff < self.cache_ttl_days:
                    # 检查 GPU 是否匹配
                    cache_gpu = cache.get('gpu', {})
                    if (cache_gpu.get('name') == self.gpu_config.name and
                        cache_gpu.get('vendor') == self.gpu_config.vendor and
                        cache_gpu.get('compute_capability') == self.gpu_config.compute_capability):
                        Logger.info(i18n.t('tuning_cache_found').format(days_diff))
                        return cache
            return None
        except Exception as e:
            Logger.debug(i18n.t('load_tuning_cache_failed').format(e))
            return None
    
    def _save_results_to_config(self, best_config: Dict[str, Any]):
        """
        保存调优结果到配置文件
        
        Args:
            best_config: 最优配置
        """
        if not self.config_file_path:
            Logger.warning(i18n.t('no_config_path'))
            return
        
        try:
            # 确保目录存在
            config_dir = os.path.dirname(self.config_file_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
                Logger.info(i18n.t('config_dir_created').format(config_dir))
            
            # 读取现有配置，如果文件不存在则创建默认配置
            config_data = {}
            has_existing_cache = False  # 标记是否已存在性能缓存
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    if self.config_file_path.endswith('.yaml') or self.config_file_path.endswith('.yml'):
                        config_data = yaml.safe_load(f) or {}
                    else:
                        config_data = json.load(f)
                has_existing_cache = 'performance_cache' in config_data
                Logger.info(i18n.t('config_file_updated').format(self.config_file_path))
            else:
                Logger.info(i18n.t('config_file_created').format(self.config_file_path))
                # 创建默认配置结构
                config_data = {
                    'server': {
                        'host': '127.0.0.1',
                        'port': 8000
                    },
                    'mode': 'auto',
                    'browser': {
                        'auto_open': True
                    },
                    'gpu': {
                        'enable_amp': True,
                        'enable_cudnn_benchmark': True,
                        'enable_tf32': True
                    },
                    'logging': {
                        'level': 'INFO',
                        'console': True,
                        'file': False
                    },
                    'model': {
                        'checkpoint': 'model_assets/sharp_2572gikvuh.pt',
                        'temp_dir': 'temp_workspace'
                    },
                    'inference': {
                        'input_size': [1536, 1536]
                    },
                    'optimization': {
                        'gradient_checkpointing': False,
                        'checkpoint_segments': 3
                    },
                    'cache': {
                        'enabled': True,
                        'size': 100
                    },
                    'redis': {
                        'enabled': False,
                        'url': 'redis://localhost:6379/0',
                        'prefix': 'mlsharp'
                    },
                    'webhook': {
                        'enabled': False,
                        'task_completed': '',
                        'task_failed': ''
                    },
                    'monitoring': {
                        'enabled': True,
                        'enable_gpu': True,
                        'metrics_path': '/metrics'
                    },
                    'performance': {
                        'max_workers': 4,
                        'max_concurrency': 10,
                        'timeout_keep_alive': 30,
                        'max_requests': 1000
                    }
                }
            
            # 更新配置
            from datetime import datetime, timezone
            config_data['performance_cache'] = {
                'last_test': datetime.now(timezone.utc).isoformat(),
                'best_config': best_config,
                'gpu': {
                    'name': self.gpu_config.name,
                    'vendor': self.gpu_config.vendor,
                    'compute_capability': self.gpu_config.compute_capability
                }
            }
            
            # 保存配置
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                if self.config_file_path.endswith('.yaml') or self.config_file_path.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            if has_existing_cache:
                Logger.success(i18n.t('tuning_results_updated').format(self.config_file_path))
            else:
                Logger.success(i18n.t('tuning_results_added').format(self.config_file_path))
        except Exception as e:
            Logger.warning(i18n.t('tuning_results_save_failed').format(e))
    
    def benchmark_optimizations(self) -> Dict[str, Any]:
        """
        基准测试各种优化配置，选择最优配置
        
        Returns:
            最优配置字典
        """
        # 检查是否有缓存的结果
        cached_results = self._load_cached_results()
        if cached_results:
            best_config = cached_results.get('best_config', {})
            if best_config:
                Logger.section("cached_config_title")
                Logger.info(i18n.t('best_config_name').format(best_config.get('name', 'N/A')))
                Logger.info(i18n.t('best_config_desc_info').format(best_config.get('description', 'N/A')))
                self._apply_config(best_config)
                return best_config
        
        Logger.section("performance_tuning_title")
        Logger.info(i18n.t('testing_optimizations'))
        
        # 测试配置列表
        test_configs = [
            {
                'name': i18n.t('tune_baseline'),
                'amp': False,
                'cudnn_benchmark': False,
                'tf32': False,
                'description': i18n.t('tune_baseline_desc')
            },
            {
                'name': i18n.t('tune_amp_only'),
                'amp': True,
                'cudnn_benchmark': False,
                'tf32': False,
                'description': i18n.t('tune_amp_only_desc')
            },
            {
                'name': i18n.t('tune_cudnn_only'),
                'amp': False,
                'cudnn_benchmark': True,
                'tf32': False,
                'description': i18n.t('tune_cudnn_only_desc')
            },
            {
                'name': i18n.t('tune_tf32_only'),
                'amp': False,
                'cudnn_benchmark': False,
                'tf32': True,
                'description': i18n.t('tune_tf32_only_desc')
            },
            {
                'name': i18n.t('tune_amp_cudnn'),
                'amp': True,
                'cudnn_benchmark': True,
                'tf32': False,
                'description': i18n.t('tune_amp_cudnn_desc')
            },
            {
                'name': i18n.t('tune_amp_tf32'),
                'amp': True,
                'cudnn_benchmark': False,
                'tf32': True,
                'description': i18n.t('tune_amp_tf32_desc')
            },
            {
                'name': i18n.t('tune_all'),
                'amp': True,
                'cudnn_benchmark': True,
                'tf32': True,
                'description': i18n.t('tune_all_desc')
            }
        ]
        
        # 根据显卡能力过滤不适用的配置
        if self.gpu_config.vendor != "NVIDIA":
            test_configs = [cfg for cfg in test_configs if not (cfg['cudnn_benchmark'] or cfg['tf32'])]
            Logger.info(i18n.t('tune_non_nvidia'))
        elif self.gpu_config.compute_capability < 60:
            test_configs = [cfg for cfg in test_configs if not cfg['cudnn_benchmark']]
            Logger.info(i18n.t('tune_skip_cudnn'))
        elif self.gpu_config.compute_capability < 80:
            test_configs = [cfg for cfg in test_configs if not cfg['tf32']]
            Logger.info(i18n.t('tune_skip_tf32'))
        
        # 执行基准测试
        results = []
        for config in test_configs:
            try:
                Logger.info(i18n.t('test_config_info').format(config['name']))
                Logger.info(i18n.t('test_config_desc').format(config['description']))
                
                # 应用配置
                self._apply_config(config)
                
                # 运行基准测试
                avg_time = self._run_benchmark()
                
                Logger.info(i18n.t('avg_inference_time_info').format(avg_time))
                
                results.append({
                    'config': config,
                    'avg_time': avg_time,
                    'throughput': 1.0 / avg_time if avg_time > 0 else 0
                })
                
            except Exception as e:
                Logger.warning(i18n.t('test_config_failed').format(e))
                continue
        
        # 选择最优配置
        if results:
            best_result = min(results, key=lambda x: x['avg_time'])
            Logger.section("tuning_results_title")
            Logger.success(i18n.t('best_config_result').format(best_result['config']['name']))
            Logger.info(i18n.t('best_config_desc_info').format(best_result['config']['description']))
            Logger.info(i18n.t('avg_inference_time_info').format(best_result['avg_time']))
            Logger.info(i18n.t('throughput_info_full').format(best_result['throughput']))
            
            # 应用最优配置
            self._apply_config(best_result['config'])
            
            self.optimization_results = {
                'best_config': best_result['config'],
                'all_results': results
            }
            
            # 保存结果到配置文件
            self._save_results_to_config(best_result['config'])
            
            return best_result['config']
        else:
            Logger.warning(i18n.t('all_tests_failed'))
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
            except Exception as e:
                Logger.debug(i18n.t('warmup_exception').format(e))
                # 继续进行，因为预热失败不是致命错误
                continue
        
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
                
                Logger.info(i18n.t('test_run_info').format(i+1, test_runs, elapsed))
                
            except Exception as e:
                Logger.warning(i18n.t('test_run_failed').format(i+1, test_runs, e))
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
        Logger.section("model_loading_title")
        Logger.info(i18n.t('model_file_info').format(self.config.checkpoint))
        
        # 检查模型文件
        if not os.path.exists(self.config.checkpoint):
            Logger.error(
                i18n.t('model_file_not_exists'),
                i18n.t('model_file_solution').format(self.config.checkpoint)
            )
            sys.exit(1)
        
        # 检查文件大小
        model_size = os.path.getsize(self.config.checkpoint) / (1024 * 1024)
        Logger.info(i18n.t('model_file_size').format(model_size))
        
        if model_size < 100:
            Logger.warning(i18n.t('model_size_warning'))
        
        try:
            from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
            from sharp.utils import io
            from sharp.utils.gaussians import Gaussians3D, SceneMetaData, save_ply, unproject_gaussians
            
            Logger.info(i18n.t('creating_predictor_msg'))
            self.predictor = create_predictor(PredictorParams())
            
            Logger.info(i18n.t('loading_weights_msg'))
            state_dict = torch.load(self.config.checkpoint, weights_only=True, map_location=self.device)
            
            Logger.info(i18n.t('loading_weights_to_predictor_msg'))
            self.predictor.load_state_dict(state_dict)
            self.predictor.eval()
            self.predictor.to(self.device)
            
            Logger.success(i18n.t('model_loaded_success'))
            Logger.info(i18n.t('device_info_full').format(self.device))
            
            # 应用梯度检查点
            if self.gradient_checkpointing and self.gpu_config.available:
                Logger.info(i18n.t('applying_gradient_checkpointing_msg'))
                self._apply_gradient_checkpointing()
                Logger.success(i18n.t('grad_checkpoint_enabled'))
            
            if self.gpu_config.available:
                memory_mb = torch.cuda.memory_allocated(self.device) / 1024**2
                Logger.info(i18n.t('vram_usage_info').format(memory_mb))
            
        except ImportError as e:
            Logger.error(
                i18n.t('sharp_import_error_msg').format(e),
                i18n.t('sharp_import_reasons')
            )
            sys.exit(1)
        except Exception as e:
            Logger.error(
                i18n.t('model_load_error_msg').format(e),
                i18n.t('model_load_check')
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
                Logger.info(i18n.t('grad_checkpoint_monodepth'))
            
            if hasattr(self.predictor, 'decoder'):
                # 包装 decoder
                original_forward = self.predictor.decoder.forward
                
                def checkpointed_forward(x):
                    return checkpoint(original_forward, x, use_reentrant=False)
                
                self.predictor.decoder.forward = checkpointed_forward
                Logger.info(i18n.t('grad_checkpoint_decoder'))
            
        except Exception as e:
            Logger.warning(i18n.t('grad_checkpoint_failed_msg').format(e))
            Logger.info(i18n.t('grad_checkpoint_fallback_msg'))
    
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
                Logger.warning(i18n.t('mixed_precision_fallback_msg').format(e))
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
            except OSError as e:
                Logger.warning(i18n.t('temp_dir_delete_failed_msg').format(e))
        os.makedirs(self.app_config.temp_dir, exist_ok=True)
        
        # 初始化 GPU
        import torch
        self.gpu_manager = GPUManager(self.gpu_config, self.args)
        self.gpu_manager.app_config = self.app_config
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
            Logger.info(i18n.t('clearing_cache'))
            self.model_manager.cache_manager.clear()
            Logger.success(i18n.t('cache_cleared_success'))
        
        # 初始化监控指标
        self.metrics_manager = init_metrics(enable_gpu=self.gpu_config.available)
        if self.gpu_config.available:
            self.metrics_manager.set_gpu_info(0, self.gpu_config.name, self.gpu_config.vendor)
        self.metrics_manager.set_input_size(*self.args.input_size)
        
        # 初始化 Redis 缓存（如果指定）
        self.redis_cache = None
        if self.args.redis_url:
            self.redis_cache = RedisCacheManager(redis_url=self.args.redis_url)
            if self.redis_cache.enabled:
                Logger.success(i18n.t('redis_cache_enabled').format(self.args.redis_url))
        
        # 初始化 Webhook 管理器（如果启用）
        self.webhook_manager = None
        if self.args.enable_webhook:
            self.webhook_manager = WebhookManager(enabled=True)
            Logger.success(i18n.t('webhook_enabled'))
        
        # 创建 FastAPI 应用
        self.app = self._create_app()
    
    # ================= Pydantic 模型定义 =================
    
    class PredictResponse(BaseModel):
        """预测响应模型"""
        status: str = Field(..., description="请求状态 (success/error)")
        url: str = Field(..., description="生成的 PLY 文件下载地址")
        processing_time: float = Field(..., description="处理时间（秒）")
        task_id: str = Field(..., description="任务 ID")
    
    class HealthResponse(BaseModel):
        """健康检查响应模型"""
        status: str = Field(..., description="服务状态 (healthy/unhealthy)")
        gpu_available: bool = Field(..., description="GPU 是否可用")
        gpu_vendor: str = Field(..., description="GPU 厂商 (NVIDIA/AMD/Intel)")
        gpu_name: str = Field(..., description="GPU 型号名称")
    
    class GPUInfo(BaseModel):
        """GPU 信息模型"""
        available: bool = Field(..., description="GPU 是否可用")
        vendor: str = Field(..., description="GPU 厂商")
        name: str = Field(..., description="GPU 型号")
        count: int = Field(..., description="GPU 数量")
        memory_mb: float = Field(..., description="当前 GPU 内存使用量（MB）")
    
    class StatsResponse(BaseModel):
        """系统统计响应模型"""
        gpu: "MLSharpApp.GPUInfo" = Field(..., description="GPU 信息")
    
    class CacheStatsResponse(BaseModel):
        """缓存统计响应模型"""
        enabled: bool = Field(..., description="缓存是否启用")
        size: int = Field(..., description="当前缓存条目数")
        max_size: int = Field(..., description="最大缓存条目数")
        hits: int = Field(..., description="缓存命中次数")
        misses: int = Field(..., description="缓存未命中次数")
        hit_rate: float = Field(..., description="缓存命中率（百分比）")
    
    class CacheClearResponse(BaseModel):
        """缓存清空响应模型"""
        status: str = Field(..., description="操作状态")
        message: str = Field(..., description="操作消息")
    
    class ErrorResponse(BaseModel):
        """统一错误响应模型"""
        error: str = Field(..., description="错误类型")
        message: str = Field(..., description="错误消息")
        status_code: int = Field(..., description="HTTP 状态码")
        path: str = Field(..., description="请求路径")
        timestamp: str = Field(..., description="错误发生时间（ISO 8601 格式）")
    
    # ================= 错误处理器 =================
    
    def _create_error_response(self, error: str, message: str, status_code: int, path: str) -> Dict[str, Any]:
        """创建标准错误响应"""
        from datetime import datetime
        return {
            "error": error,
            "message": message,
            "status_code": status_code,
            "path": path,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    def _create_app(self):
        """创建 FastAPI 应用"""
        from fastapi import FastAPI, UploadFile, File, APIRouter, Body
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI(
            title="MLSharp 3D Maker API",
            description="基于 Apple SHaRP 模型的 3D 高斯泼溅生成工具",
            version="9.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # 创建 v1 API 路由
        v1_router = APIRouter(prefix="/v1", tags=["v1"])
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        app.mount("/files", StaticFiles(directory=self.app_config.temp_dir), name="files")
        
        # ================= 异常处理器 =================

        @app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            """通用异常处理器"""
            # 记录详细错误信息到日志（用于调试）
            Logger.error(f"Internal server error [{request.url.path}]: {exc}", exc_info=exc)

            # 返回通用错误消息给客户端（避免敏感信息泄露）
            error_response = self._create_error_response(
                error="InternalServerError",
                message=i18n.t('internal_server_error'),
                status_code=500,
                path=request.url.path
            )
            return JSONResponse(
                status_code=500,
                content=error_response
            )
        
        @app.exception_handler(404)
        async def not_found_handler(request, exc):
            """404 异常处理器"""
            error_response = self._create_error_response(
                error="NotFound",
                message=i18n.t('resource_not_found'),
                status_code=404,
                path=request.url.path
            )
            return JSONResponse(
                status_code=404,
                content=error_response
            )
        
        @app.exception_handler(422)
        async def validation_error_handler(request, exc):
            """422 验证异常处理器"""
            error_response = self._create_error_response(
                error="ValidationError",
                message=i18n.t('validation_error'),
                status_code=422,
                path=request.url.path
            )
            return JSONResponse(
                status_code=422,
                content=error_response
            )
        
        @app.get("/", tags=["UI"])
        async def read_index():
            """访问 Web 界面"""
            return FileResponse(os.path.join(self.app_config.base_dir, "viewer.html"))
        
        @v1_router.post("/predict", response_model=MLSharpApp.PredictResponse, tags=["Prediction"])
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
        
        @v1_router.get("/health", response_model=MLSharpApp.HealthResponse, tags=["System"])
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
        
        @v1_router.get("/stats", response_model=MLSharpApp.StatsResponse, tags=["System"])
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
                except Exception as e:
                    Logger.warning(i18n.t('get_gpu_mem_failed').format(e))
                    stats["gpu"]["memory_mb"] = 0
            return stats
        
        @v1_router.get("/cache", response_model=MLSharpApp.CacheStatsResponse, tags=["System"])
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
        
        @v1_router.post("/cache/clear", response_model=MLSharpApp.CacheClearResponse, tags=["System"])
        async def clear_cache():
            """清空缓存
            
            清空所有缓存条目并重置统计信息。
            
            返回:
                - status: 操作状态
                - message: 操作消息
            """
            self.model_manager.cache_manager.clear()
            if self.redis_cache and self.redis_cache.enabled:
                self.redis_cache.clear()
            return {"status": "success", "message": i18n.t('cache_clear_msg')}
        
        @v1_router.get("/webhooks", tags=["Webhook"])
        async def list_webhooks():
            """获取所有已注册的 Webhook
            
            返回所有已注册的 Webhook 列表。
            
            返回:
                - enabled: Webhook 是否启用
                - webhooks: Webhook 字典（事件类型 -> URL）
            """
            if not self.webhook_manager:
                return {
                    "enabled": False,
                    "webhooks": {},
                    "message": i18n.t('webhook_not_enabled')
                }
            return {
                "enabled": self.webhook_manager.enabled,
                "webhooks": self.webhook_manager.webhooks
            }
        
        @v1_router.post("/webhooks", tags=["Webhook"])
        async def register_webhook(webhook_data: Dict[str, str] = Body(..., examples={
            "example": {
                "event_type": "task_completed",
                "url": "https://example.com/webhook/completed"
            }
        })):
            """注册 Webhook
            
            注册一个新的 Webhook 用于接收事件通知。
            
            - **event_type**: 事件类型（task_completed, task_failed）
            - **url**: Webhook URL
            
            返回:
                - status: 操作状态
                - message: 操作消息
            """
            if not self.webhook_manager:
                return {
                    "status": "error",
                    "message": i18n.t('webhook_not_enabled')
                }
            event_type = webhook_data.get("event_type")
            url = webhook_data.get("url")
            
            if not event_type or not url:
                return {
                    "status": "error",
                    "message": i18n.t('missing_params')
                }
            
            self.webhook_manager.register_webhook(event_type, url)
            return {
                "status": "success",
                "message": i18n.t('webhook_registered_resp').format(event_type, url)
            }
        
        @v1_router.delete("/webhooks/{event_type}", tags=["Webhook"])
        async def unregister_webhook(event_type: str):
            """注销 Webhook
            
            注销指定事件类型的 Webhook。
            
            - **event_type**: 事件类型
            
            返回:
                - status: 操作状态
                - message: 操作消息
            """
            if not self.webhook_manager:
                return {
                    "status": "error",
                    "message": i18n.t('webhook_not_enabled')
                }
            self.webhook_manager.unregister_webhook(event_type)
            return {
                "status": "success",
                "message": i18n.t('webhook_unregistered_info').format(event_type)
            }
        
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

            # 增加活跃任务计数（使用原子操作）
            is_predict_request = request.url.path in ("/api/predict", "/v1/predict")
            if is_predict_request:
                self.metrics_manager.active_tasks.inc()

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
                # 减少活跃任务计数（使用原子操作）
                if is_predict_request:
                    self.metrics_manager.active_tasks.dec()
        
        # 注册 v1 路由
        app.include_router(v1_router)
        
        return app
    
    # 允许的图片类型
    ALLOWED_IMAGE_TYPES = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp'
    }
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    MAX_FILE_SIZE_MB = 50  # 最大文件大小限制（MB）

    async def _handle_predict(self, file: UploadFile):
        """处理预测请求 - 异步优化版本"""
        from sharp.utils import io
        from sharp.utils.gaussians import save_ply

        # 文件类型验证
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''
        content_type = file.content_type or ''

        if content_type not in self.ALLOWED_IMAGE_TYPES and file_ext not in self.ALLOWED_EXTENSIONS:
            Logger.warning(i18n.t('api_unsupported_file_type').format(content_type, file_ext))
            return JSONResponse({
                "status": "error",
                "message": i18n.t('api_unsupported_file_simple').format(content_type or file_ext),
                "solution": i18n.t('api_supported_formats')
            }, status_code=400)

        # 文件大小验证（通过读取前几个字节检查）
        try:
            # 读取文件头用于验证
            header = await file.read(8)
            await file.seek(0)  # 重置文件指针

            # 检查文件签名（魔数）
            if header[:2] == b'\xff\xd8':  # JPEG
                pass
            elif header[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
                pass
            elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':  # WebP (需要读取更多)
                pass
            elif header[:2] in (b'BM', b'BA'):  # BMP
                pass
            else:
                Logger.warning(i18n.t('api_unknown_format').format(header[:8].hex()))
                # 不直接拒绝，让后续处理决定
        except Exception as e:
            Logger.warning(i18n.t('api_header_check_failed').format(e))

        task_id = str(uuid.uuid4())[:8]
        try:
            start_time = time.time()
            task_dir = os.path.join(self.app_config.temp_dir, task_id)

            # 路径安全验证：确保生成的路径在预期的临时目录内
            task_dir_real = os.path.realpath(task_dir)
            temp_dir_real = os.path.realpath(self.app_config.temp_dir)
            if not task_dir_real.startswith(temp_dir_real):
                Logger.error(i18n.t('api_path_traversal_detected').format(task_dir))
                return JSONResponse({
                    "status": "error",
                    "message": i18n.t('api_invalid_path'),
                    "solution": i18n.t('api_contact_support')
                }, status_code=400)

            output_dir = os.path.join(task_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存上传的文件 - 使用 asyncio.to_thread
            file_path = os.path.join(task_dir, "input.jpg")
            try:
                await asyncio.to_thread(self._save_file, file, file_path)
                Logger.info(i18n.t('task_file_saved').format(task_id, file_path))
            except Exception as e:
                Logger.error(i18n.t('api_save_upload_failed').format(task_id, e))
                elapsed_time = time.time() - start_time
                self.metrics_manager.record_predict_request("error", elapsed_time)
                return JSONResponse({
                    "status": "error",
                    "message": i18n.t('api_save_upload_msg').format(str(e)),
                    "solution": i18n.t('api_check_format_support')
                }, status_code=500)
            
            # 加载图像 - 使用 asyncio.to_thread
            load_start = time.time()
            try:
                image, _, f_px = await asyncio.to_thread(io.load_rgb, Path(file_path))
            except Exception as e:
                Logger.error(i18n.t('api_load_image_failed_task').format(task_id, e))
                elapsed_time = time.time() - start_time
                self.metrics_manager.record_predict_request("error", elapsed_time)
                return JSONResponse({
                    "status": "error",
                    "message": i18n.t('api_load_image_msg').format(str(e)),
                    "solution": i18n.t('api_check_image_format')
                }, status_code=500)
            height, width = image.shape[:2]
            load_time = time.time() - load_start
            Logger.info(i18n.t('task_image_info_full').format(task_id, width, height, f_px, load_time))
            self.metrics_manager.record_predict_stage("image_load", load_time)
            
            # 检查图片尺寸
            if width > 4096 or height > 4096:
                Logger.warning(i18n.t('api_image_too_large_task').format(task_id, width, height))
            
            # 检查 Redis 缓存
            cached_result = None
            if self.redis_cache and self.redis_cache.enabled:
                try:
                    cached_result = self.redis_cache.get(image, f_px)
                except Exception as e:
                    Logger.warning(i18n.t('api_redis_failed_task').format(task_id, e))
                    # 即使Redis失败，也继续尝试本地缓存
                    cached_result = None
            
            if cached_result is not None:
                # 使用缓存结果保存 PLY
                output_ply_path = os.path.join(output_dir, "output.ply")
                save_start = time.time()
                try:
                    await asyncio.to_thread(save_ply, cached_result, f_px, (height, width), output_ply_path)
                    save_time = time.time() - save_start
                    Logger.info(i18n.t('task_cache_hit_info').format(task_id, save_time))
                except Exception as e:
                    Logger.error(i18n.t('api_cache_ply_failed').format(task_id, e))
                    elapsed_time = time.time() - start_time
                    self.metrics_manager.record_predict_request("error", elapsed_time)
                    return JSONResponse({
                        "status": "error",
                        "message": i18n.t('api_cache_ply_msg').format(str(e)),
                        "solution": i18n.t('api_check_disk')
                    }, status_code=500)
                
                # 重命名
                final_ply = os.path.join(task_dir, "output.ply")
                try:
                    await asyncio.to_thread(os.rename, output_ply_path, final_ply)
                except OSError as e:
                    Logger.error(i18n.t('api_rename_failed_task').format(task_id, e))
                    # 如果重命名失败，直接使用原路径
                    final_ply = output_ply_path
                
                elapsed_time = time.time() - start_time
                Logger.info(i18n.t('task_completion_info').format(task_id, elapsed_time))
                
                # 记录预测指标
                self.metrics_manager.record_predict_request("success", elapsed_time)
                self.metrics_manager.record_predict_stage("total", elapsed_time)
                
                download_url = f"/files/{task_id}/output.ply"
                
                # 发送 Webhook 通知（任务完成）
                if self.webhook_manager:
                    try:
                        await self.webhook_manager.notify_task_completed(task_id, download_url, elapsed_time)
                    except Exception as e:
                        Logger.warning(i18n.t('api_webhook_failed_task').format(task_id, e))
                
                return {"status": "success", "url": download_url, "processing_time": elapsed_time, "task_id": task_id}
            
            # 预测 - GPU 推理在单独线程中执行
            Logger.info(i18n.t('task_inference_start').format(task_id))
            inference_start = time.time()
            try:
                gaussians = await asyncio.to_thread(self.model_manager.predict, image, f_px)
                if self.gpu_config.available:
                    torch.cuda.synchronize()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    Logger.critical(i18n.t('api_vram_insufficient_task').format(task_id, e))
                    elapsed_time = time.time() - start_time
                    self.metrics_manager.record_predict_request("error", elapsed_time)
                    return JSONResponse({
                        "status": "error",
                        "message": i18n.t('api_vram_insufficient'),
                        "solution": i18n.t('api_vram_solution')
                    }, status_code=507)  # 507 Insufficient Storage
                else:
                    raise  # 重新抛出其他RuntimeError
            except Exception as e:
                Logger.error(i18n.t('api_inference_failed_task').format(task_id, e))
                elapsed_time = time.time() - start_time
                self.metrics_manager.record_predict_request("error", elapsed_time)
                
                # 发送 Webhook 通知（任务失败）
                if self.webhook_manager:
                    await self.webhook_manager.notify_task_failed(task_id, str(e))
                
                return JSONResponse({
                    "status": "error",
                    "message": i18n.t('api_inference_failed').format(str(e)),
                    "solution": i18n.t('api_retry_small_image')
                }, status_code=500)
            
            inference_time = time.time() - inference_start
            Logger.info(i18n.t('task_inference_complete_info').format(task_id, inference_time))
            self.metrics_manager.record_predict_stage("inference", inference_time)
            
            # 保存到 Redis 缓存
            if self.redis_cache and self.redis_cache.enabled:
                try:
                    self.redis_cache.set(image, f_px, gaussians, ttl=3600)
                    Logger.info(i18n.t('task_redis_cache_info').format(task_id))
                except Exception as e:
                    Logger.warning(i18n.t('api_redis_cache_failed_task').format(task_id, e))
            
            # 保存 PLY - 使用 asyncio.to_thread
            output_ply_path = os.path.join(output_dir, "output.ply")
            save_start = time.time()
            try:
                await asyncio.to_thread(save_ply, gaussians, f_px, (height, width), output_ply_path)
                save_time = time.time() - save_start
                Logger.info(i18n.t('task_ply_save_info').format(task_id, save_time))
                self.metrics_manager.record_predict_stage("ply_save", save_time)
            except Exception as e:
                Logger.error(i18n.t('api_ply_save_failed_task').format(task_id, e))
                elapsed_time = time.time() - start_time
                self.metrics_manager.record_predict_request("error", elapsed_time)
                return JSONResponse({
                    "status": "error",
                    "message": i18n.t('api_ply_save_failed').format(str(e)),
                    "solution": i18n.t('api_check_disk')
                }, status_code=500)
            
            # 重命名 - 异步文件操作
            final_ply = os.path.join(task_dir, "output.ply")
            try:
                await asyncio.to_thread(os.rename, output_ply_path, final_ply)
            except OSError as e:
                Logger.error(i18n.t('api_rename_failed_task').format(task_id, e))
                # 如果重命名失败，直接使用原路径
                final_ply = output_ply_path
            
            elapsed_time = time.time() - start_time
            Logger.info(i18n.t('task_complete_info').format(task_id, elapsed_time))
            
            # 记录预测指标
            self.metrics_manager.record_predict_request("success", elapsed_time)
            self.metrics_manager.record_predict_stage("total", elapsed_time)
            
            download_url = f"/files/{task_id}/output.ply"
            
            # 发送 Webhook 通知（任务完成）
            if self.webhook_manager:
                await self.webhook_manager.notify_task_completed(task_id, download_url, elapsed_time)
            
            return {"status": "success", "url": download_url, "processing_time": elapsed_time, "task_id": task_id}
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                Logger.error(i18n.t('api_vram_insufficient_task').format(task_id, e))
                elapsed_time = time.time() - start_time
                self.metrics_manager.record_predict_request("error", elapsed_time)
                return JSONResponse({
                    "status": "error",
                    "message": i18n.t('api_vram_insufficient'),
                    "solution": i18n.t('api_vram_solution')
                }, status_code=507)
            raise
        except Exception as e:
            Logger.error(i18n.t('api_processing_failed_task').format(task_id, e))
            elapsed_time = time.time() - start_time
            self.metrics_manager.record_predict_request("error", elapsed_time)
            
            # 发送 Webhook 通知（任务失败）
            if self.webhook_manager:
                await self.webhook_manager.notify_task_failed(task_id, str(e))
            
            return JSONResponse({
                "status": "error",
                "message": i18n.t('api_processing_failed').format(str(e)),
                "solution": i18n.t('api_retry_small_image')
            }, status_code=500)
    
    def _save_file(self, upload_file, file_path):
        """保存上传的文件

        Args:
            upload_file: FastAPI UploadFile 对象
            file_path: 保存路径
        """
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
        finally:
            # 确保关闭上传文件句柄
            upload_file.file.close()
    
    def print_startup_banner(self):
        """打印启动横幅"""
        print("\n" + "=" * 60)
        print(" " * 20 + "MLSharp")
        print(" " * 12 + i18n.t('banner_title'))
        print("=" * 60)
        print()
        print(i18n.t('banner_modes'))
        print("  ✓ " + i18n.t('banner_nvidia'))
        print("  ✓ " + i18n.t('banner_amd'))
        print("  ✓ " + i18n.t('banner_intel'))
        print("  ✓ " + i18n.t('banner_cpu'))
        print()
        print("=" * 60)
        print()
    
    def print_system_info(self):
        """打印系统信息"""
        Logger.section("system_info_title")
        Logger.info(i18n.t('os_info_full').format(platform.system(), platform.release()))
        Logger.info(i18n.t('python_version_info').format(sys.version.split()[0]))
        Logger.info(i18n.t('working_directory_info').format(self.app_config.base_dir))
    
    def print_service_info(self):
        """打印服务信息"""
        Logger.section("web_service_title")
        
        Logger.info(i18n.t('input_size_info_full').format(self.args.input_size[0], self.args.input_size[1]))
        
        if self.gpu_config.available:
            Logger.success(i18n.t('gpu_accel_enabled'))
            Logger.info(i18n.t('gpu_vendor_info').format(self.gpu_config.vendor))
            Logger.info(i18n.t('gpu_model_info').format(self.gpu_config.name))
            if self.gpu_config.vendor == "NVIDIA":
                Logger.info(i18n.t('compute_capability_simple').format(self.gpu_config.compute_capability))
                Logger.info(i18n.t('amp_enabled_info') if self.gpu_config.use_amp else i18n.t('amp_disabled_info'))
                Logger.info(i18n.t('cudnn_enabled_info') if self.gpu_config.use_cudnn_benchmark else i18n.t('cudnn_disabled_info'))
                Logger.info(i18n.t('tf32_enabled_info') if self.gpu_config.use_tf32 else i18n.t('tf32_disabled_info'))
            elif self.gpu_config.vendor == "AMD":
                Logger.info(i18n.t('rocm_accel'))
        else:
            Logger.warning(i18n.t('cpu_mode_simple'))
            Logger.info(i18n.t('cpu_cores_info').format(os.cpu_count()))
            Logger.info(i18n.t('multithreading_opt_enabled'))
        
        # 缓存信息
        if self.args.enable_cache:
            Logger.success(i18n.t('inference_cache_enabled').format(self.args.cache_size))
        else:
            Logger.info(i18n.t('cache_disabled_info'))
        
        print()
        service_url = f"http://{self.args.host}:{self.args.port}"
        Logger.success(i18n.t('service_url_info').format(service_url))
        if not self.args.no_browser:
            Logger.info(i18n.t('browser_open_msg'))
        Logger.info(i18n.t('ctrl_c_stop'))
        print()
    
    def open_browser(self):
        """打开浏览器"""
        service_url = f"http://{self.args.host}:{self.args.port}"
        time.sleep(2)
        try:
            webbrowser.open(service_url)
        except Exception as e:
            Logger.warning(i18n.t('browser_open_failed').format(e))
            Logger.info(i18n.t('manual_access_info').format(service_url))
    
    def cleanup(self):
        """清理资源"""
        Logger.info(i18n.t('cleaning_resources'))
        
        # 停止 GPU 监控
        if self.metrics_manager:
            self.metrics_manager.stop_monitoring()
            Logger.info(i18n.t('gpu_monitor_stopped_log'))
        
        # 停止 GPU 自动内存监控
        if self.gpu_manager:
            self.gpu_manager.stop_auto_monitor()
            Logger.info(i18n.t('gpu_auto_monitor_stopped'))
        
        # 关闭 Webhook 客户端
        if self.webhook_manager:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.webhook_manager.close())
                else:
                    asyncio.run(self.webhook_manager.close())
                Logger.info(i18n.t('webhook_closed_log'))
            except Exception as e:
                Logger.warning(i18n.t('webhook_close_failed_log').format(e))
        
        Logger.info(i18n.t('resources_cleaned'))
    
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
            Logger.section("service_stopped_title")
            Logger.info(i18n.t('thanks_using'))
        except Exception as e:
            Logger.error(i18n.t('service_start_failed_log').format(e))
            sys.exit(1)
        finally:
            self.cleanup()


# ================= 主程序入口 =================
if __name__ == "__main__":
    # 初始化日志系统
    log_system = Logger()
    
    app = MLSharpApp()
    app.run()
