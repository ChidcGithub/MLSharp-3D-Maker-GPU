# -*- coding: utf-8 -*-
"""
GPU 检测和兼容性工具模块
提供跨平台的 GPU 检测功能
"""
import subprocess
import platform

def detect_gpu_vendor_wmi():
    """通过 WMI 检测显卡厂商（Windows）"""
    try:
        # 首先尝试使用 PowerShell Get-CimInstance（Windows 11 推荐）
        result = subprocess.run(
            ['powershell', '-Command', 'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for line in lines:
                name = line.strip().lower()
                if 'nvidia' in name or 'geforce' in name or 'quadro' in name or 'tesla' in name or 'rtx' in name or 'gtx' in name:
                    return 'NVIDIA'
                elif 'amd' in name or 'radeon' in name or 'rx' in name:
                    return 'AMD'
                elif 'intel' in name or 'iris' in name or 'uhd' in name or 'arc' in name:
                    return 'Intel'
        else:
            # 回退到 wmic 命令（Windows 10 及更早版本）
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]
                for line in lines:
                    name = line.strip().lower()
                    if 'nvidia' in name or 'geforce' in name or 'quadro' in name or 'tesla' in name or 'rtx' in name or 'gtx' in name:
                        return 'NVIDIA'
                    elif 'amd' in name or 'radeon' in name or 'rx' in name:
                        return 'AMD'
                    elif 'intel' in name or 'iris' in name or 'uhd' in name or 'arc' in name:
                        return 'Intel'
    except Exception as e:
        pass
    return 'Unknown'

def check_rocm_available():
    """检查 ROCm 是否可用"""
    try:
        import torch
        # ROCm 使用与 CUDA 相同的接口
        if torch.cuda.is_available():
            # 检查是否有 HIP 符号（ROCm 特有）
            # 注意：CUDA 版本的 PyTorch 也可能有 hip 属性，需要检查其值是否为 None
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return True
            # 检查设备名称是否包含 AMD
            device_name = torch.cuda.get_device_name(0).lower()
            if 'amd' in device_name or 'radeon' in device_name:
                return True
        return False
    except Exception:
        return False

def get_gpu_info():
    """获取 GPU 详细信息"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        gpu_info = {
            'name': torch.cuda.get_device_name(0),
            'count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
            'is_rocm': check_rocm_available(),
        }
        
        props = torch.cuda.get_device_properties(0)
        gpu_info.update({
            'compute_capability': props.major * 10 + props.minor,
            'major': props.major,
            'minor': props.minor,
            'memory_gb': props.total_memory / 1024**3,
            'multi_processor_count': props.multi_processor_count,
        })
        
        return gpu_info
    except Exception:
        return None

def get_gpu_vendor(gpu_name=None):
    """根据 GPU 名称获取厂商"""
    if gpu_name is None:
        gpu_info = get_gpu_info()
        if gpu_info:
            gpu_name = gpu_info.get('name', '')
        else:
            return 'Unknown'
    
    name_lower = gpu_name.lower()
    if 'nvidia' in name_lower or 'geforce' in name_lower or 'quadro' in name_lower or 'tesla' in name_lower or 'rtx' in name_lower or 'gtx' in name_lower:
        return 'NVIDIA'
    elif 'amd' in name_lower or 'radeon' in name_lower or 'rx' in name_lower:
        return 'AMD'
    elif 'intel' in name_lower or 'iris' in name_lower or 'uhd' in name_lower or 'arc' in name_lower:
        return 'Intel'
    return 'Unknown'

def supports_amp(compute_capability):
    """检查是否支持混合精度推理"""
    return compute_capability >= 53

def supports_cudnn_benchmark(compute_capability):
    """检查是否支持 cuDNN Benchmark"""
    return compute_capability >= 60

def supports_tf32(major):
    """检查是否支持 TensorFloat32"""
    return major >= 8

def supports_bf16(major):
    """检查是否支持 BFloat16"""
    return major >= 8

def get_performance_level(compute_capability):
    """获取 GPU 性能等级"""
    if compute_capability >= 80:
        return '顶级'
    elif compute_capability >= 70:
        return '优秀'
    elif compute_capability >= 60:
        return '良好'
    elif compute_capability >= 53:
        return '一般'
    elif compute_capability >= 40:
        return '较差'
    else:
        return '不推荐'

def get_vram_status(memory_gb):
    """获取显存状态"""
    if memory_gb >= 12:
        return '充足', 'green'
    elif memory_gb >= 8:
        return '足够', 'green'
    elif memory_gb >= 6:
        return '基本够用', 'yellow'
    elif memory_gb >= 4:
        return '紧张', 'yellow'
    else:
        return '不足', 'red'
