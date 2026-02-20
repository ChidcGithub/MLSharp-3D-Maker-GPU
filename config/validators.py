# -*- coding: utf-8 -*-
"""
配置验证模块
使用 Pydantic 进行配置验证
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator, constr


# ================= GPU 配置验证 =================

class GPUConfigModel(BaseModel):
    """GPU 配置模型"""
    
    enable_amp: bool = True
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True
    enable_auto_gc: bool = True
    auto_gc_interval: int = Field(default=30, ge=1, le=300)
    auto_gc_threshold: float = Field(default=85.0, ge=0.0, le=100.0)
    enable_smart_reclaim: bool = True
    
    @field_validator('auto_gc_threshold')
    @classmethod
    def validate_threshold(cls, v):
        if v < 0 or v > 100:
            raise ValueError('auto_gc_threshold 必须在 0-100 之间')
        return v


# ================= 服务器配置验证 =================

class ServerConfigModel(BaseModel):
    """服务器配置模型"""
    
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    
    @field_validator('host')
    @classmethod
    def validate_host(cls, v):
        if not v:
            raise ValueError('host 不能为空')
        return v


# ================= 浏览器配置验证 =================

class BrowserConfigModel(BaseModel):
    """浏览器配置模型"""
    
    auto_open: bool = True


# ================= 日志配置验证 =================

class LoggingConfigModel(BaseModel):
    """日志配置模型"""
    
    level: str = Field(default="INFO", pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    console: bool = True
    file: bool = False


# ================= 模型配置验证 =================

class ModelConfigModel(BaseModel):
    """模型配置模型"""
    
    checkpoint: str = Field(..., min_length=1)
    temp_dir: str = Field(..., min_length=1)
    
    @field_validator('checkpoint')
    @classmethod
    def validate_checkpoint(cls, v):
        if not v:
            raise ValueError('checkpoint 不能为空')
        return v
    
    @field_validator('temp_dir')
    @classmethod
    def validate_temp_dir(cls, v):
        if not v:
            raise ValueError('temp_dir 不能为空')
        return v


# ================= 推理配置验证 =================

class InferenceConfigModel(BaseModel):
    """推理配置模型"""
    
    input_size: List[int] = Field(default=[1536, 1536], min_items=2, max_items=2)
    
    @field_validator('input_size')
    @classmethod
    def validate_input_size(cls, v):
        if len(v) != 2:
            raise ValueError('input_size 必须包含 2 个元素')
        width, height = v
        
        # 检查是否为正方形
        if width != height:
            raise ValueError(f'input_size 必须为正方形，当前为 {width}x{height}')
        
        # 检查是否能被 64 整除
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError(f'input_size 必须能被 64 整除，当前为 {width}x{height}')
        
        # 检查最大尺寸
        max_size = 1536
        if width > max_size or height > max_size:
            raise ValueError(f'input_size 不能超过 {max_size}x{max_size}，当前为 {width}x{height}')
        
        return v


# ================= 优化配置验证 =================

class OptimizationConfigModel(BaseModel):
    """优化配置模型"""
    
    gradient_checkpointing: bool = False
    checkpoint_segments: int = Field(default=3, ge=1, le=10)


# ================= 缓存配置验证 =================

class CacheConfigModel(BaseModel):
    """缓存配置模型"""
    
    enabled: bool = True
    size: int = Field(default=100, ge=1, le=1000)


# ================= Redis 配置验证 =================

class RedisConfigModel(BaseModel):
    """Redis 配置模型"""
    
    enabled: bool = False
    url: Optional[str] = Field(default=None, min_length=1)
    prefix: str = Field(default="mlsharp", min_length=1)
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v, info):
        if info.data.get('enabled') and not v:
            raise ValueError('Redis 启用时必须提供 url')
        return v


# ================= Webhook 配置验证 =================

class WebhookConfigModel(BaseModel):
    """Webhook 配置模型"""
    
    enabled: bool = False
    task_completed: Optional[str] = Field(default=None, min_length=1)
    task_failed: Optional[str] = Field(default=None, min_length=1)


# ================= 监控配置验证 =================

class MonitoringConfigModel(BaseModel):
    """监控配置模型"""
    
    enabled: bool = True
    enable_gpu: bool = True
    metrics_path: str = Field(default="/metrics", min_length=1)


# ================= 性能配置验证 =================

class PerformanceConfigModel(BaseModel):
    """性能配置模型"""
    
    max_workers: int = Field(default=4, ge=1, le=16)
    max_concurrency: int = Field(default=10, ge=1, le=100)
    timeout_keep_alive: int = Field(default=30, ge=1, le=300)
    max_requests: int = Field(default=1000, ge=1)


# ================= 性能缓存配置验证 =================

class PerformanceCacheConfigModel(BaseModel):
    """性能缓存配置模型"""
    
    last_test: Optional[str] = None
    best_config: Optional[Dict[str, Any]] = None
    gpu: Optional[Dict[str, Any]] = None


# ================= 完整配置模型 =================

class AppConfigModel(BaseModel):
    """完整应用配置模型"""
    
    server: ServerConfigModel = ServerConfigModel()
    mode: str = Field(default="auto", pattern=r'^(auto|gpu|cpu|nvidia|amd)$')
    browser: BrowserConfigModel = BrowserConfigModel()
    gpu: GPUConfigModel = GPUConfigModel()
    logging: LoggingConfigModel = LoggingConfigModel()
    model: ModelConfigModel
    inference: InferenceConfigModel = InferenceConfigModel()
    optimization: OptimizationConfigModel = OptimizationConfigModel()
    cache: CacheConfigModel = CacheConfigModel()
    redis: RedisConfigModel = RedisConfigModel()
    webhook: WebhookConfigModel = WebhookConfigModel()
    monitoring: MonitoringConfigModel = MonitoringConfigModel()
    performance: PerformanceConfigModel = PerformanceConfigModel()
    performance_cache: PerformanceCacheConfigModel = PerformanceCacheConfigModel()
    
    model_config = {"extra": "forbid"}  # 禁止额外字段


# ================= 配置验证函数 =================

def validate_config(config_dict: Dict[str, Any]) -> AppConfigModel:
    """
    验证配置字典
    
    Args:
        config_dict: 配置字典
        
    Returns:
        验证后的配置模型
        
    Raises:
        ValidationError: 配置验证失败
    """
    return AppConfigModel(**config_dict)


def validate_gpu_config(config_dict: Dict[str, Any]) -> GPUConfigModel:
    """
    验证 GPU 配置
    
    Args:
        config_dict: GPU 配置字典
        
    Returns:
        验证后的 GPU 配置模型
    """
    return GPUConfigModel(**config_dict)


def validate_inference_config(config_dict: Dict[str, Any]) -> InferenceConfigModel:
    """
    验证推理配置
    
    Args:
        config_dict: 推理配置字典
        
    Returns:
        验证后的推理配置模型
    """
    return InferenceConfigModel(**config_dict)


def validate_server_config(config_dict: Dict[str, Any]) -> ServerConfigModel:
    """
    验证服务器配置
    
    Args:
        config_dict: 服务器配置字典
        
    Returns:
        验证后的服务器配置模型
    """
    return ServerConfigModel(**config_dict)