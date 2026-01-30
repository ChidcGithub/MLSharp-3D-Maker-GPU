# -*- coding: utf-8 -*-
"""
MLSharp-3D-Maker - 监控指标模块
提供 Prometheus 兼容的性能监控指标
"""
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from prometheus_client.multiprocess import MultiProcessCollector
import time
import threading
from typing import Optional
import torch


class MetricsManager:
    """监控指标管理器"""
    
    def __init__(self, enable_gpu: bool = True):
        """
        初始化监控指标管理器
        
        Args:
            enable_gpu: 是否启用 GPU 监控
        """
        self.enable_gpu = enable_gpu
        self.registry = CollectorRegistry()
        
        # HTTP 请求计数器
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        # 请求响应时间直方图
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request latency',
            ['method', 'endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
            registry=self.registry
        )
        
        # 推理请求计数器
        self.predict_requests_total = Counter(
            'predict_requests_total',
            'Total prediction requests',
            ['status'],
            registry=self.registry
        )
        
        # 推理响应时间直方图
        self.predict_duration_seconds = Histogram(
            'predict_duration_seconds',
            'Prediction processing time',
            buckets=[5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 90.0, 120.0],
            registry=self.registry
        )
        
        # 推理各阶段耗时直方图
        self.predict_stage_duration_seconds = Histogram(
            'predict_stage_duration_seconds',
            'Prediction stage processing time',
            ['stage'],  # stages: image_load, inference, ply_save, total
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            registry=self.registry
        )
        
        # GPU 内存使用量仪表板
        self.gpu_memory_used_mb = Gauge(
            'gpu_memory_used_mb',
            'GPU memory used in MB',
            ['device_id'],
            registry=self.registry
        )
        
        # GPU 利用率仪表板
        self.gpu_utilization_percent = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['device_id'],
            registry=self.registry
        )
        
        # 活跃任务数
        self.active_tasks = Gauge(
            'active_tasks',
            'Number of active prediction tasks',
            registry=self.registry
        )
        
        # 应用信息
        self.app_info = Info(
            'app_info',
            'Application information',
            registry=self.registry
        )
        self.app_info.info({
            'version': '7.1',
            'name': 'MLSharp-3D-Maker',
            'description': '3D Gaussian Splatting Generation Tool'
        })
        
        # GPU 信息
        self.gpu_info = Gauge(
            'gpu_info',
            'GPU information',
            ['device_id', 'name', 'vendor'],
            registry=self.registry
        )
        
        # 输入尺寸信息
        self.input_size_info = Gauge(
            'input_size_info',
            'Input image size',
            registry=self.registry
        )
        
        # 初始化 GPU 监控线程
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        
        if self.enable_gpu:
            self._start_gpu_monitoring()
    
    def _start_gpu_monitoring(self):
        """启动 GPU 监控线程"""
        if not torch.cuda.is_available():
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._gpu_monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
    
    def _gpu_monitoring_loop(self):
        """GPU 监控循环"""
        import pynvml
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            while self._monitoring_active:
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # 获取 GPU 内存使用量
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_memory_used_mb.labels(device_id=str(i)).set(
                        mem_info.used / 1024 / 1024
                    )
                    
                    # 获取 GPU 利用率
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_utilization_percent.labels(device_id=str(i)).set(
                        utilization.gpu
                    )
                
                time.sleep(5)  # 每 5 秒更新一次
                
        except Exception as e:
            print(f"[WARNING] GPU monitoring failed: {e}")
            self._monitoring_active = False
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2)
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """
        记录 HTTP 请求
        
        Args:
            method: HTTP 方法
            endpoint: 端点路径
            status: HTTP 状态码
            duration: 请求耗时（秒）
        """
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_predict_request(self, status: str, duration: float):
        """
        记录预测请求
        
        Args:
            status: 请求状态 (success/error)
            duration: 请求耗时（秒）
        """
        self.predict_requests_total.labels(status=status).inc()
        self.predict_duration_seconds.observe(duration)
    
    def record_predict_stage(self, stage: str, duration: float):
        """
        记录预测各阶段耗时
        
        Args:
            stage: 阶段名称 (image_load, inference, ply_save, total)
            duration: 阶段耗时（秒）
        """
        self.predict_stage_duration_seconds.labels(stage=stage).observe(duration)
    
    def set_active_tasks(self, count: int):
        """
        设置活跃任务数
        
        Args:
            count: 活跃任务数量
        """
        self.active_tasks.set(count)
    
    def set_input_size(self, width: int, height: int):
        """
        设置输入尺寸信息
        
        Args:
            width: 输入宽度
            height: 输入高度
        """
        self.input_size_info.set(width * height)
    
    def set_gpu_info(self, device_id: int, name: str, vendor: str):
        """
        设置 GPU 信息
        
        Args:
            device_id: 设备 ID
            name: GPU 名称
            vendor: 厂商名称
        """
        self.gpu_info.labels(
            device_id=str(device_id),
            name=name,
            vendor=vendor
        ).set(1)
    
    def get_metrics(self) -> bytes:
        """
        获取 Prometheus 格式的指标数据
        
        Returns:
            Prometheus 格式的指标数据
        """
        return generate_latest(self.registry)


# 全局指标管理器实例
_metrics_manager: Optional[MetricsManager] = None


def get_metrics_manager() -> MetricsManager:
    """获取全局指标管理器实例"""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager


def init_metrics(enable_gpu: bool = True) -> MetricsManager:
    """
    初始化指标管理器
    
    Args:
        enable_gpu: 是否启用 GPU 监控
        
    Returns:
        指标管理器实例
    """
    global _metrics_manager
    _metrics_manager = MetricsManager(enable_gpu=enable_gpu)
    return _metrics_manager