# -*- coding: utf-8 -*-
"""
MLSharp-3D-Maker 基础测试
验证核心功能正常工作
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试核心模块导入"""
    print("[TEST] 测试模块导入...")
    try:
        import app
        print("  ✓ app.py 导入成功")
    except Exception as e:
        print(f"  ✗ app.py 导入失败: {e}")
        return False
    
    try:
        import gpu_utils
        print("  ✓ gpu_utils.py 导入成功")
    except Exception as e:
        print(f"  ✗ gpu_utils.py 导入失败: {e}")
        return False
    
    try:
        import metrics
        print("  ✓ metrics.py 导入成功")
    except Exception as e:
        print(f"  ✗ metrics.py 导入失败: {e}")
        return False
    
    try:
        from config import validators
        print("  ✓ config/validators.py 导入成功")
    except Exception as e:
        print(f"  ✗ config/validators.py 导入失败: {e}")
        return False
    
    return True

def test_config_validation():
    """测试配置验证"""
    print("\n[TEST] 测试配置验证...")
    try:
        from config.validators import (
            GPUConfigModel,
            ServerConfigModel,
            InferenceConfigModel,
            AppConfigModel
        )
        
        # 测试 GPU 配置
        gpu_config = GPUConfigModel(
            enable_amp=True,
            enable_cudnn_benchmark=True,
            enable_tf32=True,
            auto_gc_threshold=85.0
        )
        print("  ✓ GPUConfigModel 验证通过")
        
        # 测试服务器配置
        server_config = ServerConfigModel(
            host="127.0.0.1",
            port=8000
        )
        print("  ✓ ServerConfigModel 验证通过")
        
        # 测试推理配置
        inference_config = InferenceConfigModel(
            input_size=[1536, 1536]
        )
        print("  ✓ InferenceConfigModel 验证通过")
        
        return True
    except Exception as e:
        print(f"  ✗ 配置验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_detection():
    """测试 GPU 检测功能"""
    print("\n[TEST] 测试 GPU 检测...")
    try:
        import gpu_utils
        
        # 测试 GPU 厂商检测
        vendor = gpu_utils.detect_gpu_vendor_wmi()
        print(f"  ✓ GPU 厂商检测: {vendor}")
        
        # 测试 ROCm 检测
        rocm_available = gpu_utils.check_rocm_available()
        print(f"  ✓ ROCm 可用性: {rocm_available}")
        
        # 测试 GPU 信息获取
        gpu_info = gpu_utils.get_gpu_info()
        if gpu_info:
            print(f"  ✓ GPU 信息获取成功")
            print(f"    - 名称: {gpu_info.get('name', 'N/A')}")
            print(f"    - 计算能力: {gpu_info.get('compute_capability', 'N/A')}")
        else:
            print("  ⚠ 无可用 GPU，使用 CPU 模式")
        
        return True
    except Exception as e:
        print(f"  ✗ GPU 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics():
    """测试监控指标"""
    print("\n[TEST] 测试监控指标...")
    try:
        import metrics
        
        # 初始化指标管理器
        metrics_manager = metrics.init_metrics(enable_gpu=False)
        print("  ✓ MetricsManager 初始化成功")
        
        # 测试指标记录
        metrics_manager.record_http_request("GET", "/health", 200, 0.1)
        print("  ✓ HTTP 请求指标记录成功")
        
        metrics_manager.record_predict_request("success", 15.5)
        print("  ✓ 预测请求指标记录成功")
        
        metrics_manager.set_active_tasks(0)
        print("  ✓ 活跃任务数设置成功")
        
        # 获取指标
        metrics_data = metrics_manager.get_metrics()
        print(f"  ✓ 指标数据获取成功 ({len(metrics_data)} bytes)")
        
        # 停止监控
        metrics_manager.stop_monitoring()
        print("  ✓ 监控停止成功")
        
        return True
    except Exception as e:
        print(f"  ✗ 监控指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("MLSharp-3D-Maker 单元测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("配置验证", test_config_validation),
        ("GPU 检测", test_gpu_detection),
        ("监控指标", test_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] 测试 '{name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印测试结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s} {status}")
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 通过")
    print("=" * 60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())