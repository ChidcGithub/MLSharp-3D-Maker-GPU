# -*- coding: utf-8 -*-
"""
国际化（i18n）模块
提供多语言支持功能
"""
import os
from typing import Dict, Optional


class I18n:
    """国际化类"""
    
    def __init__(self, default_lang: str = 'zh'):
        """
        初始化国际化类
        
        Args:
            default_lang: 默认语言 ('zh' 或 'en')
        """
        self.default_lang = default_lang
        self.current_lang = default_lang
        self.translations = {
            'zh': {
                # 错误消息
                'model_file_not_exists': '模型文件不存在!',
                'model_file_solution': '请确保模型文件位于: {}\n下载地址: 请查看项目 README 或联系开发者',
                'model_size_warning': '模型文件大小异常(太小),可能已损坏或不完整\n建议: 重新下载模型文件',
                'sharp_import_error': 'Sharp 模块导入失败: {}',
                'sharp_import_solution': '''可能的原因:
1. Sharp 库未安装
2. 模型文件路径错误
3. Python 环境配置不正确

解决方案:
- 检查 model_assets/ 文件夹是否存在
- 重新安装依赖: pip install -r requirements.txt
- 确保使用正确的 Python 环境''',
                'model_load_error': '模型加载失败: {}',
                'model_load_solution': '''请检查:
1. 模型文件是否完整
2. PyTorch 版本是否兼容
3. 是否有足够的内存/显存
4. Python 环境是否正确配置''',
                'gradient_checkpointing_success': '梯度检查点已启用（显存占用将减少，但推理速度可能略微降低）',
                'gpu_memory_insufficient': '显存不足,请使用较小的图片',
                'gpu_memory_solution': '建议使用小于 1024x1024 的图片,或关闭其他占用显存的程序',
                'processing_failed': '处理失败: {}',
                'processing_solution': '请尝试重新启动程序或使用较小的图片',
                
                # 日志消息
                'model_loading': '模型加载',
                'model_file': '模型文件: {}',
                'model_file_size': '模型文件大小: {:.2f} MB',
                'creating_predictor': '正在创建预测器...',
                'loading_model_weights': '正在加载模型权重...',
                'loading_weights_to_predictor': '正在加载权重到预测器...',
                'model_load_success': '模型加载完成!',
                'device_info': '设备: {}',
                'applying_gradient_checkpointing': '正在应用梯度检查点...',
                'gpu_memory_usage': '显存占用: {:.2f} MB',
                'gpu_acceleration_enabled': 'GPU 加速已启用',
                'gpu_vendor': 'GPU 厂商: {}',
                'gpu_model': 'GPU 型号: {}',
                'compute_capability': '计算能力: {}',
                'amp_enabled': '[优化] 混合精度推理: 已启用',
                'amp_disabled': '[优化] 混合精度推理: 已禁用',
                'cudnn_enabled': '[优化] cuDNN Benchmark: 已启用',
                'cudnn_disabled': '[优化] cuDNN Benchmark: 已禁用',
                'tf32_enabled': '[优化] TensorFloat32: 已启用',
                'tf32_disabled': '[优化] TensorFloat32: 已禁用',
                'cpu_mode': '使用 CPU 模式',
                'cpu_cores': 'CPU 核心数: {}',
                'multithreading_enabled': '[优化] 多线程优化: 已启用',
                'cache_enabled': '[缓存] 推理缓存: 已启用（最大 {} 条）',
                'cache_disabled': '[缓存] 推理缓存: 已禁用',
                'service_address': '服务地址: {}',
                'auto_browser': '浏览器将自动打开...',
                'press_ctrl_c': '按 Ctrl+C 停止服务',
                'auto_open_failed': '无法自动打开浏览器: {}',
                'manual_access': '请手动访问: {}',
                'cleanup_resources': '正在清理资源...',
                'gpu_monitoring_stopped': 'GPU 监控已停止',
                'gpu_auto_gc_stopped': 'GPU 自动内存监控已停止',
                'webhook_client_closed': 'Webhook 客户端已关闭',
                'webhook_close_failed': '关闭 Webhook 客户端失败: {}',
                'cleanup_complete': '资源清理完成',
                'service_stopped': '服务已停止',
                'thank_you': '感谢使用 MLSharp!',
                'service_start_failed': '服务启动失败: {}',
                'input_size_mismatch': '输入尺寸宽度和高度不相等 ({}x{}),模型使用正方形输入',
                'input_size_adjusted': '已调整为 {}x{}',
                'input_size_exceeds_max': '输入尺寸 {}x{} 超过最大支持尺寸 {}x{}',
                'patch_splitting_error': 'SPN 编码器在更大尺寸下会出现补丁分割错误',
                'input_size_not_divisible': '输入尺寸 {}x{} 不能被 64 整除',
                'adjusted_to': '已调整为 {}x{}',
                'input_size_still_exceeds': '调整后的尺寸 {}x{} 仍然超过最大支持尺寸',
                'gpu_init': 'GPU 初始化',
                'user_specified_mode': '用户指定启动模式: {}',
                'auto_detect_mode': '自动检测模式',
                'force_cpu_mode': '强制使用 CPU 模式',
                'amd_gpu_detected': '检测到 AMD GPU: {}',
                'nvidia_gpu_detected': '检测到 NVIDIA GPU: {}',
                'intel_gpu_detected': '检测到 Intel GPU: {}',
                'unknown_gpu_detected': '检测到未知 GPU: {}',
                'cuda_version': 'CUDA/ROCm 版本: {}',
                'gpu_count': 'GPU 数量: {}',
                'force_nvidia_mode': '强制使用 NVIDIA 模式，但检测到 {} GPU',
                'set_nvidia_mode': '已强制设置为 NVIDIA 模式',
                'force_amd_mode': '强制使用 AMD 模式，但检测到 {} GPU',
                'set_amd_mode': '已强制设置为 AMD 模式',
                'compute_capability_info': '计算能力: {}',
                'gpu_memory_info': '显存: {:.2f} GB',
                'low_vram_warning': '警告: 显存不足 4GB,可能导致性能问题',
                'wmi_detection_failed': 'WMI 检测失败: {}',
                'rocm_detection_failed': 'ROCm 检测失败: {}',
                'cuda_unavailable_cpu_mode': '使用 CPU 模式\n   原因: CUDA/ROCm 不可用',
                'amd_no_rocm': '检测到 AMD 显卡,但 PyTorch 未编译 ROCm 支持\n   解决方案: 安装 ROCm 版本的 PyTorch',
                'nvidia_no_cuda': '检测到 NVIDIA 显卡,但 CUDA 不可用\n   请检查:\n     1. 是否安装 NVIDIA 显卡驱动\n     2. 显卡是否支持 CUDA\n     3. PyTorch 是否编译了 CUDA 支持',
                'intel_gpu': '检测到 Intel 显卡\n   Intel GPU 暂不支持 GPU 加速',
                'no_supported_gpu': '未检测到支持的 GPU',
                'gpu_memory_query_failed': '获取 GPU 内存信息失败: {}',
                'gpu_cache_cleared': 'GPU 缓存已清理 - 设备 {}',
                'gpu_cache_freed': '  释放显存: {:.2f} MB',
                'gpu_cache_clear_failed': '清理 GPU 缓存失败: {}',
                'gpu_garbage_collected': 'GPU 垃圾回收完成 - 设备 {}',
                'gpu_memory_recovered': '  回收显存: {:.2f} MB',
                'gpu_gc_failed': 'GPU 垃圾回收失败: {}',
                'gpu_memory_high_usage': 'GPU 显存使用率过高: {:.1f}%',
                'gpu_memory_total': '  总显存: {:.2f} MB',
                'gpu_memory_used': '  已用: {:.2f} MB',
                'gpu_memory_free': '  可用: {:.2f} MB',
                'gpu_perform_recovery': '  执行智能内存回收...',
                'smart_memory_recovery_failed': '智能内存回收失败: {}',
                'gpu_monitoring_not_running': 'GPU 不可用，跳过自动内存监控',
                'gpu_monitor_running': '自动内存监控已在运行',
                'gpu_monitor_started': 'GPU 自动内存监控已启动 (间隔: {}s, 阈值: {}%)',
                'gpu_monitor_exception': '内存监控异常: {}',
                'gpu_monitor_stopped': 'GPU 自动内存监控已停止',
                'cache_cleared': '缓存已清空',
                'cache_status': '缓存状态: {}',
                'cache_size_info': '缓存大小: {}/{}',
                'cache_hits': '命中次数: {}',
                'cache_misses': '未命中次数: {}',
                'cache_hit_rate': '命中率: {:.1f}%',
                'redis_connected': 'Redis 缓存已连接: {}',
                'redis_not_installed': 'redis 模块未安装，Redis 缓存将不可用',
                'redis_install_cmd': '安装命令: pip install redis',
                'redis_connection_failed': 'Redis 连接失败: {}',
                'redis_unavailable': 'Redis 缓存将不可用，使用本地缓存代替',
                'redis_cache_cleared': 'Redis 缓存已清空: {} 个键',
                'redis_cache_empty': 'Redis 缓存为空',
                'webhook_client_init': 'Webhook 客户端已初始化',
                'webhook_not_installed': 'httpx 模块未安装，Webhook 功能将不可用',
                'webhook_install_cmd': '安装命令: pip install httpx',
                'webhook_disabled': 'Webhook 未启用，无法注册',
                'webhook_registered': 'Webhook 已注册: {} -> {}',
                'webhook_unregistered': 'Webhook 已注销: {}',
                'webhook_sent_success': 'Webhook 发送成功: {} -> {}',
                'webhook_sent_failed': 'Webhook 发送失败: {} -> {} (状态码: {})',
                'cache_hit_info': '缓存命中! PLY保存完成,耗时: {:.2f}s',
                'task_completion_time': '处理完成（缓存）,总耗时: {:.2f}秒',
                'prediction_start': '开始推理...',
                'prediction_complete': '推理完成,耗时: {:.2f}秒',
                'result_cached_to_redis': '结果已缓存到 Redis',
                'ply_saved': 'PLY保存完成,耗时: {:.2f}s',
                'task_complete': '处理完成,总耗时: {:.2f}秒',
                'rename_failed': '重命名文件失败: {}',
                
                # 系统信息
                'system_info': '系统信息',
                'os_info': '操作系统: {} {}',
                'python_version': 'Python 版本: {}',
                'working_directory': '工作目录: {}',
                'web_service': 'Web 服务',
                'input_size_info': '输入尺寸: {}x{}',
                
                # 性能调优
                'performance_tuning': '性能自动调优',
                'testing_configurations': '正在测试不同优化配置...',
                'test_config': '测试配置: {}',
                'config_description': '  描述: {}',
                'avg_inference_time': '  平均推理时间: {:.3f} 秒',
                'test_failed': '  测试失败: {}',
                'tuning_results': '调优结果',
                'best_config': '最优配置: {}',
                'best_config_desc': '  描述: {}',
                'avg_inference_time_result': '  平均推理时间: {:.3f} 秒',
                'throughput_info': '  吞吐量: {:.2f} FPS',
                'tuning_complete': '性能自动调优完成！',
                'config_applied': '已应用最优配置',
                'tuning_failed_default': '所有配置测试失败，使用默认配置',
                'tuning_failed': '性能自动调优失败: {}',
                'use_default_config': '使用默认配置',
                'valid_cache_found': '发现有效的性能调优缓存（{} 天前）',
                'using_cached_config': '使用缓存的性能配置',
                'config_name': '配置名称: {}',
                'config_description_cached': '描述: {}',
                
                # API响应
                'api_cache_cleared': '缓存已清空',
                'webhook_not_enabled': 'Webhook 未启用',
                'missing_params': '缺少必要参数: event_type 和 url',
                'webhook_registered_resp': 'Webhook 已注册: {} -> {}',
                
                # 其他
                'amp_only_optimization': '仅启用混合精度推理',
                'cudnn_only_optimization': '仅启用 cuDNN 自动调优',
                'tf32_only_optimization': '仅启用 TensorFloat32',
                'amp_cudnn_optimization': 'AMP 和 cuDNN 自动调优',
                'amp_tf32_optimization': 'AMP 和 TensorFloat32',
                'all_optimizations': '启用所有优化',
                
                # 进度和状态
                'task_processing': '[Task {}] 已保存文件: {}',
                'task_image_info': '[Task {}] 图像信息: {}x{}, 焦距: {} (加载耗时: {:.2f}s)',
                'task_large_image': '[Task {}] 图片尺寸过大 ({}x{}),可能导致性能问题',
                'task_cache_hit': '[Task {}] 缓存命中! PLY保存完成,耗时: {:.2f}s',
                'task_cache_complete': '[Task {}] 处理完成（缓存）,总耗时: {:.2f}秒',
                'task_start_inference': '[Task {}] 开始推理...',
                'task_inference_complete': '[Task {}] 推理完成,耗时: {:.2f}秒',
                'task_redis_cache': '[Task {}] 结果已缓存到 Redis',
                'task_ply_save_complete': '[Task {}] PLY保存完成,耗时: {:.2f}s',
                'task_complete_total': '[Task {}] 处理完成,总耗时: {:.2f}秒',
                'task_vram_insufficient': '[Task {}] 显存不足: {}',
                'task_processing_failed': '[Task {}] 处理失败: {}',
                
                # 配置验证错误
                'input_size_error': 'input_size 必须包含 2 个元素',
                'input_size_square_error': 'input_size 必须为正方形，当前为 {}x{}',
                'input_size_divisible_error': 'input_size 必须能被 64 整除，当前为 {}x{}',
                'input_size_max_error': 'input_size 不能超过 {}x{}，当前为 {}x{}',
                'threshold_range_error': 'auto_gc_threshold 必须在 0-100 之间',
                'host_empty_error': 'host 不能为空',
                'checkpoint_empty_error': 'checkpoint 不能为空',
                'temp_dir_empty_error': 'temp_dir 不能为空',
                'redis_url_required': 'Redis 启用时必须提供 url',
                
                # 统计信息
                'points_label': 'Points',
                'fps_label': 'FPS',
                'stat_points': '点云数',
                'stat_fps': '帧率',
                'ctrl_scale': '缩放',
                'ctrl_opacity': '透明度',
                'ctrl_exposure': '曝光',
                'ctrl_speed': '速度',
                'flow_amp': '强度',
                'flow_freq': '频率',
                'flow_speed': '流速',
                'import_main': '导入模型',
                'import_sub': '拖拽 .PLY 文件',
                'nav_move': '移动',
                'nav_lift': '升降',
                'nav_fast': '加速',
                'nav_look': '视角',
                'loading_init': '系统初始化中',
                'tutorial_text': '[ WASD ] 飞行与探索',
                'loading_processing': '正在处理作品',
                'loading_downloading': '正在下载几何数据',
                'loading_parsing': '数据解析中...',
                'loading_generating': '生成演示模型',
                'loading_importing': '正在导入模型',
                'loading_download_general': '下载中...',
                'btn_flow_tooltip': '散开 / 处理 (湍流效果)',
                'btn_download_tooltip': '下载模型 (.ply)',
                
                # 英文标签（在中文环境中保持一致性）
                'subtitle_en': 'HIGH FIDELITY VIEWER .01',
                'scale_en': 'Scale',
                'opacity_en': 'Opacity',
                'exposure_en': 'Exposure',
                'speed_en': 'Speed',
                'amp_en': 'Amp',
                'freq_en': 'Freq',
                'flow_speed_en': 'Flow Spd',
                'import_main_en': 'Import',
                'import_sub_en': 'Drag & Drop Image/.PLY',
                'move_en': 'Move',
                'lift_en': 'Lift',
                'fast_en': 'Fast',
                'look_en': 'Look',
                'init_en': 'INITIALIZING SYSTEM',
                'tutorial_en': '[ WASD ] TO FLY & COLLECT',
                
                # 新增：配置加载相关
                'load_config_failed': '加载配置文件失败: {}',
                'use_default_config_and_args': '使用默认配置和命令行参数',
                
                # 新增：日志标题
                'gpu_init_title': 'GPU 初始化',
                'cache_stats_title': '缓存统计',
                'cached_config_title': '使用缓存的性能配置',
                'performance_tuning_title': '性能自动调优',
                'tuning_results_title': '调优结果',
                'model_loading_title': '模型加载',
                'system_info_title': '系统信息',
                'web_service_title': 'Web 服务',
                'service_stopped_title': '服务已停止',
                
                # 新增：用户指定模式
                'user_specified_mode_info': '用户指定启动模式: {}',
                
                # 新增：CUDA/ROCm版本
                'cuda_version_info': '   CUDA/ROCm 版本: {}',
                
                # 新增：GPU数量
                'gpu_count_info': '   GPU 数量: {}',
                
                # 新增：计算能力
                'compute_capability_info_full': '   计算能力: {}',
                
                # 新增：GPU内存
                'gpu_memory_info_full': '   显存: {:.2f} GB',
                
                # 新增：GC间隔和阈值
                'gc_interval_info': '  检查间隔: {} 秒',
                'gc_threshold_info': '  清理阈值: {}%',
                
                # 新增：默认配置文件
                'using_default_config_file': '使用默认配置文件: {}',
                
                # 新增：显存释放
                'vram_freed_info': '  释放显存: {:.2f} MB',
                
                # 新增：显存回收
                'vram_recovered_info': '  回收显存: {:.2f} MB',
                
                # 新增：显存统计
                'vram_total_info': '  总显存: {:.2f} MB',
                'vram_used_info': '  已用: {:.2f} MB',
                'vram_free_info': '  可用: {:.2f} MB',
                
                # 新增：监控启动
                'gpu_monitoring_started': 'GPU 自动内存监控已启动 (间隔: {}s, 阈值: {}%)',
                
                # 新增：缓存状态
                'cache_enabled_status': '缓存状态: 已启用',
                'cache_disabled_status': '缓存状态: 已禁用',
                'cache_size_info_full': '缓存大小: {}/{}',
                'cache_hits_info': '命中次数: {}',
                'cache_misses_info': '未命中次数: {}',
                'cache_hit_rate_info': '命中率: {:.1f}%',
                
                # 新增：Redis连接
                'redis_connected_info': 'Redis 缓存已连接: {}',
                
                # 新增：Redis清除
                'redis_cleared_info': 'Redis 缓存已清空: {} 个键',
                
                # 新增：Webhook
                'webhook_registered_info': 'Webhook 已注册: {} -> {}',
                'webhook_unregistered_info': 'Webhook 已注销: {}',
                'webhook_sent_info': 'Webhook 发送成功: {} -> {}',
                
                # 新增：调优缓存
                'tuning_cache_found': '发现有效的性能调优缓存（{} 天前）',
                
                # 新增：配置目录
                'config_dir_created': '已创建配置目录: {}',
                
                # 新增：配置文件更新
                'config_file_updated': '配置文件已存在，更新性能调优缓存: {}',
                'config_file_created': '配置文件不存在，自动创建新配置文件: {}',
                
                # 新增：最佳配置
                'best_config_name': '配置名称: {}',
                'best_config_desc_info': '描述: {}',
                
                # 新增：测试配置
                'test_config_info': '测试配置: {}',
                'test_config_desc': '  描述: {}',
                'avg_inference_time_info': '  平均推理时间: {:.3f} 秒',
                
                # 新增：吞吐量
                'throughput_info_full': '  吞吐量: {:.2f} FPS',
                
                # 新增：测试运行
                'test_run_info': '  运行 {}/{}: {:.3f} 秒',
                
                # 新增：模型文件
                'model_file_info': '模型文件: {}',
                
                # 新增：设备信息
                'device_info_full': '设备: {}',
                
                # 新增：显存使用
                'vram_usage_info': '显存占用: {:.2f} MB',
                
                # 新增：任务处理
                'task_file_saved': '[Task {}] 已保存文件: {}',
                
                # 新增：任务图像信息
                'task_image_info_full': '[Task {}] 图像信息: {}x{}, 焦距: {} (加载耗时: {:.2f}s)',
                
                # 新增：任务缓存命中
                'task_cache_hit_info': '[Task {}] 缓存命中! PLY保存完成,耗时: {:.2f}s',
                
                # 新增：任务完成
                'task_completion_info': '[Task {}] 处理完成（缓存）,总耗时: {:.2f}秒',
                
                # 新增：任务推理
                'task_inference_start': '[Task {}] 开始推理...',
                'task_inference_complete_info': '[Task {}] 推理完成,耗时: {:.2f}秒',
                'task_redis_cache_info': '[Task {}] 结果已缓存到 Redis',
                'task_ply_save_info': '[Task {}] PLY保存完成,耗时: {:.2f}s',
                'task_complete_info': '[Task {}] 处理完成,总耗时: {:.2f}秒',
                
                # 新增：系统信息
                'os_info_full': '操作系统: {} {}',
                'python_version_info': 'Python 版本: {}',
                'working_directory_info': '工作目录: {}',
                'input_size_info_full': '输入尺寸: {}x{}',
                
                # 新增：GPU厂商/型号
                'gpu_vendor_info': 'GPU 厂商: {}',
                'gpu_model_info': 'GPU 型号: {}',
                
                # 新增：优化状态
                'amp_enabled_info': '[优化] 混合精度推理: 已启用',
                'amp_disabled_info': '[优化] 混合精度推理: 已禁用',
                'cudnn_enabled_info': '[优化] cuDNN Benchmark: 已启用',
                'cudnn_disabled_info': '[优化] cuDNN Benchmark: 已禁用',
                'tf32_enabled_info': '[优化] TensorFloat32: 已启用',
                'tf32_disabled_info': '[优化] TensorFloat32: 已禁用',
                
                # 新增：CPU核心
                'cpu_cores_info': 'CPU 核心数: {}',
                
                # 新增：缓存禁用
                'cache_disabled_info': '[缓存] 推理缓存: 已禁用',
                
                # 新增：手动访问
                'manual_access_info': '请手动访问: {}',
                
                # 新增：CPU优化
                'cpu_optimization_enabled': 'CPU 优化已启用({} 核心)',
                
                # 新增：自动GC状态
                'auto_gc_enabled': '  自动垃圾回收: 已启用',
                'auto_gc_disabled': '  自动垃圾回收: 已禁用',
                
                # 新增：cuDNN状态
                'cudnn_enabled_status': '  cuDNN Benchmark: 已启用',
                'cudnn_disabled_capability': '  cuDNN Benchmark: 已禁用(显卡计算能力不足)',
                
                # 新增：TF32状态
                'tf32_enabled_status': '  TensorFloat32: 已启用',
                'tf32_disabled_support': '  TensorFloat32: 已禁用(显卡不支持)',
                
                # 新增：AMP状态
                'amp_enabled_status': '  混合精度推理 (AMP): 已启用',
                'amp_disabled_capability': '  混合精度推理 (AMP): 已禁用(显卡计算能力不足)',
                
                # 新增：监控运行
                'monitoring_already_running': '自动内存监控已在运行',
                
                # 新增：测试运行失败
                'test_run_failed': '  运行 {}/{} 失败: {}',
                
                # 新增：梯度检查点成功
                'grad_checkpoint_enabled': '梯度检查点已启用（显存占用将减少，但推理速度可能略微降低）',
                
                # 新增：Redis缓存启用
                'redis_cache_enabled': 'Redis 缓存已启用: {}',
                
                # 新增：Webhook启用
                'webhook_enabled': 'Webhook 通知已启用',
                
                # 新增：GPU加速启用
                'gpu_accel_enabled': 'GPU 加速已启用',
                
                # 新增：多线程优化
                'multithreading_opt_enabled': '[优化] 多线程优化: 已启用',
                
                # 新增：推理缓存
                'inference_cache_enabled': '[缓存] 推理缓存: 已启用（最大 {} 条）',
            },
            'en': {
                # Error messages
                'model_file_not_exists': 'Model file does not exist!',
                'model_file_solution': 'Please ensure model file is located at: {}\nDownload URL: Please check project README or contact developer',
                'model_size_warning': 'Model file size abnormal (too small), may be corrupted or incomplete\nSuggestion: Re-download model file',
                'sharp_import_error': 'Sharp module import failed: {}',
                'sharp_import_solution': '''Possible reasons:
1. Sharp library not installed
2. Model file path error
3. Python environment configuration incorrect

Solutions:
- Check if model_assets/ folder exists
- Reinstall dependencies: pip install -r requirements.txt
- Ensure correct Python environment is used''',
                'model_load_error': 'Model loading failed: {}',
                'model_load_solution': '''Please check:
1. Model file integrity
2. PyTorch version compatibility
3. Sufficient memory/VRAM available
4. Python environment properly configured''',
                'gradient_checkpointing_success': 'Gradient checkpointing enabled (VRAM usage will reduce, but inference speed may slightly decrease)',
                'gpu_memory_insufficient': 'Insufficient VRAM, please use smaller images',
                'gpu_memory_solution': 'Suggested to use images smaller than 1024x1024, or close other VRAM-consuming programs',
                'processing_failed': 'Processing failed: {}',
                'processing_solution': 'Please try restarting the program or use smaller images',
                
                # Log messages
                'model_loading': 'Model Loading',
                'model_file': 'Model file: {}',
                'model_file_size': 'Model file size: {:.2f} MB',
                'creating_predictor': 'Creating predictor...',
                'loading_model_weights': 'Loading model weights...',
                'loading_weights_to_predictor': 'Loading weights to predictor...',
                'model_load_success': 'Model loaded successfully!',
                'device_info': 'Device: {}',
                'applying_gradient_checkpointing': 'Applying gradient checkpointing...',
                'gpu_memory_usage': 'VRAM usage: {:.2f} MB',
                'gpu_acceleration_enabled': 'GPU acceleration enabled',
                'gpu_vendor': 'GPU vendor: {}',
                'gpu_model': 'GPU model: {}',
                'compute_capability': 'Compute capability: {}',
                'amp_enabled': '[Optimization] Mixed Precision Inference: Enabled',
                'amp_disabled': '[Optimization] Mixed Precision Inference: Disabled',
                'cudnn_enabled': '[Optimization] cuDNN Benchmark: Enabled',
                'cudnn_disabled': '[Optimization] cuDNN Benchmark: Disabled',
                'tf32_enabled': '[Optimization] TensorFloat32: Enabled',
                'tf32_disabled': '[Optimization] TensorFloat32: Disabled',
                'cpu_mode': 'Using CPU mode',
                'cpu_cores': 'CPU cores: {}',
                'multithreading_enabled': '[Optimization] Multithreading optimization: Enabled',
                'cache_enabled': '[Cache] Inference cache: Enabled (max {} entries)',
                'cache_disabled': '[Cache] Inference cache: Disabled',
                'service_address': 'Service address: {}',
                'auto_browser': 'Browser will auto-open...',
                'press_ctrl_c': 'Press Ctrl+C to stop service',
                'auto_open_failed': 'Unable to auto-open browser: {}',
                'manual_access': 'Please access manually: {}',
                'cleanup_resources': 'Cleaning up resources...',
                'gpu_monitoring_stopped': 'GPU monitoring stopped',
                'gpu_auto_gc_stopped': 'GPU auto memory monitoring stopped',
                'webhook_client_closed': 'Webhook client closed',
                'webhook_close_failed': 'Failed to close Webhook client: {}',
                'cleanup_complete': 'Resource cleanup complete',
                'service_stopped': 'Service stopped',
                'thank_you': 'Thank you for using MLSharp!',
                'service_start_failed': 'Service startup failed: {}',
                'input_size_mismatch': 'Input width and height not equal ({}x{}), model uses square input',
                'input_size_adjusted': 'Adjusted to {}x{}',
                'input_size_exceeds_max': 'Input size {}x{} exceeds maximum supported size {}x{}',
                'patch_splitting_error': 'SPN encoder will have patch splitting errors with larger sizes',
                'input_size_not_divisible': 'Input size {}x{} not divisible by 64',
                'adjusted_to': 'Adjusted to {}x{}',
                'input_size_still_exceeds': 'Adjusted size {}x{} still exceeds maximum supported size',
                'gpu_init': 'GPU Initialization',
                'user_specified_mode': 'User specified startup mode: {}',
                'auto_detect_mode': 'Auto-detect mode',
                'force_cpu_mode': 'Force CPU mode',
                'amd_gpu_detected': 'Detected AMD GPU: {}',
                'nvidia_gpu_detected': 'Detected NVIDIA GPU: {}',
                'intel_gpu_detected': 'Detected Intel GPU: {}',
                'unknown_gpu_detected': 'Detected Unknown GPU: {}',
                'cuda_version': 'CUDA/ROCm version: {}',
                'gpu_count': 'GPU count: {}',
                'force_nvidia_mode': 'Force NVIDIA mode, but detected {} GPU',
                'set_nvidia_mode': 'Forced NVIDIA mode set',
                'force_amd_mode': 'Force AMD mode, but detected {} GPU',
                'set_amd_mode': 'Forced AMD mode set',
                'compute_capability_info': 'Compute capability: {}',
                'gpu_memory_info': 'VRAM: {:.2f} GB',
                'low_vram_warning': 'Warning: VRAM less than 4GB, may cause performance issues',
                'wmi_detection_failed': 'WMI detection failed: {}',
                'rocm_detection_failed': 'ROCm detection failed: {}',
                'cuda_unavailable_cpu_mode': 'Using CPU mode\n   Reason: CUDA/ROCm unavailable',
                'amd_no_rocm': 'Detected AMD GPU, but PyTorch not compiled with ROCm support\n   Solution: Install ROCm version of PyTorch',
                'nvidia_no_cuda': 'Detected NVIDIA GPU, but CUDA unavailable\n   Please check:\n     1. NVIDIA GPU driver installed\n     2. GPU supports CUDA\n     3. PyTorch compiled with CUDA support',
                'intel_gpu': 'Detected Intel GPU\n   Intel GPU does not support GPU acceleration currently',
                'no_supported_gpu': 'No supported GPU detected',
                'gpu_memory_query_failed': 'Failed to query GPU memory: {}',
                'gpu_cache_cleared': 'GPU cache cleared - Device {}',
                'gpu_cache_freed': '  Freed VRAM: {:.2f} MB',
                'gpu_cache_clear_failed': 'Failed to clear GPU cache: {}',
                'gpu_garbage_collected': 'GPU garbage collection complete - Device {}',
                'gpu_memory_recovered': '  Recovered VRAM: {:.2f} MB',
                'gpu_gc_failed': 'GPU garbage collection failed: {}',
                'gpu_memory_high_usage': 'GPU VRAM usage too high: {:.1f}%',
                'gpu_memory_total': '  Total VRAM: {:.2f} MB',
                'gpu_memory_used': '  Used: {:.2f} MB',
                'gpu_memory_free': '  Free: {:.2f} MB',
                'gpu_perform_recovery': '  Performing smart memory recovery...',
                'smart_memory_recovery_failed': 'Smart memory recovery failed: {}',
                'gpu_monitoring_not_running': 'GPU unavailable, skipping auto memory monitoring',
                'gpu_monitor_running': 'Auto memory monitoring already running',
                'gpu_monitor_started': 'GPU auto memory monitoring started (interval: {}s, threshold: {}%)',
                'gpu_monitor_exception': 'Memory monitoring exception: {}',
                'gpu_monitor_stopped': 'GPU auto memory monitoring stopped',
                'cache_cleared': 'Cache cleared',
                'cache_status': 'Cache status: {}',
                'cache_size_info': 'Cache size: {}/{}',
                'cache_hits': 'Hits: {}',
                'cache_misses': 'Misses: {}',
                'cache_hit_rate': 'Hit rate: {:.1f}%',
                'redis_connected': 'Redis cache connected: {}',
                'redis_not_installed': 'Redis module not installed, Redis cache will be unavailable',
                'redis_install_cmd': 'Install command: pip install redis',
                'redis_connection_failed': 'Redis connection failed: {}',
                'redis_unavailable': 'Redis cache will be unavailable, using local cache instead',
                'redis_cache_cleared': 'Redis cache cleared: {} keys',
                'redis_cache_empty': 'Redis cache empty',
                'webhook_client_init': 'Webhook client initialized',
                'webhook_not_installed': 'httpx module not installed, Webhook functionality will be unavailable',
                'webhook_install_cmd': 'Install command: pip install httpx',
                'webhook_disabled': 'Webhook not enabled, cannot register',
                'webhook_registered': 'Webhook registered: {} -> {}',
                'webhook_unregistered': 'Webhook unregistered: {}',
                'webhook_sent_success': 'Webhook sent successfully: {} -> {}',
                'webhook_sent_failed': 'Webhook sending failed: {} -> {} (Status code: {})',
                'cache_hit_info': 'Cache hit! PLY save complete, duration: {:.2f}s',
                'task_completion_time': 'Processing complete (cache), total duration: {:.2f}s',
                'prediction_start': 'Starting inference...',
                'prediction_complete': 'Inference complete, duration: {:.2f}s',
                'result_cached_to_redis': 'Result cached to Redis',
                'ply_saved': 'PLY saved, duration: {:.2f}s',
                'task_complete': 'Processing complete, total duration: {:.2f}s',
                'rename_failed': 'Failed to rename file: {}',
                
                # System info
                'system_info': 'System Information',
                'os_info': 'Operating System: {} {}',
                'python_version': 'Python Version: {}',
                'working_directory': 'Working Directory: {}',
                'web_service': 'Web Service',
                'input_size_info': 'Input size: {}x{}',
                
                # Performance tuning
                'performance_tuning': 'Performance Auto-Tuning',
                'testing_configurations': 'Testing different optimization configurations...',
                'test_config': 'Test configuration: {}',
                'config_description': '  Description: {}',
                'avg_inference_time': '  Average inference time: {:.3f} seconds',
                'test_failed': '  Test failed: {}',
                'tuning_results': 'Tuning Results',
                'best_config': 'Best configuration: {}',
                'best_config_desc': '  Description: {}',
                'avg_inference_time_result': '  Average inference time: {:.3f} seconds',
                'throughput_info': '  Throughput: {:.2f} FPS',
                'tuning_complete': 'Performance auto-tuning completed!',
                'config_applied': 'Optimal configuration applied',
                'tuning_failed_default': 'All configuration tests failed, using default configuration',
                'tuning_failed': 'Performance auto-tuning failed: {}',
                'use_default_config': 'Using default configuration',
                'valid_cache_found': 'Found valid performance tuning cache ({} days ago)',
                'using_cached_config': 'Using cached performance configuration',
                'config_name': 'Configuration name: {}',
                'config_description_cached': 'Description: {}',
                
                # API responses
                'api_cache_cleared': 'Cache cleared',
                'webhook_not_enabled': 'Webhook not enabled',
                'missing_params': 'Missing required parameters: event_type and url',
                'webhook_registered_resp': 'Webhook registered: {} -> {}',
                
                # Others
                'amp_only_optimization': 'Only enable mixed precision inference',
                'cudnn_only_optimization': 'Only enable cuDNN auto-tuning',
                'tf32_only_optimization': 'Only enable TensorFloat32',
                'amp_cudnn_optimization': 'AMP and cuDNN auto-tuning',
                'amp_tf32_optimization': 'AMP and TensorFloat32',
                'all_optimizations': 'Enable all optimizations',
                
                # Progress and status
                'task_processing': '[Task {}] File saved: {}',
                'task_image_info': '[Task {}] Image info: {}x{}, focal: {} (load duration: {:.2f}s)',
                'task_large_image': '[Task {}] Image size too large ({}x{}), may cause performance issues',
                'task_cache_hit': '[Task {}] Cache hit! PLY save complete, duration: {:.2f}s',
                'task_cache_complete': '[Task {}] Processing complete (cache), total duration: {:.2f}s',
                'task_start_inference': '[Task {}] Starting inference...',
                'task_inference_complete': '[Task {}] Inference complete, duration: {:.2f}s',
                'task_redis_cache': '[Task {}] Result cached to Redis',
                'task_ply_save_complete': '[Task {}] PLY save complete, duration: {:.2f}s',
                'task_complete_total': '[Task {}] Processing complete, total duration: {:.2f}s',
                'task_vram_insufficient': '[Task {}] Insufficient VRAM: {}',
                'task_processing_failed': '[Task {}] Processing failed: {}',
                
                # Configuration validation errors
                'input_size_error': 'input_size must contain 2 elements',
                'input_size_square_error': 'input_size must be square, currently {}x{}',
                'input_size_divisible_error': 'input_size must be divisible by 64, currently {}x{}',
                'input_size_max_error': 'input_size cannot exceed {}x{}, currently {}x{}',
                'threshold_range_error': 'auto_gc_threshold must be between 0-100',
                'host_empty_error': 'host cannot be empty',
                'checkpoint_empty_error': 'checkpoint cannot be empty',
                'temp_dir_empty_error': 'temp_dir cannot be empty',
                'redis_url_required': 'URL must be provided when Redis is enabled',
                
                # Stats
                'points_label': 'Points',
                'fps_label': 'FPS',
                'stat_points': 'Points',
                'stat_fps': 'FPS',
                'ctrl_scale': 'Scale',
                'ctrl_opacity': 'Opacity',
                'ctrl_exposure': 'Exposure',
                'ctrl_speed': 'Speed',
                'flow_amp': 'Amp',
                'flow_freq': 'Freq',
                'flow_speed': 'Flow Spd',
                'import_main': 'Import',
                'import_sub': 'Drag & Drop Image/.PLY',
                'nav_move': 'Move',
                'nav_lift': 'Lift',
                'nav_fast': 'Fast',
                'nav_look': 'Look',
                'loading_init': 'INITIALIZING SYSTEM',
                'tutorial_text': '[ WASD ] TO FLY & COLLECT',
                'loading_processing': 'PROCESSING ARTWORK',
                'loading_downloading': 'DOWNLOADING GEOMETRY',
                'loading_parsing': 'PARSING DATA...',
                'loading_generating': 'GENERATING ARTWORK',
                'loading_importing': 'IMPORTING MODEL',
                'loading_download_general': 'DOWNLOADING...',
                'btn_flow_tooltip': 'Disperse / Reset Flow',
                'btn_download_tooltip': 'Download Model (.ply)',
                
                # Chinese labels (for consistency in English environment)
                'subtitle_en': 'HIGH FIDELITY VIEWER .01',
                'scale_en': 'Scale',
                'opacity_en': 'Opacity',
                'exposure_en': 'Exposure',
                'speed_en': 'Speed',
                'amp_en': 'Amp',
                'freq_en': 'Freq',
                'flow_speed_en': 'Flow Spd',
                'import_main_en': 'Import',
                'import_sub_en': 'Drag & Drop Image/.PLY',
                'move_en': 'Move',
                'lift_en': 'Lift',
                'fast_en': 'Fast',
                'look_en': 'Look',
                'init_en': 'INITIALIZING SYSTEM',
                'tutorial_en': '[ WASD ] TO FLY & COLLECT',
                
                # New: Config loading related
                'load_config_failed': 'Failed to load config file: {}',
                'use_default_config_and_args': 'Using default config and command-line args',
                
                # New: Log titles
                'gpu_init_title': 'GPU Initialization',
                'cache_stats_title': 'Cache Statistics',
                'cached_config_title': 'Using Cached Performance Configuration',
                'performance_tuning_title': 'Performance Auto-Tuning',
                'tuning_results_title': 'Tuning Results',
                'model_loading_title': 'Model Loading',
                'system_info_title': 'System Information',
                'web_service_title': 'Web Service',
                'service_stopped_title': 'Service Stopped',
                
                # New: User specified mode
                'user_specified_mode_info': 'User specified startup mode: {}',
                
                # New: CUDA/ROCm version
                'cuda_version_info': '   CUDA/ROCm version: {}',
                
                # New: GPU count
                'gpu_count_info': '   GPU count: {}',
                
                # New: Compute capability
                'compute_capability_info_full': '   Compute capability: {}',
                
                # New: GPU memory
                'gpu_memory_info_full': '   VRAM: {:.2f} GB',
                
                # New: GC interval and threshold
                'gc_interval_info': '  Check interval: {} seconds',
                'gc_threshold_info': '  Cleanup threshold: {}%',
                
                # New: Default config file
                'using_default_config_file': 'Using default config file: {}',
                
                # New: VRAM freed
                'vram_freed_info': '  Freed VRAM: {:.2f} MB',
                
                # New: VRAM recovered
                'vram_recovered_info': '  Recovered VRAM: {:.2f} MB',
                
                # New: VRAM stats
                'vram_total_info': '  Total VRAM: {:.2f} MB',
                'vram_used_info': '  Used: {:.2f} MB',
                'vram_free_info': '  Free: {:.2f} MB',
                
                # New: Monitoring start
                'gpu_monitoring_started': 'GPU auto memory monitoring started (interval: {}s, threshold: {}%)',
                
                # New: Cache status
                'cache_enabled_status': 'Cache status: Enabled',
                'cache_disabled_status': 'Cache status: Disabled',
                'cache_size_info_full': 'Cache size: {}/{}',
                'cache_hits_info': 'Hits: {}',
                'cache_misses_info': 'Misses: {}',
                'cache_hit_rate_info': 'Hit rate: {:.1f}%',
                
                # New: Redis connection
                'redis_connected_info': 'Redis cache connected: {}',
                
                # New: Redis cleared
                'redis_cleared_info': 'Redis cache cleared: {} keys',
                
                # New: Webhook
                'webhook_registered_info': 'Webhook registered: {} -> {}',
                'webhook_unregistered_info': 'Webhook unregistered: {}',
                'webhook_sent_info': 'Webhook sent successfully: {} -> {}',
                
                # New: Tuning cache
                'tuning_cache_found': 'Found valid performance tuning cache ({} days ago)',
                
                # New: Config directory
                'config_dir_created': 'Config directory created: {}',
                
                # New: Config file updated
                'config_file_updated': 'Config file exists, updating performance tuning cache: {}',
                'config_file_created': 'Config file does not exist, auto-creating new config file: {}',
                
                # New: Best config
                'best_config_name': 'Configuration name: {}',
                'best_config_desc_info': 'Description: {}',
                
                # New: Test configuration
                'test_config_info': 'Test configuration: {}',
                'test_config_desc': '  Description: {}',
                'avg_inference_time_info': '  Average inference time: {:.3f} seconds',
                
                # New: Throughput
                'throughput_info_full': '  Throughput: {:.2f} FPS',
                
                # New: Test run
                'test_run_info': '  Run {}/{}: {:.3f} seconds',
                
                # New: Model file
                'model_file_info': 'Model file: {}',
                
                # New: Device info
                'device_info_full': 'Device: {}',
                
                # New: VRAM usage
                'vram_usage_info': 'VRAM usage: {:.2f} MB',
                
                # New: Task processing
                'task_file_saved': '[Task {}] File saved: {}',
                
                # New: Task image info
                'task_image_info_full': '[Task {}] Image info: {}x{}, focal: {} (load time: {:.2f}s)',
                
                # New: Task cache hit
                'task_cache_hit_info': '[Task {}] Cache hit! PLY save complete, duration: {:.2f}s',
                
                # New: Task completion
                'task_completion_info': '[Task {}] Processing complete (cache), total duration: {:.2f}s',
                
                # New: Task inference
                'task_inference_start': '[Task {}] Starting inference...',
                'task_inference_complete_info': '[Task {}] Inference complete, duration: {:.2f}s',
                'task_redis_cache_info': '[Task {}] Result cached to Redis',
                'task_ply_save_info': '[Task {}] PLY save complete, duration: {:.2f}s',
                'task_complete_info': '[Task {}] Processing complete, total duration: {:.2f}s',
                
                # New: System info
                'os_info_full': 'Operating System: {} {}',
                'python_version_info': 'Python Version: {}',
                'working_directory_info': 'Working Directory: {}',
                'input_size_info_full': 'Input size: {}x{}',
                
                # New: GPU vendor/model
                'gpu_vendor_info': 'GPU vendor: {}',
                'gpu_model_info': 'GPU model: {}',
                
                # New: Optimization status
                'amp_enabled_info': '[Optimization] Mixed Precision Inference: Enabled',
                'amp_disabled_info': '[Optimization] Mixed Precision Inference: Disabled',
                'cudnn_enabled_info': '[Optimization] cuDNN Benchmark: Enabled',
                'cudnn_disabled_info': '[Optimization] cuDNN Benchmark: Disabled',
                'tf32_enabled_info': '[Optimization] TensorFloat32: Enabled',
                'tf32_disabled_info': '[Optimization] TensorFloat32: Disabled',
                
                # New: CPU cores
                'cpu_cores_info': 'CPU cores: {}',
                
                # New: Cache disabled
                'cache_disabled_info': '[Cache] Inference cache: Disabled',
                
                # New: Manual access
                'manual_access_info': 'Please access manually: {}',
                
                # New: CPU optimization
                'cpu_optimization_enabled': 'CPU optimization enabled ({} cores)',
                
                # New: Auto GC status
                'auto_gc_enabled': '  Automatic garbage collection: Enabled',
                'auto_gc_disabled': '  Automatic garbage collection: Disabled',
                
                # New: cuDNN status
                'cudnn_enabled_status': '  cuDNN Benchmark: Enabled',
                'cudnn_disabled_capability': '  cuDNN Benchmark: Disabled (GPU compute capability insufficient)',
                
                # New: TF32 status
                'tf32_enabled_status': '  TensorFloat32: Enabled',
                'tf32_disabled_support': '  TensorFloat32: Disabled (GPU does not support)',
                
                # New: AMP status
                'amp_enabled_status': '  Mixed Precision Inference (AMP): Enabled',
                'amp_disabled_capability': '  Mixed Precision Inference (AMP): Disabled (GPU compute capability insufficient)',
                
                # New: Monitoring running
                'monitoring_already_running': '  Auto memory monitoring already running',
                
                # New: Test run failed
                'test_run_failed': '  Run {}/{} failed: {}',
                
                # New: Gradient checkpointing success
                'grad_checkpoint_enabled': 'Gradient checkpointing enabled (VRAM usage will reduce, but inference speed may slightly decrease)',
                
                # New: Redis cache enabled
                'redis_cache_enabled': 'Redis cache enabled: {}',
                
                # New: Webhook enabled
                'webhook_enabled': 'Webhook notification enabled',
                
                # New: GPU acceleration enabled
                'gpu_accel_enabled': 'GPU acceleration enabled',
                
                # New: Multithreading optimization
                'multithreading_opt_enabled': '[Optimization] Multithreading optimization: Enabled',
                
                # New: Inference cache
                'inference_cache_enabled': '[Cache] Inference cache: Enabled (max {} entries)',
            }
        }
    
    def set_language(self, lang: str):
        """
        设置当前语言
        
        Args:
            lang: 语言代码 ('zh' 或 'en')
        """
        if lang in self.translations:
            self.current_lang = lang
        else:
            self.current_lang = self.default_lang
    
    def t(self, key: str, *args, **kwargs) -> str:
        """
        获取翻译文本
        
        Args:
            key: 翻译键
            *args: 位置参数，用于字符串格式化
            **kwargs: 关键字参数，用于字符串格式化
            
        Returns:
            翻译后的文本
        """
        # 获取当前语言的翻译，如果不存在则使用默认语言
        translation = self.translations.get(self.current_lang, {}).get(key)
        if translation is None:
            translation = self.translations.get(self.default_lang, {}).get(key, key)
        
        # 如果提供了格式化参数，则进行格式化
        if args or kwargs:
            try:
                translation = translation.format(*args, **kwargs)
            except (TypeError, ValueError, KeyError):
                # 如果格式化失败，返回原始翻译
                pass
        
        return translation


# 全局 i18n 实例
i18n = I18n()
