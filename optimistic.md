# MLSharp 性能优化方案

本文档详细介绍了 MLSharp 可以实施的各种性能优化方案，以进一步提升推理速度和减少内存占用。

## 已实现的优化

### 1. 混合精度推理 (AMP)
- **显存减少**: 40-50%
- **速度提升**: 20-30%
- **状态**: ✅ 已实现

### 2. TensorFloat32 (TF32)
- **适用**: NVIDIA Ampere 架构 (RTX 30/40 系列)
- **速度提升**: 1.5-2倍
- **状态**: ✅ 已实现

### 3. cuDNN Benchmark
- **速度提升**: 10-20%
- **状态**: ✅ 已实现

### 4. 输入尺寸控制
- **显存减少**: 30-70%
- **状态**: ✅ 已实现

### 5. CPU 多线程优化
- **适用**: CPU 模式
- **状态**: ✅ 已实现

### 6. 非阻塞数据传输
- **状态**: ✅ 已实现

### 7. 内存监控
- **状态**: ✅ 已实现 (Prometheus 集成)

---

## 待实施的优化方案

### 🚀 方案 1: 模型量化 (Model Quantization)

**描述**: 将模型权重从 FP32 转换为 INT8，大幅减少显存占用。

**优势**:
- 显存占用减少 75%（FP32 → INT8）
- 推理速度提升 2-4 倍
- 支持 CPU 和 GPU 加速

**实现方案**:
```python
import torch.quantization as quantization

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# 静态量化（需要校准）
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
quantized_model = torch.quantization.prepare(model)
quantized_model = torch.quantization.convert(quantized_model)
```

**预期效果**:
- 显存: 5GB → 1.25GB
- 速度: 提升 2-3 倍
- 精度损失: < 1%

**实施难度**: ⭐⭐⭐

---

### ⚡ 方案 2: 模型剪枝 (Model Pruning)

**描述**: 移除模型中不重要的权重，减少模型大小。

**优势**:
- 模型大小减少 30-50%
- 推理速度提升
- 精度损失可控

**实现方案**:
```python
import torch.nn.utils.prune as prune

# 非结构化剪枝
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)

# 结构化剪枝（更高效）
prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
```

**预期效果**:
- 模型大小: 减少 30-50%
- 速度: 提升 1.5-2 倍
- 精度损失: 1-2%

**实施难度**: ⭐⭐⭐⭐⭐

---

### 📦 方案 3: 批处理优化 (Batch Processing)

**描述**: 支持批量处理多个图像，提高 GPU 利用率。

**优势**:
- GPU 利用率提升 50-80%
- 单图像处理成本降低
- 适合批量任务

**实现方案**:
```python
def predict_batch(self, images: List[np.ndarray], f_px_list: List[float]):
    batch_size = len(images)
    
    # 准备批量输入
    images_pt = torch.stack([
        torch.from_numpy(img).permute(2, 0, 1) / 255.0
        for img in images
    ]).to(self.device)
    
    disparity_factors = torch.tensor([f / w for f, w in zip(f_px_list, 
        [img.shape[1] for img in images])], dtype=torch.float32, device=self.device)
    
    # 批量推理
    gaussians_batch = self.predictor(images_pt, disparity_factors)
    
    return gaussians_batch
```

**API 端点**:
```python
@app.post("/api/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        result = await self._handle_predict(file)
        results.append(result)
    return results
```

**预期效果**:
- 批量大小 4: 速度提升 50-80%
- 批量大小 8: 速度提升 80-120%

**实施难度**: ⭐

---

### 🔄 方案 4: 推理缓存 (Inference Caching)

**描述**: 缓存相似图像的推理结果，避免重复计算。

**优势**:
- 相似图像处理速度提升 90%+
- 减少重复计算
- 适合相似场景处理

**实现方案**:
```python
from functools import lru_cache
import hashlib

class ModelManager:
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_image_hash(self, image: np.ndarray) -> str:
        """计算图像哈希"""
        return hashlib.md5(image.tobytes()).hexdigest()
    
    def predict_with_cache(self, image: np.ndarray, f_px: float):
        """带缓存的预测"""
        image_hash = self._get_image_hash(image)
        
        # 检查缓存
        if image_hash in self.cache:
            self.cache_hits += 1
            Logger.info(f"缓存命中: {self.cache_hits}/{self.cache_hits + self.cache_misses}")
            return self.cache[image_hash]
        
        # 执行推理
        self.cache_misses += 1
        result = self.predict(image, f_px)
        
        # 缓存结果
        self.cache[image_hash] = result
        
        # 限制缓存大小
        if len(self.cache) > 100:
            self.cache.pop(next(iter(self.cache)))
        
        return result
```

**预期效果**:
- 缓存命中率 30%: 速度提升 30%
- 缓存命中率 50%: 速度提升 50%
- 缓存命中率 80%: 速度提升 80%

**实施难度**: ⭐

---

### 🧠 方案 5: 知识蒸馏 (Knowledge Distillation)

**描述**: 使用更大的教师模型训练更小的学生模型。

**优势**:
- 模型大小减少 50-70%
- 保持高精度
- 推理速度提升 2-3 倍

**实现方案**:
```python
def distillation_loss(student_output, teacher_output, labels, T=2.0, alpha=0.5):
    """知识蒸馏损失函数"""
    # 软损失（知识蒸馏）
    soft_loss = nn.KLDivLoss()(
        F.log_softmax(student_output/T, dim=1),
        F.softmax(teacher_output/T, dim=1)
    ) * (T*T * alpha)
    
    # 硬损失（真实标签）
    hard_loss = F.cross_entropy(student_output, labels) * (1 - alpha)
    
    return soft_loss + hard_loss

# 训练流程
for batch in dataloader:
    inputs, labels = batch
    
    # 教师模型推理（不计算梯度）
    with torch.no_grad():
        teacher_output = teacher_model(inputs)
    
    # 学生模型推理
    student_output = student_model(inputs)
    
    # 计算蒸馏损失
    loss = distillation_loss(student_output, teacher_output, labels)
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

**预期效果**:
- 模型大小: 减少 50-70%
- 速度: 提升 2-3 倍
- 精度损失: < 2%

**实施难度**: ⭐⭐⭐⭐⭐

---

### 🎯 方案 6: 梯度检查点 (Gradient Checkpointing)

**描述**: 以计算换内存，减少中间激活值的存储。

**优势**:
- 显存占用减少 30-50%
- 适合超大模型
- 推理速度略微降低（可接受）

**实现方案**:
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
    
    def forward(self, x):
        # 使用检查点重新计算中间激活值
        x = checkpoint(self.original_model.block1, x)
        x = checkpoint(self.original_model.block2, x)
        x = checkpoint(self.original_model.block3, x)
        return x

# 应用到预测器
self.checkpointed_predictor = CheckpointedModel(self.predictor)
```

**配置选项**:
```yaml
# config.yaml
optimization:
  gradient_checkpointing: false  # 默认关闭
  checkpoint_segments: 3         # 检查点分段数
```

**命令行参数**:
```bash
# 启用梯度检查点
python app.py --gradient-checkpointing

# 设置检查点分段数
python app.py --gradient-checkpointing --checkpoint-segments 4
```

**预期效果**:
- 显存: 减少 30-50%
- 速度: 降低 10-20%
- 适用: 显存不足时

**实施难度**: ⭐⭐

---

### 🗜️ 方案 7: 动态输入尺寸 (Dynamic Input Size)

**描述**: 根据图像内容自动选择最优输入尺寸。

**优势**:
- 简单图像使用较小尺寸，节省显存
- 复杂图像使用较大尺寸，保证质量
- 智能平衡质量和性能

**实现方案**:
```python
import cv2
import numpy as np

def analyze_image_complexity(self, image: np.ndarray) -> Tuple[int, int]:
    """
    分析图像复杂度并返回推荐的输入尺寸
    
    Args:
        image: 输入图像 (H, W, 3)
    
    Returns:
        推荐的输入尺寸 (width, height)
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 计算边缘检测
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # 计算图像熵（信息量）
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # 计算颜色复杂度
    if len(image.shape) == 3:
        colors = np.unique(image.reshape(-1, 3), axis=0)
        color_complexity = len(colors) / (256 * 256 * 256)
    else:
        color_complexity = 0
    
    # 综合评分 (0-1)
    complexity_score = (
        edge_density * 0.4 +      # 边缘密度权重 40%
        entropy / 8.0 * 0.4 +     # 熵权重 40%
        color_complexity * 0.2     # 颜色复杂度权重 20%
    )
    
    Logger.info(f"图像复杂度分析: 边缘密度={edge_density:.3f}, 熵={entropy:.3f}, 颜色复杂度={color_complexity:.3f}")
    Logger.info(f"综合复杂度评分: {complexity_score:.3f}")
    
    # 根据复杂度选择尺寸
    if complexity_score < 0.15:
        recommended_size = (512, 512)
        Logger.info(f"推荐输入尺寸: 512x512 (简单图像)")
    elif complexity_score < 0.30:
        recommended_size = (768, 768)
        Logger.info(f"推荐输入尺寸: 768x768 (中等复杂度)")
    elif complexity_score < 0.50:
        recommended_size = (1024, 1024)
        Logger.info(f"推荐输入尺寸: 1024x1024 (复杂图像)")
    else:
        recommended_size = (1536, 1536)
        Logger.info(f"推荐输入尺寸: 1536x1536 (高度复杂)")
    
    return recommended_size
```

**配置选项**:
```yaml
# config.yaml
inference:
  input_size: [1536, 1536]  # 默认/最大尺寸
  dynamic_input_size: false  # 动态输入尺寸（默认关闭）
  min_input_size: 512        # 最小输入尺寸
  max_input_size: 1536       # 最大输入尺寸
```

**命令行参数**:
```bash
# 启用动态输入尺寸
python app.py --dynamic-input-size

# 设置最小/最大尺寸
python app.py --dynamic-input-size --min-input-size 512 --max-input-size 1536
```

**预期效果**:
- 简单图像: 显存减少 60-70%
- 中等图像: 显存减少 30-40%
- 复杂图像: 无损失
- 平均: 显存减少 30-50%

**实施难度**: ⭐⭐

---

### 📊 方案 8: 渐进式推理 (Progressive Inference)

**描述**: 先使用低分辨率快速推理，再根据需要逐步提高分辨率。

**优势**:
- 快速预览（低分辨率）
- 按需提升质量
- 节省不必要的计算

**实现方案**:
```python
def progressive_predict(self, image: np.ndarray, max_size: int = 1536):
    """
    渐进式推理
    
    Args:
        image: 输入图像
        max_size: 最大输入尺寸
    
    Returns:
        高斯结果
    """
    sizes = [512, 768, 1024, max_size]
    
    # 第一阶段：低分辨率快速推理
    gaussians = self.predict(image, input_size=(sizes[0], sizes[0]))
    
    # 检查是否需要更高分辨率
    if self.needs_high_quality(gaussians):
        # 第二阶段：中等分辨率
        gaussians = self.predict(image, input_size=(sizes[1], sizes[1]))
        
        if self.needs_high_quality(gaussians):
            # 第三阶段：高分辨率
            gaussians = self.predict(image, input_size=(max_size, max_size))
    
    return gaussians

def needs_high_quality(self, gaussians):
    """判断是否需要更高质量"""
    # 可以基于高斯数量、分布等指标判断
    return len(gaussians) < 10000  # 示例阈值
```

**预期效果**:
- 快速预览: 速度提升 4-6 倍
- 按需提升: 平均节省 50% 计算
- 用户体验: 更好的交互体验

**实施难度**: ⭐⭐⭐

---

### 🔧 方案 9: ONNX 导出和优化

**描述**: 导出 ONNX 格式并进行优化。

**优势**:
- 跨平台兼容性
- 推理速度提升 20-30%
- 支持 TensorRT 加速

**实现方案**:
```python
import torch.onnx

def export_to_onnx(self, output_path: str = "model.onnx"):
    """导出模型到 ONNX 格式"""
    self.predictor.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 1536, 1536, device=self.device)
    dummy_disparity = torch.tensor([1.0], device=self.device)
    
    # 导出 ONNX
    torch.onnx.export(
        self.predictor,
        (dummy_input, dummy_disparity),
        output_path,
        opset_version=14,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        input_names=['input', 'disparity'],
        output_names=['gaussians']
    )
    
    Logger.success(f"模型已导出到 {output_path}")

# 使用 ONNX Runtime 推理
import onnxruntime as ort

class ONNXPredictor:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path)
    
    def predict(self, image: np.ndarray, f_px: float):
        """使用 ONNX Runtime 推理"""
        # 预处理
        image_pt = torch.from_numpy(image).permute(2, 0, 1) / 255.0
        image_pt = image_pt.unsqueeze(0).numpy()
        
        # 推理
        outputs = self.session.run(
            None,
            {
                'input': image_pt,
                'disparity': np.array([f_px], dtype=np.float32)
            }
        )
        
        return outputs[0]
```

**预期效果**:
- 速度: 提升 20-30%
- 兼容性: 支持更多平台
- 部署: 更容易部署

**实施难度**: ⭐⭐

---

### 🚄 方案 10: TensorRT 加速

**描述**: 使用 NVIDIA TensorRT 进行推理加速。

**优势**:
- 推理速度提升 3-5 倍
- 显存占用减少
- 自动优化计算图

**实现方案**:
```python
import tensorrt as trt
from torch2trt import torch2trt

def convert_to_tensorrt(self, max_batch_size: int = 1):
    """转换为 TensorRT 模型"""
    self.predictor.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 1536, 1536).cuda()
    dummy_disparity = torch.tensor([1.0]).cuda()
    
    # 转换为 TensorRT
    self.predictor_trt = torch2trt(
        self.predictor,
        [dummy_input, dummy_disparity],
        fp16_mode=True,
        max_workspace_size=1 << 30,  # 1GB
        max_batch_size=max_batch_size
    )
    
    Logger.success("模型已转换为 TensorRT 格式")

def predict_with_tensorrt(self, image: np.ndarray, f_px: float):
    """使用 TensorRT 推理"""
    # 预处理
    image_pt = torch.from_numpy(image).cuda().half().permute(2, 0, 1) / 255.0
    image_pt = image_pt.unsqueeze(0)
    disparity = torch.tensor([f_px]).cuda().half()
    
    # 推理
    with torch.no_grad():
        gaussians = self.predictor_trt(image_pt, disparity)
    
    return gaussians
```

**预期效果**:
- 速度: 提升 3-5 倍
- 显存: 减少 30%
- 适用: NVIDIA GPU

**实施难度**: ⭐⭐⭐⭐

---

## 优化方案优先级

### 🔥 高优先级（立即可实施）

1. **批处理优化** - 最容易实现，效果显著
2. **推理缓存** - 实现简单，适合重复场景
3. **梯度检查点** - 显存不足时的救命稻草
4. **动态输入尺寸** - 智能化，提升用户体验

### ⭐ 中优先级（需要一定开发）

5. **模型量化** - 效果显著，需要测试精度
6. **ONNX 导出** - 提升兼容性，支持更多平台
7. **TensorRT 加速** - NVIDIA GPU 的终极优化

### 💡 低优先级（长期规划）

8. **模型剪枝** - 需要重新训练
9. **知识蒸馏** - 需要训练流程
10. **渐进式推理** - 复杂度高，应用场景有限

---

## 实施建议

### 短期（1-2周）
```bash
✅ 梯度检查点 - 显存优化
✅ 批处理优化 - 性能提升
✅ 推理缓存 - 重复场景优化
```

### 中期（1-2月）
```bash
🔄 动态输入尺寸 - 智能化优化
🔄 模型量化 - 显存大幅减少
🔄 ONNX 导出 - 跨平台支持
```

### 长期（3-6月）
```bash
⏳ TensorRT 集成 - 极致性能
⏳ 模型剪枝和蒸馏 - 模型优化
⏳ 完整的优化工具链
```

---

## 性能对比表

| 优化方案 | 显存减少 | 速度提升 | 实施难度 | 优先级 |
|---------|---------|---------|---------|--------|
| 已实现优化 | 40-50% | 20-30% | - | - |
| 梯度检查点 | 30-50% | -10-20% | ⭐⭐ | 🔥 高 |
| 批处理优化 | 0% | 50-80% | ⭐ | 🔥 高 |
| 推理缓存 | 0% | 90%+ | ⭐ | 🔥 高 |
| 动态输入尺寸 | 30-50% | 20-30% | ⭐⭐ | 🔥 高 |
| 模型量化 | 75% | 2-3倍 | ⭐⭐⭐ | ⭐ 中 |
| ONNX 导出 | 20% | 20-30% | ⭐⭐ | ⭐ 中 |
| TensorRT | 30% | 3-5倍 | ⭐⭐⭐⭐ | ⭐ 中 |
| 模型剪枝 | 30-50% | 1.5-2倍 | ⭐⭐⭐⭐⭐ | 💡 低 |
| 知识蒸馏 | 50-70% | 2-3倍 | ⭐⭐⭐⭐⭐ | 💡 低 |
| 渐进式推理 | 50% | 4-6倍 | ⭐⭐⭐ | 💡 低 |

---

## 配置文件示例

```yaml
# config.yaml
optimization:
  # 梯度检查点
  gradient_checkpointing: false
  checkpoint_segments: 3
  
  # 动态输入尺寸
  dynamic_input_size: false
  min_input_size: 512
  max_input_size: 1536
  
  # 批处理
  enable_batch_processing: false
  max_batch_size: 4
  
  # 推理缓存
  enable_cache: false
  cache_size: 100
  
  # 模型量化
  enable_quantization: false
  quantization_mode: "dynamic"  # dynamic, static
  
  # TensorRT
  enable_tensorrt: false
  tensorrt_fp16: true
```

---

## 总结

这些优化方案可以显著提升 MLSharp 的性能，特别是在处理大量请求或有限显存的环境下。建议根据实际需求和资源情况，选择合适的优化方案逐步实施。

**预期总体提升**:
- 显存占用: 减少 50-70%
- 推理速度: 提升 3-5 倍
- 吞吐量: 提升 5-10 倍

**关键成功因素**:
1. 根据硬件配置选择合适的优化方案
2. 平衡性能和精度
3. 充分测试和验证
4. 监控性能指标
5. 逐步实施和优化