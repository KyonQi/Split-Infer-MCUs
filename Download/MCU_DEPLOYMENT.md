# MobileNetV2 量化推理 - MCU部署指南

本文档说明如何将Python模拟的量化推理代码部署到Teensy 4.1 MCU上。

## 目录
- [硬件要求](#硬件要求)
- [权重存储方案](#权重存储方案)
- [部署步骤](#部署步骤)
- [内存布局](#内存布局)
- [性能优化建议](#性能优化建议)

---

## 硬件要求

### Teensy 4.1 规格
- **MCU**: NXP i.MX RT1062 (ARM Cortex-M7 @ 600MHz)
- **Flash**: 7.9 MB (程序存储空间)
- **RAM**: 1024 KB (SRAM)
- **可选PSRAM**: 8 MB (需要焊接PSRAM芯片)

### MobileNetV2 量化模型大小
根据 `export_weights_for_mcu.py` 导出结果：
- **INT8权重**: ~3.4 MB
- **INT32偏置**: ~0.4 MB  
- **总计**: **约3.8 MB**

✅ **结论**: 权重可以完全放入Flash (PROGMEM)

---

## 权重存储方案

### 方案对比

| 存储方案 | 容量 | 访问速度 | 优点 | 缺点 | 推荐度 |
|---------|------|---------|------|------|--------|
| **PROGMEM** (Flash) | 7.9 MB | 慢 (~100ns) | 容量大，不占RAM | 需要特殊读取函数 | ⭐⭐⭐⭐⭐ |
| **EXTMEM** (PSRAM) | 8 MB | 中等 (~70ns) | 快速访问 | 需要焊接芯片 | ⭐⭐⭐⭐ |
| **RAM** | 1024 KB | 快 (~10ns) | 最快访问 | 容量不足 | ❌ 不可行 |

### 推荐方案：PROGMEM (Flash)

**优势**：
1. ✅ 无需额外硬件改动
2. ✅ 容量足够（3.8MB < 7.9MB）
3. ✅ 节省RAM用于激活值存储
4. ✅ 权重只读，Flash适合

**注意事项**：
- 使用 `pgm_read_byte()` 读取权重
- 使用 `pgm_read_dword()` 读取偏置

---

## 部署步骤

### 1. 导出权重文件

在Python项目目录运行：

```bash
cd /home/kyonqi/Project/RustProjects/Python_Sim_Infer

# 方案A: 使用PROGMEM (推荐)
python export_weights_for_mcu.py \
    --storage progmem \
    --output-dir ../PlatformIO_MCU/Download/include

# 方案B: 使用EXTMEM (如果有PSRAM)
python export_weights_for_mcu.py \
    --storage extmem \
    --output-dir ../PlatformIO_MCU/Download/include
```

**生成的文件**：
- `include/weights.h` - 权重和偏置数组
- `include/quant_params.h` - 量化参数 (scale, zero_point)
- `include/layer_config.h` - 层配置信息

### 2. 配置PlatformIO

编辑 `platformio.ini`:

```ini
[env:teensy41]
platform = teensy
board = teensy41
framework = arduino
monitor_speed = 115200

# 如果使用PROGMEM，添加这个标志
build_flags = 
    -std=c++11 
    -Wall
    -DUSE_PROGMEM      ; 启用PROGMEM读取函数
    -O3                ; 最高优化等级
    -mcpu=cortex-m7
    -mfloat-abi=hard
    -mfpu=fpv5-d16

# 如果使用EXTMEM，改为
# build_flags = -DUSE_EXTMEM
```

### 3. 包含头文件

在 `src/main.cpp` 中：

```cpp
#include <Arduino.h>
#include "weights.h"
#include "quant_params.h"
#include "layer_config.h"

void setup() {
    Serial.begin(115200);
    
    // 验证权重加载
    Serial.print("Total layers: ");
    Serial.println(NUM_LAYERS);
    
    // 打印第一层权重统计
    const LayerWeights* layer0 = &layer_weights[0];
    Serial.print("Layer 0 weight size: ");
    Serial.println(layer0->weight_size);
}
```

### 4. 编译和上传

```bash
cd /home/kyonqi/Project/RustProjects/PlatformIO_MCU/Download

# 编译
pio run

# 上传到Teensy
pio run --target upload

# 查看串口输出
pio device monitor
```

---

## 内存布局

### Teensy 4.1 内存分配

```
┌─────────────────────────────────────┐
│  Flash (7.9 MB)                     │
│  ┌─────────────────────────────┐   │
│  │ Program Code (~500KB)       │   │
│  ├─────────────────────────────┤   │
│  │ Weights (3.8MB) - PROGMEM   │   │ ← 权重存储在这里
│  ├─────────────────────────────┤   │
│  │ Layer Configs (~10KB)       │   │
│  └─────────────────────────────┘   │
│  Free: ~3.6 MB                      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  RAM (1024 KB)                      │
│  ┌─────────────────────────────┐   │
│  │ Stack & Heap (~100KB)       │   │
│  ├─────────────────────────────┤   │
│  │ Activation Buffer A (400KB) │   │ ← 激活值缓冲区
│  ├─────────────────────────────┤   │
│  │ Activation Buffer B (400KB) │   │ ← 乒乓操作
│  ├─────────────────────────────┤   │
│  │ Residual Buffer (100KB)     │   │ ← 残差连接
│  └─────────────────────────────┘   │
│  Free: ~24 KB                       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  PSRAM (8 MB) - 可选                │
│  └─────────────────────────────┘   │ ← EXTMEM方案使用
└─────────────────────────────────────┘
```

### 激活值内存需求

MobileNetV2各阶段特征图大小：
```
Input:      224×224×3    = 150 KB
Conv1:      112×112×32   = 401 KB  ← 最大！
Bottleneck: 56×56×64     = 200 KB
...
Output:     1×1×1000     = 1 KB
```

**策略**：使用乒乓缓冲区（2×400KB），交替读写

---

## 性能优化建议

### 1. 编译器优化
```ini
build_flags = 
    -O3                    # 最高优化
    -march=armv7e-m        # ARM架构优化
    -mfpu=fpv5-d16         # 使用FPU
    -mfloat-abi=hard       # 硬件浮点
```

### 2. 权重访问优化

**PROGMEM读取优化**：
```cpp
// 慢速版本（每次调用pgm_read_byte）
for (int i = 0; i < size; i++) {
    val = pgm_read_byte(&weights[i]);
}

// 快速版本（批量读取到缓存）
int8_t cache[32];
memcpy_P(cache, &weights[offset], 32);
for (int i = 0; i < 32; i++) {
    val = cache[i];
}
```

### 3. 卷积优化

**Im2Col + GEMM**：
- 将卷积转换为矩阵乘法
- 利用ARM的DSP指令（SIMD）

**深度可分离卷积专用优化**：
- 单通道操作，减少内存跨度
- 使用DMA批量传输

### 4. 量化计算优化

使用ARM CMSIS-NN库：
```cpp
#include <arm_nnfunctions.h>

arm_convolve_HWC_q7_basic(
    input, input_dim, input_ch,
    weights, output_ch, kernel_dim,
    padding, stride,
    bias, bias_shift, output_shift,
    output, output_dim, buffer
);
```

### 5. 降低精度（如果需要）

| 精度 | 权重大小 | 精度损失 |
|------|---------|---------|
| INT8 | 3.8 MB | 基准 |
| INT4 | 1.9 MB | ~1-2% |
| Binary | 0.5 MB | ~5-10% |

---

## 测试验证

### 1. 权重完整性测试

```cpp
void verify_weights() {
    // 读取第一个权重
    int8_t w = read_weight(layer_weights[0].weights, 0);
    Serial.print("First weight: ");
    Serial.println(w);
    
    // 计算权重校验和
    int32_t checksum = 0;
    for (uint32_t i = 0; i < layer_weights[0].weight_size; i++) {
        checksum += read_weight(layer_weights[0].weights, i);
    }
    Serial.print("Checksum: ");
    Serial.println(checksum);
}
```

### 2. 性能基准测试

```cpp
void benchmark() {
    uint32_t start = micros();
    
    // 执行单层推理
    execute_layer(0, input_buffer, output_buffer);
    
    uint32_t elapsed = micros() - start;
    Serial.print("Layer 0 time: ");
    Serial.print(elapsed);
    Serial.println(" us");
}
```

---

## 故障排查

### 问题1: 编译失败 - "Section .text exceeds flash size"
**原因**: 权重太大，超出Flash容量  
**解决**: 
- 使用 `-O3` 优化
- 确认使用 `PROGMEM` 而非 RAM
- 考虑模型压缩（量化到INT4）

### 问题2: 运行时崩溃
**原因**: RAM不足或栈溢出  
**解决**:
- 检查激活值缓冲区大小
- 使用 `DMAMEM` 分配大数组
- 减小缓冲区或使用PSRAM

### 问题3: 推理结果不正确
**原因**: 权重读取错误或量化参数不匹配  
**解决**:
- 验证 `pgm_read_byte` 正确使用
- 对比Python模拟的中间结果
- 检查scale/zero_point是否匹配

---

## 下一步

1. ✅ **完成权重导出** - 使用 `export_weights_for_mcu.py`
2. 📝 **实现卷积核心** - 编写 INT8 卷积函数
3. 🔧 **集成CMSIS-NN** - 使用ARM优化库
4. 📊 **性能测试** - 测量每层延迟
5. 🚀 **端到端推理** - 完整模型推理

---

## 参考资源

- [Teensy 4.1 文档](https://www.pjrc.com/store/teensy41.html)
- [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [量化推理原理](https://arxiv.org/abs/1712.05877)
