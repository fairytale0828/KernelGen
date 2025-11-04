# KernelGen 使用指南

## 快速开始

### 1. 环境设置

```bash
# 设置API密钥
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# 安装依赖
pip install -r requirements.txt
```

### 2. 基本使用

```bash
# 生成Level 1 Problem 19 (ReLU)的Triton kernel
python generate_kernel.py --level 1 --problem-id 19

# 生成Level 2 Problem 40的kernel并进行性能评估
python generate_kernel.py --level 2 --problem-id 40 --evaluate

# 多轮迭代生成，寻找最佳kernel
python generate_kernel.py --level 1 --problem-id 19 --iterations 5 --evaluate
```

### 3. 高级选项

```bash
# 自定义LLM参数
python generate_kernel.py --level 2 --problem-id 40 \
    --server-type deepseek \
    --model-name deepseek-coder \
    --temperature 0.1

# 自定义输出目录
python generate_kernel.py --level 1 --problem-id 19 \
    --output-dir my_kernels \
    --log-level DEBUG
```

## 支持的算子

### Level 1 - 基础算子
- Problem 1: Square matrix multiplication
- Problem 19: ReLU
- Problem 21: Sigmoid  
- Problem 23: Softmax
- Problem 40: LayerNorm
- 等等...

### Level 2 - 融合算子
- Problem 40: Matmul + Scaling + ResidualAdd
- Problem 1: Conv2D + ReLU + BiasAdd
- 等等...

### Level 3 - 模型块
- Problem 1: MLP
- Problem 4: LeNet5
- 等等...

### Level 4 - 完整模型
- Problem 1: GPT-Neo-2.7B
- 等等...

## 输出结果

生成的结果保存在指定目录下：

```
generated_kernels/level_1_problem_19_1234567890/
├── generated_kernel.py    # 生成的Triton kernel代码
└── results.json          # 详细结果和性能数据
```

### generated_kernel.py 示例

```python
# Generated Triton Kernel
# Level: 1
# Problem ID: 19
# Problem Name: 19_ReLU.py
# Generated at: 2024-01-01 12:00:00

import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def relu_triton(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### results.json 示例

```json
{
  "problem_info": {
    "problem_id": 19,
    "name": "19_ReLU.py",
    "level": 1,
    "code": "import torch..."
  },
  "results": {
    "best_kernel": "...",
    "best_speedup": 1.45,
    "successful_iterations": 3,
    "total_iterations": 5
  }
}
```

## 常见问题

### Q: 如何找到可用的problem_id？

A: 查看KernelBench数据集或使用以下命令：

```bash
# 查看Level 1的所有问题
ls KernelBench/KernelBench/level1/

# 或者查看HuggingFace数据集
python -c "
from datasets import load_dataset
ds = load_dataset('ScalingIntelligence/KernelBench')
print([item['problem_id'] for item in ds['level_1']])
"
```

### Q: 生成失败怎么办？

A: 检查以下几点：
1. API密钥是否正确设置
2. 网络连接是否正常
3. 增加`--log-level DEBUG`查看详细日志
4. 尝试不同的temperature参数

### Q: 如何提高生成质量？

A: 
1. 增加迭代次数：`--iterations 10`
2. 启用性能评估：`--evaluate`
3. 调整temperature：`--temperature 0.1`
4. 使用更强的模型

### Q: 支持哪些LLM？

A: 目前支持：
- DeepSeek (推荐)
- OpenAI GPT系列

可以通过修改`src/kernelgen/llm/client.py`添加更多LLM支持。

## 性能优化建议

1. **GPU选择**: 使用高性能GPU (V100, A100, H100等)
2. **批量测试**: 对同一算子进行多轮迭代
3. **参数调优**: 尝试不同的temperature和max_tokens
4. **形状优化**: 针对特定输入形状优化kernel

## 扩展开发

### 添加新的算子类型

1. 在`prompts/triton_prompts.py`中添加专门的提示模板
2. 在`performance_benchmark.py`中添加相应的性能测试方法
3. 更新`kernel_generator.py`中的算子类型判断逻辑

### 集成新的LLM

1. 在`llm/client.py`中添加新的API客户端
2. 更新配置文件支持新的server_type
3. 添加相应的环境变量和认证逻辑