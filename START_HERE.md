# 🚀 开始使用 - 多Worker进化式搜索

## 最简单的启动方式

### 方式1: 使用现有入口（推荐）

```bash
# 使用多Worker搜索模式
python generate_kernel.py --level 2 --problem-id 40 --use-multi-worker --workers 4 --rounds 10
```

### 方式2: 快速测试（2-3分钟）

```bash
# 使用较少的worker和轮次进行快速测试
python generate_kernel.py --level 2 --problem-id 40 --use-multi-worker --workers 2 --rounds 3
```

### 方式3: 标准模式（不使用多Worker）

```bash
# 使用原有的迭代优化模式
python generate_kernel.py --level 2 --problem-id 40 --iterations 5
```

## 命令行参数说明

### 基础参数
- `--level`: KernelBench级别 (1-4)
- `--problem-id`: 问题ID
- `--problem-range`: 问题范围，如 `1-10`
- `--problem-list`: 问题列表，如 `1,5,10,19`

### 多Worker搜索参数（新功能）
- `--use-multi-worker`: 启用多Worker进化式搜索
- `--workers`: 并行worker数量（默认4）
- `--rounds`: 搜索轮次（默认10）
- `--ub`: 上界阈值（默认2.0）
- `--lb`: 下界阈值（默认1.2）
- `--dt`: 退化阈值（默认0.5）

### LLM参数
- `--server-type`: LLM服务器类型（默认deepseek）
- `--model-name`: 模型名称（默认deepseek-coder）
- `--temperature`: 温度参数（默认0.0）

### 其他参数
- `--output-dir`: 输出目录（默认generated_kernels）
- `--log-level`: 日志级别（DEBUG/INFO/WARNING/ERROR）

## 使用示例

### 1. 快速测试单个问题

```bash
python generate_kernel.py \
  --level 2 \
  --problem-id 40 \
  --use-multi-worker \
  --workers 2 \
  --rounds 3
```

### 2. 标准搜索单个问题

```bash
python generate_kernel.py \
  --level 2 \
  --problem-id 40 \
  --use-multi-worker \
  --workers 4 \
  --rounds 10
```

### 3. 深度搜索单个问题

```bash
python generate_kernel.py \
  --level 2 \
  --problem-id 40 \
  --use-multi-worker \
  --workers 8 \
  --rounds 20 \
  --ub 2.5 \
  --lb 1.5
```

### 4. 批量处理多个问题

```bash
# 处理问题40-42
python generate_kernel.py \
  --level 2 \
  --problem-range 40-42 \
  --use-multi-worker \
  --workers 4 \
  --rounds 10
```

### 5. 处理指定问题列表

```bash
python generate_kernel.py \
  --level 2 \
  --problem-list 40,41,42 \
  --use-multi-worker \
  --workers 4 \
  --rounds 10
```

## 配置建议

### GPU内存充足 (>16GB)
```bash
--workers 8 --rounds 15
```

### GPU内存有限 (8-16GB)
```bash
--workers 4 --rounds 10
```

### 快速原型测试
```bash
--workers 2 --rounds 3
```

## 查看结果

### 生成的文件位置

```bash
# 单个问题结果
generated_kernels/level_2/problem_40/

# 批量结果
generated_kernels/langchain_level_2_batch_*/

# 知识库
generated_kernels/knowledge_base.pkl
```

### 查看最佳kernel

```bash
# 查看生成的kernel
cat generated_kernels/level_2/problem_40/generated_kernel_*.py

# 查看结果摘要
cat generated_kernels/level_2/problem_40/results_*.json
```

## 对比两种模式

### 标准迭代模式（原有）
```bash
python generate_kernel.py --level 2 --problem-id 40 --iterations 5
```
- ✅ 简单直接
- ✅ 资源占用少
- ❌ 只尝试一种优化方向
- ❌ 不支持策略学习

### 多Worker搜索模式（新功能）
```bash
python generate_kernel.py --level 2 --problem-id 40 --use-multi-worker --workers 4 --rounds 10
```
- ✅ 并行尝试多种策略
- ✅ 自动策略融合
- ✅ 长期学习优化
- ✅ 更高的成功率
- ❌ 资源占用较多
- ❌ 运行时间较长

## 常见问题

### Q: 如何选择使用哪种模式？

**快速测试**: 使用标准模式
```bash
python generate_kernel.py --level 2 --problem-id 40 --iterations 3
```

**追求最佳性能**: 使用多Worker模式
```bash
python generate_kernel.py --level 2 --problem-id 40 --use-multi-worker --workers 4 --rounds 10
```

### Q: 多Worker模式需要多长时间？

- 快速测试（2 workers, 3 rounds）: 2-3分钟
- 标准搜索（4 workers, 10 rounds）: 10-15分钟
- 深度搜索（8 workers, 20 rounds）: 30-40分钟

### Q: GPU内存不足怎么办？

减少worker数量:
```bash
python generate_kernel.py --level 2 --problem-id 40 --use-multi-worker --workers 2 --rounds 5
```

### Q: 如何查看详细日志？

```bash
python generate_kernel.py --level 2 --problem-id 40 --use-multi-worker --log-level DEBUG
```

### Q: 知识库在哪里？

```bash
# 默认位置
generated_kernels/knowledge_base.pkl

# 查看知识库内容
python -c "
from kernelgen.core import KnowledgeBase
kb = KnowledgeBase()
kb.load('generated_kernels/knowledge_base.pkl')
print(kb.get_statistics_summary())
"
```

## 下一步

1. ✅ 运行快速测试验证安装
2. ✅ 尝试标准搜索
3. ✅ 查看生成的kernel
4. ✅ 尝试批量处理
5. ✅ 查看知识库学习效果

## 获取帮助

- 📖 完整文档: `docs/MULTI_WORKER_SEARCH.md`
- 🎯 快速指南: `QUICK_START.md`
- 📋 示例代码: `examples/multi_worker_search_example.py`
- 🔧 实现报告: `MULTI_WORKER_IMPLEMENTATION_COMPLETE.md`

## 验证安装

```bash
# 测试导入
python test_import.py

# 运行基础测试
python examples/test_multi_worker_basic.py
```

---

**现在就开始！** 🚀

```bash
python generate_kernel.py --level 2 --problem-id 40 --use-multi-worker --workers 2 --rounds 3
```
