"""
批量生成多个算子的示例
"""

import asyncio
import logging
import os
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.generator import KernelGenerator
from core.task import Task

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def batch_generate_kernels():
    """批量生成多个常用算子的Triton kernel"""
    
    # 检查API密钥
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.error("请设置DEEPSEEK_API_KEY环境变量")
        return
    
    # 定义要生成的算子列表
    operators = [
        {
            "op_name": "gelu",
            "task_desc": """
            实现GELU激活函数的Triton kernel。
            GELU(x) = x * Φ(x)，其中Φ(x)是标准正态分布的累积分布函数。
            近似公式：GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            """,
            "input_shapes": [(1024, 768), (2048, 1024)],
            "dtype": "float32"
        },
        {
            "op_name": "layer_norm",
            "task_desc": """
            实现Layer Normalization的Triton kernel。
            对每个样本的特征维度进行归一化：
            y = (x - mean) / sqrt(var + eps) * gamma + beta
            其中mean和var是在最后一个维度上计算的。
            """,
            "input_shapes": [(32, 512), (64, 768)],
            "dtype": "float32"
        },
        {
            "op_name": "matrix_add",
            "task_desc": """
            实现矩阵加法的Triton kernel。
            支持广播机制，计算 C = A + B。
            要求高效的内存访问模式和向量化操作。
            """,
            "input_shapes": [(1024, 1024), (2048, 2048)],
            "dtype": "float32"
        }
    ]
    
    # 创建生成器
    generator = KernelGenerator()
    
    # 记录开始时间
    start_time = time.time()
    
    print("开始批量生成Triton kernels...")
    print("="*60)
    
    results = []
    
    # 顺序生成每个算子
    for i, op_config in enumerate(operators, 1):
        print(f"\n[{i}/{len(operators)}] 生成 {op_config['op_name']} kernel...")
        
        try:
            # 创建任务
            task = Task(
                task_id=f"task_{i}",
                op_name=op_config["op_name"],
                task_desc=op_config["task_desc"],
                input_shapes=op_config["input_shapes"],
                dtype=op_config["dtype"]
            )
            
            # 执行任务
            result = await task.run()
            results.append((op_config["op_name"], result))
            
            if result.success:
                print(f"✓ {op_config['op_name']} 生成成功")
                if result.performance_metrics:
                    avg_time = result.performance_metrics.get("avg_time_ms", "N/A")
                    print(f"  平均执行时间: {avg_time} ms")
            else:
                print(f"✗ {op_config['op_name']} 生成失败: {result.error_message}")
                
        except Exception as e:
            print(f"✗ {op_config['op_name']} 生成异常: {str(e)}")
            results.append((op_config["op_name"], None))
    
    # 统计结果
    total_time = time.time() - start_time
    successful = sum(1 for _, result in results if result and result.success)
    
    print("\n" + "="*60)
    print("批量生成完成！")
    print("="*60)
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"成功生成: {successful}/{len(operators)} 个算子")
    
    # 显示详细结果
    print("\n详细结果:")
    print("-"*60)
    for op_name, result in results:
        if result and result.success:
            exec_time = result.execution_time or 0
            print(f"✓ {op_name:<15} - 成功 (耗时: {exec_time:.2f}s)")
        else:
            error = result.error_message if result else "执行异常"
            print(f"✗ {op_name:<15} - 失败 ({error[:50]}...)")
    
    # 保存成功的kernel代码
    print(f"\n保存生成的kernel代码...")
    os.makedirs("generated_kernels", exist_ok=True)
    
    for op_name, result in results:
        if result and result.success and result.kernel_code:
            filename = f"generated_kernels/{op_name}_kernel.py"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {op_name.upper()} Triton Kernel\n")
                f.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(result.kernel_code)
            print(f"  {op_name} -> {filename}")


if __name__ == "__main__":
    asyncio.run(batch_generate_kernels())