"""
高级Softmax算子生成示例
"""

import asyncio
import logging
import os
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.generator import KernelGenerator

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def generate_softmax_kernel():
    """生成Softmax函数的Triton kernel"""
    
    # 检查API密钥
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.error("请设置DEEPSEEK_API_KEY环境变量")
        return
    
    # 创建kernel生成器，使用自定义配置
    config = {
        "generation": {
            "max_iterations": 8,  # 增加迭代次数，因为softmax更复杂
            "enable_performance_test": True
        },
        "verification": {
            "timeout": 600  # 增加验证超时时间
        }
    }
    
    generator = KernelGenerator(config=config)
    
    # 定义任务描述
    task_desc = """
    实现一个高性能的Softmax函数Triton kernel。
    
    功能要求：
    - 输入：二维张量 x，形状为 [batch_size, seq_len]
    - 输出：二维张量 y，形状与输入相同
    - 计算：y[i,j] = exp(x[i,j] - max(x[i,:])) / sum(exp(x[i,k] - max(x[i,:])) for k in range(seq_len))
    
    数值稳定性：
    - 使用减去最大值的技巧避免数值溢出
    - 确保输出概率和为1
    - 处理极端值情况（如全零输入）
    
    性能优化：
    - 每个线程块处理一行或多行数据
    - 使用共享内存存储中间结果
    - 优化归约操作（max和sum）
    - 考虑向量化和内存合并访问
    
    实现细节：
    - 支持不同的序列长度
    - 处理边界条件
    - 添加数值检查和错误处理
    - 考虑不同数据类型的精度要求
    """
    
    try:
        logger.info("开始生成Softmax kernel...")
        
        # 生成kernel
        result = await generator.generate_kernel(
            op_name="softmax",
            task_desc=task_desc,
            input_shapes=[(32, 128), (64, 256), (128, 512)],  # 多种batch和序列长度
            dtype="float32"
        )
        
        if result.get("success", True):
            print("\n" + "="*60)
            print("Softmax Kernel生成成功！")
            print("="*60)
            
            print("\n算法设计摘要：")
            print("-"*40)
            design = result.get("design", {})
            if design:
                print(f"算法分析: {design.get('algorithm_analysis', 'N/A')[:200]}...")
                print(f"设计策略: {design.get('design_strategy', 'N/A')[:200]}...")
                
                opt_suggestions = design.get('optimization_suggestions', [])
                if opt_suggestions:
                    print("\n优化建议:")
                    for i, suggestion in enumerate(opt_suggestions[:3], 1):
                        print(f"{i}. {suggestion}")
            
            print("\n生成的Triton Kernel代码：")
            print("-"*40)
            print(result["kernel_code"])
            
            if result.get("performance_metrics"):
                print("\n性能指标：")
                print("-"*40)
                for metric, value in result["performance_metrics"].items():
                    print(f"{metric}: {value}")
            
            print(f"\n总迭代次数: {result.get('iterations', 'N/A')}")
            
        else:
            print(f"Kernel生成失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        logger.error(f"生成过程出现异常: {str(e)}")


if __name__ == "__main__":
    asyncio.run(generate_softmax_kernel())