#!/usr/bin/env python3
"""
Level 1 Problem 19 - ReLU算子实现示例
演示如何使用KernelGen为ReLU算子生成Triton kernel
"""

import os
import sys
import yaml
import logging

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kernelgen.core.kernel_generator import KernelGenerator
from kernelgen.evaluation.evaluator import KernelEvaluator

def create_relu_config():
    """创建ReLU算子的配置"""
    config = {
        "dataset": {
            "source": "huggingface",
            "name": "ScalingIntelligence/KernelBench", 
            "level": 1,
            "problem_ids": [19]  # ReLU算子
        },
        "generation": {
            "max_iterations": 10,
            "early_stop_threshold": 1.2,
            "min_successful_iterations": 3,
            "backend": "triton",
            "llm": {
                "server_type": "deepseek",
                "model_name": "deepseek-coder",
                "temperature": 0.0,
                "max_tokens": 4096
            }
        },
        "performance": {
            "device": "cuda",
            "warmup_runs": 10,
            "benchmark_runs": 100
        },
        "output": {
            "base_dir": "relu_runs",
            "run_name": "relu_experiment",
            "save_all_kernels": True,
            "save_best_only": False,
            "save_logs": True,
            "save_performance_data": True
        },
        "evaluation": {
            "num_correct_trials": 5,
            "num_perf_trials": 100,
            "pass_at_k_values": [1, 3, 5]
        },
        "concurrency": {
            "num_workers": 1,
            "api_query_interval": 0.1
        }
    }
    return config

def generate_relu_kernel():
    """生成ReLU kernel"""
    print("🚀 开始生成ReLU Triton kernel")
    print("=" * 50)
    
    # 检查环境
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 请设置DEEPSEEK_API_KEY环境变量")
        return False
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  警告: CUDA不可用，将在CPU模式下运行")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 创建配置
    config = create_relu_config()
    
    # 显示配置信息
    print(f"数据集: Level {config['dataset']['level']} Problem {config['dataset']['problem_ids'][0]}")
    print(f"算子类型: ReLU激活函数")
    print(f"最大迭代: {config['generation']['max_iterations']}")
    print(f"LLM: {config['generation']['llm']['server_type']}/{config['generation']['llm']['model_name']}")
    print()
    
    try:
        # 创建生成器
        generator = KernelGenerator(config)
        
        # 批量生成
        print("开始生成ReLU kernel...")
        summary = generator.generate_batch()
        
        # 显示结果
        print("\n" + "=" * 50)
        print("🎯 ReLU Kernel生成结果:")
        print("=" * 50)
        print(f"总结果数: {summary['total_results']}")
        print(f"成功结果数: {summary['successful_results']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        print(f"平均加速比: {summary['average_speedup']:.2f}x")
        print(f"最大加速比: {summary['max_speedup']:.2f}x")
        print(f"执行时间: {summary['execution_time']:.1f}秒")
        
        # 显示最佳结果
        if 19 in summary['problem_stats']:
            stats = summary['problem_stats'][19]
            print(f"\nReLU算子统计:")
            print(f"  成功迭代: {stats['successful']}/{stats['total']}")
            print(f"  最佳加速比: {stats['best_speedup']:.2f}x")
        
        return summary
        
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        return None

def evaluate_relu_kernel():
    """评估生成的ReLU kernel"""
    print("\n🧪 开始评估ReLU kernel")
    print("=" * 50)
    
    config = create_relu_config()
    
    try:
        # 创建评估器
        evaluator = KernelEvaluator(config)
        
        # 检查是否有生成的kernel
        run_dir = os.path.join(config["output"]["base_dir"], config["output"]["run_name"])
        if not os.path.exists(run_dir):
            print("❌ 未找到生成的kernel，请先运行生成")
            return None
        
        # 评估
        print("开始评估ReLU kernel...")
        eval_summary = evaluator.evaluate_batch()
        
        # 显示结果
        print("✅ ReLU Kernel评估完成!")
        print(f"   评估的kernel数: {eval_summary['total_evaluated']}")
        print(f"   编译成功率: {eval_summary['compilation_success_rate']:.1f}%")
        print(f"   正确性通过率: {eval_summary['correctness_pass_rate']:.1f}%")
        
        if 'pass_at_k' in eval_summary:
            print("   Pass@K结果:")
            for k, rate in eval_summary['pass_at_k'].items():
                print(f"     {k}: {rate:.1f}%")
        
        if 'performance_stats' in eval_summary and eval_summary['performance_stats']:
            perf = eval_summary['performance_stats']
            print(f"   平均加速比: {perf.get('avg_speedup', 0):.2f}x")
            print(f"   最大加速比: {perf.get('max_speedup', 0):.2f}x")
        
        return eval_summary
        
    except Exception as e:
        print(f"❌ 评估失败: {str(e)}")
        return None

def show_relu_kernel_code():
    """显示生成的ReLU kernel代码"""
    config = create_relu_config()
    run_dir = os.path.join(config["output"]["base_dir"], config["output"]["run_name"])
    kernel_file = os.path.join(run_dir, "problem_19_best_kernel.py")
    
    if os.path.exists(kernel_file):
        print("\n📄 生成的最佳ReLU Triton Kernel:")
        print("=" * 50)
        with open(kernel_file, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print("❌ 未找到生成的kernel文件")

def create_manual_relu_example():
    """创建手动ReLU Triton kernel示例"""
    relu_kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    input_ptr,  # 输入张量指针
    output_ptr, # 输出张量指针
    n_elements, # 元素总数
    BLOCK_SIZE: tl.constexpr, # 块大小
):
    """
    ReLU Triton kernel实现
    """
    # 计算当前线程块的起始位置
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码，确保不越界
    mask = offsets < n_elements
    
    # 加载输入数据
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # 计算ReLU: max(0, x)
    output = tl.maximum(x, 0.0)
    
    # 存储输出数据
    tl.store(output_ptr + offsets, output, mask=mask)

def relu_triton(x):
    """
    ReLU Triton wrapper函数
    """
    # 确保输入在GPU上
    assert x.is_cuda, "输入张量必须在CUDA设备上"
    
    # 创建输出张量
    output = torch.empty_like(x)
    
    # 计算元素总数
    n_elements = x.numel()
    
    # 选择合适的块大小
    BLOCK_SIZE = 1024
    
    # 计算网格大小
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 启动kernel
    relu_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# 测试函数
def test_relu_kernel():
    """测试ReLU kernel的正确性"""
    import torch
    
    # 创建测试数据
    batch_size = 4096
    dim = 393216
    x = torch.randn(batch_size, dim, device='cuda')
    
    # PyTorch结果
    pytorch_result = torch.relu(x)
    
    # Triton结果
    triton_result = relu_triton(x)
    
    # 比较结果
    max_diff = torch.max(torch.abs(pytorch_result - triton_result))
    print(f"最大差异: {max_diff.item()}")
    
    # 检查是否相等
    is_close = torch.allclose(pytorch_result, triton_result, rtol=1e-5, atol=1e-5)
    print(f"结果是否相等: {is_close}")
    
    return is_close

if __name__ == "__main__":
    print("测试手动实现的ReLU kernel:")
    test_result = test_relu_kernel()
    print(f"测试结果: {'✅ 通过' if test_result else '❌ 失败'}")
'''
    
    print("\n📝 手动ReLU Triton Kernel示例:")
    print("=" * 50)
    print(relu_kernel_code)
    
    # 保存示例代码
    example_file = "manual_relu_kernel.py"
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(relu_kernel_code)
    print(f"\n💾 示例代码已保存到: {example_file}")

def main():
    """主函数"""
    print("🎯 Level 1 Problem 19 - ReLU算子实现")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    print("选择操作模式:")
    print("1. 自动生成ReLU kernel")
    print("2. 评估生成的kernel")
    print("3. 显示生成的kernel代码")
    print("4. 查看手动实现示例")
    print("5. 完整流程（生成+评估）")
    
    try:
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == "1":
            summary = generate_relu_kernel()
            if summary:
                print("\n✅ ReLU kernel生成完成!")
        
        elif choice == "2":
            eval_summary = evaluate_relu_kernel()
            if eval_summary:
                print("\n✅ ReLU kernel评估完成!")
        
        elif choice == "3":
            show_relu_kernel_code()
        
        elif choice == "4":
            create_manual_relu_example()
        
        elif choice == "5":
            print("🚀 开始完整流程...")
            summary = generate_relu_kernel()
            if summary and summary['successful_results'] > 0:
                eval_summary = evaluate_relu_kernel()
                show_relu_kernel_code()
                print("\n🎉 完整流程完成!")
            else:
                print("\n❌ 生成失败，跳过评估")
        
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")

if __name__ == "__main__":
    main()