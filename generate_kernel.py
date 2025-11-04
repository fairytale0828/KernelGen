#!/usr/bin/env python3
"""
通用Kernel生成脚本
支持通过level和problem_id从KernelBench数据集生成Triton kernel

使用方法:
python generate_kernel.py --level 2 --problem-id 40
python generate_kernel.py --level 1 --problem-id 19 --iterations 5
python generate_kernel.py --level 2 --problem-id 40 --evaluate
"""

import os
import sys
import argparse
import logging
import time
import json
from typing import Dict, Any, Optional

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kernelgen.dataset import KernelBenchLoader
from kernelgen.llm import LLMClient
from kernelgen.prompts import get_triton_generation_prompt
from kernelgen.core.performance_benchmark import TritonPerformanceBenchmark
from kernelgen.core.iterative_optimizer import IterativeOptimizer

def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def validate_environment():
    """验证环境"""
    # 检查API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
    
    # 检查CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU模式")
        return cuda_available
    except ImportError:
        raise ImportError("PyTorch未安装")

def create_config(level: int, problem_id: int, **kwargs) -> Dict[str, Any]:
    """创建配置"""
    config = {
        "dataset": {
            "source": "huggingface",
            "name": "ScalingIntelligence/KernelBench",
            "level": level,
            "problem_ids": [problem_id]
        },
        "generation": {
            "max_iterations": kwargs.get("iterations", 10),
            "early_stop_threshold": kwargs.get("threshold", 1.2),
            "min_successful_iterations": kwargs.get("min_success", 3),
            "backend": "triton",
            "llm": {
                "server_type": kwargs.get("server_type", "deepseek"),
                "model_name": kwargs.get("model_name", "deepseek-coder"),
                "temperature": kwargs.get("temperature", 0.0),
                "max_tokens": kwargs.get("max_tokens", 4096)
            }
        },
        "performance": {
            "device": "cuda" if kwargs.get("cuda_available", True) else "cpu",
            "warmup_runs": kwargs.get("warmup_runs", 10),
            "benchmark_runs": kwargs.get("benchmark_runs", 100)
        },
        "output": {
            "base_dir": kwargs.get("output_dir", "generated_kernels"),
            "run_name": f"level_{level}_problem_{problem_id}_{int(time.time())}",
            "save_all_kernels": True,
            "save_logs": True,
            "save_performance_data": True
        }
    }
    return config

def load_problem_from_kernelbench(level: int, problem_id: int) -> Dict[str, Any]:
    """从KernelBench加载问题"""
    print(f"📚 从KernelBench加载 Level {level} Problem {problem_id}")
    
    # 创建数据集配置
    dataset_config = {
        "source": "huggingface",
        "name": "ScalingIntelligence/KernelBench",
        "level": level,
        "problem_ids": [problem_id]
    }
    
    # 加载数据集
    loader = KernelBenchLoader(dataset_config)
    loader.load_dataset()
    
    # 获取问题信息
    problem_info = loader.get_problem_by_id(problem_id)
    
    print(f"✅ 成功加载问题: {problem_info['name']}")
    return problem_info

def generate_triton_kernel(pytorch_code: str, llm_config: Dict[str, Any]) -> Optional[str]:
    """使用LLM生成Triton kernel"""
    print("🤖 使用LLM生成Triton kernel...")
    
    # 创建LLM客户端
    client = LLMClient(llm_config)
    
    # 构造提示
    prompt = get_triton_generation_prompt(pytorch_code)
    system_prompt = "You are an expert Triton GPU kernel programmer. Generate high-performance, correct Triton kernels."
    
    # 生成代码
    try:
        generated_text = client.generate(prompt, system_prompt)
        kernel_code = client.extract_code_block(generated_text, "python")
        
        if kernel_code:
            print("✅ Triton kernel生成成功")
            return kernel_code
        else:
            print("❌ 未能从生成文本中提取代码")
            return None
            
    except Exception as e:
        print(f"❌ LLM生成失败: {str(e)}")
        return None

def validate_and_benchmark_kernel(kernel_code: str, pytorch_code: str, 
                                 performance_config: Dict[str, Any]) -> Dict[str, Any]:
    """验证和基准测试kernel"""
    print("🧪 验证和测试生成的kernel...")
    
    # 创建性能测试器
    benchmark = TritonPerformanceBenchmark(
        device=performance_config["device"],
        warmup_runs=performance_config["warmup_runs"],
        benchmark_runs=performance_config["benchmark_runs"]
    )
    
    results = {
        "syntax_valid": False,
        "compilation_success": False,
        "performance_results": None,
        "error_message": None
    }
    
    try:
        # 1. 语法验证
        is_valid, error_msg = benchmark.validate_kernel_syntax(kernel_code)
        if not is_valid:
            results["error_message"] = f"语法验证失败: {error_msg}"
            return results
        
        results["syntax_valid"] = True
        print("✅ 语法验证通过")
        
        # 2. 编译测试
        compiled_func = benchmark.compile_and_load_kernel(kernel_code, "generated_kernel")
        if not compiled_func:
            results["error_message"] = "编译失败"
            return results
        
        results["compilation_success"] = True
        print("✅ 编译成功")
        
        # 3. 创建PyTorch参考函数
        pytorch_func = create_pytorch_reference(pytorch_code)
        if not pytorch_func:
            results["error_message"] = "无法创建PyTorch参考函数"
            return results
        
        # 4. 生成测试输入
        test_inputs = generate_test_inputs(pytorch_code, performance_config["device"])
        if not test_inputs:
            results["error_message"] = "无法生成测试输入"
            return results
        
        # 5. 性能测试
        perf_results = benchmark.benchmark_general(compiled_func, pytorch_func, test_inputs)
        results["performance_results"] = perf_results
        
        if perf_results["success"]:
            speedup = perf_results["speedups"][0] if perf_results["speedups"] else 0
            print(f"✅ 性能测试完成，加速比: {speedup:.2f}x")
        else:
            print(f"⚠️  性能测试失败: {perf_results.get('error', '未知错误')}")
        
    except Exception as e:
        results["error_message"] = f"测试过程异常: {str(e)}"
        print(f"❌ 测试失败: {str(e)}")
    
    return results

def create_pytorch_reference(pytorch_code: str):
    """创建PyTorch参考函数"""
    try:
        # 执行PyTorch代码
        exec_globals = {}
        exec(pytorch_code, exec_globals)
        
        # 获取模型类和初始化参数
        model_class = exec_globals.get("Model")
        get_init_inputs = exec_globals.get("get_init_inputs")
        
        if not model_class or not get_init_inputs:
            return None
        
        # 创建模型实例
        init_inputs = get_init_inputs()
        model = model_class(*init_inputs)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"创建PyTorch参考函数失败: {str(e)}")
        return None

def generate_test_inputs(pytorch_code: str, device: str):
    """生成测试输入"""
    try:
        # 执行PyTorch代码获取输入生成函数
        exec_globals = {}
        exec(pytorch_code, exec_globals)
        
        get_inputs = exec_globals.get("get_inputs")
        if not get_inputs:
            return None
        
        inputs = get_inputs()
        
        # 将输入移动到指定设备
        if device == "cuda":
            import torch
            inputs = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        
        return inputs
        
    except Exception as e:
        print(f"生成测试输入失败: {str(e)}")
        return None

def save_results(kernel_code: str, results: Dict[str, Any], config: Dict[str, Any], 
                problem_info: Dict[str, Any]):
    """保存结果"""
    output_dir = os.path.join(config["output"]["base_dir"], config["output"]["run_name"])
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存生成的kernel
    kernel_file = os.path.join(output_dir, "generated_kernel.py")
    with open(kernel_file, 'w', encoding='utf-8') as f:
        f.write(f"# Generated Triton Kernel\n")
        f.write(f"# Level: {config['dataset']['level']}\n")
        f.write(f"# Problem ID: {config['dataset']['problem_ids'][0]}\n")
        f.write(f"# Problem Name: {problem_info['name']}\n")
        f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(kernel_code)
    
    # 保存结果摘要
    summary = {
        "problem_info": problem_info,
        "config": config,
        "results": results,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_file = os.path.join(output_dir, "results.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 结果已保存到: {output_dir}")
    return output_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="通用Triton Kernel生成器")
    
    # 必需参数
    parser.add_argument("--level", type=int, required=True, 
                       help="KernelBench级别 (1-4)")
    parser.add_argument("--problem-id", type=int, required=True,
                       help="问题ID")
    
    # 可选参数
    parser.add_argument("--iterations", type=int, default=1,
                       help="生成迭代次数 (默认: 1)")
    parser.add_argument("--server-type", default="deepseek",
                       help="LLM服务器类型 (默认: deepseek)")
    parser.add_argument("--model-name", default="deepseek-coder",
                       help="LLM模型名称 (默认: deepseek-coder)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="LLM温度参数 (默认: 0.0)")
    parser.add_argument("--output-dir", default="generated_kernels",
                       help="输出目录 (默认: generated_kernels)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别 (默认: INFO)")
    parser.add_argument("--evaluate", action="store_true",
                       help="是否进行性能评估")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    print("🚀 通用Triton Kernel生成器")
    print("=" * 50)
    print(f"Level: {args.level}")
    print(f"Problem ID: {args.problem_id}")
    print(f"LLM: {args.server_type}/{args.model_name}")
    print(f"迭代次数: {args.iterations}")
    print()
    
    try:
        # 1. 验证环境
        cuda_available = validate_environment()
        
        # 2. 创建配置
        config = create_config(
            args.level, args.problem_id,
            iterations=args.iterations,
            server_type=args.server_type,
            model_name=args.model_name,
            temperature=args.temperature,
            output_dir=args.output_dir,
            cuda_available=cuda_available
        )
        
        # 3. 加载问题
        problem_info = load_problem_from_kernelbench(args.level, args.problem_id)
        pytorch_code = problem_info["code"]
        
        print(f"📄 PyTorch算子描述:")
        print("-" * 30)
        print(pytorch_code[:500] + "..." if len(pytorch_code) > 500 else pytorch_code)
        print()
        
        # 4. 智能迭代优化
        if args.evaluate:
            print("🧠 使用智能迭代优化...")
            
            # 创建LLM客户端和性能测试器
            llm_client = LLMClient(config["generation"]["llm"])
            benchmark = TritonPerformanceBenchmark(
                device=config["performance"]["device"],
                warmup_runs=config["performance"]["warmup_runs"],
                benchmark_runs=config["performance"]["benchmark_runs"]
            )
            
            # 创建迭代优化器
            optimizer = IterativeOptimizer(llm_client, benchmark)
            
            # 运行迭代优化
            optimization_result = optimizer.optimize_kernel(pytorch_code, args.problem_id, config["generation"])
            
            if optimization_result["success"]:
                best_kernel = optimization_result["best_kernel"]
                best_speedup = optimization_result["best_speedup"]
                all_results = optimization_result["iteration_history"]
                
                print(f"\n🎯 智能优化完成!")
                print(f"   总迭代次数: {optimization_result['total_iterations']}")
                print(f"   成功迭代次数: {optimization_result['successful_iterations']}")
                print(f"   成功率: {optimization_result['success_rate']:.1f}%")
                print(f"   最佳加速比: {best_speedup:.2f}x")
            else:
                print(f"❌ 智能优化失败: {optimization_result['error_message']}")
                best_kernel = None
                best_speedup = 0.0
                all_results = []
        else:
            print("🔄 简单迭代生成...")
            best_kernel = None
            best_speedup = 0.0
            all_results = []
            
            # 简单迭代生成（不评估）
            for iteration in range(1, args.iterations + 1):
                print(f"🔄 第 {iteration}/{args.iterations} 轮生成")
                print("-" * 30)
                
                # 生成kernel
                kernel_code = generate_triton_kernel(pytorch_code, config["generation"]["llm"])
                
                if not kernel_code:
                    print(f"❌ 第 {iteration} 轮生成失败")
                    continue
                
                print(f"📝 生成的Triton kernel (前200字符):")
                print(kernel_code[:200] + "..." if len(kernel_code) > 200 else kernel_code)
                print()
                
                # 不评估时，保存第一个成功生成的kernel
                if not best_kernel:
                    best_kernel = kernel_code
                    all_results.append({
                        "iteration": iteration,
                        "kernel_code": kernel_code,
                        "compilation_success": None,
                        "runtime_success": None,
                        "correctness_success": None,
                        "performance_metrics": None
                    })
                
                print()
        
        # 5. 保存结果
        if best_kernel:
            if args.evaluate:
                # 使用优化结果
                final_results = {
                    "best_kernel": best_kernel,
                    "best_speedup": best_speedup,
                    "all_iterations": all_results,
                    "total_iterations": len(all_results),
                    "successful_iterations": len([r for r in all_results if r.get("correctness_success", False)])
                }
            else:
                # 简单生成结果
                final_results = {
                    "best_kernel": best_kernel,
                    "best_speedup": best_speedup,
                    "all_iterations": all_results,
                    "total_iterations": args.iterations,
                    "successful_iterations": len([r for r in all_results if r.get("kernel_code")])
                }
            
            output_dir = save_results(best_kernel, final_results, config, problem_info)
            
            # 显示最终结果
            print("🎯 生成完成!")
            print("=" * 50)
            print(f"成功迭代: {final_results['successful_iterations']}/{args.iterations}")
            if args.evaluate and best_speedup > 0:
                print(f"最佳加速比: {best_speedup:.2f}x")
            print(f"结果保存在: {output_dir}")
            
            # 显示使用建议
            print(f"\n📋 使用生成的kernel:")
            print(f"   查看代码: cat {output_dir}/generated_kernel.py")
            print(f"   查看结果: cat {output_dir}/results.json")
            
        else:
            print("❌ 所有迭代都失败了")
            return False
        
        return True
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        logging.error("执行失败", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)