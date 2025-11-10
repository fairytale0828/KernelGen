#!/usr/bin/env python3
"""
基于LangChain的多Agent协作Triton Kernel生成器
使用LangChain框架重构的现代化Agent系统

使用方法:
python generate_kernel_langchain.py --level 1 --problem-id 19 --iterations 5
python generate_kernel_langchain.py --level 2 --problem-id 1 --iterations 10
"""

import os
import sys
import argparse
import logging
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kernelgen.llm.chat_models import create_chat_model
from kernelgen.chains.orchestration_chain import OrchestrationChain
from kernelgen.database import KernelBenchLoader

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
            "min_successful_iterations": kwargs.get("min_success", 2),
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
            "run_name": f"langchain_level_{level}_problem_{problem_id}_{int(time.time())}",
            "save_all_kernels": True,
            "save_logs": True,
            "save_performance_data": True
        }
    }
    return config

def save_results(result_summary: Dict[str, Any], config: Dict[str, Any]):
    """保存结果"""
    output_dir = os.path.join(config["output"]["base_dir"], config["output"]["run_name"])
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存生成的kernel
    if result_summary.get("best_kernel"):
        kernel_file = os.path.join(output_dir, "generated_kernel.py")
        with open(kernel_file, 'w', encoding='utf-8') as f:
            f.write(f"# Generated Triton Kernel (LangChain Multi-Agent)\n")
            f.write(f"# Level: {config['dataset']['level']}\n")
            f.write(f"# Problem ID: {config['dataset']['problem_ids'][0]}\n")
            f.write(f"# Operation: {result_summary.get('operation_name', 'Unknown')}\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Best Speedup: {result_summary.get('best_speedup', 0):.2f}x\n\n")
            f.write(result_summary["best_kernel"])
    
    # 保存结果摘要
    summary_file = os.path.join(output_dir, "results.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 结果已保存到: {output_dir}")
    return output_dir

async def generate_kernel_async(level: int, problem_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """异步生成kernel"""
    
    # 1. 初始化数据库加载器
    print("📊 初始化KernelBench数据库...")
    db_loader = KernelBenchLoader()
    
    try:
        # 验证问题是否存在
        problem_info = db_loader.get_problem(level, problem_id)
        print(f"✅ 加载问题: {problem_info['operation_name']} (Level {level} Problem {problem_id})")
        
        # 验证问题可执行性
        if not db_loader.validate_problem(level, problem_id):
            raise ValueError(f"问题验证失败: Level {level} Problem {problem_id}")
        
    except Exception as e:
        print(f"❌ 数据库加载失败: {e}")
        return {"success": False, "error": str(e)}
    
    # 2. 初始化LangChain聊天模型
    print("🤖 初始化LangChain聊天模型...")
    llm_config = config["generation"]["llm"]
    chat_model = create_chat_model(llm_config)
    
    # 3. 初始化编排链
    print("🎭 初始化LangChain编排链...")
    orchestration_chain = OrchestrationChain(chat_model, config)
    
    # 4. 开始生成过程
    print(f"🚀 开始生成Triton kernel...")
    print(f"   问题: {problem_info['operation_name']}")
    print(f"   级别: {level}")
    print(f"   问题ID: {problem_id}")
    print(f"   最大迭代: {config['generation']['max_iterations']}")
    
    start_time = time.time()
    
    try:
        # 执行生成
        result = await orchestration_chain.generate_kernel(level, problem_id)
        
        # 记录结果
        generation_time = time.time() - start_time
        
        # 构建最终结果
        final_result = {
            "success": result.get("success", False),
            "level": level,
            "problem_id": problem_id,
            "operation_name": result.get("operation_name", "Unknown"),
            "generation_time_seconds": generation_time,
            "total_iterations": result.get("total_iterations", 0),
            "successful_iterations": result.get("successful_iterations", 0),
            "best_speedup": result.get("best_speedup", 0),
            "best_kernel": result.get("best_kernel", ""),
            "best_kernel_path": result.get("best_kernel_path"),
            "session_summary_file": result.get("session_summary_file"),
            "iteration_summary": result.get("iteration_summary", []),
            "langchain_result": result
        }
        
        return final_result
        
    except Exception as e:
        print(f"❌ 生成过程出错: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "level": level,
            "problem_id": problem_id,
            "generation_time_seconds": time.time() - start_time
        }
    
    finally:
        # 清理资源
        db_loader.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于LangChain的多Agent协作Triton Kernel生成器")
    
    # 必需参数
    parser.add_argument("--level", type=int, required=True, 
                       help="KernelBench级别 (1-4)")
    parser.add_argument("--problem-id", type=int, required=True,
                       help="问题ID")
    
    # 可选参数
    parser.add_argument("--iterations", type=int, default=5,
                       help="最大迭代次数 (默认: 5)")
    parser.add_argument("--threshold", type=float, default=1.2,
                       help="早停阈值 (默认: 1.2)")
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
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    print("🚀 基于LangChain的多Agent协作Triton Kernel生成器")
    print("=" * 70)
    print(f"Level: {args.level}")
    print(f"Problem ID: {args.problem_id}")
    print(f"LLM: {args.server_type}/{args.model_name}")
    print(f"最大迭代次数: {args.iterations}")
    print(f"早停阈值: {args.threshold}x")
    print(f"框架: LangChain")
    print()
    
    try:
        # 1. 验证环境
        cuda_available = validate_environment()
        
        # 2. 创建配置
        config = create_config(
            args.level, args.problem_id,
            iterations=args.iterations,
            threshold=args.threshold,
            server_type=args.server_type,
            model_name=args.model_name,
            temperature=args.temperature,
            output_dir=args.output_dir,
            cuda_available=cuda_available
        )
        
        # 3. 开始异步生成
        print(f"📚 开始处理 Level {args.level} Problem {args.problem_id}")
        start_time = time.time()
        
        result_summary = asyncio.run(generate_kernel_async(args.level, args.problem_id, config))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 4. 显示结果
        print("\n🎯 生成完成!")
        print("=" * 70)
        
        if result_summary["success"]:
            print(f"✅ 成功生成kernel")
            print(f"   问题: {result_summary['operation_name']}")
            print(f"   总迭代次数: {result_summary['total_iterations']}")
            print(f"   成功迭代次数: {result_summary['successful_iterations']}")
            print(f"   最佳加速比: {result_summary['best_speedup']:.2f}x")
            print(f"   总耗时: {total_time:.1f}秒")
            
            # 保存结果
            output_dir = save_results(result_summary, config)
            
            print(f"\n📋 结果文件:")
            print(f"   最佳kernel: {result_summary.get('best_kernel_path', 'N/A')}")
            print(f"   会话摘要: {result_summary.get('session_summary_file', 'N/A')}")
            print(f"   输出目录: {output_dir}")
            
        else:
            print(f"❌ 生成失败")
            if "error" in result_summary:
                print(f"   错误: {result_summary['error']}")
            print(f"   总耗时: {total_time:.1f}秒")
            
            # 即使失败也尝试保存部分结果
            if result_summary.get("best_kernel"):
                output_dir = save_results(result_summary, config)
                print(f"   部分结果已保存到: {output_dir}")
        
        # 5. 显示详细统计
        if result_summary.get('iteration_summary'):
            print(f"\n📊 迭代详情:")
            for iter_info in result_summary['iteration_summary']:
                status = "✅" if iter_info['success'] else "❌"
                speedup = iter_info.get('speedup', 0.0)
                print(f"   迭代 {iter_info['iteration']}: {status} 加速比 {speedup:.2f}x")
        
        return result_summary["success"]
        
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