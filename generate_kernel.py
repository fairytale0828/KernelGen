#!/usr/bin/env python3
"""
基于LangChain的多Agent协作Triton Kernel生成器
使用LangChain框架重构的现代化Agent系统

支持单个问题和批量生成两种模式：

单个问题生成:
python generate_kernel.py --level 1 --problem-id 19 --iterations 5

批量生成(整个级别):
python generate_kernel.py --level 1 --iterations 5

批量生成(指定问题范围):
python generate_kernel.py --level 1 --problem-range 1-10 --iterations 5

批量生成(指定问题列表):
python generate_kernel.py --level 1 --problem-list 1,5,10,19 --iterations 5
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

def create_config(level: int, problem_ids: list, **kwargs) -> Dict[str, Any]:
    """创建配置"""
    config = {
        "dataset": {
            "source": "huggingface",
            "name": "ScalingIntelligence/KernelBench",
            "level": level,
            "problem_ids": problem_ids
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
            "run_name": f"langchain_level_{level}_batch_{int(time.time())}",
            "save_all_kernels": True,
            "save_logs": True,
            "save_performance_data": True
        }
    }
    return config

def save_single_result(result_summary: Dict[str, Any], config: Dict[str, Any], problem_id: int):
    """保存单个问题的结果"""
    output_dir = os.path.join(config["output"]["base_dir"], f"level_{config['dataset']['level']}", f"problem_{problem_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存生成的kernel
    if result_summary.get("best_kernel"):
        kernel_file = os.path.join(output_dir, f"generated_kernel_{int(time.time())}.py")
        with open(kernel_file, 'w', encoding='utf-8') as f:
            f.write(f"# Generated Triton Kernel (LangChain Multi-Agent)\n")
            f.write(f"# Level: {config['dataset']['level']}\n")
            f.write(f"# Problem ID: {problem_id}\n")
            f.write(f"# Operation: {result_summary.get('operation_name', 'Unknown')}\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Best Speedup: {result_summary.get('best_speedup', 0):.2f}x\n\n")
            f.write(result_summary["best_kernel"])
    
    # 保存结果摘要
    summary_file = os.path.join(output_dir, f"results_{int(time.time())}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False, default=str)
    
    return output_dir

def save_batch_results(batch_results: list, config: Dict[str, Any]):
    """保存批量结果"""
    output_dir = os.path.join(config["output"]["base_dir"], config["output"]["run_name"])
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    total_problems = len(batch_results)
    successful = [r for r in batch_results if r.get("success", False)]
    failed = [r for r in batch_results if not r.get("success", False)]
    
    # 保存批量摘要
    batch_summary = {
        "batch_info": {
            "level": config["dataset"]["level"],
            "total_problems": total_problems,
            "successful_count": len(successful),
            "failed_count": len(failed),
            "success_rate": len(successful) / total_problems if total_problems > 0 else 0,
            "generated_at": datetime.now().isoformat()
        },
        "performance_summary": {
            "avg_speedup": sum(r.get("best_speedup", 0) for r in successful) / len(successful) if successful else 0,
            "max_speedup": max((r.get("best_speedup", 0) for r in successful), default=0),
            "min_speedup": min((r.get("best_speedup", 0) for r in successful), default=0),
            "avg_iterations": sum(r.get("total_iterations", 0) for r in successful) / len(successful) if successful else 0
        },
        "detailed_results": batch_results
    }
    
    summary_file = os.path.join(output_dir, "batch_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存成功和失败的问题列表
    if successful:
        success_file = os.path.join(output_dir, "successful_problems.txt")
        with open(success_file, 'w') as f:
            for r in successful:
                f.write(f"Problem {r['problem_id']}: {r.get('operation_name', 'Unknown')} - {r.get('best_speedup', 0):.2f}x\n")
    
    if failed:
        failed_file = os.path.join(output_dir, "failed_problems.txt")
        with open(failed_file, 'w') as f:
            for r in failed:
                f.write(f"Problem {r['problem_id']}: {r.get('operation_name', 'Unknown')} - {r.get('error', 'Unknown error')}\n")
    
    print(f"💾 批量结果已保存到: {output_dir}")
    return output_dir, batch_summary

def get_problem_ids(level: int, problem_id: int = None, problem_range: str = None, problem_list: str = None) -> list:
    """获取要处理的问题ID列表"""
    from kernelgen.database import KernelBenchLoader
    
    db_loader = KernelBenchLoader()
    
    try:
        # 获取该级别的所有问题
        all_problems = db_loader.get_problems_by_level(level)
        all_problem_ids = [p['problem_id'] for p in all_problems]
        
        if problem_id is not None:
            # 单个问题
            if problem_id not in all_problem_ids:
                raise ValueError(f"Problem {problem_id} not found in level {level}")
            return [problem_id]
        
        elif problem_range is not None:
            # 问题范围 (e.g., "1-10")
            try:
                start, end = map(int, problem_range.split('-'))
                range_ids = list(range(start, end + 1))
                valid_ids = [pid for pid in range_ids if pid in all_problem_ids]
                if not valid_ids:
                    raise ValueError(f"No valid problems found in range {problem_range}")
                return valid_ids
            except ValueError as e:
                raise ValueError(f"Invalid problem range format: {problem_range}. Use format like '1-10'")
        
        elif problem_list is not None:
            # 问题列表 (e.g., "1,5,10,19")
            try:
                list_ids = [int(pid.strip()) for pid in problem_list.split(',')]
                valid_ids = [pid for pid in list_ids if pid in all_problem_ids]
                if not valid_ids:
                    raise ValueError(f"No valid problems found in list {problem_list}")
                return valid_ids
            except ValueError as e:
                raise ValueError(f"Invalid problem list format: {problem_list}. Use format like '1,5,10,19'")
        
        else:
            # 整个级别的所有问题
            return all_problem_ids
    
    finally:
        db_loader.close()

async def generate_kernel_async(level: int, problem_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """异步生成kernel"""
    
    # 1. 初始化数据库加载器
    print("初始化KernelBench数据库...")
    db_loader = KernelBenchLoader()
    
    try:
        # 验证问题是否存在
        problem_info = db_loader.get_problem(level, problem_id)
        print(f"加载问题: {problem_info['operation_name']} (Level {level} Problem {problem_id})")
        
        # 验证问题可执行性
        if not db_loader.validate_problem(level, problem_id):
            raise ValueError(f"问题验证失败: Level {level} Problem {problem_id}")
        
    except Exception as e:
        print(f"数据库加载失败: {e}")
        return {"success": False, "error": str(e)}
    
    # 2. 初始化LangChain聊天模型
    llm_config = config["generation"]["llm"]
    chat_model = create_chat_model(llm_config)
    
    # 3. 初始化编排链
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

async def generate_batch_async(level: int, problem_ids: list, config: Dict[str, Any]) -> Dict[str, Any]:
    """批量生成kernels"""
    print(f"🚀 开始批量生成 Level {level} 的 {len(problem_ids)} 个问题")
    print(f"   问题ID列表: {problem_ids}")
    print(f"   最大迭代: {config['generation']['max_iterations']}")
    
    batch_results = []
    start_time = time.time()
    
    for i, problem_id in enumerate(problem_ids):
        print(f"\n{'='*60}")
        print(f"进度: {i+1}/{len(problem_ids)} - Problem {problem_id}")
        
        try:
            # 为每个问题生成kernel
            result = await generate_kernel_async(level, problem_id, config)
            
            # 保存单个结果
            if result.get("success"):
                save_single_result(result, config, problem_id)
                print(f"✅ Problem {problem_id}: {result.get('operation_name', 'Unknown')} - {result.get('best_speedup', 0):.2f}x")
            else:
                print(f"❌ Problem {problem_id}: {result.get('error', 'Unknown error')}")
            
            batch_results.append(result)
            
            # 显示当前统计
            successful = [r for r in batch_results if r.get("success", False)]
            print(f"📊 当前统计: {len(successful)}/{len(batch_results)} 成功 ({len(successful)/len(batch_results)*100:.1f}%)")
            
        except Exception as e:
            print(f"💥 Problem {problem_id} 执行错误: {e}")
            batch_results.append({
                "success": False,
                "level": level,
                "problem_id": problem_id,
                "error": str(e),
                "generation_time_seconds": 0
            })
    
    # 计算总体统计
    total_time = time.time() - start_time
    successful = [r for r in batch_results if r.get("success", False)]
    
    batch_summary = {
        "success": len(successful) > 0,
        "level": level,
        "total_problems": len(problem_ids),
        "successful_count": len(successful),
        "failed_count": len(problem_ids) - len(successful),
        "success_rate": len(successful) / len(problem_ids),
        "total_time_seconds": total_time,
        "avg_speedup": sum(r.get("best_speedup", 0) for r in successful) / len(successful) if successful else 0,
        "results": batch_results
    }
    
    return batch_summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于LangChain的多Agent协作Triton Kernel生成器")
    
    # 必需参数
    parser.add_argument("--level", type=int, required=True, 
                       help="KernelBench级别 (1-4)")
    
    # 问题选择参数 (互斥)
    problem_group = parser.add_mutually_exclusive_group()
    problem_group.add_argument("--problem-id", type=int,
                              help="单个问题ID")
    problem_group.add_argument("--problem-range", type=str,
                              help="问题范围，格式: '1-10'")
    problem_group.add_argument("--problem-list", type=str,
                              help="问题列表，格式: '1,5,10,19'")
    
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
    parser.add_argument("--timeout", type=int, default=1800,
                       help="每个问题的超时时间(秒) (默认: 1800)")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    print("基于LangChain的多Agent协作Triton Kernel生成器")
    print("=" * 70)
    
    try:
        # 1. 验证环境
        cuda_available = validate_environment()
        
        # 2. 获取要处理的问题ID列表
        problem_ids = get_problem_ids(
            args.level, 
            args.problem_id, 
            args.problem_range, 
            args.problem_list
        )
        
        # 3. 判断是单个问题还是批量处理
        is_batch = len(problem_ids) > 1
        
        print(f"Level: {args.level}")
        if is_batch:
            print(f"批量模式: {len(problem_ids)} 个问题")
            print(f"问题ID: {problem_ids}")
        else:
            print(f"单个问题模式: Problem {problem_ids[0]}")
        print(f"LLM: {args.server_type}/{args.model_name}")
        print(f"最大迭代次数: {args.iterations}")
        print(f"早停阈值: {args.threshold}x")
        print(f"超时时间: {args.timeout}秒")
        print()
        
        # 4. 创建配置
        config = create_config(
            args.level, problem_ids,
            iterations=args.iterations,
            threshold=args.threshold,
            server_type=args.server_type,
            model_name=args.model_name,
            temperature=args.temperature,
            output_dir=args.output_dir,
            cuda_available=cuda_available
        )
        
        start_time = time.time()
        
        if is_batch:
            # 批量处理
            print(f"🚀 开始批量处理 Level {args.level}")
            
            batch_result = asyncio.run(generate_batch_async(args.level, problem_ids, config))
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 显示批量结果
            print(f"\n🎉 批量生成完成!")
            print("=" * 70)
            print(f"   总问题数: {batch_result['total_problems']}")
            print(f"   成功: {batch_result['successful_count']} ({batch_result['success_rate']*100:.1f}%)")
            print(f"   失败: {batch_result['failed_count']}")
            print(f"   平均加速比: {batch_result['avg_speedup']:.2f}x")
            print(f"   总耗时: {total_time:.1f}秒")
            
            # 保存批量结果
            output_dir, summary = save_batch_results(batch_result['results'], config)
            
            print(f"\n📊 详细统计:")
            if summary['performance_summary']['avg_speedup'] > 0:
                print(f"   最大加速比: {summary['performance_summary']['max_speedup']:.2f}x")
                print(f"   最小加速比: {summary['performance_summary']['min_speedup']:.2f}x")
                print(f"   平均迭代数: {summary['performance_summary']['avg_iterations']:.1f}")
            
            print(f"\n💾 结果文件:")
            print(f"   批量摘要: {output_dir}/batch_summary.json")
            if batch_result['successful_count'] > 0:
                print(f"   成功列表: {output_dir}/successful_problems.txt")
            if batch_result['failed_count'] > 0:
                print(f"   失败列表: {output_dir}/failed_problems.txt")
            
            return batch_result['success']
        
        else:
            # 单个问题处理
            problem_id = problem_ids[0]
            print(f"🎯 开始处理 Level {args.level} Problem {problem_id}")
            
            result_summary = asyncio.run(generate_kernel_async(args.level, problem_id, config))
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 显示单个结果
            print(f"\n🎉 生成完成!")
            print("=" * 70)
            
            if result_summary["success"]:
                print(f"✅ 成功生成kernel")
                print(f"   问题: {result_summary['operation_name']}")
                print(f"   总迭代次数: {result_summary['total_iterations']}")
                print(f"   成功迭代次数: {result_summary['successful_iterations']}")
                print(f"   最佳加速比: {result_summary['best_speedup']:.2f}x")
                print(f"   总耗时: {total_time:.1f}秒")
                
                # 保存结果
                output_dir = save_single_result(result_summary, config, problem_id)
                
                print(f"\n💾 结果文件:")
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
                    output_dir = save_single_result(result_summary, config, problem_id)
                    print(f"   部分结果已保存到: {output_dir}")
            
            # 显示详细统计
            if result_summary.get('iteration_summary'):
                print(f"\n📋 迭代详情:")
                for iter_info in result_summary['iteration_summary']:
                    status = "✅" if iter_info['success'] else "❌"
                    speedup = iter_info.get('speedup', 0.0)
                    print(f"   迭代 {iter_info['iteration']}: {status} 加速比 {speedup:.2f}x")
            
            return result_summary["success"]
        
    except KeyboardInterrupt:
        return False
    except Exception as e:
        logging.error("执行失败", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)