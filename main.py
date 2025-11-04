#!/usr/bin/env python3
"""
KernelGen主入口脚本
支持生成和评估Triton kernel
"""

import os
import sys
import yaml
import argparse
import logging
from typing import Dict, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kernelgen.core.kernel_generator import KernelGenerator
from kernelgen.evaluation.evaluator import KernelEvaluator

def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def validate_config(config: Dict[str, Any]):
    """验证配置"""
    required_sections = ["dataset", "generation", "performance", "output"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必要部分: {section}")
    
    # 检查API密钥
    server_type = config["generation"]["llm"]["server_type"]
    key_mapping = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY"
    }
    
    env_key = key_mapping.get(server_type)
    if env_key and not os.getenv(env_key):
        raise ValueError(f"未设置环境变量: {env_key}")

def generate_kernels(config: Dict[str, Any]) -> Dict[str, Any]:
    """生成kernel"""
    print("🚀 开始生成Triton kernels")
    print("=" * 60)
    
    # 显示配置信息
    print(f"数据集: {config['dataset']['source']} Level {config['dataset']['level']}")
    print(f"问题ID: {config['dataset'].get('problem_ids', 'All')}")
    print(f"最大迭代: {config['generation']['max_iterations']}")
    print(f"LLM: {config['generation']['llm']['server_type']}/{config['generation']['llm']['model_name']}")
    print(f"输出目录: {config['output']['base_dir']}/{config['output']['run_name']}")
    print()
    
    # 创建生成器并运行
    generator = KernelGenerator(config)
    summary = generator.generate_batch()
    
    # 显示结果
    print("\n" + "=" * 60)
    print("🎯 生成结果摘要:")
    print("=" * 60)
    print(f"总结果数: {summary['total_results']}")
    print(f"成功结果数: {summary['successful_results']}")
    print(f"成功率: {summary['success_rate']:.1f}%")
    print(f"平均加速比: {summary['average_speedup']:.2f}x")
    print(f"最大加速比: {summary['max_speedup']:.2f}x")
    print(f"执行时间: {summary['execution_time']:.1f}秒")
    
    # 按问题显示统计
    print(f"\n📊 按问题统计:")
    for problem_id, stats in summary['problem_stats'].items():
        print(f"  Problem {problem_id}: {stats['successful']}/{stats['total']} 成功, "
              f"最佳加速比: {stats['best_speedup']:.2f}x")
    
    return summary

def evaluate_kernels(config: Dict[str, Any]) -> Dict[str, Any]:
    """评估kernel"""
    print("🧪 开始评估生成的kernels")
    print("=" * 60)
    
    # 创建评估器并运行
    evaluator = KernelEvaluator(config)
    eval_summary = evaluator.evaluate_batch()
    
    # 显示评估结果
    print("\n" + "=" * 60)
    print("📈 评估结果摘要:")
    print("=" * 60)
    print(f"评估的kernel数: {eval_summary.get('total_evaluated', 0)}")
    print(f"编译成功率: {eval_summary.get('compilation_success_rate', 0):.1f}%")
    print(f"正确性通过率: {eval_summary.get('correctness_pass_rate', 0):.1f}%")
    
    # 显示pass@k结果
    if 'pass_at_k' in eval_summary:
        print(f"\nPass@K结果:")
        for k, rate in eval_summary['pass_at_k'].items():
            print(f"  {k}: {rate:.1f}%")
    
    return eval_summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="KernelGen - Triton Kernel生成器")
    parser.add_argument("--config", "-c", default="config.yaml", 
                       help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--mode", "-m", choices=["generate", "evaluate", "both"], 
                       default="generate", help="运行模式")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    # 支持命令行覆盖配置
    parser.add_argument("--level", type=int, help="数据集级别")
    parser.add_argument("--problem-ids", nargs="+", type=int, help="指定问题ID")
    parser.add_argument("--max-iterations", type=int, help="最大迭代次数")
    parser.add_argument("--run-name", help="运行名称")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 命令行参数覆盖配置
        if args.level:
            config["dataset"]["level"] = args.level
        if args.problem_ids:
            config["dataset"]["problem_ids"] = args.problem_ids
        if args.max_iterations:
            config["generation"]["max_iterations"] = args.max_iterations
        if args.run_name:
            config["output"]["run_name"] = args.run_name
        
        # 验证配置
        validate_config(config)
        
        # 执行相应模式
        if args.mode in ["generate", "both"]:
            generate_summary = generate_kernels(config)
            
            if args.mode == "both":
                print("\n" + "🔄 切换到评估模式...")
                time.sleep(1)
        
        if args.mode in ["evaluate", "both"]:
            evaluate_summary = evaluate_kernels(config)
        
        print("\n✅ 任务完成!")
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        logging.error(f"执行失败: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()