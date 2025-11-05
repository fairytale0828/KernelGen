#!/usr/bin/env python3
"""
测试多Agent协作系统
"""

import os
import sys
import logging

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_agent_imports():
    """测试Agent导入"""
    print("🧪 测试Agent导入...")
    
    try:
        from kernelgen.agents import (
            BaseAgent, DesignAgent, CodeAgent, OptimizeAgent, DebugAgent, 
            AgentCoordinator
        )
        print("✅ 所有Agent导入成功")
        return True
    except Exception as e:
        print(f"❌ Agent导入失败: {str(e)}")
        return False

def test_llm_client():
    """测试LLM客户端"""
    print("🧪 测试LLM客户端...")
    
    try:
        from kernelgen.llm import LLMClient
        
        # 创建测试配置
        config = {
            "server_type": "deepseek",
            "model_name": "deepseek-coder", 
            "temperature": 0.0,
            "max_tokens": 1000
        }
        
        client = LLMClient(config)
        print("✅ LLM客户端创建成功")
        return True
    except Exception as e:
        print(f"❌ LLM客户端测试失败: {str(e)}")
        return False

def test_dataset_loader():
    """测试数据集加载器"""
    print("🧪 测试数据集加载器...")
    
    try:
        from kernelgen.dataset import KernelBenchLoader
        
        # 创建测试配置
        config = {
            "source": "huggingface",
            "name": "ScalingIntelligence/KernelBench",
            "level": 1,
            "problem_ids": [19]
        }
        
        loader = KernelBenchLoader(config)
        print("✅ 数据集加载器创建成功")
        return True
    except Exception as e:
        print(f"❌ 数据集加载器测试失败: {str(e)}")
        return False

def test_performance_benchmark():
    """测试性能基准测试器"""
    print("🧪 测试性能基准测试器...")
    
    try:
        from kernelgen.core.performance_benchmark import TritonPerformanceBenchmark
        
        benchmark = TritonPerformanceBenchmark(device="cpu")  # 使用CPU避免CUDA依赖
        print("✅ 性能基准测试器创建成功")
        return True
    except Exception as e:
        print(f"❌ 性能基准测试器测试失败: {str(e)}")
        return False

def test_agent_coordinator():
    """测试Agent协调器"""
    print("🧪 测试Agent协调器...")
    
    try:
        from kernelgen.agents import AgentCoordinator
        from kernelgen.llm import LLMClient
        
        # 创建测试配置
        llm_config = {
            "server_type": "deepseek",
            "model_name": "deepseek-coder",
            "temperature": 0.0,
            "max_tokens": 1000
        }
        
        system_config = {
            "generation": {
                "max_iterations": 3,
                "early_stop_threshold": 1.2,
                "min_successful_iterations": 1
            },
            "performance": {
                "device": "cpu",
                "warmup_runs": 1,
                "benchmark_runs": 1
            }
        }
        
        llm_client = LLMClient(llm_config)
        coordinator = AgentCoordinator(llm_client, system_config)
        
        print("✅ Agent协调器创建成功")
        return True
    except Exception as e:
        print(f"❌ Agent协调器测试失败: {str(e)}")
        return False

def test_environment_check():
    """测试环境检查"""
    print("🧪 测试环境检查...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print(f"❌ Python版本过低: {sys.version}")
        return False
    else:
        print(f"✅ Python版本: {sys.version}")
    
    # 检查必要的包
    required_packages = ['torch', 'datasets', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"❌ 缺少必要的包: {missing_packages}")
        return False
    
    # 检查API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        print("✅ DEEPSEEK_API_KEY 已设置")
    else:
        print("⚠️  DEEPSEEK_API_KEY 未设置 (运行时需要)")
    
    return True

def main():
    """主测试函数"""
    print("🚀 多Agent系统测试")
    print("=" * 50)
    
    # 设置日志
    logging.basicConfig(level=logging.WARNING)  # 减少日志输出
    
    tests = [
        ("环境检查", test_environment_check),
        ("Agent导入", test_agent_imports),
        ("LLM客户端", test_llm_client),
        ("数据集加载器", test_dataset_loader),
        ("性能基准测试器", test_performance_benchmark),
        ("Agent协调器", test_agent_coordinator),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {str(e)}")
    
    print(f"\n📊 测试结果")
    print("=" * 50)
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！多Agent系统准备就绪")
        print("\n🚀 可以运行以下命令测试:")
        print("python generate_kernel_multi_agent.py --level 1 --problem-id 19 --iterations 3")
        return True
    else:
        print("❌ 部分测试失败，请检查环境配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)