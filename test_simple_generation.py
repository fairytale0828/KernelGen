#!/usr/bin/env python3
"""
简化的多Agent测试 - 测试单个Agent功能
"""

import os
import sys
import asyncio
import logging

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_design_agent():
    """测试Design Agent"""
    print("🧪 测试Design Agent...")
    
    try:
        from kernelgen.agents import DesignAgent
        from kernelgen.llm import LLMClient
        
        # 创建LLM客户端
        llm_config = {
            "server_type": "deepseek",
            "model_name": "deepseek-coder",
            "temperature": 0.0,
            "max_tokens": 2000
        }
        
        llm_client = LLMClient(llm_config)
        design_agent = DesignAgent(llm_client, {})
        
        # 简单的PyTorch ReLU代码
        pytorch_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

def get_inputs():
    x = torch.rand(1024, 512)
    return [x]

def get_init_inputs():
    return []
"""
        
        task_info = {
            "pytorch_code": pytorch_code,
            "problem_info": {
                "name": "ReLU",
                "problem_id": 19,
                "level": 1
            }
        }
        
        print("📐 调用Design Agent...")
        result_text, _, _ = await design_agent.run(task_info)
        
        print("✅ Design Agent执行成功")
        print(f"📄 结果长度: {len(result_text)} 字符")
        print(f"📋 结果预览: {result_text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Design Agent测试失败: {str(e)}")
        return False

async def test_code_agent():
    """测试Code Agent"""
    print("\n🧪 测试Code Agent...")
    
    try:
        from kernelgen.agents import CodeAgent
        from kernelgen.llm import LLMClient
        
        # 创建LLM客户端
        llm_config = {
            "server_type": "deepseek",
            "model_name": "deepseek-coder",
            "temperature": 0.0,
            "max_tokens": 2000
        }
        
        llm_client = LLMClient(llm_config)
        code_agent = CodeAgent(llm_client, {})
        
        # 模拟设计结果
        task_info = {
            "design_plan": {
                "architecture": "elementwise",
                "memory_strategy": "coalesced_access",
                "recommended_block_size": 256,
                "optimization_focus": ["memory_bandwidth"]
            },
            "implementation_guide": {
                "code_structure": "kernel_with_wrapper",
                "key_functions": ["relu_kernel", "relu_wrapper"],
                "performance_tips": ["use_vectorization"]
            },
            "pytorch_code": "return torch.relu(x)",
            "operator_analysis": {
                "operator_type": "elementwise",
                "computational_complexity": "O(1)",
                "memory_access_pattern": "elementwise"
            }
        }
        
        print("💻 调用Code Agent...")
        result_text, _, _ = await code_agent.run(task_info)
        
        print("✅ Code Agent执行成功")
        print(f"📄 结果长度: {len(result_text)} 字符")
        print(f"📋 结果预览: {result_text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Code Agent测试失败: {str(e)}")
        return False

async def main():
    """主测试函数"""
    print("🚀 简化多Agent测试")
    print("=" * 50)
    
    # 设置日志
    logging.basicConfig(level=logging.WARNING)
    
    # 检查API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 请设置DEEPSEEK_API_KEY环境变量")
        return False
    
    print("✅ DEEPSEEK_API_KEY 已设置")
    
    tests = [
        ("Design Agent", test_design_agent),
        ("Code Agent", test_code_agent),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if await test_func():
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
        print("🎉 所有测试通过！")
        return True
    else:
        print("❌ 部分测试失败")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)