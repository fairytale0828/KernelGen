"""
基础ReLU算子生成示例 - 使用PyTorch算子代码作为输入
"""

import asyncio
import logging
import os
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.task import Task
from core.async_pool.device_pool import DevicePool
from core.async_pool.task_pool import TaskPool
from config.config_manager import ConfigManager

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_op_name():
    return 'relu'


def get_task_desc():
    """返回PyTorch算子代码描述，参考aikg模式"""
    return '''
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ReLU激活函数模型
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算ReLU激活函数
        Args:
            x: 输入张量
        Returns:
            ReLU激活后的张量
        """
        return torch.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim, dtype=torch.float32)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
'''


async def run_pytorch_triton_single():
    """运行单个PyTorch到Triton的转换任务"""
    
    # 检查API密钥
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.error("请设置DEEPSEEK_API_KEY环境变量")
        return
    
    op_name = get_op_name()
    task_desc = get_task_desc()

    # 创建任务池和设备池
    task_pool = TaskPool()
    device_pool = DevicePool([0])  # 使用GPU 0，如果没有CUDA则自动使用CPU
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # 设置工作流配置路径
    config["workflow_config_path"] = "config/default_workflow.yaml"
    
    # 设置文档目录配置
    config["docs_dir"] = {
        "designer": "resources/docs/triton_docs",
        "coder": "resources/docs/triton_docs"
    }

    try:
        # 创建任务
        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id="0",
            config=config,
            device_pool=device_pool,
            dtype="float32",
            workflow="default_workflow"
        )

        # 执行任务
        task_pool.create_task(task.run)
        results = await task_pool.wait_all()
        
        # 处理结果
        for op_name, success, task_info in results:
            if success:
                print("\n" + "="*60)
                print(f"✅ Task {op_name} 执行成功！")
                print("="*60)
                
                # 显示生成的代码
                coder_result = task_info.get('coder_result')
                if coder_result:
                    print("\n🔧 生成的Triton Kernel代码：")
                    print("-"*50)
                    print(coder_result)
                
                # 显示性能指标
                performance_metrics = task_info.get('performance_metrics')
                if performance_metrics:
                    print("\n📊 性能指标：")
                    print("-"*30)
                    for metric, value in performance_metrics.items():
                        print(f"{metric}: {value}")
                
                # 显示验证结果
                verifier_result = task_info.get('verifier_result', False)
                print(f"\n✅ 验证结果: {'通过' if verifier_result else '失败'}")
                
            else:
                print(f"❌ Task {op_name} 执行失败")
                error_info = task_info.get('error_log', '未知错误')
                print(f"错误信息: {error_info}")
                
    except Exception as e:
        logger.error(f"任务执行异常: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_pytorch_triton_single())