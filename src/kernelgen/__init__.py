"""
KernelGen - 基于多Agent的Triton Kernel生成器

这是一个专门为NVIDIA GPU设计的Triton kernel自动生成系统，
通过多个AI Agent的协作来完成从PyTorch算子到高性能Triton kernel的转换。
"""

import logging
import os

# 定义通用的日志格式
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
log_datefmt = '%Y-%m-%d %H:%M:%S'

# 根据KERNELGEN_LOG_LEVEL环境变量设置日志级别
glog_level = os.getenv('KERNELGEN_LOG_LEVEL', '1')  # 默认为1 (INFO)
level_map = {
    '0': logging.DEBUG,
    '1': logging.INFO,
    '2': logging.WARNING,
    '3': logging.ERROR
}
log_level = level_map.get(glog_level, logging.INFO)

logging.basicConfig(
    level=log_level,
    format=log_format,
    datefmt=log_datefmt
)

logger = logging.getLogger(__name__)

# 版本信息
__version__ = "0.1.0"
__author__ = "KernelGen Team"

# 导出主要类
from .core.generator import KernelGenerator
from .core.task import Task

__all__ = [
    "KernelGenerator",
    "Task"
]

def get_project_root():
    """获取项目根目录的绝对路径
    
    Returns:
        str: 项目根目录的绝对路径
    """
    return os.path.dirname(os.path.abspath(__file__))