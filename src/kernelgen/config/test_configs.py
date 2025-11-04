"""
测试配置管理模块
"""
from typing import Dict, List, Tuple, Any
import torch

class TestConfigs:
    """测试配置管理器"""
    
    # PyTorch算子模板
    PYTORCH_OPERATORS = {
        "matmul": '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    矩阵乘法模型
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        计算矩阵乘法 C = A @ B
        
        Args:
            a: 输入矩阵A，形状 [M, K]
            b: 输入矩阵B，形状 [K, N]
            
        Returns:
            输出矩阵C，形状 [M, N]
        """
        return torch.mm(a, b)

# 测试参数
M, N, K = 1024, 1024, 1024

def get_inputs():
    a = torch.randn(M, K, dtype=torch.float32, device="cuda")
    b = torch.randn(K, N, dtype=torch.float32, device="cuda")
    return [a, b]

def get_init_inputs():
    return []
''',
        
        "relu": '''
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

# 测试参数
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, dtype=torch.float32, device="cuda")
    return [x]

def get_init_inputs():
    return []
''',
        
        "sigmoid": '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sigmoid激活函数模型
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算Sigmoid激活函数
        
        Args:
            x: 输入张量
            
        Returns:
            Sigmoid激活后的张量
        """
        return torch.sigmoid(x)

# 测试参数
batch_size = 32
dim = 8192

def get_inputs():
    x = torch.randn(batch_size, dim, dtype=torch.float32, device="cuda")
    return [x]

def get_init_inputs():
    return []
''',
        
        "softmax": '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Softmax激活函数模型
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算Softmax激活函数
        
        Args:
            x: 输入张量，形状 [batch_size, seq_len]
            
        Returns:
            Softmax激活后的张量
        """
        return torch.softmax(x, dim=-1)

# 测试参数
batch_size = 32
seq_len = 512

def get_inputs():
    x = torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")
    return [x]

def get_init_inputs():
    return []
''',
        
        "layer_norm": '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Layer Normalization模型
    """
    def __init__(self, normalized_shape):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算Layer Normalization
        
        Args:
            x: 输入张量
            
        Returns:
            归一化后的张量
        """
        return self.layer_norm(x)

# 测试参数
batch_size = 32
hidden_dim = 768

def get_inputs():
    x = torch.randn(batch_size, hidden_dim, dtype=torch.float32, device="cuda")
    return [x]

def get_init_inputs():
    return [hidden_dim]
'''
    }
    
    # 测试形状配置
    TEST_SHAPES = {
        "matmul": [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096)
        ],
        "relu": [
            (1024, 512),
            (2048, 1024),
            (4096, 2048),
            (8192, 4096)
        ],
        "sigmoid": [
            (1024, 512),
            (2048, 1024),
            (4096, 2048)
        ],
        "softmax": [
            (32, 512),
            (64, 1024),
            (128, 2048)
        ],
        "layer_norm": [
            (32, 768),
            (64, 1024),
            (128, 1536)
        ]
    }
    
    # 迭代测试配置
    ITERATION_CONFIGS = {
        "default": {
            "max_iterations": 20,
            "early_stop_threshold": 1.2,  # 加速比阈值
            "min_successful_iterations": 3,  # 最少成功迭代次数
            "save_all_kernels": True,
            "save_best_only": False
        },
        "quick": {
            "max_iterations": 10,
            "early_stop_threshold": 1.1,
            "min_successful_iterations": 2,
            "save_all_kernels": False,
            "save_best_only": True
        },
        "thorough": {
            "max_iterations": 50,
            "early_stop_threshold": 1.5,
            "min_successful_iterations": 5,
            "save_all_kernels": True,
            "save_best_only": False
        }
    }
    
    # 性能测试配置
    PERFORMANCE_CONFIGS = {
        "default": {
            "warmup_runs": 10,
            "benchmark_runs": 100,
            "device": "cuda",
            "dtype": torch.float32
        },
        "quick": {
            "warmup_runs": 5,
            "benchmark_runs": 50,
            "device": "cuda",
            "dtype": torch.float32
        },
        "precise": {
            "warmup_runs": 20,
            "benchmark_runs": 200,
            "device": "cuda",
            "dtype": torch.float32
        }
    }
    
    # Agent配置
    AGENT_CONFIGS = {
        "default": {
            "docs_dir": {
                "designer": "resources/docs",
                "coder": "resources/docs"
            },
            "agent_model_config": {
                "designer": "deepseek_default",
                "coder": "deepseek_coder",
                "default": "deepseek_default"
            }
        }
    }
    
    # 输出配置
    OUTPUT_CONFIGS = {
        "default": {
            "output_dir": "generated_kernels",
            "save_logs": True,
            "save_performance_data": True,
            "save_source_code": True,
            "log_level": "INFO"
        }
    }
    
    @classmethod
    def get_operator_config(cls, op_name: str) -> str:
        """获取算子的PyTorch代码模板"""
        return cls.PYTORCH_OPERATORS.get(op_name, cls.PYTORCH_OPERATORS["relu"])
    
    @classmethod
    def get_test_shapes(cls, op_name: str) -> List[Tuple]:
        """获取算子的测试形状"""
        return cls.TEST_SHAPES.get(op_name, cls.TEST_SHAPES["relu"])
    
    @classmethod
    def get_iteration_config(cls, config_name: str = "default") -> Dict[str, Any]:
        """获取迭代配置"""
        return cls.ITERATION_CONFIGS.get(config_name, cls.ITERATION_CONFIGS["default"])
    
    @classmethod
    def get_performance_config(cls, config_name: str = "default") -> Dict[str, Any]:
        """获取性能测试配置"""
        return cls.PERFORMANCE_CONFIGS.get(config_name, cls.PERFORMANCE_CONFIGS["default"])
    
    @classmethod
    def get_agent_config(cls, config_name: str = "default") -> Dict[str, Any]:
        """获取Agent配置"""
        return cls.AGENT_CONFIGS.get(config_name, cls.AGENT_CONFIGS["default"])
    
    @classmethod
    def get_output_config(cls, config_name: str = "default") -> Dict[str, Any]:
        """获取输出配置"""
        return cls.OUTPUT_CONFIGS.get(config_name, cls.OUTPUT_CONFIGS["default"])
    
    @classmethod
    def create_full_config(cls, op_name: str, iteration_config: str = "default",
                          performance_config: str = "default", 
                          agent_config: str = "default",
                          output_config: str = "default") -> Dict[str, Any]:
        """创建完整的测试配置"""
        return {
            "operator": {
                "name": op_name,
                "pytorch_code": cls.get_operator_config(op_name),
                "test_shapes": cls.get_test_shapes(op_name)
            },
            "iteration": cls.get_iteration_config(iteration_config),
            "performance": cls.get_performance_config(performance_config),
            "agent": cls.get_agent_config(agent_config),
            "output": cls.get_output_config(output_config)
        }