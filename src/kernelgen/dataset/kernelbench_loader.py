"""
KernelBench数据集加载器
集成HuggingFace数据集和本地数据集
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

class KernelBenchLoader:
    """KernelBench数据集加载器"""
    
    def __init__(self, dataset_config: Dict[str, Any]):
        """
        初始化数据集加载器
        
        Args:
            dataset_config: 数据集配置
        """
        self.source = dataset_config["source"]
        self.dataset_name = dataset_config["name"]
        self.level = dataset_config["level"]
        self.problem_ids = dataset_config.get("problem_ids", [])
        
        self.dataset = None
        self.problems_cache = {}
        
        logger.info(f"初始化KernelBench加载器: source={self.source}, level={self.level}")
    
    def load_dataset(self):
        """加载数据集"""
        if self.source == "huggingface":
            logger.info(f"从HuggingFace加载数据集: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name)
            self.dataset = dataset[f"level_{self.level}"]
        elif self.source == "local":
            logger.info(f"从本地加载Level {self.level}数据集")
            self.dataset = self._construct_local_dataset(self.level)
        else:
            raise ValueError(f"不支持的数据源: {self.source}")
        
        logger.info(f"数据集加载完成，共{len(self.dataset)}个问题")
    
    def _construct_local_dataset(self, level: int) -> List[str]:
        """构建本地数据集"""
        # 假设KernelBench在相对路径下
        kernelbench_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "..", "..", 
            "KernelBench", "KernelBench", f"level{level}"
        )
        
        if not os.path.exists(kernelbench_path):
            raise FileNotFoundError(f"本地KernelBench路径不存在: {kernelbench_path}")
        
        dataset = []
        for filename in os.listdir(kernelbench_path):
            if filename.endswith(".py"):
                filepath = os.path.join(kernelbench_path, filename)
                dataset.append(filepath)
        
        # 按数字前缀排序
        dataset.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
        return dataset
    
    def get_problem_by_id(self, problem_id: int) -> Dict[str, Any]:
        """
        根据问题ID获取问题信息
        
        Args:
            problem_id: 问题ID
            
        Returns:
            包含问题信息的字典
        """
        if problem_id in self.problems_cache:
            return self.problems_cache[problem_id]
        
        if self.source == "huggingface":
            problem_row = self.dataset.filter(
                lambda x: x["problem_id"] == problem_id
            )
            if len(problem_row) == 0:
                raise ValueError(f"问题ID {problem_id} 不存在")
            
            problem_info = {
                "problem_id": problem_id,
                "name": problem_row["name"][0],
                "code": problem_row["code"][0],
                "level": self.level
            }
        
        elif self.source == "local":
            # 本地数据集是0索引的
            problem_idx = problem_id - 1
            if problem_idx >= len(self.dataset):
                raise ValueError(f"问题ID {problem_id} 超出范围")
            
            problem_path = self.dataset[problem_idx]
            problem_name = os.path.basename(problem_path)
            
            # 验证问题编号
            file_problem_id = int(problem_name.split("_")[0])
            if file_problem_id != problem_id:
                raise ValueError(f"文件中的问题编号({file_problem_id})与请求的问题ID({problem_id})不匹配")
            
            with open(problem_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            problem_info = {
                "problem_id": problem_id,
                "name": problem_name,
                "code": code,
                "level": self.level,
                "path": problem_path
            }
        
        # 缓存结果
        self.problems_cache[problem_id] = problem_info
        return problem_info
    
    def get_all_problem_ids(self) -> List[int]:
        """获取所有问题ID"""
        if self.source == "huggingface":
            return [item["problem_id"] for item in self.dataset]
        elif self.source == "local":
            problem_ids = []
            for path in self.dataset:
                filename = os.path.basename(path)
                problem_id = int(filename.split("_")[0])
                problem_ids.append(problem_id)
            return problem_ids
    
    def get_target_problem_ids(self) -> List[int]:
        """获取目标问题ID列表"""
        if self.problem_ids:
            return self.problem_ids
        else:
            return self.get_all_problem_ids()
    
    def extract_model_info(self, code: str) -> Dict[str, Any]:
        """
        从PyTorch代码中提取模型信息
        
        Args:
            code: PyTorch模型代码
            
        Returns:
            模型信息字典
        """
        model_info = {
            "has_model_class": "class Model" in code,
            "has_get_inputs": "def get_inputs" in code,
            "has_get_init_inputs": "def get_init_inputs" in code,
            "imports": [],
            "model_params": {}
        }
        
        # 提取导入语句
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                model_info["imports"].append(line)
        
        # 提取模型参数（简单解析）
        try:
            # 查找全局变量定义
            for line in lines:
                line = line.strip()
                if '=' in line and not line.startswith('#') and not line.startswith('def '):
                    parts = line.split('=')
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        var_value = parts[1].strip()
                        # 尝试解析简单的数值
                        try:
                            if var_value.isdigit():
                                model_info["model_params"][var_name] = int(var_value)
                            elif var_value.replace('.', '').isdigit():
                                model_info["model_params"][var_name] = float(var_value)
                            else:
                                model_info["model_params"][var_name] = var_value
                        except:
                            model_info["model_params"][var_name] = var_value
        except Exception as e:
            logger.warning(f"解析模型参数失败: {str(e)}")
        
        return model_info
    
    def validate_problem(self, problem_info: Dict[str, Any]) -> bool:
        """
        验证问题是否有效
        
        Args:
            problem_info: 问题信息
            
        Returns:
            是否有效
        """
        code = problem_info["code"]
        model_info = self.extract_model_info(code)
        
        required_components = [
            model_info["has_model_class"],
            model_info["has_get_inputs"],
            model_info["has_get_init_inputs"]
        ]
        
        is_valid = all(required_components)
        
        if not is_valid:
            logger.warning(f"问题 {problem_info['problem_id']} 验证失败: 缺少必要组件")
        
        return is_valid