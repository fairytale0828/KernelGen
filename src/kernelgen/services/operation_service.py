"""
操作类型推断服务 - 基于LLM的智能推断
"""

import logging
import json
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class OperationTypeService:
    """操作类型推断服务"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._setup_prompt()
    
    def _setup_prompt(self):
        """设置操作类型推断的prompt"""
        self.operation_inference_prompt = PromptTemplate(
            input_variables=["pytorch_code", "operation_name"],
            template="""你是一个专业的GPU kernel操作类型分析专家。请分析以下PyTorch代码，推断其核心操作类型。

## 操作名称
{operation_name}

## PyTorch代码
```python
{pytorch_code}
```

## 支持的操作类型
- matmul: 矩阵乘法相关操作（包括batch_matmul, gemm等）
- conv2d: 2D卷积相关操作（包括conv2d, depthwise_conv等）
- conv1d: 1D卷积操作
- conv3d: 3D卷积操作
- elementwise: 逐元素操作（如add, mul, relu, sigmoid等）
- reduction: 归约操作（如sum, mean, max, min, softmax等）
- transpose: 转置和重排操作
- pooling: 池化操作（如maxpool, avgpool等）
- normalization: 归一化操作（如batchnorm, layernorm等）
- attention: 注意力机制相关操作
- embedding: 嵌入层操作
- composite: 复合操作（多个基础操作的组合）

## 分析要求
1. 仔细分析PyTorch代码中的Model.forward()方法
2. 识别主要的计算操作
3. 如果包含多个操作，判断是否为复合操作
4. 选择最符合的操作类型

请返回JSON格式的结果：
```json
{{
    "primary_operation": "主要操作类型",
    "secondary_operations": ["次要操作1", "次要操作2"],
    "is_composite": true/false,
    "confidence": 0.95,
    "reasoning": "推理过程说明"
}}
```"""
        )
        
        self.chain = self.operation_inference_prompt | self.llm | StrOutputParser()
    
    async def infer_operation_type(self, pytorch_code: str, operation_name: str = "") -> Dict[str, Any]:
        """
        推断操作类型
        
        Args:
            pytorch_code: PyTorch代码
            operation_name: 操作名称（可选）
            
        Returns:
            操作类型推断结果
        """
        try:
            # 调用LLM进行推断
            response = await self.chain.ainvoke({
                "pytorch_code": pytorch_code,
                "operation_name": operation_name or "Unknown"
            })
            
            # 解析响应
            result = self._parse_inference_response(response)
            
            logger.info(f"推断操作类型: {result.get('primary_operation', 'unknown')} (置信度: {result.get('confidence', 0.0)})")
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"操作类型推断失败: {e}")
            
            # 使用fallback逻辑
            fallback_result = self._fallback_inference(pytorch_code, operation_name)
            
            return {
                "success": False,
                "error": str(e),
                "fallback_result": fallback_result
            }
    
    def _parse_inference_response(self, response: str) -> Dict[str, Any]:
        """解析LLM推断响应"""
        try:
            # 尝试直接解析JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取JSON代码块
            import re
            json_pattern = r'```json\s*({.*?})\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 如果都失败了，尝试从文本中提取信息
            return self._extract_from_text(response)
    
    def _extract_from_text(self, response: str) -> Dict[str, Any]:
        """从文本响应中提取操作类型信息"""
        response_lower = response.lower()
        
        # 定义操作类型关键词
        operation_keywords = {
            "matmul": ["matmul", "matrix multiplication", "gemm", "dot product"],
            "conv2d": ["conv2d", "convolution", "conv", "filter"],
            "elementwise": ["elementwise", "element-wise", "add", "multiply", "relu"],
            "reduction": ["reduction", "sum", "mean", "max", "min", "softmax"],
            "transpose": ["transpose", "permute", "reshape"],
            "pooling": ["pool", "maxpool", "avgpool"],
            "normalization": ["norm", "batchnorm", "layernorm"],
            "attention": ["attention", "self-attention", "multi-head"],
            "embedding": ["embedding", "embed"],
            "composite": ["composite", "multiple", "combination"]
        }
        
        # 计算每种操作类型的匹配分数
        scores = {}
        for op_type, keywords in operation_keywords.items():
            score = sum(1 for keyword in keywords if keyword in response_lower)
            if score > 0:
                scores[op_type] = score
        
        # 选择得分最高的操作类型
        if scores:
            primary_operation = max(scores, key=scores.get)
            confidence = min(scores[primary_operation] * 0.2, 0.8)  # 基于匹配数量的置信度
        else:
            primary_operation = "unknown"
            confidence = 0.1
        
        return {
            "primary_operation": primary_operation,
            "secondary_operations": [],
            "is_composite": "composite" in response_lower or "multiple" in response_lower,
            "confidence": confidence,
            "reasoning": "基于关键词匹配的fallback推断"
        }
    
    def _fallback_inference(self, pytorch_code: str, operation_name: str) -> Dict[str, Any]:
        """fallback推断逻辑"""
        code_lower = pytorch_code.lower()
        name_lower = operation_name.lower()
        
        # 基于代码内容的简单推断
        if any(keyword in code_lower for keyword in ["matmul", "mm", "bmm", "@"]):
            return {
                "primary_operation": "matmul",
                "secondary_operations": [],
                "is_composite": False,
                "confidence": 0.7,
                "reasoning": "基于代码中的矩阵乘法关键词"
            }
        elif any(keyword in code_lower for keyword in ["conv2d", "conv"]):
            return {
                "primary_operation": "conv2d",
                "secondary_operations": [],
                "is_composite": False,
                "confidence": 0.7,
                "reasoning": "基于代码中的卷积关键词"
            }
        elif any(keyword in code_lower for keyword in ["relu", "sigmoid", "tanh", "+"]):
            return {
                "primary_operation": "elementwise",
                "secondary_operations": [],
                "is_composite": False,
                "confidence": 0.6,
                "reasoning": "基于代码中的逐元素操作关键词"
            }
        else:
            return {
                "primary_operation": "unknown",
                "secondary_operations": [],
                "is_composite": False,
                "confidence": 0.1,
                "reasoning": "无法从代码中识别明确的操作类型"
            }
    
    def get_operation_hierarchy(self, operation_type: str) -> Dict[str, Any]:
        """获取操作类型的层次结构信息"""
        hierarchy = {
            "matmul": {
                "category": "compute_intensive",
                "complexity": "medium",
                "memory_pattern": "structured",
                "parallelization": "high"
            },
            "conv2d": {
                "category": "compute_intensive", 
                "complexity": "high",
                "memory_pattern": "structured",
                "parallelization": "high"
            },
            "elementwise": {
                "category": "memory_bound",
                "complexity": "low",
                "memory_pattern": "simple",
                "parallelization": "very_high"
            },
            "reduction": {
                "category": "mixed",
                "complexity": "medium",
                "memory_pattern": "irregular",
                "parallelization": "medium"
            }
        }
        
        return hierarchy.get(operation_type, {
            "category": "unknown",
            "complexity": "unknown",
            "memory_pattern": "unknown", 
            "parallelization": "unknown"
        })