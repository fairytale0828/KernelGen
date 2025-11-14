"""
知识库服务 - 基于RAG的知识检索
"""

import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeItem:
    """知识项"""
    operation_type: str
    best_practices: List[str]
    optimization_tips: List[str]
    common_issues: List[str]
    implementation_patterns: List[str]
    performance_considerations: List[str]

class KnowledgeBaseService:
    """知识库服务"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        初始化知识库服务
        
        Args:
            knowledge_base_path: 知识库文件路径，支持YAML格式
        """
        self.knowledge_base_path = knowledge_base_path or "knowledge_base/triton_knowledge.yaml"
        self._knowledge_cache: Dict[str, KnowledgeItem] = {}
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """加载知识库"""
        try:
            kb_path = Path(self.knowledge_base_path)
            if not kb_path.exists():
                logger.warning(f"知识库文件不存在: {self.knowledge_base_path}")
                self._create_default_knowledge_base()
                return
            
            with open(kb_path, 'r', encoding='utf-8') as f:
                if kb_path.suffix.lower() == '.yaml' or kb_path.suffix.lower() == '.yml':
                    knowledge_data = yaml.safe_load(f)
                else:
                    knowledge_data = json.load(f)
            
            # 解析知识库数据
            for op_type, data in knowledge_data.get('operations', {}).items():
                self._knowledge_cache[op_type] = KnowledgeItem(
                    operation_type=op_type,
                    best_practices=data.get('best_practices', []),
                    optimization_tips=data.get('optimization_tips', []),
                    common_issues=data.get('common_issues', []),
                    implementation_patterns=data.get('implementation_patterns', []),
                    performance_considerations=data.get('performance_considerations', [])
                )
            
            logger.info(f"成功加载知识库，包含 {len(self._knowledge_cache)} 个操作类型")
            
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            self._create_default_knowledge_base()
    
    def _create_default_knowledge_base(self):
        """创建默认知识库"""
        default_knowledge = {
            "matmul": KnowledgeItem(
                operation_type="matmul",
                best_practices=[
                    "使用分块算法优化内存访问",
                    "利用共享内存缓存数据",
                    "确保内存合并访问"
                ],
                optimization_tips=[
                    "选择合适的BLOCK_SIZE平衡并行度和内存使用",
                    "使用tl.dot进行高效矩阵乘法",
                    "考虑使用混合精度计算"
                ],
                common_issues=[
                    "分块大小不当导致性能下降",
                    "内存访问不连续",
                    "共享内存使用不当"
                ],
                implementation_patterns=[
                    "标准分块矩阵乘法",
                    "流水线化计算"
                ],
                performance_considerations=[
                    "内存带宽利用率",
                    "计算强度优化"
                ]
            ),
            "conv2d": KnowledgeItem(
                operation_type="conv2d",
                best_practices=[
                    "使用简单直接的实现，避免过度复杂化",
                    "每个线程处理一个输出元素",
                    "正确处理4D张量索引"
                ],
                optimization_tips=[
                    "合理设计grid和block大小",
                    "优化4D张量内存访问",
                    "使用适当的数值精度"
                ],
                common_issues=[
                    "4D张量索引计算错误",
                    "边界条件处理不当",
                    "内存访问模式不优化"
                ],
                implementation_patterns=[
                    "直接卷积实现",
                    "im2col算法"
                ],
                performance_considerations=[
                    "内存访问局部性",
                    "计算与内存访问平衡"
                ]
            ),
            "elementwise": KnowledgeItem(
                operation_type="elementwise",
                best_practices=[
                    "内存合并访问",
                    "避免分支分歧",
                    "向量化操作"
                ],
                optimization_tips=[
                    "使用合适的BLOCK_SIZE(256/512)",
                    "使用tl.where避免分支",
                    "优化mask使用"
                ],
                common_issues=[
                    "mask使用不当",
                    "BLOCK_SIZE不合理",
                    "分支导致性能下降"
                ],
                implementation_patterns=[
                    "标准elementwise模式",
                    "向量化处理"
                ],
                performance_considerations=[
                    "内存带宽限制",
                    "线程利用率"
                ]
            )
        }
        
        self._knowledge_cache = default_knowledge
        logger.info("使用默认知识库")
    
    def get_knowledge(self, operation_type: str) -> Optional[KnowledgeItem]:
        """获取指定操作类型的知识"""
        return self._knowledge_cache.get(operation_type)
    
    def get_knowledge_context_string(self, operation_type: str) -> str:
        """获取知识库上下文字符串，用于prompt"""
        knowledge = self.get_knowledge(operation_type)
        
        if not knowledge:
            return f"## {operation_type.upper()}操作相关知识\n暂无相关知识库信息"
        
        context = f"## {operation_type.upper()}操作相关知识\n"
        
        if knowledge.best_practices:
            context += "### 最佳实践\n"
            for practice in knowledge.best_practices:
                context += f"- {practice}\n"
        
        if knowledge.optimization_tips:
            context += "### 优化技巧\n"
            for tip in knowledge.optimization_tips:
                context += f"- {tip}\n"
        
        if knowledge.common_issues:
            context += "### 常见问题\n"
            for issue in knowledge.common_issues:
                context += f"- {issue}\n"
        
        if knowledge.implementation_patterns:
            context += "### 实现模式\n"
            for pattern in knowledge.implementation_patterns:
                context += f"- {pattern}\n"
        
        return context
    
    def search_relevant_knowledge(self, query: str, operation_type: str = None) -> List[str]:
        """
        基于查询搜索相关知识（简化版RAG）
        
        Args:
            query: 查询字符串
            operation_type: 操作类型过滤
            
        Returns:
            相关知识列表
        """
        relevant_knowledge = []
        
        # 简化的关键词匹配（实际RAG会使用向量相似度）
        query_lower = query.lower()
        
        knowledge_items = [self._knowledge_cache[operation_type]] if operation_type and operation_type in self._knowledge_cache else self._knowledge_cache.values()
        
        for knowledge in knowledge_items:
            # 搜索最佳实践
            for practice in knowledge.best_practices:
                if any(keyword in practice.lower() for keyword in query_lower.split()):
                    relevant_knowledge.append(f"最佳实践: {practice}")
            
            # 搜索优化技巧
            for tip in knowledge.optimization_tips:
                if any(keyword in tip.lower() for keyword in query_lower.split()):
                    relevant_knowledge.append(f"优化技巧: {tip}")
            
            # 搜索常见问题
            for issue in knowledge.common_issues:
                if any(keyword in issue.lower() for keyword in query_lower.split()):
                    relevant_knowledge.append(f"常见问题: {issue}")
        
        return relevant_knowledge[:5]  # 返回最相关的5条
    
    def get_all_operation_types(self) -> List[str]:
        """获取所有支持的操作类型"""
        return list(self._knowledge_cache.keys())