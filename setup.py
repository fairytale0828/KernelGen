"""
KernelGen - 多Worker进化式搜索架构
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "README_MULTI_WORKER.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="kernelgen",
    version="2.0.0",
    description="多Worker进化式Triton Kernel优化系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="KernelGen Team",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "pyyaml>=6.0",
        "datasets>=2.0.0",
        "scipy>=1.9.0",  # 用于置信区间计算
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kernelgen-search=kernelgen.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
