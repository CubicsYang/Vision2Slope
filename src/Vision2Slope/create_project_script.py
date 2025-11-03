"""
Vision2Slope Project Creation Script
运行此脚本将创建完整的项目结构和所有文件
"""

import os
from pathlib import Path

# 项目文件内容
FILES = {
    "vision2slope/__init__.py": '''"""
Vision2Slope: Integrated Pipeline for Road Slope Analysis
========================================================

A comprehensive pipeline for road slope analysis from street view images.

Author: Cubics Yang
Date: June 2025
"""

from .config import PipelineConfig, VisualizationConfig
from .pipeline import Vision2SlopePipeline
from .models import SegmentationModel
from .detectors import SkewDetector
from .correctors import ImageCorrector
from .analyzers import RoadSlopeAnalyzer
from .utils import Utils
from .core.types import ProcessingResult, ProcessingStatus, ProcessingStage

__version__ = "1.0.0"
__all__ = [
    "PipelineConfig",
    "VisualizationConfig",
    "Vision2SlopePipeline",
    "SegmentationModel",
    "SkewDetector",
    "ImageCorrector",
    "RoadSlopeAnalyzer",
    "Utils",
    "ProcessingResult",
    "ProcessingStatus",
    "ProcessingStage",
]
''',

    "requirements.txt": '''torch>=2.0.0
transformers>=4.30.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
Pillow>=10.0.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
tqdm>=4.65.0
''',

    "README.md": '''# Vision2Slope

完整的文档请查看上面的 artifacts 中的 README.md 文件
''',

    ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
output/
results/
*.log
'''
}

def create_project(base_dir="vision2slope_project"):
    """创建完整的项目结构"""
    base_path = Path(base_dir)
    
    print(f"Creating project in: {base_path.absolute()}")
    
    # 创建目录结构
    dirs = [
        "vision2slope",
        "examples",
        "tests",
        "docs",
    ]
    
    for dir_name in dirs:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # 创建文件
    for file_path, content in FILES.items():
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created file: {full_path}")
    
    # 创建占位文件
    placeholder_files = [
        "tests/__init__.py",
        "docs/API.md",
    ]
    
    for file_path in placeholder_files:
        full_path = base_path / file_path
        full_path.touch()
        print(f"✓ Created placeholder: {full_path}")
    
    print("\n" + "="*60)
    print("Project created successfully!")
    print("="*60)
    print(f"\n请按照以下步骤完成项目设置：")
    print(f"\n1. 进入项目目录:")
    print(f"   cd {base_dir}")
    print(f"\n2. 从上面的 artifacts 中复制以下文件的完整内容到对应位置:")
    print(f"   - vision2slope/config.py")
    print(f"   - vision2slope/data_types.py")
    print(f"   - vision2slope/utils.py")
    print(f"   - vision2slope/models.py")
    print(f"   - vision2slope/detectors.py")
    print(f"   - vision2slope/correctors.py")
    print(f"   - vision2slope/analyzers.py")
    print(f"   - vision2slope/visualizers.py")
    print(f"   - vision2slope/pipeline.py")
    print(f"   - vision2slope/cli.py")
    print(f"   - main.py")
    print(f"   - examples/example_usage.py")
    print(f"   - setup.py")
    print(f"\n3. 安装依赖:")
    print(f"   pip install -r requirements.txt")
    print(f"\n4. 安装项目:")
    print(f"   pip install -e .")
    print(f"\n5. 运行示例:")
    print(f"   python main.py --help")

if __name__ == "__main__":
    create_project()
