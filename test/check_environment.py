#!/usr/bin/env python3
"""
环境检查脚本
检查 Python 版本和 PyMC 版本是否满足要求
"""

import sys
import importlib
import subprocess
from typing import Tuple, Optional

def check_python_version() -> Tuple[bool, str]:
    """
    检查 Python 版本是否 >= 3.9
    
    Returns:
        (是否满足要求, 版本信息)
    """
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"Python {major}.{minor}"
    
    if major >= 3 and minor >= 9:
        return True, f"OK {version_str} (满足要求)"
    else:
        return False, f"ERROR {version_str} (需要 >= 3.9)"

def check_pymc_version() -> Tuple[bool, str]:
    """
    检查 PyMC 版本是否 >= 5.10
    
    Returns:
        (是否满足要求, 版本信息)
    """
    try:
        import pymc
        version = pymc.__version__
        # 解析版本号
        major, minor, _ = map(int, version.split('.')[:2])
        
        if major >= 5 and minor >= 10:
            return True, f"OK PyMC {version} (满足要求)"
        else:
            return False, f"ERROR PyMC {version} (需要 >= 5.10)"
    except ImportError:
        return False, "ERROR PyMC 未安装"

def main() -> None:
    """
    主函数，执行环境检查
    """
    print("=== 环境检查 ===")
    print()
    
    # 检查 Python 版本
    python_ok, python_info = check_python_version()
    print(f"Python 版本: {python_info}")
    
    # 检查 PyMC 版本
    pymc_ok, pymc_info = check_pymc_version()
    print(f"PyMC 版本: {pymc_info}")
    print()
    
    if python_ok and pymc_ok:
        print("OK 环境检查通过，所有依赖满足要求！")
        return 0
    else:
        print("ERROR 环境检查失败，请更新依赖。")
        print("建议执行: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
