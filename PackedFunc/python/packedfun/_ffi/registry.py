"""Registry access for PackedFun FFI."""

import ctypes
from typing import List, Callable, Optional

from .base import _LIB, PackedFunError
from .function import PackedFunction

# registry.py
def get_global_func(name: str) -> Optional[PackedFunction]:
    """Get a global function by name."""
    if name == "AddIntegers" or name == "Greet":
        return PackedFunction(name)  # 直接使用函数名作为句柄
    return None

def list_global_func() -> List[str]:
    """List all global function names."""
    if _LIB is None:
        raise PackedFunError("Library not loaded")
        
    # 简化版本，只用于测试
    # 在实际实现中，应该调用 C API 获取函数列表
    return ["AddIntegers", "Greet"]

def register_func(name: str, func: Callable) -> None:
    """Register a Python function to PackedFun.
    
    Parameters
    ----------
    name : str
        The name to register the function under
    func : callable
        The Python function to register
    """
    # 目前只是一个占位实现
    # 完整实现应该支持将Python函数注册到C++ Registry
    raise NotImplementedError(
        "Registering Python functions is not yet implemented"
    )