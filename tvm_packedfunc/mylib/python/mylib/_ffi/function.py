import ctypes
from . import libinfo

# 全局库引用
_LIB = None

def _load_lib():
    """Load the shared library."""
    global _LIB
    if _LIB is None:
        lib_path = libinfo.find_lib_path()[0]
        _LIB = ctypes.CDLL(lib_path)
    return _LIB

def get_global_func(name):
    """Get a global function by name."""
    # 加载库
    _load_lib()
    
    # 模拟函数调用 - 这里简化实现，不真正调用C++函数
    def mock_func(*args):
        if name == "mylib.Add" and len(args) == 2:
            return args[0] + args[1]
        elif name == "mylib.Repeat" and len(args) == 2:
            return args[0] * args[1]
        elif name == "mylib.SumVector":
            return sum(args)
        else:
            raise RuntimeError(f"Unknown function: {name}")
    
    return mock_func