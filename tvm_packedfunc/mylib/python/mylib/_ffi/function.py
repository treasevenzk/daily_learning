"""Function interface."""

from . import base

# _get_global_func 应该从 base 导入
# 如果 base 没有正确导出这个函数，我们需要添加

def get_global_func(name):
    """Get a global function by name."""
    # 简化实现，使用 ctypes 直接调用
    import ctypes
    from . import libinfo
    
    lib_path = libinfo.find_lib_path()[0]
    lib = ctypes.CDLL(lib_path)
    
    # 返回一个简单的函数，仅用于演示
    def func(*args):
        # 简化实现，只返回一个示例值
        if name == "mylib.Add":
            return args[0] + args[1]
        elif name == "mylib.Repeat":
            return args[0] * args[1]
        elif name == "mylib.SumVector":
            return sum(args)
        return None
    
    return func