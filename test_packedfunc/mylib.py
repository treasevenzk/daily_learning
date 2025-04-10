# mylib.py
import ctypes
import numpy as np
from typing import Any
import os

class PackedFuncWrapper:
    def __init__(self, lib, func_name: str):
        self.lib = lib
        self.func_name = func_name
        # 设置函数参数类型
        self.lib.CallFunction.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.lib.CallFunction.restype = ctypes.c_int
    
    def __call__(self, *args):
        # 将参数转换为C类型
        c_args = [arg if isinstance(arg, (int, float)) else ctypes.c_int(arg) for arg in args]
        result = self.lib.CallFunction(self.func_name.encode(), *c_args)
        return result

class Library:
    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.InitLibrary()
    
    def get_function(self, func_name: str) -> PackedFuncWrapper:
        return PackedFuncWrapper(self.lib, func_name)

# 使用示例
if __name__ == "__main__":
    # 加载库
    lib = Library("./libmylib.so")
    
    # 获取并调用add函数
    add_func = lib.get_function("add")
    result = add_func(10, 20)
    print(f"Add result: {result}")  # 应该输出: 30