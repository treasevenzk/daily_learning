"""Base definitions for PackedFun FFI."""

import os
import ctypes
from enum import IntEnum

# 自定义异常类
class PackedFunError(Exception):
    """Error thrown by PackedFun functions."""
    pass

def _load_lib():
    # 首先尝试从本地 _lib 目录加载
    curr_path = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(curr_path, '..', '_lib')
    
    # 优先尝试 C API 库
    lib_name = 'libpackedfun_c_api.so'
    full_path = os.path.join(lib_path, lib_name)
    if os.path.exists(full_path):
        try:
            return ctypes.CDLL(full_path)
        except (OSError, ImportError) as e:
            print(f"Failed to load {full_path}: {e}")
    
    # 如果失败，尝试其他可能的名称
    possible_lib_names = ['packedfun_c_api.so', 'libpackedfun_c_api']
    
    for lib_name in possible_lib_names:
        try:
            return ctypes.CDLL(os.path.join(lib_path, lib_name))
        except (OSError, ImportError) as e:
            print(f"Failed to load {lib_name}: {e}")
            continue
            
    raise ImportError(
        "Cannot find libpackedfun_c_api.so. Make sure it's placed in the packedfun/_lib directory."
    )

try:
    _LIB = _load_lib()
except ImportError as e:
    print(f"Library loading error: {e}")
    # 创建一个空的占位符以允许导入继续
    _LIB = None

class TypeCode(IntEnum):
    """Type codes for PackedFun values."""
    INT = 0
    FLOAT = 1
    STR = 2
    BYTES = 3
    HANDLE = 4
    NULL = 5
    NODE_HANDLE = 6

# 仅当库成功加载时才定义函数原型
if _LIB is not None:
    # C API function prototypes
    class PackedValueHandle(ctypes.Structure):
        """Handle to a PackedValue."""
        pass

    # Function prototypes for C API
    _LIB.PackedFunCreateInt.argtypes = [ctypes.c_int]
    _LIB.PackedFunCreateInt.restype = ctypes.POINTER(PackedValueHandle)

    _LIB.PackedFunCreateFloat.argtypes = [ctypes.c_float]
    _LIB.PackedFunCreateFloat.restype = ctypes.POINTER(PackedValueHandle)

    _LIB.PackedFunCreateString.argtypes = [ctypes.c_char_p]
    _LIB.PackedFunCreateString.restype = ctypes.POINTER(PackedValueHandle)

    _LIB.PackedFunGetInt.argtypes = [ctypes.POINTER(PackedValueHandle)]
    _LIB.PackedFunGetInt.restype = ctypes.c_int

    _LIB.PackedFunGetFloat.argtypes = [ctypes.POINTER(PackedValueHandle)]
    _LIB.PackedFunGetFloat.restype = ctypes.c_float

    _LIB.PackedFunGetString.argtypes = [ctypes.POINTER(PackedValueHandle)]
    _LIB.PackedFunGetString.restype = ctypes.c_char_p

    _LIB.PackedFunGetTypeCode.argtypes = [ctypes.POINTER(PackedValueHandle)]
    _LIB.PackedFunGetTypeCode.restype = ctypes.c_int

    _LIB.PackedFunDeleteValue.argtypes = [ctypes.POINTER(PackedValueHandle)]
    _LIB.PackedFunDeleteValue.restype = None

    _LIB.PackedFunGetFuncName.argtypes = [ctypes.c_void_p]
    _LIB.PackedFunGetFuncName.restype = ctypes.c_char_p