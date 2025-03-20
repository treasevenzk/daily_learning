"""Base definitions for FFI."""

import os
import sys
import numpy as np

# 定义lib_path函数
def find_lib_path():
    """Find MyLib dynamic library files."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # 在各种可能的位置搜索库
    lib_search_paths = [
        # 本地开发路径
        os.path.join(curr_path, "../../../build/lib"),
        os.path.join(curr_path, "../../../lib"),
        # 安装路径
        os.path.join(sys.prefix, "lib"),
        # 系统路径
        "/usr/local/lib",
        "/usr/lib"
    ]
    
    # 根据不同平台确定库名称
    if sys.platform.startswith("win32"):
        lib_name = "libmylib.pyd"
    elif sys.platform.startswith("darwin"):
        lib_name = "libmylib.so"
    else:
        lib_name = "libmylib.so"
    
    # 在所有可能的路径中搜索库
    lib_paths = [os.path.join(path, lib_name) for path in lib_search_paths]
    
    # 返回找到的第一个库路径，如果找不到则抛出异常
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            return [lib_path]
    
    raise RuntimeError(
        "Cannot find the MyLib library. " + 
        "List of candidates:\n" + "\n".join(lib_paths)
    )

# 获取库路径
_get_lib_path = find_lib_path

# 尝试导入C++库
try:
    # 使用绝对导入避免混淆
    from mylib._ffi.libmylib import get_global_func as _get_global_func
except ImportError:
    if not os.path.exists(_get_lib_path()[0]):
        raise ImportError(
            "Cannot find the C++ library. Make sure it is built correctly. "
            "You may need to run the build script first."
        )
    # 如果导入失败但库存在，尝试将库添加到Python路径
    lib_path = os.path.dirname(_get_lib_path()[0])
    if lib_path not in sys.path:
        sys.path.append(lib_path)
    try:
        from mylib._ffi.libmylib import get_global_func as _get_global_func
    except ImportError as e:
        raise ImportError(
            f"Failed to import libmylib: {str(e)}. "
            "Make sure the library is built correctly."
        ) from e

    