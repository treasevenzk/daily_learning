import os
import sys

def find_lib_path():
    """Find MyLib dynamic library files."""
    # 从当前文件位置推导项目根目录
    curr_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(curr_path, "../../.."))
    
    # 直接指定库文件的确切位置
    lib_path = os.path.join(root_path, "build/lib/libmylib.so")
    
    # 检查文件是否存在
    if os.path.exists(lib_path):
        return [lib_path]
    
    # 如果找不到，提供更多调试信息
    print("Current directory:", curr_path)
    print("Root directory:", root_path)
    print("Expected library path:", lib_path)
    print("File exists:", os.path.exists(lib_path))
    
    raise RuntimeError(f"Cannot find the MyLib library at {lib_path}")