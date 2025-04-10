# test_basic.py
import os
import sys

# 打印 Python 路径和工作目录
print("Python path:", sys.path)
print("Current directory:", os.getcwd())

try:
    # 尝试导入包
    import packedfun
    print("Successfully imported packedfun")
    
    # 导入具体函数
    from packedfun._ffi import list_global_func
    print("Available functions:", list_global_func())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()