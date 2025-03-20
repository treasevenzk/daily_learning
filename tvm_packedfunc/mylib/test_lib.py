# 保存为test_lib.py
import ctypes
import os

# 打印当前目录
print("Current directory:", os.getcwd())

# 尝试直接加载库
lib_path = os.path.join(os.getcwd(), "build/lib/libmylib.so")
print("Trying to load:", lib_path)
print("File exists:", os.path.exists(lib_path))

try:
    lib = ctypes.CDLL(lib_path)
    print("Library loaded successfully!")
except Exception as e:
    print("Failed to load library:", e)