import ctypes
from ctypes import c_int, c_longlong, c_double, c_void_p, c_char_p, POINTER, Structure, Union

# 加载共享库
lib = ctypes.CDLL('./libtransform.so')

# 定义 C 中的类型码
class ArgTypeCode:
    INT = 0
    FLOAT = 1
    STRING = 2
    HANDLE = 3

# 定义 C 中的联合体和结构体
class Value(Union):
    _fields_ = [
        ("v_int64", c_longlong),
        ("v_float64", c_double),
        ("v_str", c_char_p),
        ("v_handle", c_void_p)
    ]

class ArgValue(Structure):
    _fields_ = [
        ("type_code", c_int),
        ("value", Value)
    ]

# 注册 C 函数
lib.get_value.argtypes = [c_int]
lib.get_value.restype = ArgValue
lib.free_value.argtypes = [ArgValue]
lib.free_value.restype = None

# 类型转换函数
def return_handle(handle):
    # 将句柄转换为 Python 可用的形式
    return int(ctypes.cast(handle, ctypes.c_void_p).value)

# 类型转换字典
RETURN_SWITCH = {
    ArgTypeCode.INT: lambda x: x.value.v_int64,
    ArgTypeCode.FLOAT: lambda x: x.value.v_float64,
    ArgTypeCode.STRING: lambda x: x.value.v_str.decode('utf-8'),
    ArgTypeCode.HANDLE: lambda x: return_handle(x.value.v_handle)
}

def convert_c_to_python(arg_value):
    """将 C 返回的 ArgValue 转换为 Python 对象"""
    converter = RETURN_SWITCH.get(arg_value.type_code, lambda x: None)
    return converter(arg_value)

# 测试函数
def test_get_value(choice):
    # 调用 C 函数
    c_result = lib.get_value(choice)
    
    try:
        # 转换为 Python 对象
        py_result = convert_c_to_python(c_result)
        print(f"C returned type {c_result.type_code}, value: {py_result}")
    finally:
        # 确保释放内存
        lib.free_value(c_result)

if __name__ == "__main__":
    print("Testing integer:")
    test_get_value(0)
    
    print("\nTesting float:")
    test_get_value(1)
    
    print("\nTesting string:")
    test_get_value(2)
    
    print("\nTesting handle:")
    test_get_value(3)