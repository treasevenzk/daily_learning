"""Python API for MyLib"""

from ._ffi.function import get_global_func


def add(a, b):
    return get_global_func("mylib.Add")(a, b)

def repeat(text, times):
    return get_global_func("mylib.Repeat")(text, times)

def sum_vector(arr):
    return get_global_func("mylib.SumVector")(*arr)