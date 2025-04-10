from ctypes import CFUNCTYPE, CDLL, c_int, c_double, POINTER
import sys

lib = CDLL('./libexample.so')

lib.process_data.argtypes = [POINTER(c_int), c_int, CFUNCTYPE(c_int, c_int, c_double)]
lib.process_data.restype = None

@CFUNCTYPE(c_int, c_int, c_double)
def python_callback(value, scaled_value):
    print(f"Python: Received {value} and {scaled_value:.2f}")
    return value * 2

if __name__ == "__main__":

    data = (c_int * 5)(10, 20, 30, 40, 50)

    print("Starting C data processing ...")
    lib.process_data(data, len(data), python_callback)
    print("Processing completed")