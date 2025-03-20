import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import mylib

def main():
    # 使用Add函数
    result = mylib.add(10, 20)
    print(f"10 + 20 = {result}")
    
    # 使用Repeat函数
    text = mylib.repeat("Hello ", 3)
    print(f"Repeat 'Hello ' 3 times: {text}")
    
    # 使用SumVector函数
    arr = [1, 2, 3, 4, 5]
    total = mylib.sum_vector(arr)
    print(f"Sum of {arr} = {total}")

if __name__ == "__main__":
    main()