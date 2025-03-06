import tvm
from tvm import te
import numpy as np
import time

# 定义矩阵乘法计算 - 未优化版本
def matmul_basic(N, M, K):
    A = te.placeholder((N, K), name='A')
    B = te.placeholder((K, M), name='B')
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = te.create_schedule(C.op)
    return s, [A, B, C]

# 定义矩阵乘法计算 - 优化版本
def matmul_optimized(N, M, K):
    # 声明输入张量
    A = te.placeholder((N, K), name='A')
    B = te.placeholder((K, M), name='B')
    
    # 创建归约轴
    k = te.reduce_axis((0, K), name='k')
    
    # 定义矩阵乘法计算
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    
    # 创建默认的调度
    s = te.create_schedule(C.op)
    
    # 获取输出和循环变量
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 8, 8)
    
    # 将归约轴分割，创建向量化机会
    ko, ki = s[C].split(k, factor=4)
    
    # 重新排序循环顺序，适合向量化
    s[C].reorder(xo, yo, ko, xi, yi, ki)
    
    # 对内层循环进行向量化
    s[C].vectorize(ki)
    
    # 并行化外层循环
    s[C].parallel(xo)
    
    return s, [A, B, C]

# 运行基准测试的函数
def benchmark_matmul(func, a_tvm, b_tvm, c_tvm, repeat=10):
    # 预热运行
    func(a_tvm, b_tvm, c_tvm)
    
    # 测量时间
    total_time = 0
    for i in range(repeat):
        start = time.time()
        func(a_tvm, b_tvm, c_tvm)
        total_time += time.time() - start
    
    return total_time / repeat

# 矩阵大小示例
N, M, K = 1024, 1024, 1024

# 创建输入数据
a_np = np.random.uniform(size=(N, K)).astype(np.float32)
b_np = np.random.uniform(size=(K, M)).astype(np.float32)
c_np = np.zeros((N, M), dtype=np.float32)

# 创建TVM设备
ctx = tvm.cpu(0)
a_tvm = tvm.nd.array(a_np, ctx)
b_tvm = tvm.nd.array(b_np, ctx)
c_tvm = tvm.nd.array(c_np, ctx)

# 对比numpy原生矩阵乘法性能
start_numpy = time.time()
c_numpy = np.matmul(a_np, b_np)
numpy_time = time.time() - start_numpy
print(f"NumPy 矩阵乘法时间: {numpy_time*1000:.2f} ms")

# 基本版本
s_basic, args_basic = matmul_basic(N, M, K)
func_basic = tvm.build(s_basic, args_basic, target="llvm", name="matmul_basic")
c_basic = tvm.nd.array(np.zeros((N, M), dtype=np.float32), ctx)
basic_time = benchmark_matmul(func_basic, a_tvm, b_tvm, c_basic)
print(f"TVM 基本版本时间: {basic_time*1000:.2f} ms")

# 优化版本
s_opt, args_opt = matmul_optimized(N, M, K)
func_opt = tvm.build(s_opt, args_opt, target="llvm", name="matmul_optimized")
c_opt = tvm.nd.array(np.zeros((N, M), dtype=np.float32), ctx)
opt_time = benchmark_matmul(func_opt, a_tvm, b_tvm, c_opt)
print(f"TVM 优化版本时间: {opt_time*1000:.2f} ms")

# 计算加速比
speedup_vs_basic = basic_time / opt_time
speedup_vs_numpy = numpy_time / opt_time
print(f"相对于基本版本的加速比: {speedup_vs_basic:.2f}x")
print(f"相对于NumPy的加速比: {speedup_vs_numpy:.2f}x")

# 验证结果正确性
np.testing.assert_allclose(c_numpy, c_opt.asnumpy(), rtol=1e-5)
print("结果验证: 通过! TVM计算结果与NumPy一致")

# 分析内存访问优化
print("\n优化分析:")
print("1. 分块(Tiling): 提高了缓存局部性，减少内存访问")
print("2. 向量化(Vectorize): 利用CPU SIMD指令，每个指令处理多个数据元素")
print("3. 并行化(Parallel): 利用多核CPU同时计算")
print("4. 循环重排序(Reorder): 优化内存访问模式，减少缓存未命中")

# 输出生成的IR代码以查看底层优化
print("\n优化版本的低级IR代码:")
print(tvm.lower(s_opt, args_opt, simple_mode=True))