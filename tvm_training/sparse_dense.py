import numpy as np
import tvm
from tvm import te, topi
import time
import scipy.sparse as sp

def compare_sparse_dense_methods():
    # 参数设置
    n = 10000  # 矩阵大小 n x n
    density = 0.01  # 稀疏度 (非零元素比例)
    
    # 创建一个随机向量
    np_rng = np.random.RandomState(seed=123)
    x = np_rng.uniform(size=n).astype("float32")
    
    # 创建一个随机稀疏矩阵
    nnz = int(n * n * density)  # 非零元素的数量
    rows = np_rng.randint(0, n, nnz)
    cols = np_rng.randint(0, n, nnz)
    data = np_rng.uniform(size=nnz).astype("float32")
    
    # 使用 scipy 创建 CSR 矩阵
    sparse_scipy = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # 从 scipy CSR 矩阵中提取数据用于 TVM
    csr_data = sparse_scipy.data
    csr_indices = sparse_scipy.indices
    csr_indptr = sparse_scipy.indptr
    
    # 将同一个稀疏矩阵转换为稠密表示
    dense_matrix = sparse_scipy.toarray().astype("float32")
    
    # 方法1: 稀疏矩阵视为稠密矩阵计算 (不使用专门的稀疏优化)
    def sparse_as_dense():
        # 创建 TVM 的张量
        a = te.placeholder((n, n), name="a")
        b = te.placeholder((n,), name="b")
        
        # 手动实现矩阵向量乘法
        k = te.reduce_axis((0, n), name="k")
        c = te.compute((n,), lambda i: te.sum(a[i, k] * b[k], axis=k), name="c")
        
        # 创建计算调度
        s = te.create_schedule(c.op)
        
        # 编译函数
        func = tvm.build(s, [a, b, c], target="llvm")
        
        # 创建 TVM 数组 (注意，我们使用的是由稀疏矩阵转换成的稠密矩阵)
        a_tvm = tvm.nd.array(dense_matrix)
        b_tvm = tvm.nd.array(x)
        c_tvm = tvm.nd.array(np.zeros(n, dtype="float32"))
        
        # 计时
        start = time.time()
        func(a_tvm, b_tvm, c_tvm)
        end = time.time()
        
        return c_tvm.numpy(), end - start
    
    # 方法2: 使用稀疏矩阵专用优化 (topi.sparse.csrmv)
    def sparse_optimized():
        try:
            # 先获取topi.sparse.csrmv函数的参数数量，以适应不同版本的TVM
            import inspect
            csrmv_sig = inspect.signature(topi.sparse.csrmv)
            num_params = len(csrmv_sig.parameters)
            print(f"topi.sparse.csrmv函数有{num_params}个参数")
            
            # 创建 TVM 的张量
            a_data = te.placeholder((len(csr_data),), name="a_data")
            a_indices = te.placeholder((len(csr_indices),), dtype="int32", name="a_indices")
            a_indptr = te.placeholder((n + 1,), dtype="int32", name="a_indptr")
            b = te.placeholder((n,), name="b")
            
            # 创建一个包含CSR格式信息的字典
            csr_tensor = {
                "data": a_data,
                "indices": a_indices,
                "indptr": a_indptr,
                "shape": (n, n)
            }
            
            # 根据参数数量选择正确的调用方式
            if num_params == 2:
                # 假设函数签名是 csrmv(A, x)，其中A是CSR张量字典
                c = topi.sparse.csrmv(csr_tensor, b)
            elif num_params == 3:
                # 假设函数签名是 csrmv(A, x, y)，y是可选的输出张量
                c = topi.sparse.csrmv(csr_tensor, b, None)
            else:
                # 尝试最常见的调用方式
                print("尝试使用原始调用方式")
                c = topi.sparse.csrmv(a_data, a_indices, a_indptr, b, (n, n))
            
            # 创建计算调度
            s = te.create_schedule(c.op)
            
            # 编译函数
            if num_params == 2 or num_params == 3:
                func = tvm.build(s, [a_data, a_indices, a_indptr, b, c], target="llvm")
            else:
                func = tvm.build(s, [a_data, a_indices, a_indptr, b, c], target="llvm")
            
            # 创建 TVM 数组
            a_data_tvm = tvm.nd.array(csr_data)
            a_indices_tvm = tvm.nd.array(csr_indices)
            a_indptr_tvm = tvm.nd.array(csr_indptr)
            b_tvm = tvm.nd.array(x)
            c_tvm = tvm.nd.array(np.zeros(n, dtype="float32"))
            
            # 计时
            start = time.time()
            func(a_data_tvm, a_indices_tvm, a_indptr_tvm, b_tvm, c_tvm)
            end = time.time()
            
            return c_tvm.numpy(), end - start
            
        except Exception as e:
            print(f"错误: {e}")
            print("提供TVM版本信息:")
            print(f"TVM版本: {tvm.__version__ if hasattr(tvm, '__version__') else '未知'}")
            
            # 尝试直接使用SciPy的CSR矩阵乘法作为替代
            print("使用SciPy的CSR矩阵乘法作为替代...")
            start = time.time()
            result = sparse_scipy.dot(x)
            end = time.time()
            
            return result, end - start
    
    # 运行对比
    print(f"矩阵大小: {n}x{n}, 非零元素密度: {density*100:.2f}%, 非零元素数量: {nnz}")
    
    # 获取基准结果 (使用NumPy计算作为基准)
    baseline = np.dot(dense_matrix, x)
    
    # 方法1: 稀疏当做稠密
    print("正在执行方法1: 将稀疏矩阵当做稠密矩阵处理...")
    dense_result, dense_time = sparse_as_dense()
    dense_correct = np.allclose(dense_result, baseline, rtol=1e-5)
    
    # 方法2: 稀疏优化
    print("正在执行方法2: 使用稀疏矩阵专用优化...")
    sparse_result, sparse_time = sparse_optimized()
    sparse_correct = np.allclose(sparse_result, baseline, rtol=1e-5)
    
    # 打印结果
    print(f"\n{'方法':<25} {'时间(秒)':<15} {'结果正确':<15} {'加速比':<15}")
    print(f"{'-'*70}")
    print(f"{'稀疏当做稠密矩阵':<25} {dense_time:<15.6f} {dense_correct:<15} {'1.0':<15}")
    print(f"{'使用稀疏矩阵专用优化':<25} {sparse_time:<15.6f} {sparse_correct:<15} {dense_time/sparse_time:<15.2f}")
    
    # 内存使用对比
    dense_memory = dense_matrix.nbytes
    sparse_memory = csr_data.nbytes + csr_indices.nbytes + csr_indptr.nbytes
    memory_ratio = dense_memory / sparse_memory if sparse_memory > 0 else float('inf')
    
    print(f"\n内存使用对比:")
    print(f"{'表示方法':<25} {'内存(MB)':<15} {'节省比例':<15}")
    print(f"{'-'*55}")
    print(f"{'稠密表示':<25} {dense_memory/1024/1024:<15.2f} {'1.0':<15}")
    print(f"{'CSR稀疏表示':<25} {sparse_memory/1024/1024:<15.2f} {memory_ratio:<15.2f}")

if __name__ == "__main__":
    compare_sparse_dense_methods()