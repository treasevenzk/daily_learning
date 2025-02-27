import tvm
from tvm import te
import numpy as np
import time
import tvm.topi.x86

def optimized_matrix_multiply():
    # 定义目标 - 使用skylake兼容模式
    target = tvm.target.Target("llvm -mcpu=skylake-avx512")
    
    # 定义矩阵大小 - 确保能被16整除，适合张量化
    M, N, K = 1024, 1024, 1024
    
    # 创建目标上下文
    with tvm.target.Target(target):
        # 定义输入和输出的占位符 - 注意类型一致性
        A = te.placeholder((M, K), name="A", dtype="uint8")
        B = te.placeholder((N, K), name="B", dtype="int8")
        
        # 定义归约轴
        k = te.reduce_axis((0, K), name="k")
        
        # 定义矩阵乘法计算 - 注意这里是 A[i, k] 和 B[j, k]
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k].astype("int32") * B[j, k].astype("int32"), axis=k),
            name="C"
        )
        
        # 创建默认调度
        s = te.create_schedule(C.op)
        
        # 定义一个简单的函数来创建内联函数
        def create_skylake_intrinsic():
            int32_lanes = 16  # AVX512
            num_int8_elements = 4
            data = te.placeholder((num_int8_elements,), dtype="uint8", name="data")
            kernel = te.placeholder((int32_lanes, num_int8_elements), dtype="int8", name="kernel")
            k = te.reduce_axis((0, num_int8_elements), name="k")
            C = te.compute(
                (int32_lanes,),
                lambda i: te.sum(data[k].astype("int32") * kernel[i, k].astype("int32"), axis=k),
                name="C",
            )
            a_buffer = tvm.tir.decl_buffer(
                data.shape, dtype="uint8", name="a_buffer", offset_factor=1, strides=[1]
            )
            b_buffer = tvm.tir.decl_buffer(
                kernel.shape, dtype="int8", name="b_buffer", offset_factor=1, strides=[te.var("ldw"), 1]
            )
            
            def _intrin_func(ins, outs):
                def _instr(index):
                    # Simple implementation without special instructions
                    ib = tvm.tir.ir_builder.create()
                    if index == 1:
                        ib.emit(outs[0].vstore(0, tvm.tir.const(0, "int32x16")))
                        return ib.get()
                    
                    # Use vectorized operations instead of special intrinsics
                    a = ins[0].vload([0], "uint8x4")
                    b = ins[1].vload([0, 0], "int8x64")
                    
                    # Convert to int32 and perform multiplication and sum
                    a_int32 = a.astype("int32x16")
                    b_int32 = tvm.tir.call_intrin("int32x16", "tir.reinterpret", b)
                    
                    # Perform vector operations
                    mul_res = a_int32 * b_int32
                    result = mul_res
                    
                    if index == 0:
                        ib.emit(outs[0].vstore(0, result))
                    else:
                        ib.emit(outs[0].vstore(0, result + outs[0].vload([0], "int32x16")))
                    return ib.get()
                
                return _instr(0), _instr(1), _instr(2)
            
            # Return tensor intrinsic
            return te.decl_tensor_intrin(
                C.op, 
                _intrin_func, 
                binds={data: a_buffer, kernel: b_buffer}, 
                default_buffer_params={"offset_factor": 1}
            )
                
        # 获取轴
        i, j = C.op.axis
        
        # 缓存优化
        packedB = s.cache_read(B, "global", [C])
        CC = s.cache_write(C, "global")
        
        # 分块的大小必须与张量化要求对齐
        block_size = 16
        
        # 创建遵循张量化模式的分块
        i_outer, i_inner = s[C].split(i, factor=block_size)
        j_outer, j_inner = s[C].split(j, factor=block_size) 
        
        # 设置计算顺序
        s[C].reorder(i_outer, j_outer, i_inner, j_inner)
        
        # 并行化外部循环
        s[C].parallel(i_outer)
        
        # 放置计算位置
        s[CC].compute_at(s[C], j_outer)
        
        # 处理缓存计算的轴
        ii, jj = CC.op.axis
        
        # 分割归约轴，确保与张量化单元大小匹配
        k_outer, k_inner = s[CC].split(k, factor=4)  # 每4个元素一组
        
        # 重新排序轴以准备张量化
        s[CC].reorder(k_outer, ii, k_inner, jj)
        
        # 尝试张量化
        try:
            intrinsic = create_skylake_intrinsic()
            s[CC].tensorize(k_inner, intrinsic)
            print("成功应用张量化!")
        except Exception as e:
            print(f"张量化失败，回退到向量化: {e}")
            s[CC].vectorize(jj)
        
        # 调度缓存读取
        s[packedB].compute_at(s[CC], k_outer)
        px, py = packedB.op.axis
        s[packedB].vectorize(py)
        
        # 构建函数
        func = tvm.build(s, [A, B, C], target=target)
    
    # 创建随机测试数据
    a_np = np.random.randint(0, 255, size=(M, K)).astype(np.uint8)
    b_np = np.random.randint(-127, 127, size=(N, K)).astype(np.int8)
    
    # 计算参考结果
    c_np = np.matmul(a_np.astype(np.int32), b_np.transpose().astype(np.int32))
    
    # 在TVM上运行
    dev = tvm.device(target.kind.name, 0)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype=np.int32), dev)
    
    # 预热
    func(a, b, c)
    
    # 测量性能
    num_runs = 3
    timer = time.time()
    for _ in range(num_runs):
        func(a, b, c)
    dev.sync()
    tvm_time = (time.time() - timer) / num_runs
    
    # 验证结果
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)
    
    # 与NumPy基准进行比较
    timer = time.time()
    for _ in range(num_runs):
        np.matmul(a_np.astype(np.int32), b_np.transpose().astype(np.int32))
    numpy_time = (time.time() - timer) / num_runs
    
    print(f"矩阵大小: {M}x{K} * {K}x{N}")
    print(f"TVM 张量化版本用时: {tvm_time*1000:.2f} ms")
    print(f"NumPy 基准版本用时: {numpy_time*1000:.2f} ms")
    print(f"加速比: {numpy_time/tvm_time:.2f}x")
    
    return func

if __name__ == "__main__":
    optimized_matrix_multiply()