import numpy as np
import tvm
from tvm import te
import tvm.testing
from tvm.contrib import nvcc
import tvm.autotvm as autotvm

# 检查GPU是否可用
if not tvm.runtime.enabled("cuda"):
    print("CUDA不可用，请确保已安装CUDA并配置TVM支持CUDA")
    exit(-1)

# 定义矩阵大小
M, K, N = 1024, 1024, 1024

# 定义计算
def matmul_basic(M, K, N):
    """返回基础矩阵乘法算子的计算描述"""
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: 
                   te.sum(A[i, k] * B[k, j], axis=k), name="C")
    return A, B, C

# 简化版调度，避免冲突的线程绑定
def schedule_matmul_gpu(A, B, C):
    """为GPU创建一个简化的调度"""
    s = te.create_schedule(C.op)
    
    # 获取缓存写入
    CC = s.cache_write(C, "local")
    
    # 定位迭代变量
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    
    # 简化版瓦片设计
    block_size = 32
    
    # 分割迭代空间
    by, yi = s[C].split(C.op.axis[0], factor=block_size)
    bx, xi = s[C].split(C.op.axis[1], factor=block_size)
    
    # 绑定到块
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    
    # 进一步划分为线程
    tyz, yi = s[C].split(yi, nparts=4)  # 使用固定的线程数
    txz, xi = s[C].split(xi, nparts=8)  # 使用固定的线程数
    
    # 绑定到线程
    s[C].bind(tyz, thread_y)
    s[C].bind(txz, thread_x)
    
    # 调整循环顺序
    s[C].reorder(by, bx, tyz, txz, yi, xi)
    
    # 缓存写入调度
    s[CC].compute_at(s[C], txz)
    yo, xo = CC.op.axis
    
    # 分割reduction轴
    k = CC.op.reduce_axis[0]
    ko, ki = s[CC].split(k, factor=8)
    
    # 调整循环顺序
    s[CC].reorder(ko, ki, yo, xo)
    
    # 添加共享内存缓存
    AA = s.cache_read(A, "shared", [CC])
    BB = s.cache_read(B, "shared", [CC])
    
    # 在线程块内加载
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    
    # 矢量化计算，简单起见不再绑定额外的线程
    s[C].vectorize(xi)
    
    return s

# 编译和运行矩阵乘法
def run_matmul_gpu(M, K, N):
    # 获取基本计算
    A, B, C = matmul_basic(M, K, N)
    
    # 应用优化调度
    s = schedule_matmul_gpu(A, B, C)
    
    # 构建CUDA内核
    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(s, [A, B, C], "cuda", name="matmul")
    
    # 创建输入数据
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    
    # 分配设备内存
    dev = tvm.cuda(0)
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(np.zeros((M, N), dtype=np.float32), dev)
    
    # 执行内核
    func(a_tvm, b_tvm, c_tvm)
    
    # 验证结果
    c_np = np.matmul(a_np, b_np)
    tvm.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
    
    # 评估性能
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    time_ms = evaluator(a_tvm, b_tvm, c_tvm).mean * 1000
    
    # 计算GFLOPS
    gflops = 2 * M * N * K / (time_ms / 1000) / 1e9
    print(f"矩阵大小: {M}x{K} * {K}x{N}")
    print(f"执行时间: {time_ms:.3f} ms")
    print(f"计算性能: {gflops:.2f} GFLOPS")
    
    return gflops

# 使用AutoTVM进行自动调优
def tune_and_evaluate(M, K, N, tuning_rounds=200):
    @autotvm.template("matmul_auto")
    def matmul_auto(M, K, N):
        A = te.placeholder((M, K), name="A")
        B = te.placeholder((K, N), name="B")
        k = te.reduce_axis((0, K), name="k")
        C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
        
        s = te.create_schedule(C.op)
        
        # 允许自动调优器优化这些参数
        cfg = autotvm.get_config()
        
        # 定义搜索空间
        block_x = te.thread_axis("blockIdx.x")
        block_y = te.thread_axis("blockIdx.y")
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")
        
        # 使用固定的分割因子，简化调优空间
        # 定义候选的分割因子
        cfg.define_knob("block_row", [8, 16, 32, 64])
        cfg.define_knob("block_col", [8, 16, 32, 64])
        cfg.define_knob("thread_row", [2, 4, 8])
        cfg.define_knob("thread_col", [2, 4, 8, 16])
        cfg.define_knob("k_factor", [4, 8, 16, 32])
        
        # 缓存写入
        CC = s.cache_write(C, "local")
        
        # 创建循环嵌套
        mo, no = C.op.axis[0], C.op.axis[1]
        
        # 使用cfg获取的实际值进行分割
        block_row = cfg["block_row"].val
        block_col = cfg["block_col"].val
        thread_row = cfg["thread_row"].val
        thread_col = cfg["thread_col"].val
        
        # 分割循环
        by, ty = s[C].split(mo, factor=block_row)
        ty, mi = s[C].split(ty, factor=thread_row)
        bx, tx = s[C].split(no, factor=block_col)
        tx, ni = s[C].split(tx, factor=thread_col)
        
        # 绑定到线程和块
        s[C].bind(by, block_y)
        s[C].bind(bx, block_x)
        s[C].bind(ty, thread_y)
        s[C].bind(tx, thread_x)
        
        # 计算在本地内存中
        s[CC].compute_at(s[C], tx)
        mc, nc = CC.op.axis
        kc = CC.op.reduce_axis[0]
        k_factor = cfg["k_factor"].val
        ko, ki = s[CC].split(kc, factor=k_factor)
        
        # 共享内存
        AA = s.cache_read(A, "shared", [CC])
        BB = s.cache_read(B, "shared", [CC])
        
        # 在reduction循环级别计算共享缓存
        s[AA].compute_at(s[CC], ko)
        s[BB].compute_at(s[CC], ko)
        
        # 矢量化，向量化循环可以由硬件限制决定
        cfg.define_knob("vector_n", [1, 2, 4, 8, 16])
        vector_n = cfg["vector_n"].val
        if vector_n > 1:
            s[C].vectorize(ni)
        
        return s, [A, B, C]
    
    # 创建TVM调优任务
    task = autotvm.task.create("matmul_auto", args=(M, K, N), target="cuda")
    
    # 配置调优过程
    config_space = task.config_space
    print(f"配置空间大小: {len(config_space)}")
    
    # 创建调优器
    tuner = autotvm.tuner.XGBTuner(task)
    
    # 调优记录文件
    log_file = "matmul_gpu_tune.log"
    
    # 开始调优
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=10, repeat=3, timeout=4, min_repeat_ms=150)
    )
    
    # 开始调优过程
    tuner.tune(
        n_trial=tuning_rounds,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )
    
    # 应用最佳配置
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            s, args = matmul_auto(M, K, N)
            # 获取并编译优化后的函数
            best_func = tvm.build(s, args, target="cuda")
    
    # 评估最佳配置
    print("使用最佳调优配置的性能:")
    # 创建输入数据
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    
    # 分配设备内存
    dev = tvm.cuda(0)
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(np.zeros((M, N), dtype=np.float32), dev)
    
    # 执行内核
    best_func(a_tvm, b_tvm, c_tvm)
    
    # 评估性能
    evaluator = best_func.time_evaluator(best_func.entry_name, dev, number=10)
    time_ms = evaluator(a_tvm, b_tvm, c_tvm).mean * 1000
    
    # 计算GFLOPS
    gflops = 2 * M * N * K / (time_ms / 1000) / 1e9
    print(f"矩阵大小: {M}x{K} * {K}x{N}")
    print(f"最佳执行时间: {time_ms:.3f} ms")
    print(f"最佳计算性能: {gflops:.2f} GFLOPS")
    
    return gflops

# 主函数
if __name__ == "__main__":
    print("使用手动优化的矩阵乘法:")
    manual_gflops = run_matmul_gpu(M, K, N)
    
    print("\n开始AutoTVM自动调优过程...")
    auto_gflops = tune_and_evaluate(M, K, N, tuning_rounds=200)
    
    print(f"\n性能提升: {auto_gflops/manual_gflops:.2f}x")