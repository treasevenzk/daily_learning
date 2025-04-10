import numpy as np
import tvm
from tvm import te, autotvm
from tvm.autotvm.measure import MeasureInput, MeasureResult, create_measure_batch

# 简化版的矩阵乘法模板，不在模板内部分割迭代变量
@autotvm.template("matmul_simple")
def matmul_simple(N, L, M, dtype):
    """简单的矩阵乘法模板，让AutoTVM系统自动应用配置"""
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)
    
    k = te.reduce_axis((0, L), name='k')
    C = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name='C'
    )
    
    s = te.create_schedule(C.op)
    
    # 获取循环变量
    i, j = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    
    # 创建配置空间
    cfg = autotvm.get_config()
    
    # 定义配置参数
    cfg.define_knob("tile_i", [8, 16])
    cfg.define_knob("tile_j", [8, 16])
    
    # 注意：不在这里应用配置，让AutoTVM框架负责应用
    # 只声明配置空间，不进行实际的split操作
    
    # 在这里提供tuning提示，但不实际执行分割
    # 这些提示会被AutoTVM在运行时使用
    cfg.define_split("tile_i", i, num_outputs=2)
    cfg.define_split("tile_j", j, num_outputs=2)
    
    return s, [A, B, C]

def basic_measure_demo():
    """简单的MeasureInput和create_measure_batch的演示"""
    # 参数设置
    N, L, M = 64, 64, 64
    dtype = "float32"
    target = tvm.target.Target("llvm")
    
    # 创建任务
    task = autotvm.task.create("matmul_simple", args=(N, L, M, dtype), target=target)
    print(f"创建任务: {task}")
    
    # 输出配置空间信息
    print(f"配置空间大小: {len(task.config_space)}")
    
    # 获取几个配置
    num_configs = min(3, len(task.config_space))
    configs = [task.config_space.get(i) for i in range(num_configs)]
    
    # 创建MeasureInput实例
    inputs = []
    for config in configs:
        measure_input = MeasureInput(target=task.target, task=task, config=config)
        inputs.append(measure_input)
        print(f"创建MeasureInput: {measure_input}")
    
    # 设置测量选项
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=3, repeat=1, min_repeat_ms=0, timeout=4)
    )
    
    # 这里我们只演示API的使用，不实际执行
    print("\n===== 测量API演示 =====")
    print("1. 创建任务")
    print("2. 选择配置")
    print("3. 创建MeasureInput")
    print("4. 设置测量选项")
    print("5. 创建测量批处理函数:")
    print("   measure_batch = create_measure_batch(task, measure_option)")
    print("6. 执行测量:")
    print("   results = measure_batch(inputs)")

    # 创建测试数据（仅用于验证）
    print("\n===== 创建测试数据 =====")
    dev = tvm.device(target.kind.name, 0)
    a_np = np.random.uniform(size=(N, L)).astype(dtype)
    b_np = np.random.uniform(size=(L, M)).astype(dtype)
    c_np = np.dot(a_np, b_np)  # 基准结果
    
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(np.zeros((N, M), dtype=dtype), dev)
    
    # 使用默认调度
    print("\n===== 使用默认调度 =====")
    s, args = matmul_simple(N, L, M, dtype)
    func = tvm.build(s, args, target=target)
    func(a_tvm, b_tvm, c_tvm)
    
    # 验证结果
    np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
    print("矩阵乘法结果验证成功!")

def main():
    print("===== TVM AutoTVM 矩阵乘法示例 =====")
    print("这个示例展示了如何使用MeasureInput和measure_option")
    
    try:
        basic_measure_demo()
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        
    print("\n===== 结束演示 =====")
    print("注意: 为了避免序列化问题，我们没有实际执行create_measure_batch")
    print("在实际应用中，完整的调用序列如下:")
    print("  task = autotvm.task.create(...)")
    print("  measure_option = autotvm.measure_option(...)")
    print("  measure_batch = create_measure_batch(task, measure_option)")
    print("  inputs = [MeasureInput(target, task, config) for config in configs]")
    print("  results = measure_batch(inputs)")

if __name__ == "__main__":
    main()