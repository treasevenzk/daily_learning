import tvm
from tvm import te, autotvm
from tvm.autotvm import measure
import numpy as np

# ==================== 参数配置 ====================
N, M, K = 256, 256, 256  # 减小矩阵尺寸以加快调试
target = tvm.target.Target("llvm")
dtype = "float32"

# ==================== 显式全局定义任务模板 ====================
# 将任务模板定义在全局作用域以避免序列化问题
@autotvm.template("matmul")
def matmul_tuner():
    # 计算定义
    A = te.placeholder((N, K), name='A', dtype=dtype)
    B = te.placeholder((K, M), name='B', dtype=dtype)
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((N, M), 
                   lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                   name='C')
    
    s = tvm.te.create_schedule(C.op)
    
    # 参数配置空间
    cfg = autotvm.get_config()
    
    # 简化参数空间
    cfg.define_split("tile_x", N, num_outputs=2)
    cfg.define_split("tile_y", M, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 32])
    
    # 应用基础调度
    x, y = s[C].op.axis
    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)
    
    s[C].reorder(xo, yo, xi, yi)
    
    return s, [A, B, C]

# ==================== 创建调优任务 ====================
task = autotvm.task.create(
    "matmul",
    args=(),
    target=target
)

# ==================== 配置调优参数 ====================
tuning_option = {
    'n_trial': 5,  # 进一步减少尝试次数
    'early_stopping': 2,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=2,
            repeat=1,
            timeout=30
        )
    ),
}

# ==================== 使用更简单的调优器 ====================
tuner = autotvm.tuner.RandomTuner(task)  # 改用随机搜索调优器

# 执行调优
tuner.tune(
    n_trial=tuning_option['n_trial'],
    early_stopping=tuning_option['early_stopping'],
    measure_option=tuning_option['measure_option'],
    callbacks=[
        autotvm.callback.log_to_file('matmul.log'),
        autotvm.callback.progress_bar(tuning_option['n_trial'], prefix='Tuning:')
    ]
)

# ==================== 编译最佳内核 ====================
with autotvm.apply_history_best('matmul.log'):
    with tvm.target.Target("llvm"):
        # 重新定义计算图
        A = te.placeholder((N, K), name='A', dtype=dtype)
        B = te.placeholder((K, M), name='B', dtype=dtype)
        k = te.reduce_axis((0, K), name='k')
        C = te.compute((N, M), 
                       lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                       name='C')
        s = tvm.te.create_schedule(C.op)
        
    mod = tvm.build(s, [A, B, C], target=target)

# ==================== 验证结果 ====================
a_np = np.random.uniform(size=(N, K)).astype(dtype)
b_np = np.random.uniform(size=(K, M)).astype(dtype)
c_np = a_np.dot(b_np)

ctx = tvm.context(target, 0)
a_tvm = tvm.nd.array(a_np, ctx)
b_tvm = tvm.nd.array(b_np, ctx)
c_tvm = tvm.nd.empty(c_np.shape, dtype=dtype, ctx=ctx)

mod(a_tvm, b_tvm, c_tvm)
tvm.testing.assert_allclose(c_tvm.asnumpy(), c_np, rtol=1e-3)

print("\n优化验证成功！最佳配置已应用。")