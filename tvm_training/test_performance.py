import tvm
from tvm import te, auto_scheduler
import numpy as np

# 1. 定义计算
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    
    # 计算填充
    pad_h = padding
    pad_w = padding
    
    # 计算输出形状
    OH = (H + 2 * pad_h - KH) // stride + 1
    OW = (W + 2 * pad_w - KW) // stride + 1
    
    # 填充数据
    data_pad = te.compute(
        (N, CI, H + 2 * pad_h, W + 2 * pad_w),
        lambda n, c, h, w: 
            tvm.tir.if_then_else(
                tvm.tir.all(h >= pad_h, h < H + pad_h, w >= pad_w, w < W + pad_w),
                data[n, c, h - pad_h, w - pad_w],
                tvm.tir.const(0.0, data.dtype)
            ),
        name="data_pad"
    )
    
    # 定义规约轴
    rc = te.reduce_axis((0, CI), name="rc")
    rh = te.reduce_axis((0, KH), name="rh")
    rw = te.reduce_axis((0, KW), name="rw")
    
    # 计算卷积
    conv = te.compute(
        (N, CO, OH, OW),
        lambda n, co, h, w:
            te.sum(
                data_pad[n, rc, h * stride + rh, w * stride + rw] * kernel[co, rc, rh, rw],
                axis=[rc, rh, rw]
            ),
        name="conv"
    )
    
    return data, kernel, conv

# 2. 手动创建不同的调度配置
def get_conv_configs(s, data, kernel, conv, target):
    # 生成几种不同的调度配置
    configs = []
    
    # 配置1: 基本调度
    s1 = te.create_schedule(conv.op)
    configs.append(s1)
    
    # 配置2: 使用向量化
    s2 = te.create_schedule(conv.op)
    n, co, h, w = s2[conv].op.axis
    rc, rh, rw = s2[conv].op.reduce_axis
    co, coi = s2[conv].split(co, factor=16)
    s2[conv].reorder(n, co, h, w, coi, rc, rh, rw)
    s2[conv].vectorize(coi)
    configs.append(s2)
    
    # 配置3: 使用并行化
    s3 = te.create_schedule(conv.op)
    n, co, h, w = s3[conv].op.axis
    s3[conv].parallel(co)
    configs.append(s3)
    
    # 配置4: 使用平铺
    s4 = te.create_schedule(conv.op)
    n, co, h, w = s4[conv].op.axis
    co, coi = s4[conv].split(co, factor=4)
    h, hi = s4[conv].split(h, factor=8)
    w, wi = s4[conv].split(w, factor=8)
    s4[conv].reorder(n, co, h, w, coi, hi, wi)
    configs.append(s4)
    
    return configs

# 3. 创建自定义的测量工具
def create_measure_batch(task, configs, target, dev):
    # 编译函数
    def build_func(config_idx):
        config = configs[config_idx]
        # 将计算和调度编译成可执行模块
        try:
            func = tvm.build(config, task, target=target)
            return func
        except Exception as e:
            print(f"编译配置 {config_idx} 失败: {e}")
            return None
    
    # 执行测量
    def measure_func(built_funcs, number=3, repeat=3):
        results = []
        for idx, func in enumerate(built_funcs):
            if func is None:
                results.append({"error": "编译失败", "time": float('inf')})
                continue
                
            # 分配输入输出数据
            data, kernel, conv = task
            # 将TVM的IntImm类型转换为Python整数
            data_shape = tuple(int(dim) for dim in data.shape)
            kernel_shape = tuple(int(dim) for dim in kernel.shape)
            conv_shape = tuple(int(dim) for dim in conv.shape)
            
            data_np = np.random.uniform(size=data_shape).astype(data.dtype)
            kernel_np = np.random.uniform(size=kernel_shape).astype(kernel.dtype)
            conv_np = np.zeros(conv_shape, dtype=conv.dtype)
            
            data_tvm = tvm.nd.array(data_np, dev)
            kernel_tvm = tvm.nd.array(kernel_np, dev)
            conv_tvm = tvm.nd.array(conv_np, dev)
            
            # 预热
            func(data_tvm, kernel_tvm, conv_tvm)
            
            # 计时
            evaluator = func.time_evaluator(func.entry_name, dev, number=number, repeat=repeat)
            time_cost = evaluator(data_tvm, kernel_tvm, conv_tvm).mean
            
            results.append({"error": None, "time": time_cost})
        
        return results
    
    # 创建批处理函数
    def measure_batch(indices):
        # 构建所有配置
        built_funcs = [build_func(idx) for idx in indices]
        # 测量所有配置
        results = measure_func(built_funcs)
        return results
    
    return measure_batch

# 主函数
def main():
    # 设置参数
    N, H, W, CO, CI, KH, KW = 1, 64, 64, 32, 16, 3, 3
    stride, padding = 1, 1
    
    # 创建目标设备
    target = tvm.target.Target("llvm")
    dev = tvm.cpu(0)
    
    # 创建计算
    data, kernel, conv = conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding)
    
    # 创建调度配置
    s = te.create_schedule(conv.op)
    configs = get_conv_configs(s, data, kernel, conv, target)
    
    # 打印配置数量
    print(f"生成了 {len(configs)} 种调度配置")
    
    # 创建测量批处理函数
    task = (data, kernel, conv)
    measure_batch = create_measure_batch(task, configs, target, dev)
    
    # 准备输入索引
    indices = list(range(len(configs)))
    
    # 执行批量测量
    print("开始测量...")
    results = measure_batch(indices)
    
    # 打印结果
    print("\n测量结果:")
    for idx, result in enumerate(results):
        if result["error"] is None:
            print(f"配置 {idx}: {result['time'] * 1000:.6f} ms")
        else:
            print(f"配置 {idx}: {result['error']}")
    
    # 找出最佳配置
    best_idx = min(range(len(results)), key=lambda i: results[i]["time"])
    print(f"\n最佳配置: {best_idx}, 时间: {results[best_idx]['time'] * 1000:.6f} ms")

if __name__ == "__main__":
    main()