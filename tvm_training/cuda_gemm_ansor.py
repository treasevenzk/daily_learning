import numpy as np
import tvm
from tvm import te, auto_scheduler
import tvm.testing
from tvm.contrib import nvcc

# 检查GPU是否可用
if not tvm.runtime.enabled("cuda"):
    print("CUDA不可用，请确保已安装CUDA并配置TVM支持CUDA")
    exit(-1)

# 定义矩阵大小
M, K, N = 1024, 1024, 1024

# 1. 定义矩阵乘法计算
@auto_scheduler.register_workload
def matmul(M, K, N):
    """返回矩阵乘法计算描述"""
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N), 
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), 
        name="C"
    )
    return [A, B, C]

# 2. 使用Ansor执行搜索和优化
def optimize_matmul_ansor(M, K, N, num_trials=1000):
    """使用Ansor自动搜索最佳调度"""
    print(f"使用Ansor优化 {M}x{K} * {K}x{N} 矩阵乘法...")
    
    # 创建任务
    task = auto_scheduler.SearchTask(func=matmul, args=(M, K, N), target="cuda")
    
    # 检查任务
    print("计算DAG:")
    print(task.compute_dag)
    
    # 定义调优日志文件
    log_file = "ansor_matmul_gpu.json"
    
    # 创建调优器
    tuner = auto_scheduler.TaskScheduler([task])
    
    # 调优选项
    measure_ctx = auto_scheduler.LocalRPCMeasureContext()
    
    # 设置搜索策略 - 使用兼容TVM 0.15.0的参数
    search_strategy = auto_scheduler.SketchPolicy(
        task, 
        # 在TVM 0.15.0中，参数名可能有所不同
        # 如果出错，简化为不带额外参数的版本
        # program_cost_model=auto_scheduler.XGBModel(),
        # init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    
    # 搜索运行参数
    tuning_options = auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,  # 搜索次数
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    
    # 开始搜索
    print("开始Ansor搜索过程...")
    tuner.tune(tuning_options, search_strategy=search_strategy)
    
    # 应用最佳结果
    print("应用最佳搜索结果...")
    sch, args = task.apply_best(log_file)
    
    # 编译生成优化后的函数
    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(sch, args, target="cuda")
    
    # 测量性能
    print("评估Ansor优化后的性能...")
    return measure_performance(func, M, K, N)

# 3. Numpy参考实现
def numpy_matmul(M, K, N):
    """使用Numpy执行矩阵乘法"""
    print(f"使用Numpy执行 {M}x{K} * {K}x{N} 矩阵乘法...")
    
    # 生成随机矩阵
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    
    # 计时
    import time
    start = time.time()
    c_np = np.matmul(a_np, b_np)
    end = time.time()
    
    # 计算性能
    time_ms = (end - start) * 1000
    gflops = 2 * M * N * K / (time_ms / 1000) / 1e9
    
    print(f"Numpy执行时间: {time_ms:.3f} ms")
    print(f"Numpy计算性能: {gflops:.2f} GFLOPS")
    
    return a_np, b_np, c_np, gflops

# 4. 测量Ansor优化函数的性能
def measure_performance(func, M, K, N):
    """测量TVM生成的函数性能"""
    # 创建输入数据
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    c_np = np.matmul(a_np, b_np)  # 参考结果
    
    # 分配设备内存
    dev = tvm.cuda(0)
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(np.zeros((M, N), dtype=np.float32), dev)
    
    # 执行内核
    func(a_tvm, b_tvm, c_tvm)
    
    # 验证结果
    tvm.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
    
    # 评估性能
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    time_ms = evaluator(a_tvm, b_tvm, c_tvm).mean * 1000
    
    # 计算GFLOPS
    gflops = 2 * M * N * K / (time_ms / 1000) / 1e9
    print(f"矩阵大小: {M}x{K} * {K}x{N}")
    print(f"Ansor执行时间: {time_ms:.3f} ms")
    print(f"Ansor计算性能: {gflops:.2f} GFLOPS")
    
    return gflops

# 5. 可视化搜索过程中的性能变化
def visualize_tuning_progress(log_file):
    """可视化调优过程"""
    try:
        import matplotlib.pyplot as plt
        import json
        
        # 检查日志文件是否存在
        import os
        if not os.path.exists(log_file):
            print(f"警告: 日志文件 '{log_file}' 不存在，无法可视化调优进度")
            return
            
        # 读取日志文件
        with open(log_file, "r") as f:
            lines = f.readlines()
            
        if not lines:
            print(f"警告: 日志文件 '{log_file}' 为空，无法可视化调优进度")
            return
            
        # 解析JSON数据
        data = []
        for line in lines:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"警告: 跳过无效的JSON行")
                continue
        
        # 提取性能数据 - 适配TVM 0.15.0的日志格式
        costs = []
        for item in data:
            if "r" in item and isinstance(item["r"], list) and len(item["r"]) > 0:
                if isinstance(item["r"][0], list) and len(item["r"][0]) > 0:
                    costs.append(item["r"][0][0])
                    
        if not costs:
            print(f"警告: 无法从日志文件中提取性能数据")
            return
            
        costs = [cost if cost > 0 else float('inf') for cost in costs]
        
        # 计算最佳性能
        best_costs = []
        best_cost = float('inf')
        for cost in costs:
            if cost < best_cost:
                best_cost = cost
            best_costs.append(best_cost)
        
        # 绘制图表
        plt.figure(figsize=(10, 6))
        plt.plot(costs, label='当前搜索')
        plt.plot(best_costs, label='历史最佳')
        plt.xlabel('搜索迭代次数')
        plt.ylabel('运行时间 (ms)')
        plt.title('Ansor调优过程中的性能变化')
        plt.legend()
        plt.grid()
        plt.yscale('log')
        plt.savefig('ansor_tuning_progress.png')
        plt.close()
        
        print("调优进度图已保存为 'ansor_tuning_progress.png'")
    except ImportError:
        print("警告: 未安装matplotlib，无法生成可视化图表")
    except Exception as e:
        print(f"生成可视化图表时出错: {e}")

# 6. 比较不同矩阵大小下的性能
def compare_performance(sizes=[(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]):
    """比较不同矩阵大小下的性能"""
    results = []
    
    for M, K, N in sizes:
        print(f"\n===== 测试矩阵大小: {M}x{K} * {K}x{N} =====")
        
        # Numpy基准
        _, _, _, numpy_gflops = numpy_matmul(M, K, N)
        
        # Ansor优化
        ansor_gflops = optimize_matmul_ansor(M, K, N, num_trials=200)
        
        # 记录结果
        speedup = ansor_gflops / numpy_gflops
        results.append({
            'size': f"{M}x{K}x{N}",
            'numpy_gflops': numpy_gflops,
            'ansor_gflops': ansor_gflops,
            'speedup': speedup
        })
    
    # 打印性能比较表格
    print("\n===== 性能比较 =====")
    print(f"{'矩阵大小':<15} {'Numpy (GFLOPS)':<20} {'Ansor (GFLOPS)':<20} {'加速比':<10}")
    print("-" * 65)
    for result in results:
        print(f"{result['size']:<15} {result['numpy_gflops']:<20.2f} {result['ansor_gflops']:<20.2f} {result['speedup']:<10.2f}x")

# 7. 展示不同优化层次的调度过程
def print_schedule_transformations(M=1024, K=1024, N=1024):
    """打印Ansor自动生成的变换过程"""
    try:
        task = auto_scheduler.SearchTask(func=matmul, args=(M, K, N), target="cuda")
        log_file = "ansor_matmul_gpu.json"
        
        # 检查日志文件是否存在
        import os
        if not os.path.exists(log_file):
            print(f"警告: 日志文件 '{log_file}' 不存在，无法获取最佳调度")
            
            # 打印低级IR - 这部分不依赖日志文件
            try:
                print("===== 优化前的Compute IR =====")
                print(task.compute_dag.print_python_code_as_str())
            except AttributeError:
                print("在TVM 0.15.0中可能无法访问compute_dag.print_python_code_as_str")
                try:
                    # 尝试使用替代方法
                    print(task.compute_dag)
                except:
                    print("无法打印计算图")
            return
            
        # 获取最佳调度
        try:
            sch, args = task.apply_best(log_file)
            
            # 打印低级IR
            print("===== 优化前的Compute IR =====")
            try:
                print(task.compute_dag.print_python_code_as_str())
            except AttributeError:
                print("在TVM 0.15.0中可能无法访问compute_dag.print_python_code_as_str")
                try:
                    # 尝试使用替代方法
                    print(task.compute_dag)
                except:
                    print("无法打印计算图")
            
            print("\n===== Ansor自动生成的优化调度 =====")
            print(sch.mod.astext(show_meta_data=False))
        except Exception as e:
            print(f"获取最佳调度时出错: {e}")
            print("这可能是因为日志文件格式不正确或为空")
    except Exception as e:
        print(f"打印调度变换时出错: {e}")
        print("在TVM 0.15.0中，Ansor的API可能与代码中使用的不完全兼容")

# 8. 主函数
def main():
    # 设置固定种子以便复现结果
    np.random.seed(0)
    
    print("TVM版本:", tvm.__version__)
    print("CUDA设备信息:")
    try:
        dev = tvm.cuda(0)
        try:
            # 这种方式在新版TVM中可能不适用
            print(dev.compute_capabilities)
        except AttributeError:
            # 使用运行时API获取设备信息
            try:
                print(f"设备名称: {dev.device_name}")
            except AttributeError:
                print("无法获取设备详细信息，但CUDA设备可用")
    except Exception as e:
        print(f"CUDA设备检查出错: {e}")
        print("如果您的GPU可用，请确保TVM已正确配置CUDA")
        return
    
    # 运行单次优化
    M, K, N = 1024, 1024, 1024
    print(f"\n===== 测试 {M}x{K} * {K}x{N} 矩阵乘法 =====")
    
    try:
        # Numpy基准
        _, _, _, numpy_gflops = numpy_matmul(M, K, N)
        
        # 降低搜索次数以加快脚本运行速度
        num_trials = 100  # 减少到100次，可根据需要增加
        print(f"执行{num_trials}次Ansor搜索...")
        
        # Ansor优化
        ansor_gflops = optimize_matmul_ansor(M, K, N, num_trials=num_trials)
        
        # 计算加速比
        speedup = ansor_gflops / numpy_gflops
        print(f"\n性能提升: {speedup:.2f}x")
        
        # 可视化调优进度
        visualize_tuning_progress("ansor_matmul_gpu.json")
        
        try:
            # 打印调度变换
            print_schedule_transformations()
        except Exception as e:
            print(f"打印调度变换时出错: {e}")
            print("跳过调度变换打印...")
        
        # 比较不同大小
        # compare_performance()
    except Exception as e:
        print(f"运行主函数时出错: {e}")
        print("可能需要调整代码以适配您的TVM版本(0.15.0)")
        print("建议: 检查Ansor API是否有变化，或参考TVM 0.15.0的官方示例")

if __name__ == "__main__":
    main()