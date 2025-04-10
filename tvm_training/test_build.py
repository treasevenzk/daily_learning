import tvm
from tvm import autotvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import numpy as np

# 定义一个简单的神经网络模型
def get_network():
    data_shape = (1, 3, 224, 224)
    input_data = tvm.relay.var("data", shape=data_shape, dtype="float32")
    weight = tvm.relay.var("weight")
    conv = tvm.relay.nn.conv2d(input_data, weight, strides=(1, 1), padding=(1, 1), channels=64, kernel_size=(3, 3))
    func = tvm.relay.Function([input_data, weight], conv)
    return func

# 创建任务
def create_tuning_task():
    func = get_network()
    tasks = autotvm.task.extract_from_program(
        func,
        target="cuda",
        target_host="llvm",
        params={"weight": np.random.uniform(-1, 1, (64, 3, 3, 3)).astype("float32")},
    )
    return tasks[0]  # 返回第一个任务

# 主函数
def main():
    # 创建配置对象
    class Config:
        def __init__(self):
            self.runner_number = 1
            self.runner_repeat = 3
            self.runner_timeout = 10
            self.build_timeout = 10
    
    config = Config()
    flush = True
    
    # 创建任务
    task = create_tuning_task()
    print(f"任务: {task}")
    
    # 设置运行器和测量选项
    runner = autotvm.measure.LocalRunner(
        number=config.runner_number,
        repeat=config.runner_repeat,
        min_repeat_ms=500,
        timeout=config.runner_timeout,
        enable_cpu_cache_flush=flush
    )
    
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=config.build_timeout),
        runner=runner
    )
    
    builder = measure_option["builder"]
    runner = measure_option["runner"]
    
    # 设置任务
    attach_objects = runner.set_task(task)
    build_kwargs = runner.get_build_kwargs()
    builder.set_task(task, build_kwargs)
    
    # 打印build_kwargs内容
    print("\n======= build_kwargs内容 =======")
    for key, value in build_kwargs.items():
        print(f"{key}: {value}")
    
    # 定义get函数
    def get():
        return build_kwargs
    
    get.n_parallel = builder.n_parallel
    get.attach_objects = attach_objects
    
    return get()

# 运行主函数
if __name__ == "__main__":
    build_kwargs = main()