import tvm
from tvm import te, autotvm, topi
import numpy as np
import logging
import sys

# 设置日志级别
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def main():
    # 1. 定义输入参数
    N, H, W, CO, CI = 1, 224, 224, 512, 512
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation = (1, 1)  # 添加 dilation 参数
    
    # 构建数据shape和参数
    data_shape = (N, CI, H, W)
    kernel_shape = (CO, CI, kernel_h, kernel_w)
    strides = (stride_h, stride_w)
    padding = (pad_h, pad_w)
    
    # 创建 placeholder
    data = te.placeholder(data_shape, name="data")
    kernel = te.placeholder(kernel_shape, name="kernel")
    
    # 创建优化任务
    task = autotvm.task.create(
        "conv2d_nchw.cuda",
        args=(data, kernel, strides, padding, dilation, "float32"),  # 添加 dilation 参数
        target="cuda"
    )
    
    print("Task config space:", task.config_space)
    
    # 2. 设置测量选项
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=5,
            repeat=3,
            timeout=100,
            min_repeat_ms=150
        )
    )
    
    # 3. 创建测量批次
    measure_batch = autotvm.measure.create_measure_batch(task, measure_option)
    
    # 4. 创建调优器
    tuner = autotvm.tuner.XGBTuner(task)
    
    # 5. 开始调优
    n_trial = 1000
    early_stopping = 100
    
    log_file = "conv2d_tuning.log"
    
    tuner.tune(
        n_trial=n_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[
            autotvm.callback.progress_bar(n_trial),
            autotvm.callback.log_to_file(log_file)
        ]
    )
    
    # 6. 应用最佳配置
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            dispatch_context = autotvm.task.DispatchContext.current
            best_config = dispatch_context.query(task.target, task.workload)
            print("\nBest config:")
            print(best_config)
            
            # 创建输入数据
            data_np = np.random.uniform(size=data_shape).astype('float32')
            weight_np = np.random.uniform(size=kernel_shape).astype('float32')
            
            # 转换为TVM张量
            data_tvm = tvm.nd.array(data_np, device=tvm.cuda(0))
            weight_tvm = tvm.nd.array(weight_np, device=tvm.cuda(0))
            
            # 创建输出tensor
            output = tvm.nd.array(
                np.zeros((N, CO, H, W), dtype='float32'), 
                device=tvm.cuda(0)
            )
            
            # 编译并运行
            with tvm.transform.PassContext(opt_level=3):
                conv = topi.cuda.conv2d_nchw(data, kernel, strides, padding, dilation)  # 添加 dilation 参数
                s = topi.cuda.schedule_conv2d_nchw([conv])
                
                # 构建函数
                func = tvm.build(s, [data, kernel, conv], "cuda")
                func(data_tvm, weight_tvm, output)
                
                print("\nOptimization completed!")

if __name__ == "__main__":
    main()