import tvm
from tvm import te, autotvm, topi
import numpy as np
import logging
import sys
import time

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def time_evaluation(func, dev, data, weight, output, repeat=100):
    func(data, weight, output)

    times = []
    for _ in range(repeat):
        start = time.time()
        func(data, weight, output)
        dev.sync()
        times.append(time.time() - start)
    
    return np.mean(times), np.std(times)

def compute_gflops(batch_size, in_channel, out_channel, height, width, kernel_size, time_cost):
    fma_ops = batch_size * out_channel * in_channel * height * width * kernel_size * kernel_size * 2
    return fma_ops / (time_cost * 1e9)

def main():
    N, H, W, CO, CI = 1, 224, 224, 512, 512
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation = (1, 1)

    data_shape = (N, CI, H, W)
    kernel_shape = (CO, CI, kernel_h, kernel_w)
    strides = (stride_h, stride_w)
    padding = (pad_h, pad_w)

    data_np = np.random.uniform(size=data_shape).astype('float32')
    weight_np = np.random.uniform(size=kernel_shape).astype('float32')

    data = te.placeholder(data_shape, name="data")
    kernel = te.placeholder(kernel_shape, name="kernel")

    print("\n=== 评估未调优版本 ===")
    with tvm.target.Target("cuda"):
        conv = topi.cuda.conv2d_nchw(data, kernel, strides, padding, dilation)
        s = topi.cuda.schedule_conv2d_nchw([conv])
        func = tvm.build(s, [data, kernel, conv], "cuda")

        dev = tvm.cuda(0)
        data_tvm = tvm.nd.array(data_np, device=dev)
        weight_tvm = tvm.nd.array(weight_np, device=dev)
        output_shape = (N, CO, H, W)
        output_tvm = tvm.nd.array(np.zeros(output_shape).astype('float32'), device=dev)

        time_mean, time_std = time_evaluation(func, dev, data_tvm, weight_tvm, output_tvm)
        gflops = compute_gflops(N, CI, CO, H, W, kernel_h, time_mean)
        print(f"未调优版本平均时间： {time_mean*1000:.3f} ms (标准差: {time_std*1000:.3f} ms)")
        print(f"未调优版本性能: {gflops:.2f} GFLOPS")



    print("\n=== 开始调优过程 ===")
    task = autotvm.task.create(
        "conv2d_nchw.cuda",
        args=(data, kernel, strides, padding, dilation, "float32"),
        target="cuda"
    )

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=5,
            repeat=3,
            timeout=100,
            min_repeat_ms=150
        )
    )

    tuner = autotvm.tuner.XGBTuner(task)
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


    print("\n=== 评估调优后版本 ===")
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            conv = topi.cuda.conv2d_nchw(data, kernel, strides, padding, dilation)
            s = topi.cuda.schedule_conv2d_nchw([conv])
            func = tvm.build(s, [data, kernel, conv], "cuda")

            dev = tvm.cuda(0)
            data_tvm = tvm.nd.array(data_np, device=dev)
            weight_tvm = tvm.nd.array(weight_np, device=dev)
            output_tvm = tvm.nd.array(np.zeros(output_shape).astype('float32'), device=dev)

            time_mean, time_stad = time_evaluation(func, dev, data_tvm, weight_tvm, output_tvm)
            gflops = compute_gflops(N, CI, CO, H, W, kernel_h, time_mean)
            print(f"调优后版本平均运行时间: {time_mean*1000:.3f} ms (标准差: {time_std*1000:.3f} ms)")
            print(f"调优后版本性能: {gflops:.2f} GFLOPS")

if __name__ == "__main__":
    main()