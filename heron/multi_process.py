import multiprocessing as mp
import os
import time
import signal
import psutil
import sys

multi = mp.get_context("fork")

class Job:
    def __init__(self, func, attach_info, timeout=10):
        self.func = func
        self.attach_info = attach_info
        self.timeout = timeout
        self.process = None
        self.queue = None

    
    def start(self, inputs):
        self.queue = multi.Queue(2)

        self.process = multi.Process(
            target=exec_in_process,
            args=(self.func, self.queue, inputs)
        )

        self.process.start()

        print(f"进程已启动 [PID: {self.process.pid}]")

    
    def get(self):
        result = None
        try:
            result = self.queue.get(block=True, timeout=self.timeout)
            print(f"成功获取到结果")
        except Exception as e:
            print(f"获取结果失败: {e}")
            result = None

        if self.process and self.process.is_alive():
            print(f"终止进程 [PID: {self.process.pid}]")
            kill_process_tree(self.process.pid)
            self.process.terminate()

        if self.process:
            self.process.join()
        
        if self.queue:
            self.queue.close()
            self.queue.join_thread()

        tmp_process = self.process
        tmp_queue = self.queue
        self.process = None
        self.queue = None

        del tmp_process
        del tmp_queue

        return result
    
def exec_in_process(func, queue, args):
    try:
        os.getpgrp()

        def signal_handler(signum, frame):
            print(f"进程 {os.getpid()} 收到信号 {signum}， 正在退出...")
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        print(f"进程 {os.getpid()} 开始执行...")
        result = func(*args)

        queue.put(result)
        print(f"进程 {os.getpid()} 执行完成，结果已放入队列")

    except Exception as e:
        error_msg = f"进程执行失败: {str(e)}"
        print(error_msg)
        try:
            queue.put(None)
        except:
            pass

def kill_process_tree(pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                print(f"终止子进程 [PID: {child.pid}]")
                child.send_signal(sig)
            except psutil.NoSuchProcess:
                pass

        try:
            print(f"终止父进程 [PID: {pid}]")
            parent.send_signal(sig)
        except psutil.NoSuchProcess:
            pass

    except psutil.NoSuchProcess:
        print(f"进程 {pid} 不存在")
    except Exception as e:
        print(f"终止进程树时出错: {e}")



if __name__ == "__main__":
    def test_function(a, b, sleep_time=2):
        process_id = os.getpid()
        print(f"进程 {process_id} 开始计算 {a} + {b}, 休眠 {sleep_time} 秒")

        time.sleep(sleep_time)

        result = a + b
        print(f"进程 {process_id} 计算完成： {a} + {b} = {result}")
        return result
    
    jobs = []
    for i in range(3):
        job = Job(test_function, f"作业 {i}", timeout=5)

        job.start((i, i*10, i+1))

        jobs.append(job)


    results = []
    for i, job in enumerate(jobs):
        print(f"\n获取作业 {i} 结果...")
        result = job.get()
        results.append(result)
        print(f"作业 {i} 结果: {result}")

    print("\n所有作业完成")
    print(f"结果列表: {results}")

    print("\n测试超时情况...")
    timeout_job = Job(test_function, "超时作业", timeout=2)
    timeout_job.start((100, 200, 5))
    result = timeout_job.get()
    print(f"超时作业结果: {result}")