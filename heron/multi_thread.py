import threading
import queue
import time
import signal
import sys

thread_registry = {}

class ThreadJob:
    def __init__(self, func, attach_info, timeout=10):
        self.func = func
        self.attach_info = attach_info
        self.timeout = timeout
        self.thread_id = None

    def start(self, inputs):
        self.queue = queue.Queue(maxsize=2)
        self.thread = threading.Thread(
            target=thread_exec,
            args=(self.func, self.queue, inputs),
            daemon=True
        )
        self.thread_id = id(self.thread)
        thread_registry[self.thread_id] = self.thread
        self.thread.start()
    
    def get(self):
        try:
            res = self.queue.get(block=True, timeout=self.timeout)
        except queue.Empty:
            print(f"作业执行超时 ({self.timeout})")

        if self.thread.is_alive():
            print("等待线程完成...")
            self.thread.join(0.1)

        if self.thread_id in thread_registry:
            del thread_registry[self.thread_id]

        del self.thread
        return res


def thread_exec(func, result_queue, args):
    try:
        res = func(*args)
        result_queue.put(res)
    except Exception as e:
        print(f"线程执行失败: {e}")
        result_queue.put(None)


def cancel_all_threads():
    print(f"尝试清理 {len(thread_registry)} 个线程...")
    for thread_id, thread in list(thread_registry.item()):
        if thread.is_alive():
            print(f"等待线程 {thread_id} 结束...")
            thread.join(0.1)
        del thread_registry[thread_id]
    print("线程清理完成")


if __name__ == "__main__":
    
    def signal_handler(sig, frame):
        print("收到终止信号， 正在清理....")
        cancel_all_threads()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    def test_function(a, b, delay=2):
        print(f"线程开始计算 {a} + {b}, 延迟 {delay} 秒")
        time.sleep(delay)
        result = a + b
        print(f"线程计算完成: {result}")
        return result
    
    jobs = []
    for i in range(5):
        job = ThreadJob(test_function, f"作业 {i}", timeout=5)
        job.start((i, i*10, i))
        jobs.append(job)

    for i, job in enumerate(jobs):
        print(f"获取作业 {i} 的结果...")
        result = job.get()
        print(f"作业 {i} 结果: {result}")

    print("所有作业完成")