import openai
import os
import time
import tiktoken
import json
from datetime import datetime



class RateLimiter:
    def __init__(self, max_tokens=100000, time_frame=60, max_requests=720):
        self.max_tokens = max_tokens
        self.time_frame = time_frame
        self.max_requests = max_requests
        self.timestamps = []
        self.tokens_used = []
        self.request_count = 0

    def add_request(self, request_text=None, request_token_count=None, current_time=None):
        if current_time is None:
            current_time = time.time()

        # 移除过期的请求记录
        while self.timestamps and self.timestamps[0] < current_time - self.time_frame:
            self.timestamps.pop(0)
            self.tokens_used.pop(0)
            self.request_count -= 1
        
        self.timestamps.append(current_time)

        # 计算token数量
        if request_text is not None:
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            num_tokens = len(encoding.encode(request_text))
        elif request_token_count is not None:
            num_tokens = request_token_count
        else:
            raise ValueError('Either request_text or request_token_count must be specified.')

        self.tokens_used.append(num_tokens)
        self.request_count += 1

        # 检查请求次数限制
        if self.request_count >= self.max_requests:
            sleep_time = (self.timestamps[0] + self.time_frame) - current_time
            print(f'[Rate Limiter] Sleeping for {sleep_time:.2f}s to avoid hitting the request limit...')
            time.sleep(sleep_time)
            self.request_count = 0
        
        # 检查token数量限制
        if sum(self.tokens_used) > self.max_tokens:
            sleep_time = (self.timestamps[0] + self.time_frame) - current_time
            print(f"[Rate Limiter] Sleeping for {sleep_time:.2f}s to avoid hitting the token limit...")
            time.sleep(sleep_time)
            self.timestamps.clear()
            self.tokens_used.clear()


class ConversationManager:
    """管理会话历史，确保不超出上下文限制"""
    def __init__(self, max_tokens=4000, system_message="", prefix=""):
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.prefix = prefix
        self.current_messages = [{"role": "system", "content": system_message}]
        self.all_examples = []
        self.prefix_sent = False
        self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        
        # 存储所有配置的历史记录，用于分析和优化
        self.config_history = []
        
        # 创建一个用于本地持久化的目录
        os.makedirs('conversation_history', exist_ok=True)
    
    def count_tokens(self, message):
        """计算消息的token数量"""
        return len(self.encoding.encode(message["content"]))
    
    def get_total_tokens(self):
        """计算当前会话的总token数量"""
        return sum(self.count_tokens(msg) for msg in self.current_messages)
    
    def add_message(self, role, content):
        """添加消息到会话，并检查是否需要压缩会话"""
        message = {"role": role, "content": content}
        self.current_messages.append(message)
        
        # 检查是否需要压缩会话
        if self.get_total_tokens() > self.max_tokens * 0.9:  # 当使用90%的token限制时进行压缩
            self.compress_conversation()
            
        return message
    
    def compress_conversation(self):
        """压缩会话以减少token使用"""
        print("[ConversationManager] Compressing conversation to reduce token usage...")
        
        # 保存当前完整会话到文件，用于记录
        self.save_conversation_snapshot()
        
        # 必须保留的消息：系统消息和前缀消息（如果已发送）
        essential_messages = [self.current_messages[0]]  # 系统消息
        
        if self.prefix_sent:
            # 查找前缀消息
            for i, msg in enumerate(self.current_messages):
                if msg["role"] == "user" and msg["content"].startswith(self.prefix[:50]):
                    essential_messages.append(msg)
                    break
        
        # 创建一个摘要消息，总结当前的示例和性能数据
        summary_content = "Previous conversation included examples with the following performance metrics:\n"
        
        # 按性能指标排序的示例
        sorted_examples = sorted(self.all_examples, key=lambda x: float(x['A']), reverse=True)
        
        # 添加顶部5个和底部5个示例摘要
        top_examples = sorted_examples[:5]
        bottom_examples = sorted_examples[-5:] if len(sorted_examples) > 5 else []
        
        summary_content += "Top performing configurations:\n"
        for ex in top_examples:
            summary_content += f"- Configuration with perf = {ex['A']}\n"
            
        if bottom_examples:
            summary_content += "\nLow performing configurations:\n"
            for ex in bottom_examples:
                summary_content += f"- Configuration with perf = {ex['A']}\n"
        
        summary_content += f"\nTotal {len(self.all_examples)} examples have been provided so far.\n"
        summary_content += "The most recent examples and their performance are kept in the conversation."
        
        summary_message = {"role": "assistant", "content": summary_content}
        
        # 获取最近的一些例子（最后10个）
        recent_examples_messages = []
        for i in range(len(self.current_messages)-1, 0, -1):
            msg = self.current_messages[i]
            if msg["role"] == "user" and "Performance:" in msg["content"] and len(recent_examples_messages) < 10:
                recent_examples_messages.append(msg)
                # 如果这是例子，也包括助手的回复
                if i + 1 < len(self.current_messages) and self.current_messages[i+1]["role"] == "assistant":
                    recent_examples_messages.append(self.current_messages[i+1])
        
        # 倒序，使其保持原来的顺序
        recent_examples_messages.reverse()
        
        # 重建当前消息列表
        self.current_messages = essential_messages + [summary_message] + recent_examples_messages
        print(f"[ConversationManager] Conversation compressed. New token count: {self.get_total_tokens()}")
    
    def save_conversation_snapshot(self):
        """将当前会话保存到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_history/conversation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "messages": self.current_messages,
                "all_examples": self.all_examples,
                "config_history": self.config_history
            }, f, indent=2)
        
        print(f"[ConversationManager] Conversation snapshot saved to {filename}")
    
    def load_conversation_snapshot(self, filename):
        """从文件加载会话"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.current_messages = data.get("messages", [])
                self.all_examples = data.get("all_examples", [])
                self.config_history = data.get("config_history", [])
                
                # 设置前缀已发送的标志
                for msg in self.current_messages:
                    if msg["role"] == "user" and msg["content"].startswith(self.prefix[:50]):
                        self.prefix_sent = True
                        break
                
                print(f"[ConversationManager] Loaded conversation from {filename}")
                print(f"[ConversationManager] Token count: {self.get_total_tokens()}")
                print(f"[ConversationManager] Examples count: {len(self.all_examples)}")
                
                return True
        except Exception as e:
            print(f"[ConversationManager] Error loading conversation: {e}")
            return False


class TvmScheduleOptimizer:
    def __init__(self, system_message="You are an AI assistant that helps optimize TVM schedules.", 
                 max_tokens=4000, 
                 persistence_path=None):
        self.rate_limiter = RateLimiter(max_tokens=100000, time_frame=60, max_requests=720)
        self.system_message = system_message
        self.prefix = self._create_prefix()
        
        # 创建会话管理器
        self.conversation = ConversationManager(
            max_tokens=max_tokens,
            system_message=system_message,
            prefix=self.prefix
        )
        
        # 如果提供了持久化路径，尝试加载之前的会话
        if persistence_path and os.path.exists(persistence_path):
            self.conversation.load_conversation_snapshot(persistence_path)
        else:
            # 初始化会话
            self.conversation.current_messages = [{"role": "system", "content": system_message}]
    
    def _create_prefix(self):
        """创建前缀说明部分"""
        prefix = f"The following are examples of TVM schedule parameter configurations and their corresponding performance metrics (perf) on an NVIDIA RTX3080 GPU. Higher perf values are better."
        prefix += f"\n\nHardware constraints for NVIDIA RTX3080:"
        prefix += f"\n- max_shared_memory_per_block: 49152"
        prefix += f"\n- max_threads_per_block: 1024"
        prefix += f"\n- max_thread_x: 1024"
        prefix += f"\n- max_thread_y: 1024"
        prefix += f"\n- max_thread_z: 64"

        prefix += f"\n\nThe allowable ranges for the schedule parameters are:"
        prefix += f"\n- densei.innertileSpatial: [1, 64] (int)"
        prefix += f"\n- densej.innertileSpatial: [1, 64] (int)"
        prefix += f"\n- densei.inner.innertileSpatial: [1, 64] (int)"
        prefix += f"\n- densej.inner.innertileSpatial: [1, 64] (int)"
        prefix += f"\n- densei.inner.inner.innertileSpatial: [1, 64] (int)"
        prefix += f"\n- densej.inner.inner.innertileSpatial: [1, 64] (int)"
        prefix += f"\n- densei.inner.inner.inner.innertileSpatial: [1, 64] (int)"
        prefix += f"\n- densej.inner.inner.inner.innertileSpatial: [1, 64] (int)"
        prefix += f"\n- dense_shared_pos: [1, 5] (int)"
        prefix += f"\n- dense.wmma.accumulator.shared_ax1: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulator.shared_offset: [0, 48] (int)"
        prefix += f"\n- dense.wmma.accumulator.sharedax0tileSpatial: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulator.sharedax1tileSpatial: [1, 64] (int)"
        prefix += f"\n- wmma_m: [8, 32] (int)"
        prefix += f"\n- wmma_k: [16, 16] (int)" 
        prefix += f"\n- wmma_n: [8, 32] (int)"
        prefix += f"\n- dense.wmma.accumulator.shared_local_pos: [0, 1] (int)"
        prefix += f"\n- dense.wmma.accumulatori.ctileAll: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulatorj.ctileAll: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulatorktileAll: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulatori.c.innertileAll: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulatorj.c.innertileAll: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulatork.innertileAll: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulatori.c.inner.innertileAll: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulatorj.c.inner.innertileAll: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulatork.inner.innertileAll: [1, 64] (int)"
        prefix += f"\n- dense.wmma.accumulator_local_pos: [0, 3] (int)"
        prefix += f"\n- dense.wmma.accumulator_shared_pos: [0, 3] (int)"
        prefix += f"\n- B.shared_ax1: [1, 64] (int)"
        prefix += f"\n- B.shared_offset: [0, 48] (int)"
        prefix += f"\n- B.shared_vectorize: [1, 8] (int)"
        prefix += f"\n- threadIdx.x: [32, 32] (int)"
        prefix += f"\n- threadIdx.y: [1, 1024] (int)"
        prefix += f"\n- A.shared_ax1: [1, 64] (int)"
        prefix += f"\n- A.shared_offset: [0, 48] (int)"
        prefix += f"\n- A.shared_vectorize: [1, 8] (int)"
        prefix += f"\n- dense_unroll_pragma: [0, 5] (int)"

        prefix += f"\n\nBelow is a reference TVM schedule template showing how parameters are used:"
        prefix += f"""

        ## Cache Tensor Core
        dense_wmma_accumulator = s.cache_write(dense, wmma.accumulator)

        ## Cache read shared
        A_shared = s.cache_read(A, shared, dense.wmma.accumulator)
        B_shared = s.cache_read(B, shared, dense.wmma.accumulator)
        A_shared_wmma_matrix_a = s.cache_read(A_shared, wmma.matrix_a, dense.wmma.accumulator)
        B_shared_wmma_matrix_b = s.cache_read(B_shared, wmma.matrix_b, dense.wmma.accumulator)

        ## Cache read shared
        dense_wmma_accumulator_shared = s.cache_read(dense_wmma_accumulator, shared, dense)

        #==--------- Start schedule STAGE dense ----------==#

        ## Unroll pragma 
        i_o, i_i = s[dense].split(i, nparts = 1)
        j_o, j_i = s[dense].split(j, nparts = 1)
        s[dense].reorder(i_o, j_o, i_i, j_i, )

        ## Bind blockIdx.x

        ## tile spatial 
        i_i_o, i_i_i = s[dense].split(i_i, nparts = 1)
        j_i_o, j_i_i = s[dense].split(j_i, nparts = 1)
        s[dense].reorder(i_i_o, j_i_o, i_i_i, j_i_i, )
        i_i_o_j_i_o_f = s[dense].fuse(i_i_o, j_i_o, )
        s[dense].bind(i_i_o_j_i_o_f, te.thread_axis("blockIdx.x"))

        ## Bind threadIdx.y

        ## tile spatial 
        i_i_i_o, i_i_i_i = s[dense].split(i_i_i, nparts = 1)
        j_i_i_o, j_i_i_i = s[dense].split(j_i_i, nparts = 1)
        s[dense].reorder(i_i_i_o, j_i_i_o, i_i_i_i, j_i_i_i, )
        i_i_i_o_j_i_i_o_f = s[dense].fuse(i_i_i_o, j_i_i_o, )
        s[dense].bind(i_i_i_o_j_i_i_o_f, te.thread_axis("threadIdx.y"))

        ## Bind threadIdx.x

        ## tile spatial 
        i_i_i_i_o, i_i_i_i_i = s[dense].split(i_i_i_i, nparts = 1)
        j_i_i_i_o, j_i_i_i_i = s[dense].split(j_i_i_i, nparts = 1)
        s[dense].reorder(i_i_i_i_o, j_i_i_i_o, i_i_i_i_i, j_i_i_i_i, )
        i_i_i_i_o_j_i_i_i_o_f = s[dense].fuse(i_i_i_i_o, j_i_i_i_o, )
        s[dense].bind(i_i_i_i_o_j_i_i_i_o_f, te.thread_axis("threadIdx.x"))

        ## Vectorize 

        ## tile spatial 
        i_i_i_i_i_o, i_i_i_i_i_i = s[dense].split(i_i_i_i_i, nparts = 1)
        j_i_i_i_i_o, j_i_i_i_i_i = s[dense].split(j_i_i_i_i, nparts = 1)
        s[dense].reorder(i_i_i_i_i_o, j_i_i_i_i_o, i_i_i_i_i_i, j_i_i_i_i_i, )
        i_i_i_i_i_i_j_i_i_i_i_i_f = s[dense].fuse(i_i_i_i_i_i, j_i_i_i_i_i, )
        s[dense].vectorize(i_i_i_i_i_i_j_i_i_i_i_i_f)

        # Var i_o length 1
        # Var j_o length 1
        # Var i_i_o_j_i_o_f length 1
        # Var i_i_i_o_j_i_i_o_f length 1
        # Var i_i_i_i_o_j_i_i_i_o_f length 1
        # Var i_i_i_i_i_o length 1
        # Var j_i_i_i_i_o length 1
        # Var i_i_i_i_i_i_j_i_i_i_i_i_f length 1
        #==--------- Start schedule STAGE dense.wmma.accumulator.shared ----------==#
        s[dense_wmma_accumulator_shared].compute_at(s[dense], j_o)

        # Var ax0 length 1
        # Var ax1 length 1
        ## Storage align 
        s[dense_wmma_accumulator_shared].storage_align(ax0, 0.000000, 1.000000)

        ## Bind threadIdx.y

        ## tile spatial 
        ax0_o, ax0_i = s[dense_wmma_accumulator_shared].split(ax0, nparts = 1)
        ax1_o, ax1_i = s[dense_wmma_accumulator_shared].split(ax1, nparts = 1)
        s[dense_wmma_accumulator_shared].reorder(ax0_o, ax1_o, ax0_i, ax1_i, )
        ax0_o_ax1_o_f = s[dense_wmma_accumulator_shared].fuse(ax0_o, ax1_o, )
        s[dense_wmma_accumulator_shared].bind(ax0_o_ax1_o_f, te.thread_axis("threadIdx.y"))

        ## Tensor core store
        ax0_i_o, ax0_i_i = s[dense_wmma_accumulator_shared].split(ax0_i, factor = 16)
        ax1_i_o, ax1_i_i = s[dense_wmma_accumulator_shared].split(ax1_i, factor = 16)
        s[dense_wmma_accumulator_shared].reorder(ax0_i_o, ax1_i_o, ax0_i_i, ax1_i_i, )
        s[dense_wmma_accumulator_shared].tensorize(ax0_i_i, intrin_wmma_store_matrix(
        [sc_n0, 1], [lc_n0, 1], (16, 16, 16), float16, [16, 16], [16, 16], 
        ))

        # Var ax0_o_ax1_o_f length 1
        # Var ax0_i_o length 1
        # Var ax1_i_o length 1
        # Var ax0_i_i length 1
        # Var ax1_i_i length 1
        #==--------- Start schedule STAGE dense.wmma.accumulator ----------==#
        s[dense_wmma_accumulator].compute_at(s[dense_wmma_accumulator_shared], ax0_o_ax1_o_f)

        # Var i_c length 1
        # Var j_c length 1
        # Var k
        ## general tile 

        ## tile 
        i_c_o, i_c_i = s[dense_wmma_accumulator].split(i_c, nparts = 1)
        j_c_o, j_c_i = s[dense_wmma_accumulator].split(j_c, nparts = 1)
        k_o, k_i = s[dense_wmma_accumulator].split(k, nparts = 1)
        s[dense_wmma_accumulator].reorder(i_c_o, j_c_o, k_o, i_c_i, j_c_i, k_i, )

        ## tile 
        i_c_i_o, i_c_i_i = s[dense_wmma_accumulator].split(i_c_i, nparts = 1)
        j_c_i_o, j_c_i_i = s[dense_wmma_accumulator].split(j_c_i, nparts = 1)
        k_i_o, k_i_i = s[dense_wmma_accumulator].split(k_i, nparts = 1)
        s[dense_wmma_accumulator].reorder(i_c_i_o, j_c_i_o, k_i_o, i_c_i_i, j_c_i_i, k_i_i, )

        ## tile 
        i_c_i_i_o, i_c_i_i_i = s[dense_wmma_accumulator].split(i_c_i_i, nparts = 1)
        j_c_i_i_o, j_c_i_i_i = s[dense_wmma_accumulator].split(j_c_i_i, nparts = 1)
        k_i_i_o, k_i_i_i = s[dense_wmma_accumulator].split(k_i_i, nparts = 1)
        s[dense_wmma_accumulator].reorder(i_c_i_i_o, j_c_i_i_o, k_i_i_o, i_c_i_i_i, j_c_i_i_i, k_i_i_i, )

        ## Tensor core compute
        i_c_i_i_i_o, i_c_i_i_i_i = s[dense_wmma_accumulator].split(i_c_i_i_i, factor = 16)
        j_c_i_i_i_o, j_c_i_i_i_i = s[dense_wmma_accumulator].split(j_c_i_i_i, factor = 16)
        k_i_i_i_o, k_i_i_i_i = s[dense_wmma_accumulator].split(k_i_i_i, factor = 16)
        s[dense_wmma_accumulator].reorder(i_c_i_i_i_o, j_c_i_i_i_o, k_i_i_i_o, i_c_i_i_i_i, j_c_i_i_i_i, k_i_i_i_i, )
        s[dense_wmma_accumulator].tensorize(i_c_i_i_i_i, intrin_wmma_gemm(
        Tensor(shape=[16, 16], op.name=wmma_A), Tensor(shape=[16, 16], op.name=wmma_B), Tensor(shape=[16, 16], op.name=wmma_C), [la_k0, 1], [lb_k0, 1], [lc_n0, 1], (16, 16, 16), 
        ))

        # Var i_c_o length 1
        # Var j_c_o length 1
        # Var k_o length 1
        # Var i_c_i_o length 1
        # Var j_c_i_o length 1
        # Var k_i_o length 1
        # Var i_c_i_i_o length 1
        # Var j_c_i_i_o length 1
        # Var k_i_i_o length 1
        # Var i_c_i_i_i_o length 1
        # Var j_c_i_i_i_o length 1
        # Var k_i_i_i_o length 1
        # Var i_c_i_i_i_i length 1
        # Var j_c_i_i_i_i length 1
        # Var k_i_i_i_i length 1
        #==--------- Start schedule STAGE B.shared.wmma.matrix_b ----------==#
        s[B_shared_wmma_matrix_b].compute_at(s[dense_wmma_accumulator], k_o)

        # Var ax0
        # Var ax1
        ## Tensor core loadB
        ax0_o, ax0_i = s[B_shared_wmma_matrix_b].split(ax0, factor = 16)
        ax1_o, ax1_i = s[B_shared_wmma_matrix_b].split(ax1, factor = 16)
        s[B_shared_wmma_matrix_b].reorder(ax0_o, ax1_o, ax0_i, ax1_i, )
        s[B_shared_wmma_matrix_b].tensorize(ax0_i, intrin_wmma_load_matrix_W(
        [lb_k0, 1], [sb_k0, 1], (16, 16, 16), col_major, [16, 16], [16, 16], float16, 
        ))

        # Var ax0_o length 1
        # Var ax1_o length 1
        # Var ax0_i length 1
        # Var ax1_i length 1
        #==--------- Start schedule STAGE B.shared ----------==#
        s[B_shared].compute_at(s[dense_wmma_accumulator], k_o)

        # Var ax0 length 1
        # Var ax1 length 1
        ## Storage align 
        s[B_shared].storage_align(ax0, 0.000000, 1.000000)
        ax0_ax1_f = s[B_shared].fuse(ax0, ax1, )
        ax0_ax1_f_o, ax0_ax1_f_i = s[B_shared].split(ax0_ax1_f, factor = 1)
        s[B_shared].vectorize(ax0_ax1_f_i)
        ax0_ax1_f_o_o, ax0_ax1_f_o_i = s[B_shared].split(ax0_ax1_f_o, factor = 1)
        s[B_shared].bind(ax0_ax1_f_o_i, te.thread_axis("threadIdx.x"))
        ax0_ax1_f_o_o_o, ax0_ax1_f_o_o_i = s[B_shared].split(ax0_ax1_f_o_o, factor = 1)
        s[B_shared].bind(ax0_ax1_f_o_o_i, te.thread_axis("threadIdx.y"))

        #==--------- Start schedule STAGE B ----------==#

        #==--------- Start schedule STAGE A.shared.wmma.matrix_a ----------==#
        s[A_shared_wmma_matrix_a].compute_at(s[dense_wmma_accumulator], k_o)

        # Var ax0 length 1
        # Var ax1 length 1
        ## Tensor core loadA
        ax0_o, ax0_i = s[A_shared_wmma_matrix_a].split(ax0, factor = 16)
        ax1_o, ax1_i = s[A_shared_wmma_matrix_a].split(ax1, factor = 16)
        s[A_shared_wmma_matrix_a].reorder(ax0_o, ax1_o, ax0_i, ax1_i, )
        s[A_shared_wmma_matrix_a].tensorize(ax0_i, intrin_wmma_load_matrix_A(
        [la_k0, 1], [sa_k0, 1], (16, 16, 16), row_major, [16, 16], [16, 16], float16, 
        ))

        # Var ax0_o length 1
        # Var ax1_o length 1
        # Var ax0_i length 1
        # Var ax1_i length 1
        #==--------- Start schedule STAGE A.shared ----------==#
        s[A_shared].compute_at(s[dense_wmma_accumulator], k_o)

        # Var ax0 length 1
        # Var ax1 length 1
        ## Storage align 
        s[A_shared].storage_align(ax0, 0.000000, 1.000000)
        ax0_ax1_f = s[A_shared].fuse(ax0, ax1, )
        ax0_ax1_f_o, ax0_ax1_f_i = s[A_shared].split(ax0_ax1_f, factor = 1)
        s[A_shared].vectorize(ax0_ax1_f_i)
        ax0_ax1_f_o_o, ax0_ax1_f_o_i = s[A_shared].split(ax0_ax1_f_o, factor = 1)
        s[A_shared].bind(ax0_ax1_f_o_i, te.thread_axis("threadIdx.x"))
        ax0_ax1_f_o_o_o, ax0_ax1_f_o_o_i = s[A_shared].split(ax0_ax1_f_o_o, factor = 1)
        s[A_shared].bind(ax0_ax1_f_o_o_i, te.thread_axis("threadIdx.y"))

        #==--------- Start schedule STAGE A ----------==#
        """

        prefix += f"\n\nBased on the above examples, schedule template, and hardware constraints, recommend a new TVM schedule parameter configuration that will MAXIMIZE performance (perf) on the NVIDIA RTX3080 GPU. The best configurations seen so far have perf values above 12.7. Consider the following optimization principles:"
        prefix += f"\n1. Optimize thread and block dimensions (threadIdx.x, threadIdx.y) to maximize GPU utilization without exceeding limits"
        prefix += f"\n2. Balance tiling sizes to optimize memory access patterns and cache utilization"
        prefix += f"\n3. Choose appropriate vectorization parameters for coalesced memory access"
        prefix += f"\n4. Select tensor core parameters (wmma_m, wmma_n, wmma_k) that align with the NVIDIA RTX3080's tensor core architecture"
        prefix += f"\n5. Consider memory alignment and offset parameters to reduce bank conflicts"
        prefix += f"\n6. Fine-tune spatial tiling parameters to optimize data locality"
        prefix += f"\n7. Use insights from the most successful examples (perf > 12.6) while avoiding configurations that resulted in 0.0 performance"

        prefix += f"\n\nYour response must only contain the predicted configuration, in the format ## configuration ##."
        prefix += f"\n\nCRITICAL: Your response must EXACTLY follow this format: begin with '## configuration ##' (all lowercase), then list all parameters with values, and end with ' ##'. Do not include any explanations or additional text. Only output the configuration in this exact format."
        
        return prefix
    
    def send_prefix(self):
        """发送前缀说明部分给模型"""
        if not self.conversation.prefix_sent:
            self.conversation.add_message("user", self.prefix)
            self.conversation.prefix_sent = True
    
    def add_examples(self, examples):
        """添加示例到会话中"""
        # 添加到所有示例列表
        self.conversation.all_examples.extend(examples)
        
        # 将示例格式化为文本
        examples_text = "\n\n".join([f"Performance: {example['A']}\nHyperparameter configuration: {example['Q']}" for example in examples])
        
        # 如果是第一批示例，先发送前缀
        if not self.conversation.prefix_sent:
            self.send_prefix()
            message = examples_text
        else:
            message = "Here are additional examples to consider:\n\n" + examples_text
        
        # 添加示例到会话
        self.conversation.add_message("user", message)
    
    def add_example_with_performance(self, config, performance):
        """添加单个示例及其性能指标"""
        example = {
            'Q': config if isinstance(config, str) else '## ' + ', '.join([f"{k}: {v}" for k, v in config.items()]) + ' ##',
            'A': str(performance)
        }
        
        # 记录到配置历史
        self.conversation.config_history.append({
            'config': config,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })
        
        # 添加到示例列表
        self.add_examples([example])
        
        # 保存新的快照
        if len(self.conversation.config_history) % 10 == 0:  # 每10个新配置保存一次
            self.conversation.save_conversation_snapshot()
    
    def get_configuration(self, query="maximize"):
        """获取优化配置"""
        # 添加查询
        query_text = f"\n\nRecommend a configuration that would likely achieve the highest possible performance:"
        self.conversation.add_message("user", query_text)
        
        # 生成响应
        MAX_RETRIES = 3
        for retry in range(MAX_RETRIES):
            try:
                start_time = time.time()
                # 创建消息列表
                messages = self.conversation.current_messages.copy()
                
                # 添加请求到速率限制器
                full_conversation = "\n".join([msg["content"] for msg in messages])
                self.rate_limiter.add_request(request_text=full_conversation, current_time=start_time)
                
                # 调用OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.8,
                    max_tokens=1000,  # 配置不需要太多token
                    top_p=0.95,
                    n=1
                )
                
                # 记录响应token数量
                self.rate_limiter.add_request(request_token_count=response['usage']['total_tokens'], current_time=time.time())
                
                # 提取响应并添加到会话历史
                message_content = response['choices'][0]['message']['content']
                self.conversation.add_message("assistant", message_content)
                
                # 计算成本
                tot_tokens = response['usage']['total_tokens']
                tot_cost = 0.0015*(response['usage']['prompt_tokens']/1000) + 0.002*(response['usage']['completion_tokens']/1000)
                
                return message_content, tot_cost, tot_tokens
            
            except Exception as e:
                print(f"[AF] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...")
                print(e)
                if retry == MAX_RETRIES - 1:
                    return None, 0, 0
    
    def extract_configuration(self, response):
        """从响应中提取配置"""
        # 查找配置的开始和结束位置
        start_markers = ["## Configuration ##\n", "## Configuration ##", "## configuration ##\n", "## configuration ##"]
        end_marker = " ##"
        
        try:
            # 查找开始位置
            start_pos = -1
            for marker in start_markers:
                pos = response.find(marker)
                if pos != -1:
                    start_pos = pos + len(marker)
                    break
            
            if start_pos == -1:
                raise ValueError("找不到配置开始标记")
            
            # 查找结束位置
            end_pos = response.find(end_marker, start_pos)
            if end_pos == -1:
                # 如果找不到结束标记，获取剩余文本
                config_text = response[start_pos:].strip()
            else:
                config_text = response[start_pos:end_pos].strip()
            
            # 分割参数对
            pairs = config_text.split(", ")
            config_dict = {}
            
            for pair in pairs:
                # 按冒号分割键值对
                if ": " in pair:
                    key, value = pair.split(": ")
                    try:
                        config_dict[key] = int(value)
                    except ValueError:
                        try:
                            config_dict[key] = float(value)
                        except ValueError:
                            config_dict[key] = value
            
            return config_dict
        except Exception as e:
            print(f"提取配置时出错: {e}")
            return None
    
    def get_multiple_configurations(self, n=5):
        """获取多个优化配置"""
        # 添加查询
        query_text = f"\n\nPlease provide {n} different configuration recommendations that would likely achieve high performance. Make each configuration as diverse as possible to explore different optimization strategies. Format each configuration exactly like previous examples."
        self.conversation.add_message("user", query_text)
        
        # 生成响应
        MAX_RETRIES = 3
        for retry in range(MAX_RETRIES):
            try:
                start_time = time.time()
                messages = self.conversation.current_messages.copy()
                full_conversation = "\n".join([msg["content"] for msg in messages])
                self.rate_limiter.add_request(request_text=full_conversation, current_time=start_time)
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.9,  # 增加温度以获得更多样化的结果
                    max_tokens=2000,  # 多个配置需要更多token
                    top_p=0.95,
                    n=1  # 在一个响应中生成多个配置
                )
                
                self.rate_limiter.add_request(request_token_count=response['usage']['total_tokens'], current_time=time.time())
                
                message_content = response['choices'][0]['message']['content']
                self.conversation.add_message("assistant", message_content)
                
                tot_tokens = response['usage']['total_tokens']
                tot_cost = 0.0015*(response['usage']['prompt_tokens']/1000) + 0.002*(response['usage']['completion_tokens']/1000)
                
                # 分割多个配置
                configs = self._split_multiple_configs(message_content)
                return configs, tot_cost, tot_tokens
            
            except Exception as e:
                print(f"[AF] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...")
                print(e)
                if retry == MAX_RETRIES - 1:
                    return None, 0, 0
    
    def _split_multiple_configs(self, content):
        """分割包含多个配置的响应"""
        configs = []
        parts = content.split("## configuration ##")
        
        for part in parts:
            if not part.strip():
                continue
                
            # 查找结束标记
            end_pos = part.find(" ##")
            if end_pos != -1:
                config_text = part[:end_pos].strip()
                configs.append("## configuration ##" + config_text + " ##")
        
        return configs
    
    def optimize_batch(self, initial_examples, iterations=10, batch_size=5, 
                       evaluate_fn=None, save_frequency=5):
        """批量优化，自动进行多次迭代
        
        Args:
            initial_examples: 初始示例列表
            iterations: 迭代次数
            batch_size: 每次迭代生成的配置数量
            evaluate_fn: 评估函数，接收配置字典，返回性能值
            save_frequency: 每多少次迭代保存一次会话状态
        """
        if not evaluate_fn:
            raise ValueError("必须提供评估函数evaluate_fn")
            
        # 添加初始示例
        self.add_examples(initial_examples)
        
        for iteration in range(iterations):
            print(f"\n===== 优化迭代 {iteration+1}/{iterations} =====")
            
            # 获取多个配置建议
            configs_text, cost, tokens = self.get_multiple_configurations(n=batch_size)
            print(f"生成了 {len(configs_text)} 个配置建议, 成本: ${cost:.4f}, 令牌: {tokens}")
            
            # 评估每个配置
            for i, config_text in enumerate(configs_text):
                print(f"\n评估配置 {i+1}/{len(configs_text)}")
                config_dict = self.extract_configuration(config_text)
                
                if config_dict:
                    try:
                        # 评估配置
                        performance = evaluate_fn(config_dict)
                        print(f"配置性能: {performance}")
                        
                        # 添加到历史
                        self.add_example_with_performance(config_dict, performance)
                    except Exception as e:
                        print(f"评估配置时出错: {e}")
                else:
                    print("无法解析配置")
            
            # 定期保存状态
            if (iteration + 1) % save_frequency == 0:
                self.conversation.save_conversation_snapshot()
                
        # 优化结束，查找最佳配置
        best_config = max(self.conversation.config_history, 
                          key=lambda x: float(x['performance']) if isinstance(x['performance'], (int, float, str)) and str(x['performance']) != '0.0' else 0)
        
        print(f"\n===== 优化完成 =====")
        print(f"最佳配置: {best_config['config']}")
        print(f"性能: {best_config['performance']}")
        
        return best_config


def dummy_evaluate_fn(config):
    """示例评估函数，在实际应用中替换为真实的评估"""
    # 这里只是一个示例，返回一个随机性能值
    # 在实际应用中，这个函数应该实际运行TVM并返回性能指标
    import random
    return random.uniform(10.0, 13.0)

def main():
    # 示例配置
    example_configs = [
        {'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 8, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 2, densej.inner.inner.inner.innertileSpatial: 1, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 48, dense.wmma.accumulator.sharedax0tileSpatial: 1, dense.wmma.accumulator.sharedax1tileSpatial: 8, wmma_m: 32, wmma_k: 16, wmma_n: 8, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 2, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 2, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 3, dense.wmma.accumulator_shared_pos: 2, B.shared_ax1: 16, B.shared_offset: 32, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 8, A.shared_ax1: 16, A.shared_offset: 8, A.shared_vectorize: 8, dense_unroll_pragma: 4 ##', 'A': '0.0'}, 
        {'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 16, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 2, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 32, dense.wmma.accumulator.sharedax0tileSpatial: 8, dense.wmma.accumulator.sharedax1tileSpatial: 2, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 4, dense.wmma.accumulator_local_pos: 3, dense.wmma.accumulator_shared_pos: 3, B.shared_ax1: 16, B.shared_offset: 0, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 16, A.shared_ax1: 16, A.shared_offset: 48, A.shared_vectorize: 2, dense_unroll_pragma: 4 ##', 'A': '12.541613'}, 
        {'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 1, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 8, densej.inner.inner.innertileSpatial: 4, densei.inner.inner.inner.innertileSpatial: 8, densej.inner.inner.inner.innertileSpatial: 16, dense_shared_pos: 2, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 24, dense.wmma.accumulator.sharedax0tileSpatial: 1, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 32, wmma_k: 16, wmma_n: 8, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 2, dense.wmma.accumulatorj.ctileAll: 8, dense.wmma.accumulatorktileAll: 4, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 1, dense.wmma.accumulator_local_pos: 0, dense.wmma.accumulator_shared_pos: 0, B.shared_ax1: 16, B.shared_offset: 24, B.shared_vectorize: 4, threadIdx.x: 32, threadIdx.y: 1, A.shared_ax1: 16, A.shared_offset: 8, A.shared_vectorize: 2, dense_unroll_pragma: 3 ##', 'A': '11.630533'}, 
        {'Q': '## densei.innertileSpatial: 2, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 4, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 2, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 48, dense.wmma.accumulator.sharedax0tileSpatial: 4, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 2, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 2, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 2, dense.wmma.accumulator_shared_pos: 2, B.shared_ax1: 16, B.shared_offset: 8, B.shared_vectorize: 8, threadIdx.x: 32, threadIdx.y: 4, A.shared_ax1: 16, A.shared_offset: 24, A.shared_vectorize: 1, dense_unroll_pragma: 1 ##', 'A': '12.792116'}
    ]

    persistence_path = "conversation_history/latest_snapshot.json"
    optimizer = TvmScheduleOptimizer(persistence_path=persistence_path)
    
    # 选择运行模式
    import argparse
    parser = argparse.ArgumentParser(description='TVM Schedule Optimizer')
    parser.add_argument('--mode', type=str, default='interactive', 
                        choices=['interactive', 'batch', 'resume'],
                        help='Running mode: interactive, batch, or resume')
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        # 交互式示例
        print("运行交互式优化示例...")
        
        # 添加初始示例
        optimizer.add_examples(example_configs)
        
        # 获取配置
        response, cost, tokens = optimizer.get_configuration()
        print(f"初始响应:\n{response}")
        print(f"成本: ${cost:.4f}, 令牌: {tokens}")
        
        config = optimizer.extract_configuration(response)
        print(f"提取的配置: {config}")
        
        # 模拟评估
        performance = dummy_evaluate_fn(config)
        print(f"模拟评估性能: {performance}")
        
        
        # 添加结果回到历史
        optimizer.add_example_with_performance(config, performance)
        
        # 添加更多示例并再次获取配置
        new_examples = [
            {'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 8, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 8, densej.inner.inner.innertileSpatial: 4, densei.inner.inner.inner.innertileSpatial: 1, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 24, dense.wmma.accumulator.sharedax0tileSpatial: 4, dense.wmma.accumulator.sharedax1tileSpatial: 2, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 1, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 4, dense.wmma.accumulator_local_pos: 2, dense.wmma.accumulator_shared_pos: 0, B.shared_ax1: 64, B.shared_offset: 0, B.shared_vectorize: 4, threadIdx.x: 32, threadIdx.y: 8, A.shared_ax1: 64, A.shared_offset: 32, A.shared_vectorize: 2, dense_unroll_pragma: 3 ##', 'A': '12.266978'},
            {'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 16, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 4, densej.inner.inner.inner.innertileSpatial: 1, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 32, dense.wmma.accumulator.sharedax0tileSpatial: 4, dense.wmma.accumulator.sharedax1tileSpatial: 4, wmma_m: 16, wmma_k: 16, wmma_n: 16, dense.wmma.accumulator.shared_local_pos: 1, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 2, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 3, dense.wmma.accumulator_shared_pos: 3, B.shared_ax1: 16, B.shared_offset: 0, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 16, A.shared_ax1: 16, A.shared_offset: 24, A.shared_vectorize: 1, dense_unroll_pragma: 5 ##', 'A': '12.672532'}
        ]
        
        optimizer.add_examples(new_examples)
        
        # 再次获取配置
        response, cost, tokens = optimizer.get_configuration()
        print(f"\n新示例后的响应:\n{response}")
        print(f"成本: ${cost:.4f}, 令牌: {tokens}")
        
        config = optimizer.extract_configuration(response)
        print(f"提取的配置: {config}")
        
        # 获取多个配置
        responses, cost, tokens = optimizer.get_multiple_configurations(n=3)
        print(f"\n获取多个配置:")
        
        for i, response in enumerate(responses):
            print(f"\n配置 {i+1}:")
            print(response)
            config = optimizer.extract_configuration(response)
            print(f"提取的配置: {config}")
        
        print(f"\n获取多个配置的成本: ${cost:.4f}, 令牌: {tokens}")
        
        # 保存最终状态
        optimizer.conversation.save_conversation_snapshot()
        print("会话快照已保存")
        
    elif args.mode == 'batch':
        # 批量优化模式
        print("运行批量优化...")
        
        # 使用示例评估函数运行10次迭代，每次生成3个配置
        best_config = optimizer.optimize_batch(
            initial_examples=example_configs,
            iterations=3,
            batch_size=3,
            evaluate_fn=dummy_evaluate_fn,
            save_frequency=1
        )
        
        print(f"批量优化完成，最佳配置:")
        print(f"配置: {best_config['config']}")
        print(f"性能: {best_config['performance']}")
        
    elif args.mode == 'resume':
        # 从上次保存的状态恢复并继续优化
        print("从上次状态恢复并继续优化...")
        
        # 检查会话是否已加载
        if len(optimizer.conversation.all_examples) == 0:
            print("没有找到有效的会话状态，添加初始示例...")
            optimizer.add_examples(example_configs)
        else:
            print(f"已加载会话状态，包含 {len(optimizer.conversation.all_examples)} 个示例")
        
        # 继续批量优化
        best_config = optimizer.optimize_batch(
            initial_examples=[],  # 已经加载了示例，不需要再添加
            iterations=3,
            batch_size=3,
            evaluate_fn=dummy_evaluate_fn,
            save_frequency=1
        )
        
        print(f"继续优化完成，最佳配置:")
        print(f"配置: {best_config['config']}")
        print(f"性能: {best_config['performance']}")

if __name__ == "__main__":
    main()
