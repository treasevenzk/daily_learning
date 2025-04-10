import numpy as np
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import time
from decision_compiler.algorithms.tree import decision_tree_classifier
from decision_compiler.utils.tree_common import convert_decision_tree


def init_model(sklearn_model, data_shape, target, dtype, out_dtype, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning):
    model = TreeModel(sklearn_model, data_shape, target, dtype, out_dtype, "tree_clf", batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    return model

def build_model(sklearn_model, data_shape, target="llvm", dtype="float32", out_dtype="float32", batch_size=None, sparse_replacing=False, dtype_converting=False, elimination=False, auto_tuning=False):
    if (batch_size == None):
        batch_size = data_shape[0]
    model = init_model(sklearn_model, data_shape, target, dtype, out_dtype, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning)
    model.build()
    return model

class BaseModel:
    def __init__(self, sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning):
        self.sklearn_model = sklearn_model
        self.data_shape = data_shape
        self.target = target
        self.dev = tvm.device(str(target), 0)
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.flag_clf = flag_clf
        self.batch_size = batch_size if batch_size is not None else data_shape[0]
        try:
            self.batch_shape = (batch_size, self.data_shape[1])
        except:
            self.batch_shape = (batch_size)
        self.sparse_replacing = sparse_replacing
        self.dtype_converting = dtype_converting
        self.auto_tuning = auto_tuning

    
    def get_mod(self):
        """
        Get IRModule from the algorithm
        """
        if isinstance(self.algo, relay.Function):
            mod = tvm.IRModule.from_expr(self.algo)
        else:
            args = relay.analysis.free_vars(self.algo)
            net = relay.Function(args, self.algo)
            mod = tvm.IRModule.from_expr(net)
        
        mod = relay.transform.InferType()(mod)
        
        # 获取参数列表和添加检查
        params = {}
        param_map = {
            "weight": 0,  # coef
            "bias": 1,
            "classes": 2
        }
        
        print("\n=== Parameter Assignment Check ===")
        for param in mod["main"].params:
            name = param.name_hint
            print(f"Processing parameter: {name}")
            if name == "data":
                continue
            elif name in param_map:
                idx = param_map[name]
                if idx < len(self.params):
                    print(f"Assigning {name} with shape {self.params[idx].shape}")
                    params[name] = self.params[idx]
                else:
                    print(f"Warning: No parameter available for {name}")
        
        return mod, params
    
    def build(self):
        print("\n=== Build Check ===")
        print("Data shape:", self.data_shape)
        print("Batch shape:", self.batch_shape)
        print("Target:", self.target)
        print("Device:", self.dev)
        print("Data type:", self.dtype)
        print("Output data type:", self.out_dtype)
        
        mod, params = self.get_mod()

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=self.target, params=params)

        self.lib = lib
        self.model = graph_executor.GraphModule(lib["default"](self.dev))

    def run(self, data, breakdown=False):
        data = data.asnumpy()
        out = np.empty([self.data_shape[0]], dtype=self.out_dtype)
        n_batch = self.data_shape[0] // self.batch_size
        load_time = 0
        exec_time = 0
        store_time = 0
        for i in range(n_batch):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            a = time.perf_counter()
            input_data = tvm.nd.array(data[start:end])
            self.model.set_input("data", input_data)
            b = time.perf_counter()
            self.model.run()
            c = time.perf_counter()
            out[start:end] = self.model.get_output(0).asnumpy().flatten()
            d = time.perf_counter()
            load_time = load_time + b - a
            exec_time = exec_time + c - b
            store_time = store_time + d - c
        if(breakdown == True):
            take_time = 0
            return load_time, exec_time, store_time, take_time, out
        else:
            return out
        

class TreeModel(BaseModel):
    def __init__(self, sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, elimination, auto_tuning):
        super(TreeModel, self).__init__(sklearn_model, data_shape, target, dtype, out_dtype, flag_clf, batch_size, sparse_replacing, dtype_converting, auto_tuning)
        self.params = self._parse_params()

        data = relay.var("data", relay.TensorType((self.batch_shape[0], self.batch_shape[1]), self.dtype))

        self._get_algo()

    def _parse_params(self):
        if (self.sparse_replacing == True):
            S_data, S_indices, S_indptr, T, B, L = convert_decision_tree(self.batch_shape[1], self.sklearn_model, self.flag_clf, self.dtype, self.target, self.sparse_replacing, self.dtype_converting)
            self.L = L.asnumpy()
            self.internal_node = T.shape[0]
            self.leaf_node = B.shape[0]
            return L, S_data, S_indices, S_indptr, T, B
        else:
            S, T, B, L = convert_decision_tree(self.batch_shape[1], self.sklearn_model, self.flag_clf, self.dtype, self.target, self.sparse_replacing, self.dtype_converting)
            self.internal_node = S.shape[0]
            self.leaf_node = B.shape[0]
            return L, S, T, B
        
    def _get_algo(self):
        func = decision_tree_classifier
        print(self.batch_shape)
        self.algo = func(self.batch_shape, self.internal_node, self.leaf_node, self.dtype, self.sparse_replacing, self.dtype_converting)
        