import tvm
from tvm import relay
from collections import namedtuple

'''def tree_gemm(data, internal_node, leaf_node, dtype, sparse_replacing, dtype_converting):
    """
    S [internal_node, X_shape] The relationship between internal node and feature
    T [internal_node, 1] Threshold for each internal node
    B [leaf_node, internal_node] The relationship between lead node and internal node
    L [leaf_node,] Label for each leaf node
    """
    if(dtype_converting == True):
        min_dtype = "int8"
    else:
        min_dtype = dtype
    index_dtype = "int32"

    batch_size = data.type_annotation.shape[0]

    if(sparse_replacing == True):
        S_data = relay.var("S_data", dtype=dtype)
        S_indices = relay.var("S_indices", dtype=index_dtype)
        S_indptr = relay.var("S_indptr", dtype=index_dtype)
        Sparse = namedtuple("Sparse", ["data", "indices", "indptr"])
        S = Sparse(S_data, S_indices, S_indptr)
        y = relay.nn.sparse_dense(data, S)
    else:
        S = relay.var("S", shape=(internal_node, data.type_annotation.shape[1]), dtype=dtype)
        y = relay.nn.dense(data, relay.var("S", dtype=dtype), units=internal_node)
    # y = tvm.relay.nn.bitserial_dense(data, relay.var("S", dtype=dtype), units=internal_node, weight_bits=1)
    # [batch_size, internal_node]
    T = relay.var("T", shape=(internal_node,), dtype=dtype)
    y = relay.greater(y, relay.var("T", shape=(internal_node,), dtype=dtype))
    #y = tvm.relay.nn.bitserial_dense(y, relay.var("B", dtype="bool"), units=leaf_node, pack_dtype="uint8", out_dtype="uint8")
    y = relay.cast(y, min_dtype)

    B = relay.var("B", shape=(leaf_node, internal_node), dtype=min_dtype)
    y = relay.nn.dense(y, relay.var("B", dtype=min_dtype), units=leaf_node, out_dtype=min_dtype)
    # [batch_size, leaf_node]
    # y = relay.argmax(y, axis=-1)
    max_val = relay.max(y, axis=1, keepdims=True)
    equal_to_max = relay.equal(y, relay.broadcast_to(max_val, [batch_size, leaf_node]))
    indices = relay.cast(equal_to_max, index_dtype)

    arange = relay.arange(
        relay.const(0, dtype=index_dtype),
        relay.const(leaf_node, dtype=index_dtype),
        relay.const(1, dtype=index_dtype),
        dtype=index_dtype
    )

    arange = relay.reshape(arange, [1, -1])
    arange = relay.broadcast_to(arange, [batch_size, leaf_node])

    y = relay.sum(relay.multiply(indices, arange), axis=1)
    y = relay.cast(y, index_dtype)
    
    # [batch_size,]
    l = relay.var("L", shape=(leaf_node,), dtype=dtype)
    y = relay.take(l, y)
    # [batch_size,]
    return y
'''


def tree_gemm(data, internal_node, leaf_node, dtype, sparse_replacing, dtype_converting):
    """
    修改函数，确保所有类型正确
    """
    if(dtype_converting == True):
        min_dtype = "int8"
    else:
        min_dtype = dtype
    index_dtype = "int32"
    
    batch_size = data.type_annotation.shape[0]
    
    if(sparse_replacing == True):
        S_data = relay.var("S_data", dtype=dtype)
        S_indices = relay.var("S_indices", dtype=index_dtype)
        S_indptr = relay.var("S_indptr", dtype=index_dtype)
        Sparse = namedtuple("Sparse", ["data", "indices", "indptr"])
        S = Sparse(S_data, S_indices, S_indptr)
        y = relay.nn.sparse_dense(data, S)
    else:
        S = relay.var("S", shape=(internal_node, data.type_annotation.shape[1]), dtype=dtype)
        y = relay.nn.dense(data, S, units=internal_node)
    
    T = relay.var("T", shape=(internal_node,), dtype=dtype)
    y = relay.greater(y, T)
    y = relay.cast(y, min_dtype)
    
    B = relay.var("B", shape=(leaf_node, internal_node), dtype=min_dtype)
    y = relay.nn.dense(y, B, units=leaf_node, out_dtype=min_dtype)
    
    # 处理 argmax 逻辑，确保所有类型正确
    max_val = relay.max(y, axis=1, keepdims=True)
    equal_to_max = relay.equal(y, relay.broadcast_to(max_val, [batch_size, leaf_node]))
    indices = relay.cast(equal_to_max, index_dtype)
    
    # 创建 arange
    arange = relay.arange(
        relay.const(0, dtype=index_dtype),
        relay.const(leaf_node, dtype=index_dtype),
        relay.const(1, dtype=index_dtype),
        dtype=index_dtype
    )
    
    # 广播 arange
    arange = relay.reshape(arange, [1, -1])
    arange = relay.broadcast_to(arange, [batch_size, leaf_node])
    
    # 计算 argmax
    y = relay.sum(relay.multiply(indices, arange), axis=1)
    # 确保 indices 是整数类型
    y = relay.cast(y, index_dtype)
    
    # take 操作
    l = relay.var("L", shape=(leaf_node,), dtype=dtype)
    y = relay.take(l, y)
    
    # 确保最终输出类型正确
    y = relay.cast(y, dtype)
    
    return y

def decision_tree_classifier(data_shape, internal_node, leaf_node, dtype, sparse_replacing, dtype_converting):
    """
    确保数据类型正确
    """
    data = relay.var("data", relay.TensorType(data_shape, dtype))
    y = tree_gemm(data, internal_node, leaf_node, dtype, sparse_replacing, dtype_converting)
    # 最终类型检查
    if isinstance(y, relay.expr.Call):
        y = relay.cast(y, dtype)
    return y