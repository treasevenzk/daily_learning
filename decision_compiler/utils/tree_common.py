import tvm
import numpy as np
from scipy.sparse import csr_matrix, bsr_matrix
from collections import namedtuple

def parse_tree(X_shape, clf, clf_flag, dtype):
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value
    T = []
    n_node = len(children_left)

    level_order_traversal = [0]
    internal_index = []
    def level_order_traverse(i):
        # Use level order traverse to number internal node 
        while len(level_order_traversal) > 0:
            node_i = level_order_traversal.pop(0)
            if(feature[node_i] >= 0):
                internal_index.append(node_i)
            if(children_left[node_i] > 0):
                level_order_traversal.append(children_left[node_i])
            if(children_right[node_i] > 0):
                level_order_traversal.append(children_right[node_i])
    
    level_order_traverse(0)
    n_internal = len(internal_index)
    dic_internal = {internal_index[i] : i for i in range(n_internal)}
    #print(internal_index)
    for i in range(n_internal):
        T.append(threshold[internal_index[i]])
    T = np.array(T)
    S = np.zeros((n_internal, X_shape), dtype=dtype)
    for i in range(n_internal):
        S[i][feature[internal_index[i]]] = 1
    #print(S)
    S = np.array(S)
    
    n_leaf = n_node - n_internal
    leaf_index = []
    mid_order_traversal = []
    
    def mid_order_traverse(i):
        # Use mid order traverse to number leaf node
        if(children_left[i] > 0):
            mid_order_traverse(children_left[i])
        mid_order_traversal.append(i)
        if(feature[i] < 0):
            leaf_index.append(i)
        if(children_right[i] > 0):
            mid_order_traverse(children_right[i])
    mid_order_traverse(0)
    dic_leaf = {leaf_index[i] : i for i in range(n_leaf)}
    
    #print(mid_order_traversal)
    #print(leaf_index)
    
    tree_path = np.ones((n_node, n_node), dtype=dtype)
    for i in range(n_node):
        if (feature[i] >= 0):
            # internal node
            tree_path[i][children_left[i]] = 0
            for j in range(n_node):
                if (tree_path[i][j] == 0 and feature[j] >=0):
                    tree_path[i][children_left[j]] = 0
                    tree_path[i][children_right[j]] = 0
    #print(tree_path)
    B = np.ones((n_leaf, n_internal), dtype=dtype)
    for i in range(n_node):
        for j in range(n_node):
            if(tree_path[i][j] == 0 and j in leaf_index):
                B[dic_leaf[j]][dic_internal[i]] = 0
    #print(B)
    L = []
    for i in range(n_leaf):
        L.append(value[leaf_index[i]][0])
    L = np.array(L)
    #print(L)
    # Note that not converting for forest
    #print(L)
    if(clf_flag == "tree_clf"):
        for i in range(L.shape[0]):
            L[i] = L[i] / np.sum(L[i])
        L = np.argmax(L, axis=1)
        L = L.astype(dtype)
    
    elif(clf_flag == "forest_clf"):
        for i in range(L.shape[0]):
            L[i] = L[i] / np.sum(L[i])
    return S, T, B, L

def dense_to_sparse(x, dtype, sparse_type):
    """
    Convert dense data to sparse data in csr format
    """
    if(sparse_type == "csr"):
        x = csr_matrix(x)
    elif(sparse_type == "bsr"):
        x = bsr_matrix(x)
    else:
        print("Unsupported sparse type")
    data = x.data.astype(dtype)
    indices = x.indices.astype("int32")
    indptr = x.indptr.astype("int32")
    data = tvm.nd.array(data)
    indices = tvm.nd.array(indices)
    indptr = tvm.nd.array(indptr)
    return data, indices, indptr

def convert_decision_tree(X_shape, clf, clf_flag, dtype, target, sparse, type_convert):
    """
    Convert sklearn decision tree to tvm gemm
    Fit for extra tree as well
    """
    S, T, B, L = parse_tree(X_shape, clf, clf_flag, dtype)
    ctx = tvm.device(target, 0)

    S = S.reshape((S.shape[0], X_shape))
    T = T.reshape(-1)
    B = B.reshape((B.shape[0], S.shape[0]))

    S = S.astype(dtype)
    T = T.astype(dtype)
    B = B.astype(dtype)
    L = L.astype(dtype)
    if(clf_flag == "tree_clf"):
        classes = clf.classes_
        L = L.astype("int32")
        L = np.take(classes, L)
        L = L.astype("float32")
    if(type_convert == True):
        B = B.astype("int8")
    T = tvm.nd.array(T)
    B = tvm.nd.array(B)
    L = tvm.nd.array(L)
    if(sparse == True):
        S_data, S_indices, S_indptr = dense_to_sparse(S, "float32", "csr")
        return S_data, S_indices, S_indptr, T, B, L
    else:
        S = tvm.nd.array(S)
        return S, T, B, L

