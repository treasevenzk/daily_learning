def compare_dicts(dict_a, dict_b):
    """
    比较两个字典，返回它们的差异
    
    返回:
        only_in_a: 只在A中存在的键
        only_in_b: 只在B中存在的键
    """
    # 获取两个字典的所有键
    keys_a = set(dict_a.keys())
    keys_b = set(dict_b.keys())
    
    # 找出只在A中存在的键
    only_in_a = keys_a - keys_b
    
    # 找出只在B中存在的键
    only_in_b = keys_b - keys_a
    
    return only_in_a, only_in_b

# 使用示例
if __name__ == "__main__":
    # 示例字典
    dict_a = {'wmma_m': Var(wmma_m, 8, 32, 16), 'wmma_m_cand0': Var(wmma_m_cand0, 0, 1, 0), 'wmma_m_cand1': Var(wmma_m_cand1, 0, 1, 0), 'wmma_m_cand2': Var(wmma_m_cand2, 0, 1, 0), 'wmma_k': Var(wmma_k, 16, 16, 16), 'wmma_n': Var(wmma_n, 8, 32, 16), 'wmma_n_cand0': Var(wmma_n_cand0, 0, 1, 0), 'wmma_n_cand1': Var(wmma_n_cand1, 0, 1, 0), 'wmma_n_cand2': Var(wmma_n_cand2, 0, 1, 0), 'wmma_m_wmma_n': Var(wmma_m_wmma_n, 1, 4096, 1), 'dense.wmma.accumulator_shared_pos': Var(dense.wmma.accumulator_shared_pos, 0, 3, 0), 'dense.wmma.accumulator_local_pos': Var(dense.wmma.accumulator_local_pos, 0, 3, 0), 'dense_shared_pos': Var(dense_shared_pos, 1, 5, 0), 'dense.wmma.accumulator.shared_local_pos': Var(dense.wmma.accumulator.shared_local_pos, 0, 1, 0), 'dense_unroll_pragma': Var(dense_unroll_pragma, 0, 5, 0), 'dense_i.outer': Var(dense_i.outer, 1, 64, 1), 'dense_i.inner': Var(dense_i.inner, 1, 64, 1), 'dense_j.outer': Var(dense_j.outer, 1, 64, 1), 'dense_j.inner': Var(dense_j.inner, 1, 64, 1), 'densei.innertileSpatial': Var(densei.innertileSpatial, 1, 64, 1), 'dense_i.inner.outer': Var(dense_i.inner.outer, 1, 64, 1), 'dense_i.inner.inner': Var(dense_i.inner.inner, 1, 64, 1), 'densej.innertileSpatial': Var(densej.innertileSpatial, 1, 64, 1), 'dense_j.inner.outer': Var(dense_j.inner.outer, 1, 64, 1), 'dense_j.inner.inner': Var(dense_j.inner.inner, 1, 64, 1), 'blockIdx.x': Var(blockIdx.x, 1, 8888888, 1), 'dense_i.inner.outer.j.inner.outer.fused': Var(dense_i.inner.outer.j.inner.outer.fused, 1, 1000000, 1), 'densei.inner.innertileSpatial': Var(densei.inner.innertileSpatial, 1, 64, 1), 'dense_i.inner.inner.outer': Var(dense_i.inner.inner.outer, 1, 64, 1), 'dense_i.inner.inner.inner': Var(dense_i.inner.inner.inner, 1, 64, 1), 'densej.inner.innertileSpatial': Var(densej.inner.innertileSpatial, 1, 64, 1), 'dense_j.inner.inner.outer': Var(dense_j.inner.inner.outer, 1, 64, 1), 'dense_j.inner.inner.inner': Var(dense_j.inner.inner.inner, 1, 64, 1), 'threadIdx.y': Var(threadIdx.y, 1, 1024, 1), 'dense_i.inner.inner.outer.j.inner.inner.outer.fused': Var(dense_i.inner.inner.outer.j.inner.inner.outer.fused, 1, 1000000, 1), 'densei.inner.inner.innertileSpatial': Var(densei.inner.inner.innertileSpatial, 1, 64, 1), 'dense_i.inner.inner.inner.outer': Var(dense_i.inner.inner.inner.outer, 1, 64, 1), 'dense_i.inner.inner.inner.inner': Var(dense_i.inner.inner.inner.inner, 1, 64, 1), 'densej.inner.inner.innertileSpatial': Var(densej.inner.inner.innertileSpatial, 1, 64, 1), 'dense_j.inner.inner.inner.outer': Var(dense_j.inner.inner.inner.outer, 1, 64, 1), 'dense_j.inner.inner.inner.inner': Var(dense_j.inner.inner.inner.inner, 1, 64, 1), 'threadIdx.x': Var(threadIdx.x, 32, 32, 1), 'dense_i.inner.inner.inner.outer.j.inner.inner.inner.outer.fused': Var(dense_i.inner.inner.inner.outer.j.inner.inner.inner.outer.fused, 1, 1000000, 1), 'dense_vectorize': Var(dense_vectorize, 1, 8, 1), 'dense_vectorize_cand0': Var(dense_vectorize_cand0, 0, 1, 0), 'dense_vectorize_cand1': Var(dense_vectorize_cand1, 0, 1, 0), 'dense_vectorize_cand2': Var(dense_vectorize_cand2, 0, 1, 0), 'dense_vectorize_cand3': Var(dense_vectorize_cand3, 0, 1, 0), 'densei.inner.inner.inner.innertileSpatial': Var(densei.inner.inner.inner.innertileSpatial, 1, 64, 1), 'dense_i.inner.inner.inner.inner.outer': Var(dense_i.inner.inner.inner.inner.outer, 1, 64, 1), 'dense_i.inner.inner.inner.inner.inner': Var(dense_i.inner.inner.inner.inner.inner, 1, 64, 1), 'densej.inner.inner.inner.innertileSpatial': Var(densej.inner.inner.inner.innertileSpatial, 1, 64, 1), 'dense_j.inner.inner.inner.inner.outer': Var(dense_j.inner.inner.inner.inner.outer, 1, 64, 1), 'dense_j.inner.inner.inner.inner.inner': Var(dense_j.inner.inner.inner.inner.inner, 1, 64, 1), 'dense_i.inner.inner.inner.inner.inner.j.inner.inner.inner.inner.inner.fused': Var(dense_i.inner.inner.inner.inner.inner.j.inner.inner.inner.inner.inner.fused, 1, 1000000, 1), 'dense.wmma.accumulator.shared_ax0': Var(dense.wmma.accumulator.shared_ax0, 1, 64, 1), 'dense_shared_pos_select0': Var(dense_shared_pos_select0, 0, 1, 0), 'dense_shared_pos_select1': Var(dense_shared_pos_select1, 0, 1, 0), 'dense_shared_pos_select2': Var(dense_shared_pos_select2, 0, 1, 0), 'dense_shared_pos_select3': Var(dense_shared_pos_select3, 0, 1, 0), 'dense_shared_pos_select4': Var(dense_shared_pos_select4, 0, 1, 0), 'dense_shared_pos_select5': Var(dense_shared_pos_select5, 0, 1, 0), 'dense.wmma.accumulator.shared_ax1': Var(dense.wmma.accumulator.shared_ax1, 1, 64, 1), 'dense.wmma.accumulator.shared_offset': Var(dense.wmma.accumulator.shared_offset, 0, 48, 0), 'dense.wmma.accumulator.shared_offset_cand0': Var(dense.wmma.accumulator.shared_offset_cand0, 0, 1, 0), 'dense.wmma.accumulator.shared_offset_cand1': Var(dense.wmma.accumulator.shared_offset_cand1, 0, 1, 0), 'dense.wmma.accumulator.shared_offset_cand2': Var(dense.wmma.accumulator.shared_offset_cand2, 0, 1, 0), 'dense.wmma.accumulator.shared_offset_cand3': Var(dense.wmma.accumulator.shared_offset_cand3, 0, 1, 0), 'dense.wmma.accumulator.shared_offset_cand4': Var(dense.wmma.accumulator.shared_offset_cand4, 0, 1, 0), 'dense.wmma.accumulator.shared_offset_cand5': Var(dense.wmma.accumulator.shared_offset_cand5, 0, 1, 0), 'dense.wmma.accumulator.shared_align_size': Var(dense.wmma.accumulator.shared_align_size, 1, 88888888, 1), 'dense.wmma.accumulator.sharedax0tileSpatial': Var(dense.wmma.accumulator.sharedax0tileSpatial, 1, 64, 1), 'dense.wmma.accumulator.shared_ax0.outer': Var(dense.wmma.accumulator.shared_ax0.outer, 1, 64, 1), 'dense.wmma.accumulator.shared_ax0.inner': Var(dense.wmma.accumulator.shared_ax0.inner, 1, 64, 1), 'dense.wmma.accumulator.sharedax1tileSpatial': Var(dense.wmma.accumulator.sharedax1tileSpatial, 1, 64, 1), 'dense.wmma.accumulator.shared_ax1.outer': Var(dense.wmma.accumulator.shared_ax1.outer, 1, 64, 1), 'dense.wmma.accumulator.shared_ax1.inner': Var(dense.wmma.accumulator.shared_ax1.inner, 1, 64, 1), 'dense.wmma.accumulator.shared_ax0.outer.ax1.outer.fused': Var(dense.wmma.accumulator.shared_ax0.outer.ax1.outer.fused, 1, 1000000, 1), 'dense.wmma.accumulator.shared_ax0.inner.outer': Var(dense.wmma.accumulator.shared_ax0.inner.outer, 1, 64, 1), 'dense.wmma.accumulator.shared_ax0.inner.inner': Var(dense.wmma.accumulator.shared_ax0.inner.inner, 1, 64, 1), 'dense.wmma.accumulator.shared_ax1.inner.outer': Var(dense.wmma.accumulator.shared_ax1.inner.outer, 1, 64, 1), 'dense.wmma.accumulator.shared_ax1.inner.inner': Var(dense.wmma.accumulator.shared_ax1.inner.inner, 1, 64, 1), 'dense.wmma.accumulator_i.c': Var(dense.wmma.accumulator_i.c, 1, 64, 1), 'dense.wmma.accumulator.shared_local_pos_select0': Var(dense.wmma.accumulator.shared_local_pos_select0, 0, 1, 0), 'dense.wmma.accumulator.shared_local_pos_select1': Var(dense.wmma.accumulator.shared_local_pos_select1, 0, 1, 0), 'dense.wmma.accumulator_j.c': Var(dense.wmma.accumulator_j.c, 1, 64, 1), 'dense.wmma.accumulatori.ctileAll': Var(dense.wmma.accumulatori.ctileAll, 1, 64, 1), 'dense.wmma.accumulator_i.c.outer': Var(dense.wmma.accumulator_i.c.outer, 1, 64, 1), 'dense.wmma.accumulator_i.c.inner': Var(dense.wmma.accumulator_i.c.inner, 1, 64, 1), 'dense.wmma.accumulatorj.ctileAll': Var(dense.wmma.accumulatorj.ctileAll, 1, 64, 1), 'dense.wmma.accumulator_j.c.outer': Var(dense.wmma.accumulator_j.c.outer, 1, 64, 1), 'dense.wmma.accumulator_j.c.inner': Var(dense.wmma.accumulator_j.c.inner, 1, 64, 1), 'dense.wmma.accumulatorktileAll': Var(dense.wmma.accumulatorktileAll, 1, 64, 1), 'dense.wmma.accumulator_k.outer': Var(dense.wmma.accumulator_k.outer, 1, 64, 1), 'dense.wmma.accumulator_k.inner': Var(dense.wmma.accumulator_k.inner, 1, 64, 1), 'dense.wmma.accumulatori.c.innertileAll': Var(dense.wmma.accumulatori.c.innertileAll, 1, 64, 1), 'dense.wmma.accumulator_i.c.inner.outer': Var(dense.wmma.accumulator_i.c.inner.outer, 1, 64, 1), 'dense.wmma.accumulator_i.c.inner.inner': Var(dense.wmma.accumulator_i.c.inner.inner, 1, 64, 1), 'dense.wmma.accumulatorj.c.innertileAll': Var(dense.wmma.accumulatorj.c.innertileAll, 1, 64, 1), 'dense.wmma.accumulator_j.c.inner.outer': Var(dense.wmma.accumulator_j.c.inner.outer, 1, 64, 1), 'dense.wmma.accumulator_j.c.inner.inner': Var(dense.wmma.accumulator_j.c.inner.inner, 1, 64, 1), 'dense.wmma.accumulatork.innertileAll': Var(dense.wmma.accumulatork.innertileAll, 1, 64, 1), 'dense.wmma.accumulator_k.inner.outer': Var(dense.wmma.accumulator_k.inner.outer, 1, 64, 1), 'dense.wmma.accumulator_k.inner.inner': Var(dense.wmma.accumulator_k.inner.inner, 1, 64, 1), 'dense.wmma.accumulatori.c.inner.innertileAll': Var(dense.wmma.accumulatori.c.inner.innertileAll, 1, 64, 1), 'dense.wmma.accumulator_i.c.inner.inner.outer': Var(dense.wmma.accumulator_i.c.inner.inner.outer, 1, 64, 1), 'dense.wmma.accumulator_i.c.inner.inner.inner': Var(dense.wmma.accumulator_i.c.inner.inner.inner, 1, 64, 1), 'dense.wmma.accumulatorj.c.inner.innertileAll': Var(dense.wmma.accumulatorj.c.inner.innertileAll, 1, 64, 1), 'dense.wmma.accumulator_j.c.inner.inner.outer': Var(dense.wmma.accumulator_j.c.inner.inner.outer, 1, 64, 1), 'dense.wmma.accumulator_j.c.inner.inner.inner': Var(dense.wmma.accumulator_j.c.inner.inner.inner, 1, 64, 1), 'dense.wmma.accumulatork.inner.innertileAll': Var(dense.wmma.accumulatork.inner.innertileAll, 1, 64, 1), 'dense.wmma.accumulator_k.inner.inner.outer': Var(dense.wmma.accumulator_k.inner.inner.outer, 1, 64, 1), 'dense.wmma.accumulator_k.inner.inner.inner': Var(dense.wmma.accumulator_k.inner.inner.inner, 1, 64, 1), 'dense.wmma.accumulator_i.c.inner.inner.inner.outer': Var(dense.wmma.accumulator_i.c.inner.inner.inner.outer, 1, 64, 1), 'dense.wmma.accumulator_i.c.inner.inner.inner.inner': Var(dense.wmma.accumulator_i.c.inner.inner.inner.inner, 1, 64, 1), 'dense.wmma.accumulator_j.c.inner.inner.inner.outer': Var(dense.wmma.accumulator_j.c.inner.inner.inner.outer, 1, 64, 1), 'dense.wmma.accumulator_j.c.inner.inner.inner.inner': Var(dense.wmma.accumulator_j.c.inner.inner.inner.inner, 1, 64, 1), 'dense.wmma.accumulator_k.inner.inner.inner.outer': Var(dense.wmma.accumulator_k.inner.inner.inner.outer, 1, 64, 1), 'dense.wmma.accumulator_k.inner.inner.inner.inner': Var(dense.wmma.accumulator_k.inner.inner.inner.inner, 1, 64, 1), 'B.shared.wmma.matrix_b_ax0': Var(B.shared.wmma.matrix_b_ax0, 1, 64, 1), 'B.shared.wmma.matrix_b_ax0.outer': Var(B.shared.wmma.matrix_b_ax0.outer, 1, 64, 1), 'B.shared.wmma.matrix_b_ax0.inner': Var(B.shared.wmma.matrix_b_ax0.inner, 1, 64, 1), 'B.shared.wmma.matrix_b_ax1': Var(B.shared.wmma.matrix_b_ax1, 1, 64, 1), 'B.shared.wmma.matrix_b_ax1.outer': Var(B.shared.wmma.matrix_b_ax1.outer, 1, 64, 1), 'B.shared.wmma.matrix_b_ax1.inner': Var(B.shared.wmma.matrix_b_ax1.inner, 1, 64, 1), 'B.shared_ax0': Var(B.shared_ax0, 1, 64, 1), 'B.shared_tmp_ax0': Var(B.shared_tmp_ax0, 1, 64, 1), 'dense.wmma.accumulator_shared_pos_select0': Var(dense.wmma.accumulator_shared_pos_select0, 0, 1, 0), 'dense.wmma.accumulator_shared_pos_select1': Var(dense.wmma.accumulator_shared_pos_select1, 0, 1, 0), 'dense.wmma.accumulator_shared_pos_select2': Var(dense.wmma.accumulator_shared_pos_select2, 0, 1, 0), 'dense.wmma.accumulator_shared_pos_select3': Var(dense.wmma.accumulator_shared_pos_select3, 0, 1, 0), 'B.shared_ax1': Var(B.shared_ax1, 1, 64, 1), 'B.shared_offset': Var(B.shared_offset, 0, 48, 0), 'B.shared_offset_cand0': Var(B.shared_offset_cand0, 0, 1, 0), 'B.shared_offset_cand1': Var(B.shared_offset_cand1, 0, 1, 0), 'B.shared_offset_cand2': Var(B.shared_offset_cand2, 0, 1, 0), 'B.shared_offset_cand3': Var(B.shared_offset_cand3, 0, 1, 0), 'B.shared_offset_cand4': Var(B.shared_offset_cand4, 0, 1, 0), 'B.shared_offset_cand5': Var(B.shared_offset_cand5, 0, 1, 0), 'B.shared_align_size': Var(B.shared_align_size, 1, 88888888, 1), 'B.shared_ax0.ax1.fused': Var(B.shared_ax0.ax1.fused, 1, 1000000, 1), 'B.shared_vectorize': Var(B.shared_vectorize, 1, 8, 1), 'B.shared_vectorize_cand0': Var(B.shared_vectorize_cand0, 0, 1, 0), 'B.shared_vectorize_cand1': Var(B.shared_vectorize_cand1, 0, 1, 0), 'B.shared_vectorize_cand2': Var(B.shared_vectorize_cand2, 0, 1, 0), 'B.shared_vectorize_cand3': Var(B.shared_vectorize_cand3, 0, 1, 0), 'A.shared.wmma.matrix_a_ax0': Var(A.shared.wmma.matrix_a_ax0, 1, 64, 1), 'dense.wmma.accumulator_local_pos_select0': Var(dense.wmma.accumulator_local_pos_select0, 0, 1, 0), 'dense.wmma.accumulator_local_pos_select1': Var(dense.wmma.accumulator_local_pos_select1, 0, 1, 0), 'dense.wmma.accumulator_local_pos_select2': Var(dense.wmma.accumulator_local_pos_select2, 0, 1, 0), 'dense.wmma.accumulator_local_pos_select3': Var(dense.wmma.accumulator_local_pos_select3, 0, 1, 0), 'A.shared.wmma.matrix_a_ax1': Var(A.shared.wmma.matrix_a_ax1, 1, 64, 1), 'A.shared.wmma.matrix_a_ax0.outer': Var(A.shared.wmma.matrix_a_ax0.outer, 1, 64, 1), 'A.shared.wmma.matrix_a_ax0.inner': Var(A.shared.wmma.matrix_a_ax0.inner, 1, 64, 1), 'A.shared.wmma.matrix_a_ax1.outer': Var(A.shared.wmma.matrix_a_ax1.outer, 1, 64, 1), 'A.shared.wmma.matrix_a_ax1.inner': Var(A.shared.wmma.matrix_a_ax1.inner, 1, 64, 1), 'A.shared_ax0': Var(A.shared_ax0, 1, 64, 1), 'A.shared_tmp_ax0': Var(A.shared_tmp_ax0, 1, 64, 1), 'A.shared_ax1': Var(A.shared_ax1, 1, 64, 1), 'A.shared_offset': Var(A.shared_offset, 0, 48, 0), 'A.shared_offset_cand0': Var(A.shared_offset_cand0, 0, 1, 0), 'A.shared_offset_cand1': Var(A.shared_offset_cand1, 0, 1, 0), 'A.shared_offset_cand2': Var(A.shared_offset_cand2, 0, 1, 0), 'A.shared_offset_cand3': Var(A.shared_offset_cand3, 0, 1, 0), 'A.shared_offset_cand4': Var(A.shared_offset_cand4, 0, 1, 0), 'A.shared_offset_cand5': Var(A.shared_offset_cand5, 0, 1, 0), 'A.shared_align_size': Var(A.shared_align_size, 1, 88888888, 1), 'A.shared_ax0.ax1.fused': Var(A.shared_ax0.ax1.fused, 1, 1000000, 1), 'A.shared_vectorize': Var(A.shared_vectorize, 1, 8, 1), 'A.shared_vectorize_cand0': Var(A.shared_vectorize_cand0, 0, 1, 0), 'A.shared_vectorize_cand1': Var(A.shared_vectorize_cand1, 0, 1, 0), 'A.shared_vectorize_cand2': Var(A.shared_vectorize_cand2, 0, 1, 0), 'A.shared_vectorize_cand3': Var(A.shared_vectorize_cand3, 0, 1, 0), 'threads': Var(threads, 1, 1024, 1), 'A.shared_shared_mem_size': Var(A.shared_shared_mem_size, 1, 24576, 1), 'B.shared_shared_mem_size': Var(B.shared_shared_mem_size, 1, 24576, 1), 'dense.wmma.accumulator.shared_shared_mem_size': Var(dense.wmma.accumulator.shared_shared_mem_size, 1, 24576, 1)}

    dict_b = {'densei.innertileSpatial': 1, 'densej.innertileSpatial': 1, 'densei.inner.innertileSpatial': 1, 'densej.inner.innertileSpatial': 1, 'densei.inner.inner.innertileSpatial': 1, 'densej.inner.inner.innertileSpatial': 1, 'densei.inner.inner.inner.innertileSpatial': 1, 'densej.inner.inner.inner.innertileSpatial': 1, 'dense_i.outer': 1, 'dense_j.outer': 1, 'dense_i.inner.outer.j.inner.outer.fused': 1, 'dense_i.inner.inner.outer.j.inner.inner.outer.fused': 1, 'dense_i.inner.inner.inner.outer.j.inner.inner.inner.outer.fused': 1, 'dense_i.inner.inner.inner.inner.outer': 1, 'dense_j.inner.inner.inner.inner.outer': 1, 'dense_i.inner.inner.inner.inner.inner.j.inner.inner.inner.inner.inner.fused': 1, 'dense_shared_pos': 1, 'dense.wmma.accumulator.shared_ax0': 1, 'dense.wmma.accumulator.shared_ax1': 1, 'dense.wmma.accumulator.shared_offset': 1, 'dense.wmma.accumulator.sharedax0tileSpatial': 1, 'dense.wmma.accumulator.sharedax1tileSpatial': 1, 'wmma_m': 1, 'wmma_k': 1, 'wmma_n': 1, 'dense.wmma.accumulator.shared_ax0.outer.ax1.outer.fused': 1, 'dense.wmma.accumulator.shared_ax0.inner.outer': 1, 'dense.wmma.accumulator.shared_ax1.inner.outer': 1, 'dense.wmma.accumulator.shared_ax0.inner.inner': 1, 'dense.wmma.accumulator.shared_ax1.inner.inner': 1, 'dense.wmma.accumulator.shared_local_pos': 1, 'dense.wmma.accumulator_i.c': 1, 'dense.wmma.accumulator_j.c': 1, 'dense.wmma.accumulatori.ctileAll': 1, 'dense.wmma.accumulatorj.ctileAll': 1, 'dense.wmma.accumulatorktileAll': 1, 'dense.wmma.accumulatori.c.innertileAll': 1, 'dense.wmma.accumulatorj.c.innertileAll': 1, 'dense.wmma.accumulatork.innertileAll': 1, 'dense.wmma.accumulatori.c.inner.innertileAll': 1, 'dense.wmma.accumulatorj.c.inner.innertileAll': 1, 'dense.wmma.accumulatork.inner.innertileAll': 1, 'dense.wmma.accumulator_i.c.outer': 1, 'dense.wmma.accumulator_j.c.outer': 1, 'dense.wmma.accumulator_k.outer': 1, 'dense.wmma.accumulator_i.c.inner.outer': 1, 'dense.wmma.accumulator_j.c.inner.outer': 1, 'dense.wmma.accumulator_k.inner.outer': 1, 'dense.wmma.accumulator_i.c.inner.inner.outer': 1, 'dense.wmma.accumulator_j.c.inner.inner.outer': 1, 'dense.wmma.accumulator_k.inner.inner.outer': 1, 'dense.wmma.accumulator_i.c.inner.inner.inner.outer': 1, 'dense.wmma.accumulator_j.c.inner.inner.inner.outer': 1, 'dense.wmma.accumulator_k.inner.inner.inner.outer': 1, 'dense.wmma.accumulator_i.c.inner.inner.inner.inner': 1, 'dense.wmma.accumulator_j.c.inner.inner.inner.inner': 1, 'dense.wmma.accumulator_k.inner.inner.inner.inner': 1, 'dense.wmma.accumulator_local_pos': 1, 'B.shared.wmma.matrix_b_ax0': 1, 'B.shared.wmma.matrix_b_ax1': 1, 'B.shared.wmma.matrix_b_ax0.outer': 1, 'B.shared.wmma.matrix_b_ax1.outer': 1, 'B.shared.wmma.matrix_b_ax0.inner': 1, 'B.shared.wmma.matrix_b_ax1.inner': 1, 'dense.wmma.accumulator_shared_pos': 1, 'B.shared_ax0': 1, 'B.shared_ax1': 1, 'B.shared_offset': 1, 'B.shared_vectorize': 1, 'threadIdx.x': 1, 'threadIdx.y': 1, 'A.shared.wmma.matrix_a_ax0': 1, 'A.shared.wmma.matrix_a_ax1': 1, 'A.shared.wmma.matrix_a_ax0.outer': 1, 'A.shared.wmma.matrix_a_ax1.outer': 1, 'A.shared.wmma.matrix_a_ax0.inner': 1, 'A.shared.wmma.matrix_a_ax1.inner': 1, 'A.shared_ax0': 1, 'A.shared_ax1': 1, 'A.shared_offset': 1, 'A.shared_vectorize': 1, 'dense_unroll_pragma': 1}
    
    
    only_in_a, only_in_b = compare_dicts(dict_a, dict_b)
    
    print("只在字典A中存在的键:", only_in_a)
    print("只在字典B中存在的键:", only_in_b)
    
    # 如果还想检查共有键但值不同的情况
    common_keys = set(dict_a.keys()) & set(dict_b.keys())
    different_values = {k: (dict_a[k], dict_b[k]) for k in common_keys if dict_a[k] != dict_b[k]}
    
    print("共有键但值不同的情况:", different_values)