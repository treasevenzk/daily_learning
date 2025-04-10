from ortools.sat.python import cp_model

model = cp_model.CpModel()
wmma_m = model.NewIntVar(8, 32, 'wmma_m')
wmma_m_cand0 = model.NewIntVar(0, 1, 'wmma_m_cand0')
wmma_m_cand1 = model.NewIntVar(0, 1, 'wmma_m_cand1')
wmma_m_cand2 = model.NewIntVar(0, 1, 'wmma_m_cand2')
wmma_k = model.NewIntVar(16, 16, 'wmma_k')
wmma_n = model.NewIntVar(8, 32, 'wmma_n')
wmma_n_cand0 = model.NewIntVar(0, 1, 'wmma_n_cand0')
wmma_n_cand1 = model.NewIntVar(0, 1, 'wmma_n_cand1')
wmma_n_cand2 = model.NewIntVar(0, 1, 'wmma_n_cand2')
wmma_m_wmma_n = model.NewIntVar(1, 4096, 'wmma_m_wmma_n')
dense_wmma_accumulator_shared_pos = model.NewIntVar(0, 3, 'dense_wmma_accumulator_shared_pos')
dense_wmma_accumulator_local_pos = model.NewIntVar(0, 3, 'dense_wmma_accumulator_local_pos')
dense_shared_pos = model.NewIntVar(1, 5, 'dense_shared_pos')
dense_wmma_accumulator_shared_local_pos = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_local_pos')
dense_unroll_pragma = model.NewIntVar(0, 5, 'dense_unroll_pragma')
dense_i_outer = model.NewIntVar(1, 1024, 'dense_i_outer')
dense_i_inner = model.NewIntVar(1, 1024, 'dense_i_inner')
dense_j_outer = model.NewIntVar(1, 8, 'dense_j_outer')
dense_j_inner = model.NewIntVar(1, 8, 'dense_j_inner')
densei_innertileSpatial = model.NewIntVar(1, 1024, 'densei_innertileSpatial')
dense_i_inner_outer = model.NewIntVar(1, 1024, 'dense_i_inner_outer')
dense_i_inner_inner = model.NewIntVar(1, 1024, 'dense_i_inner_inner')
densej_innertileSpatial = model.NewIntVar(1, 8, 'densej_innertileSpatial')
dense_j_inner_outer = model.NewIntVar(1, 8, 'dense_j_inner_outer')
dense_j_inner_inner = model.NewIntVar(1, 8, 'dense_j_inner_inner')
blockIdx_x = model.NewIntVar(1, 8888888, 'blockIdx_x')
dense_i_inner_outer_j_inner_outer_fused = model.NewIntVar(1, 1000000, 'dense_i_inner_outer_j_inner_outer_fused')
densei_inner_innertileSpatial = model.NewIntVar(1, 1024, 'densei_inner_innertileSpatial')
dense_i_inner_inner_outer = model.NewIntVar(1, 1024, 'dense_i_inner_inner_outer')
dense_i_inner_inner_inner = model.NewIntVar(1, 1024, 'dense_i_inner_inner_inner')
densej_inner_innertileSpatial = model.NewIntVar(1, 8, 'densej_inner_innertileSpatial')
dense_j_inner_inner_outer = model.NewIntVar(1, 8, 'dense_j_inner_inner_outer')
dense_j_inner_inner_inner = model.NewIntVar(1, 8, 'dense_j_inner_inner_inner')
threadIdx_y = model.NewIntVar(1, 1024, 'threadIdx_y')
dense_i_inner_inner_outer_j_inner_inner_outer_fused = model.NewIntVar(1, 1000000, 'dense_i_inner_inner_outer_j_inner_inner_outer_fused')
densei_inner_inner_innertileSpatial = model.NewIntVar(1, 1024, 'densei_inner_inner_innertileSpatial')
dense_i_inner_inner_inner_outer = model.NewIntVar(1, 1024, 'dense_i_inner_inner_inner_outer')
dense_i_inner_inner_inner_inner = model.NewIntVar(1, 1024, 'dense_i_inner_inner_inner_inner')
densej_inner_inner_innertileSpatial = model.NewIntVar(1, 8, 'densej_inner_inner_innertileSpatial')
dense_j_inner_inner_inner_outer = model.NewIntVar(1, 8, 'dense_j_inner_inner_inner_outer')
dense_j_inner_inner_inner_inner = model.NewIntVar(1, 8, 'dense_j_inner_inner_inner_inner')
threadIdx_x = model.NewIntVar(32, 32, 'threadIdx_x')
dense_i_inner_inner_inner_outer_j_inner_inner_inner_outer_fused = model.NewIntVar(1, 1000000, 'dense_i_inner_inner_inner_outer_j_inner_inner_inner_outer_fused')
dense_vectorize = model.NewIntVar(1, 8, 'dense_vectorize')
dense_vectorize_cand0 = model.NewIntVar(0, 1, 'dense_vectorize_cand0')
dense_vectorize_cand1 = model.NewIntVar(0, 1, 'dense_vectorize_cand1')
dense_vectorize_cand2 = model.NewIntVar(0, 1, 'dense_vectorize_cand2')
dense_vectorize_cand3 = model.NewIntVar(0, 1, 'dense_vectorize_cand3')
densei_inner_inner_inner_innertileSpatial = model.NewIntVar(1, 1024, 'densei_inner_inner_inner_innertileSpatial')
dense_i_inner_inner_inner_inner_outer = model.NewIntVar(1, 1024, 'dense_i_inner_inner_inner_inner_outer')
dense_i_inner_inner_inner_inner_inner = model.NewIntVar(1, 1024, 'dense_i_inner_inner_inner_inner_inner')
densej_inner_inner_inner_innertileSpatial = model.NewIntVar(1, 8, 'densej_inner_inner_inner_innertileSpatial')
dense_j_inner_inner_inner_inner_outer = model.NewIntVar(1, 8, 'dense_j_inner_inner_inner_inner_outer')
dense_j_inner_inner_inner_inner_inner = model.NewIntVar(1, 8, 'dense_j_inner_inner_inner_inner_inner')
dense_i_inner_inner_inner_inner_inner_j_inner_inner_inner_inner_inner_fused = model.NewIntVar(1, 1000000, 'dense_i_inner_inner_inner_inner_inner_j_inner_inner_inner_inner_inner_fused')
dense_wmma_accumulator_shared_ax0 = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_shared_ax0')
dense_shared_pos_select0 = model.NewIntVar(0, 1, 'dense_shared_pos_select0')
dense_shared_pos_select1 = model.NewIntVar(0, 1, 'dense_shared_pos_select1')
dense_shared_pos_select2 = model.NewIntVar(0, 1, 'dense_shared_pos_select2')
dense_shared_pos_select3 = model.NewIntVar(0, 1, 'dense_shared_pos_select3')
dense_shared_pos_select4 = model.NewIntVar(0, 1, 'dense_shared_pos_select4')
dense_shared_pos_select5 = model.NewIntVar(0, 1, 'dense_shared_pos_select5')
dense_wmma_accumulator_shared_ax1 = model.NewIntVar(1, 8, 'dense_wmma_accumulator_shared_ax1')
dense_wmma_accumulator_shared_offset = model.NewIntVar(0, 48, 'dense_wmma_accumulator_shared_offset')
dense_wmma_accumulator_shared_offset_cand0 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_offset_cand0')
dense_wmma_accumulator_shared_offset_cand1 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_offset_cand1')
dense_wmma_accumulator_shared_offset_cand2 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_offset_cand2')
dense_wmma_accumulator_shared_offset_cand3 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_offset_cand3')
dense_wmma_accumulator_shared_offset_cand4 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_offset_cand4')
dense_wmma_accumulator_shared_offset_cand5 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_offset_cand5')
dense_wmma_accumulator_shared_align_size = model.NewIntVar(1, 88888888, 'dense_wmma_accumulator_shared_align_size')
dense_wmma_accumulator_sharedax0tileSpatial = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_sharedax0tileSpatial')
dense_wmma_accumulator_shared_ax0_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_shared_ax0_outer')
dense_wmma_accumulator_shared_ax0_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_shared_ax0_inner')
dense_wmma_accumulator_sharedax1tileSpatial = model.NewIntVar(1, 8, 'dense_wmma_accumulator_sharedax1tileSpatial')
dense_wmma_accumulator_shared_ax1_outer = model.NewIntVar(1, 8, 'dense_wmma_accumulator_shared_ax1_outer')
dense_wmma_accumulator_shared_ax1_inner = model.NewIntVar(1, 8, 'dense_wmma_accumulator_shared_ax1_inner')
dense_wmma_accumulator_shared_ax0_outer_ax1_outer_fused = model.NewIntVar(1, 1000000, 'dense_wmma_accumulator_shared_ax0_outer_ax1_outer_fused')
dense_wmma_accumulator_shared_ax0_inner_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_shared_ax0_inner_outer')
dense_wmma_accumulator_shared_ax0_inner_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_shared_ax0_inner_inner')
dense_wmma_accumulator_shared_ax1_inner_outer = model.NewIntVar(1, 8, 'dense_wmma_accumulator_shared_ax1_inner_outer')
dense_wmma_accumulator_shared_ax1_inner_inner = model.NewIntVar(1, 8, 'dense_wmma_accumulator_shared_ax1_inner_inner')
dense_wmma_accumulator_i_c = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_i_c')
dense_wmma_accumulator_shared_local_pos_select0 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_local_pos_select0')
dense_wmma_accumulator_shared_local_pos_select1 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_local_pos_select1')
dense_wmma_accumulator_j_c = model.NewIntVar(1, 8, 'dense_wmma_accumulator_j_c')
dense_wmma_accumulatori_ctileAll = model.NewIntVar(1, 1024, 'dense_wmma_accumulatori_ctileAll')
dense_wmma_accumulator_i_c_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_i_c_outer')
dense_wmma_accumulator_i_c_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_i_c_inner')
dense_wmma_accumulatorj_ctileAll = model.NewIntVar(1, 8, 'dense_wmma_accumulatorj_ctileAll')
dense_wmma_accumulator_j_c_outer = model.NewIntVar(1, 8, 'dense_wmma_accumulator_j_c_outer')
dense_wmma_accumulator_j_c_inner = model.NewIntVar(1, 8, 'dense_wmma_accumulator_j_c_inner')
dense_wmma_accumulatorktileAll = model.NewIntVar(1, 1024, 'dense_wmma_accumulatorktileAll')
dense_wmma_accumulator_k_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_k_outer')
dense_wmma_accumulator_k_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_k_inner')
dense_wmma_accumulatori_c_innertileAll = model.NewIntVar(1, 1024, 'dense_wmma_accumulatori_c_innertileAll')
dense_wmma_accumulator_i_c_inner_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_i_c_inner_outer')
dense_wmma_accumulator_i_c_inner_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_i_c_inner_inner')
dense_wmma_accumulatorj_c_innertileAll = model.NewIntVar(1, 8, 'dense_wmma_accumulatorj_c_innertileAll')
dense_wmma_accumulator_j_c_inner_outer = model.NewIntVar(1, 8, 'dense_wmma_accumulator_j_c_inner_outer')
dense_wmma_accumulator_j_c_inner_inner = model.NewIntVar(1, 8, 'dense_wmma_accumulator_j_c_inner_inner')
dense_wmma_accumulatork_innertileAll = model.NewIntVar(1, 1024, 'dense_wmma_accumulatork_innertileAll')
dense_wmma_accumulator_k_inner_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_k_inner_outer')
dense_wmma_accumulator_k_inner_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_k_inner_inner')
dense_wmma_accumulatori_c_inner_innertileAll = model.NewIntVar(1, 1024, 'dense_wmma_accumulatori_c_inner_innertileAll')
dense_wmma_accumulator_i_c_inner_inner_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_i_c_inner_inner_outer')
dense_wmma_accumulator_i_c_inner_inner_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_i_c_inner_inner_inner')
dense_wmma_accumulatorj_c_inner_innertileAll = model.NewIntVar(1, 8, 'dense_wmma_accumulatorj_c_inner_innertileAll')
dense_wmma_accumulator_j_c_inner_inner_outer = model.NewIntVar(1, 8, 'dense_wmma_accumulator_j_c_inner_inner_outer')
dense_wmma_accumulator_j_c_inner_inner_inner = model.NewIntVar(1, 8, 'dense_wmma_accumulator_j_c_inner_inner_inner')
dense_wmma_accumulatork_inner_innertileAll = model.NewIntVar(1, 1024, 'dense_wmma_accumulatork_inner_innertileAll')
dense_wmma_accumulator_k_inner_inner_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_k_inner_inner_outer')
dense_wmma_accumulator_k_inner_inner_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_k_inner_inner_inner')
dense_wmma_accumulator_i_c_inner_inner_inner_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_i_c_inner_inner_inner_outer')
dense_wmma_accumulator_i_c_inner_inner_inner_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_i_c_inner_inner_inner_inner')
dense_wmma_accumulator_j_c_inner_inner_inner_outer = model.NewIntVar(1, 8, 'dense_wmma_accumulator_j_c_inner_inner_inner_outer')
dense_wmma_accumulator_j_c_inner_inner_inner_inner = model.NewIntVar(1, 8, 'dense_wmma_accumulator_j_c_inner_inner_inner_inner')
dense_wmma_accumulator_k_inner_inner_inner_outer = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_k_inner_inner_inner_outer')
dense_wmma_accumulator_k_inner_inner_inner_inner = model.NewIntVar(1, 1024, 'dense_wmma_accumulator_k_inner_inner_inner_inner')
BPad_shared_wmma_matrix_b_ax0 = model.NewIntVar(1, 8, 'BPad_shared_wmma_matrix_b_ax0')
BPad_shared_wmma_matrix_b_ax0_outer = model.NewIntVar(1, 8, 'BPad_shared_wmma_matrix_b_ax0_outer')
BPad_shared_wmma_matrix_b_ax0_inner = model.NewIntVar(1, 8, 'BPad_shared_wmma_matrix_b_ax0_inner')
BPad_shared_wmma_matrix_b_ax1 = model.NewIntVar(1, 1024, 'BPad_shared_wmma_matrix_b_ax1')
BPad_shared_wmma_matrix_b_ax1_outer = model.NewIntVar(1, 1024, 'BPad_shared_wmma_matrix_b_ax1_outer')
BPad_shared_wmma_matrix_b_ax1_inner = model.NewIntVar(1, 1024, 'BPad_shared_wmma_matrix_b_ax1_inner')
BPad_shared_ax0 = model.NewIntVar(1, 8, 'BPad_shared_ax0')
BPad_shared_tmp_ax0 = model.NewIntVar(1, 8, 'BPad_shared_tmp_ax0')
dense_wmma_accumulator_shared_pos_select0 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_pos_select0')
dense_wmma_accumulator_shared_pos_select1 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_pos_select1')
dense_wmma_accumulator_shared_pos_select2 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_pos_select2')
dense_wmma_accumulator_shared_pos_select3 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_shared_pos_select3')
BPad_shared_ax1 = model.NewIntVar(1, 1024, 'BPad_shared_ax1')
BPad_shared_offset = model.NewIntVar(0, 48, 'BPad_shared_offset')
BPad_shared_offset_cand0 = model.NewIntVar(0, 1, 'BPad_shared_offset_cand0')
BPad_shared_offset_cand1 = model.NewIntVar(0, 1, 'BPad_shared_offset_cand1')
BPad_shared_offset_cand2 = model.NewIntVar(0, 1, 'BPad_shared_offset_cand2')
BPad_shared_offset_cand3 = model.NewIntVar(0, 1, 'BPad_shared_offset_cand3')
BPad_shared_offset_cand4 = model.NewIntVar(0, 1, 'BPad_shared_offset_cand4')
BPad_shared_offset_cand5 = model.NewIntVar(0, 1, 'BPad_shared_offset_cand5')
BPad_shared_align_size = model.NewIntVar(1, 88888888, 'BPad_shared_align_size')
BPad_shared_ax0_ax1_fused = model.NewIntVar(1, 1000000, 'BPad_shared_ax0_ax1_fused')
BPad_shared_vectorize = model.NewIntVar(1, 8, 'BPad_shared_vectorize')
BPad_shared_vectorize_cand0 = model.NewIntVar(0, 1, 'BPad_shared_vectorize_cand0')
BPad_shared_vectorize_cand1 = model.NewIntVar(0, 1, 'BPad_shared_vectorize_cand1')
BPad_shared_vectorize_cand2 = model.NewIntVar(0, 1, 'BPad_shared_vectorize_cand2')
BPad_shared_vectorize_cand3 = model.NewIntVar(0, 1, 'BPad_shared_vectorize_cand3')
A_shared_wmma_matrix_a_ax0 = model.NewIntVar(1, 1024, 'A_shared_wmma_matrix_a_ax0')
dense_wmma_accumulator_local_pos_select0 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_local_pos_select0')
dense_wmma_accumulator_local_pos_select1 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_local_pos_select1')
dense_wmma_accumulator_local_pos_select2 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_local_pos_select2')
dense_wmma_accumulator_local_pos_select3 = model.NewIntVar(0, 1, 'dense_wmma_accumulator_local_pos_select3')
A_shared_wmma_matrix_a_ax1 = model.NewIntVar(1, 1024, 'A_shared_wmma_matrix_a_ax1')
A_shared_wmma_matrix_a_ax0_outer = model.NewIntVar(1, 1024, 'A_shared_wmma_matrix_a_ax0_outer')
A_shared_wmma_matrix_a_ax0_inner = model.NewIntVar(1, 1024, 'A_shared_wmma_matrix_a_ax0_inner')
A_shared_wmma_matrix_a_ax1_outer = model.NewIntVar(1, 1024, 'A_shared_wmma_matrix_a_ax1_outer')
A_shared_wmma_matrix_a_ax1_inner = model.NewIntVar(1, 1024, 'A_shared_wmma_matrix_a_ax1_inner')
A_shared_ax0 = model.NewIntVar(1, 1024, 'A_shared_ax0')
A_shared_tmp_ax0 = model.NewIntVar(1, 1024, 'A_shared_tmp_ax0')
A_shared_ax1 = model.NewIntVar(1, 1024, 'A_shared_ax1')
A_shared_offset = model.NewIntVar(0, 48, 'A_shared_offset')
A_shared_offset_cand0 = model.NewIntVar(0, 1, 'A_shared_offset_cand0')
A_shared_offset_cand1 = model.NewIntVar(0, 1, 'A_shared_offset_cand1')
A_shared_offset_cand2 = model.NewIntVar(0, 1, 'A_shared_offset_cand2')
A_shared_offset_cand3 = model.NewIntVar(0, 1, 'A_shared_offset_cand3')
A_shared_offset_cand4 = model.NewIntVar(0, 1, 'A_shared_offset_cand4')
A_shared_offset_cand5 = model.NewIntVar(0, 1, 'A_shared_offset_cand5')
A_shared_align_size = model.NewIntVar(1, 88888888, 'A_shared_align_size')
A_shared_ax0_ax1_fused = model.NewIntVar(1, 1000000, 'A_shared_ax0_ax1_fused')
A_shared_vectorize = model.NewIntVar(1, 8, 'A_shared_vectorize')
A_shared_vectorize_cand0 = model.NewIntVar(0, 1, 'A_shared_vectorize_cand0')
A_shared_vectorize_cand1 = model.NewIntVar(0, 1, 'A_shared_vectorize_cand1')
A_shared_vectorize_cand2 = model.NewIntVar(0, 1, 'A_shared_vectorize_cand2')
A_shared_vectorize_cand3 = model.NewIntVar(0, 1, 'A_shared_vectorize_cand3')
threads = model.NewIntVar(1, 1024, 'threads')
A_shared_shared_mem_size = model.NewIntVar(1, 24576, 'A_shared_shared_mem_size')
BPad_shared_shared_mem_size = model.NewIntVar(1, 24576, 'BPad_shared_shared_mem_size')
dense_wmma_accumulator_shared_shared_mem_size = model.NewIntVar(1, 24576, 'dense_wmma_accumulator_shared_shared_mem_size')
model.Add(wmma_m == 8).OnlyEnforceIf(wmma_m_cand0)
model.Add(wmma_m == 16).OnlyEnforceIf(wmma_m_cand1)
model.Add(wmma_m == 32).OnlyEnforceIf(wmma_m_cand2)
model.Add(sum([wmma_m_cand0, wmma_m_cand1, wmma_m_cand2]) == 1)
model.Add(wmma_n == 8).OnlyEnforceIf(wmma_n_cand0)
model.Add(wmma_n == 16).OnlyEnforceIf(wmma_n_cand1)
model.Add(wmma_n == 32).OnlyEnforceIf(wmma_n_cand2)
model.Add(sum([wmma_n_cand0, wmma_n_cand1, wmma_n_cand2]) == 1)
model.AddMultiplicationEquality(wmma_m_wmma_n, [wmma_m, wmma_n])
model.AddMultiplicationEquality(4096, [wmma_m_wmma_n, wmma_k])
model.AddMultiplicationEquality(1024, [dense_i_outer, dense_i_inner])
model.Add(dense_i_outer == 1)
model.AddMultiplicationEquality(8, [dense_j_outer, dense_j_inner])
model.Add(dense_j_outer == 1)
model.AddMultiplicationEquality(dense_i_inner, [dense_i_inner_outer, dense_i_inner_inner])
model.Add(dense_i_inner_outer == densei_innertileSpatial)
model.AddMultiplicationEquality(dense_j_inner, [dense_j_inner_outer, dense_j_inner_inner])
model.Add(dense_j_inner_outer == densej_innertileSpatial)
model.AddMultiplicationEquality(blockIdx_x, [densei_innertileSpatial, densej_innertileSpatial])
model.AddMultiplicationEquality(dense_i_inner_outer_j_inner_outer_fused, [dense_i_inner_outer, dense_j_inner_outer])
model.AddMultiplicationEquality(dense_i_inner_inner, [dense_i_inner_inner_outer, dense_i_inner_inner_inner])
model.Add(dense_i_inner_inner_outer == densei_inner_innertileSpatial)
model.AddMultiplicationEquality(dense_j_inner_inner, [dense_j_inner_inner_outer, dense_j_inner_inner_inner])
model.Add(dense_j_inner_inner_outer == densej_inner_innertileSpatial)
model.AddMultiplicationEquality(threadIdx_y, [densei_inner_innertileSpatial, densej_inner_innertileSpatial])
model.AddMultiplicationEquality(dense_i_inner_inner_outer_j_inner_inner_outer_fused, [dense_i_inner_inner_outer, dense_j_inner_inner_outer])
model.AddMultiplicationEquality(dense_i_inner_inner_inner, [dense_i_inner_inner_inner_outer, dense_i_inner_inner_inner_inner])
model.Add(dense_i_inner_inner_inner_outer == densei_inner_inner_innertileSpatial)
model.AddMultiplicationEquality(dense_j_inner_inner_inner, [dense_j_inner_inner_inner_outer, dense_j_inner_inner_inner_inner])
model.Add(dense_j_inner_inner_inner_outer == densej_inner_inner_innertileSpatial)
model.AddMultiplicationEquality(threadIdx_x, [densei_inner_inner_innertileSpatial, densej_inner_inner_innertileSpatial])
model.AddMultiplicationEquality(dense_i_inner_inner_inner_outer_j_inner_inner_inner_outer_fused, [dense_i_inner_inner_inner_outer, dense_j_inner_inner_inner_outer])
model.Add(dense_vectorize == 1).OnlyEnforceIf(dense_vectorize_cand0)
model.Add(dense_vectorize == 2).OnlyEnforceIf(dense_vectorize_cand1)
model.Add(dense_vectorize == 4).OnlyEnforceIf(dense_vectorize_cand2)
model.Add(dense_vectorize == 8).OnlyEnforceIf(dense_vectorize_cand3)
model.Add(sum([dense_vectorize_cand0, dense_vectorize_cand1, dense_vectorize_cand2, dense_vectorize_cand3]) == 1)
model.AddMultiplicationEquality(dense_i_inner_inner_inner_inner, [dense_i_inner_inner_inner_inner_outer, dense_i_inner_inner_inner_inner_inner])
model.Add(dense_i_inner_inner_inner_inner_outer == densei_inner_inner_inner_innertileSpatial)
model.AddMultiplicationEquality(dense_j_inner_inner_inner_inner, [dense_j_inner_inner_inner_inner_outer, dense_j_inner_inner_inner_inner_inner])
model.Add(dense_j_inner_inner_inner_inner_outer == densej_inner_inner_inner_innertileSpatial)
model.AddMultiplicationEquality(dense_i_inner_inner_inner_inner_inner_j_inner_inner_inner_inner_inner_fused, [dense_i_inner_inner_inner_inner_inner, dense_j_inner_inner_inner_inner_inner])
model.Add(dense_i_inner_inner_inner_inner_inner_j_inner_inner_inner_inner_inner_fused == dense_vectorize)
model.Add(dense_shared_pos != 0)
model.Add(dense_wmma_accumulator_shared_ax0 == dense_i_inner).OnlyEnforceIf(dense_shared_pos_select0)
model.Add(dense_shared_pos == 0).OnlyEnforceIf(dense_shared_pos_select0)
model.Add(dense_wmma_accumulator_shared_ax0 == dense_i_inner_inner).OnlyEnforceIf(dense_shared_pos_select1)
model.Add(dense_shared_pos == 1).OnlyEnforceIf(dense_shared_pos_select1)
model.Add(dense_wmma_accumulator_shared_ax0 == dense_i_inner_inner_inner).OnlyEnforceIf(dense_shared_pos_select2)
model.Add(dense_shared_pos == 2).OnlyEnforceIf(dense_shared_pos_select2)
model.Add(dense_wmma_accumulator_shared_ax0 == dense_i_inner_inner_inner_inner).OnlyEnforceIf(dense_shared_pos_select3)
model.Add(dense_shared_pos == 3).OnlyEnforceIf(dense_shared_pos_select3)
model.Add(dense_wmma_accumulator_shared_ax0 == dense_i_inner_inner_inner_inner_inner).OnlyEnforceIf(dense_shared_pos_select4)
model.Add(dense_shared_pos == 4).OnlyEnforceIf(dense_shared_pos_select4)
model.Add(dense_wmma_accumulator_shared_ax0 == 1).OnlyEnforceIf(dense_shared_pos_select5)
model.Add(dense_shared_pos == 5).OnlyEnforceIf(dense_shared_pos_select5)
model.Add(sum([dense_shared_pos_select0, dense_shared_pos_select1, dense_shared_pos_select2, dense_shared_pos_select3, dense_shared_pos_select4, dense_shared_pos_select5]) == 1)
model.Add(dense_wmma_accumulator_shared_ax1 == dense_j_inner).OnlyEnforceIf(dense_shared_pos_select0)
model.Add(dense_shared_pos == 0).OnlyEnforceIf(dense_shared_pos_select0)
model.Add(dense_wmma_accumulator_shared_ax1 == dense_j_inner_inner).OnlyEnforceIf(dense_shared_pos_select1)
model.Add(dense_shared_pos == 1).OnlyEnforceIf(dense_shared_pos_select1)
model.Add(dense_wmma_accumulator_shared_ax1 == dense_j_inner_inner_inner).OnlyEnforceIf(dense_shared_pos_select2)
model.Add(dense_shared_pos == 2).OnlyEnforceIf(dense_shared_pos_select2)
model.Add(dense_wmma_accumulator_shared_ax1 == dense_j_inner_inner_inner_inner).OnlyEnforceIf(dense_shared_pos_select3)
model.Add(dense_shared_pos == 3).OnlyEnforceIf(dense_shared_pos_select3)
model.Add(dense_wmma_accumulator_shared_ax1 == dense_j_inner_inner_inner_inner_inner).OnlyEnforceIf(dense_shared_pos_select4)
model.Add(dense_shared_pos == 4).OnlyEnforceIf(dense_shared_pos_select4)
model.Add(dense_wmma_accumulator_shared_ax1 == 1).OnlyEnforceIf(dense_shared_pos_select5)
model.Add(dense_shared_pos == 5).OnlyEnforceIf(dense_shared_pos_select5)
model.Add(sum([dense_shared_pos_select0, dense_shared_pos_select1, dense_shared_pos_select2, dense_shared_pos_select3, dense_shared_pos_select4, dense_shared_pos_select5]) == 1)
model.Add(dense_wmma_accumulator_shared_offset == 0).OnlyEnforceIf(dense_wmma_accumulator_shared_offset_cand0)
model.Add(dense_wmma_accumulator_shared_offset == 8).OnlyEnforceIf(dense_wmma_accumulator_shared_offset_cand1)
model.Add(dense_wmma_accumulator_shared_offset == 16).OnlyEnforceIf(dense_wmma_accumulator_shared_offset_cand2)
model.Add(dense_wmma_accumulator_shared_offset == 24).OnlyEnforceIf(dense_wmma_accumulator_shared_offset_cand3)
model.Add(dense_wmma_accumulator_shared_offset == 32).OnlyEnforceIf(dense_wmma_accumulator_shared_offset_cand4)
model.Add(dense_wmma_accumulator_shared_offset == 48).OnlyEnforceIf(dense_wmma_accumulator_shared_offset_cand5)
model.Add(sum([dense_wmma_accumulator_shared_offset_cand0, dense_wmma_accumulator_shared_offset_cand1, dense_wmma_accumulator_shared_offset_cand2, dense_wmma_accumulator_shared_offset_cand3, dense_wmma_accumulator_shared_offset_cand4, dense_wmma_accumulator_shared_offset_cand5]) == 1)
model.Add(sum([dense_wmma_accumulator_shared_ax1, dense_wmma_accumulator_shared_offset]) == dense_wmma_accumulator_shared_align_size)
model.AddMultiplicationEquality(dense_wmma_accumulator_shared_ax0, [dense_wmma_accumulator_shared_ax0_outer, dense_wmma_accumulator_shared_ax0_inner])
model.Add(dense_wmma_accumulator_shared_ax0_outer == dense_wmma_accumulator_sharedax0tileSpatial)
model.AddMultiplicationEquality(dense_wmma_accumulator_shared_ax1, [dense_wmma_accumulator_shared_ax1_outer, dense_wmma_accumulator_shared_ax1_inner])
model.Add(dense_wmma_accumulator_shared_ax1_outer == dense_wmma_accumulator_sharedax1tileSpatial)
model.AddMultiplicationEquality(threadIdx_y, [dense_wmma_accumulator_sharedax0tileSpatial, dense_wmma_accumulator_sharedax1tileSpatial])
model.AddMultiplicationEquality(dense_wmma_accumulator_shared_ax0_outer_ax1_outer_fused, [dense_wmma_accumulator_shared_ax0_outer, dense_wmma_accumulator_shared_ax1_outer])
model.AddMultiplicationEquality(dense_wmma_accumulator_shared_ax0_inner, [dense_wmma_accumulator_shared_ax0_inner_outer, dense_wmma_accumulator_shared_ax0_inner_inner])
model.Add(dense_wmma_accumulator_shared_ax0_inner_inner == wmma_m)
model.AddMultiplicationEquality(dense_wmma_accumulator_shared_ax1_inner, [dense_wmma_accumulator_shared_ax1_inner_outer, dense_wmma_accumulator_shared_ax1_inner_inner])
model.Add(dense_wmma_accumulator_shared_ax1_inner_inner == wmma_n)
model.Add(dense_wmma_accumulator_i_c == dense_wmma_accumulator_shared_ax0_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_local_pos_select0)
model.Add(dense_wmma_accumulator_shared_local_pos == 0).OnlyEnforceIf(dense_wmma_accumulator_shared_local_pos_select0)
model.Add(dense_wmma_accumulator_i_c == dense_wmma_accumulator_shared_ax0_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_local_pos_select1)
model.Add(dense_wmma_accumulator_shared_local_pos == 1).OnlyEnforceIf(dense_wmma_accumulator_shared_local_pos_select1)
model.Add(sum([dense_wmma_accumulator_shared_local_pos_select0, dense_wmma_accumulator_shared_local_pos_select1]) == 1)
model.Add(dense_wmma_accumulator_j_c == dense_wmma_accumulator_shared_ax1_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_local_pos_select0)
model.Add(dense_wmma_accumulator_shared_local_pos == 0).OnlyEnforceIf(dense_wmma_accumulator_shared_local_pos_select0)
model.Add(dense_wmma_accumulator_j_c == dense_wmma_accumulator_shared_ax1_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_local_pos_select1)
model.Add(dense_wmma_accumulator_shared_local_pos == 1).OnlyEnforceIf(dense_wmma_accumulator_shared_local_pos_select1)
model.Add(sum([dense_wmma_accumulator_shared_local_pos_select0, dense_wmma_accumulator_shared_local_pos_select1]) == 1)
model.AddMultiplicationEquality(dense_wmma_accumulator_i_c, [dense_wmma_accumulator_i_c_outer, dense_wmma_accumulator_i_c_inner])
model.Add(dense_wmma_accumulator_i_c_outer == dense_wmma_accumulatori_ctileAll)
model.AddMultiplicationEquality(dense_wmma_accumulator_j_c, [dense_wmma_accumulator_j_c_outer, dense_wmma_accumulator_j_c_inner])
model.Add(dense_wmma_accumulator_j_c_outer == dense_wmma_accumulatorj_ctileAll)
model.AddMultiplicationEquality(1024, [dense_wmma_accumulator_k_outer, dense_wmma_accumulator_k_inner])
model.Add(dense_wmma_accumulator_k_outer == dense_wmma_accumulatorktileAll)
model.AddMultiplicationEquality(dense_wmma_accumulator_i_c_inner, [dense_wmma_accumulator_i_c_inner_outer, dense_wmma_accumulator_i_c_inner_inner])
model.Add(dense_wmma_accumulator_i_c_inner_outer == dense_wmma_accumulatori_c_innertileAll)
model.AddMultiplicationEquality(dense_wmma_accumulator_j_c_inner, [dense_wmma_accumulator_j_c_inner_outer, dense_wmma_accumulator_j_c_inner_inner])
model.Add(dense_wmma_accumulator_j_c_inner_outer == dense_wmma_accumulatorj_c_innertileAll)
model.AddMultiplicationEquality(dense_wmma_accumulator_k_inner, [dense_wmma_accumulator_k_inner_outer, dense_wmma_accumulator_k_inner_inner])
model.Add(dense_wmma_accumulator_k_inner_outer == dense_wmma_accumulatork_innertileAll)
model.AddMultiplicationEquality(dense_wmma_accumulator_i_c_inner_inner, [dense_wmma_accumulator_i_c_inner_inner_outer, dense_wmma_accumulator_i_c_inner_inner_inner])
model.Add(dense_wmma_accumulator_i_c_inner_inner_outer == dense_wmma_accumulatori_c_inner_innertileAll)
model.AddMultiplicationEquality(dense_wmma_accumulator_j_c_inner_inner, [dense_wmma_accumulator_j_c_inner_inner_outer, dense_wmma_accumulator_j_c_inner_inner_inner])
model.Add(dense_wmma_accumulator_j_c_inner_inner_outer == dense_wmma_accumulatorj_c_inner_innertileAll)
model.AddMultiplicationEquality(dense_wmma_accumulator_k_inner_inner, [dense_wmma_accumulator_k_inner_inner_outer, dense_wmma_accumulator_k_inner_inner_inner])
model.Add(dense_wmma_accumulator_k_inner_inner_outer == dense_wmma_accumulatork_inner_innertileAll)
model.AddMultiplicationEquality(dense_wmma_accumulator_i_c_inner_inner_inner, [dense_wmma_accumulator_i_c_inner_inner_inner_outer, dense_wmma_accumulator_i_c_inner_inner_inner_inner])
model.Add(dense_wmma_accumulator_i_c_inner_inner_inner_inner == wmma_m)
model.AddMultiplicationEquality(dense_wmma_accumulator_j_c_inner_inner_inner, [dense_wmma_accumulator_j_c_inner_inner_inner_outer, dense_wmma_accumulator_j_c_inner_inner_inner_inner])
model.Add(dense_wmma_accumulator_j_c_inner_inner_inner_inner == wmma_n)
model.AddMultiplicationEquality(dense_wmma_accumulator_k_inner_inner_inner, [dense_wmma_accumulator_k_inner_inner_inner_outer, dense_wmma_accumulator_k_inner_inner_inner_inner])
model.Add(dense_wmma_accumulator_k_inner_inner_inner_inner == wmma_k)
model.Add(dense_wmma_accumulator_shared_pos <= dense_wmma_accumulator_local_pos)
model.AddMultiplicationEquality(BPad_shared_wmma_matrix_b_ax0, [BPad_shared_wmma_matrix_b_ax0_outer, BPad_shared_wmma_matrix_b_ax0_inner])
model.Add(BPad_shared_wmma_matrix_b_ax0_inner == wmma_n)
model.AddMultiplicationEquality(BPad_shared_wmma_matrix_b_ax1, [BPad_shared_wmma_matrix_b_ax1_outer, BPad_shared_wmma_matrix_b_ax1_inner])
model.Add(BPad_shared_wmma_matrix_b_ax1_inner == wmma_k)
model.Add(BPad_shared_tmp_ax0 == dense_wmma_accumulator_j_c_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select0)
model.Add(dense_wmma_accumulator_shared_pos == 0).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select0)
model.Add(BPad_shared_tmp_ax0 == dense_wmma_accumulator_j_c_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select1)
model.Add(dense_wmma_accumulator_shared_pos == 1).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select1)
model.Add(BPad_shared_tmp_ax0 == dense_wmma_accumulator_j_c_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select2)
model.Add(dense_wmma_accumulator_shared_pos == 2).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select2)
model.Add(BPad_shared_tmp_ax0 == dense_wmma_accumulator_j_c_inner_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select3)
model.Add(dense_wmma_accumulator_shared_pos == 3).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select3)
model.Add(sum([dense_wmma_accumulator_shared_pos_select0, dense_wmma_accumulator_shared_pos_select1, dense_wmma_accumulator_shared_pos_select2, dense_wmma_accumulator_shared_pos_select3]) == 1)
model.AddMultiplicationEquality(BPad_shared_ax0, [BPad_shared_tmp_ax0, dense_wmma_accumulator_sharedax1tileSpatial])
model.Add(BPad_shared_ax1 == dense_wmma_accumulator_k_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select0)
model.Add(dense_wmma_accumulator_shared_pos == 0).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select0)
model.Add(BPad_shared_ax1 == dense_wmma_accumulator_k_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select1)
model.Add(dense_wmma_accumulator_shared_pos == 1).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select1)
model.Add(BPad_shared_ax1 == dense_wmma_accumulator_k_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select2)
model.Add(dense_wmma_accumulator_shared_pos == 2).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select2)
model.Add(BPad_shared_ax1 == dense_wmma_accumulator_k_inner_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select3)
model.Add(dense_wmma_accumulator_shared_pos == 3).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select3)
model.Add(sum([dense_wmma_accumulator_shared_pos_select0, dense_wmma_accumulator_shared_pos_select1, dense_wmma_accumulator_shared_pos_select2, dense_wmma_accumulator_shared_pos_select3]) == 1)
model.Add(BPad_shared_offset == 0).OnlyEnforceIf(BPad_shared_offset_cand0)
model.Add(BPad_shared_offset == 8).OnlyEnforceIf(BPad_shared_offset_cand1)
model.Add(BPad_shared_offset == 16).OnlyEnforceIf(BPad_shared_offset_cand2)
model.Add(BPad_shared_offset == 24).OnlyEnforceIf(BPad_shared_offset_cand3)
model.Add(BPad_shared_offset == 32).OnlyEnforceIf(BPad_shared_offset_cand4)
model.Add(BPad_shared_offset == 48).OnlyEnforceIf(BPad_shared_offset_cand5)
model.Add(sum([BPad_shared_offset_cand0, BPad_shared_offset_cand1, BPad_shared_offset_cand2, BPad_shared_offset_cand3, BPad_shared_offset_cand4, BPad_shared_offset_cand5]) == 1)
model.Add(sum([BPad_shared_ax1, BPad_shared_offset]) == BPad_shared_align_size)
model.AddMultiplicationEquality(BPad_shared_ax0_ax1_fused, [BPad_shared_ax0, BPad_shared_ax1])
model.Add(BPad_shared_vectorize == 1).OnlyEnforceIf(BPad_shared_vectorize_cand0)
model.Add(BPad_shared_vectorize == 2).OnlyEnforceIf(BPad_shared_vectorize_cand1)
model.Add(BPad_shared_vectorize == 4).OnlyEnforceIf(BPad_shared_vectorize_cand2)
model.Add(BPad_shared_vectorize == 8).OnlyEnforceIf(BPad_shared_vectorize_cand3)
model.Add(sum([BPad_shared_vectorize_cand0, BPad_shared_vectorize_cand1, BPad_shared_vectorize_cand2, BPad_shared_vectorize_cand3]) == 1)
model.Add(A_shared_wmma_matrix_a_ax0 == dense_wmma_accumulator_i_c_inner).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select0)
model.Add(dense_wmma_accumulator_local_pos == 0).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select0)
model.Add(A_shared_wmma_matrix_a_ax0 == dense_wmma_accumulator_i_c_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select1)
model.Add(dense_wmma_accumulator_local_pos == 1).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select1)
model.Add(A_shared_wmma_matrix_a_ax0 == dense_wmma_accumulator_i_c_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select2)
model.Add(dense_wmma_accumulator_local_pos == 2).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select2)
model.Add(A_shared_wmma_matrix_a_ax0 == dense_wmma_accumulator_i_c_inner_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select3)
model.Add(dense_wmma_accumulator_local_pos == 3).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select3)
model.Add(sum([dense_wmma_accumulator_local_pos_select0, dense_wmma_accumulator_local_pos_select1, dense_wmma_accumulator_local_pos_select2, dense_wmma_accumulator_local_pos_select3]) == 1)
model.Add(A_shared_wmma_matrix_a_ax1 == dense_wmma_accumulator_k_inner).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select0)
model.Add(dense_wmma_accumulator_local_pos == 0).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select0)
model.Add(A_shared_wmma_matrix_a_ax1 == dense_wmma_accumulator_k_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select1)
model.Add(dense_wmma_accumulator_local_pos == 1).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select1)
model.Add(A_shared_wmma_matrix_a_ax1 == dense_wmma_accumulator_k_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select2)
model.Add(dense_wmma_accumulator_local_pos == 2).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select2)
model.Add(A_shared_wmma_matrix_a_ax1 == dense_wmma_accumulator_k_inner_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select3)
model.Add(dense_wmma_accumulator_local_pos == 3).OnlyEnforceIf(dense_wmma_accumulator_local_pos_select3)
model.Add(sum([dense_wmma_accumulator_local_pos_select0, dense_wmma_accumulator_local_pos_select1, dense_wmma_accumulator_local_pos_select2, dense_wmma_accumulator_local_pos_select3]) == 1)
model.AddMultiplicationEquality(A_shared_wmma_matrix_a_ax0, [A_shared_wmma_matrix_a_ax0_outer, A_shared_wmma_matrix_a_ax0_inner])
model.Add(A_shared_wmma_matrix_a_ax0_inner == wmma_m)
model.AddMultiplicationEquality(A_shared_wmma_matrix_a_ax1, [A_shared_wmma_matrix_a_ax1_outer, A_shared_wmma_matrix_a_ax1_inner])
model.Add(A_shared_wmma_matrix_a_ax1_inner == wmma_k)
model.Add(A_shared_tmp_ax0 == dense_wmma_accumulator_i_c_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select0)
model.Add(dense_wmma_accumulator_shared_pos == 0).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select0)
model.Add(A_shared_tmp_ax0 == dense_wmma_accumulator_i_c_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select1)
model.Add(dense_wmma_accumulator_shared_pos == 1).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select1)
model.Add(A_shared_tmp_ax0 == dense_wmma_accumulator_i_c_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select2)
model.Add(dense_wmma_accumulator_shared_pos == 2).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select2)
model.Add(A_shared_tmp_ax0 == dense_wmma_accumulator_i_c_inner_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select3)
model.Add(dense_wmma_accumulator_shared_pos == 3).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select3)
model.Add(sum([dense_wmma_accumulator_shared_pos_select0, dense_wmma_accumulator_shared_pos_select1, dense_wmma_accumulator_shared_pos_select2, dense_wmma_accumulator_shared_pos_select3]) == 1)
model.AddMultiplicationEquality(A_shared_ax0, [A_shared_tmp_ax0, dense_wmma_accumulator_sharedax0tileSpatial])
model.Add(A_shared_ax1 == dense_wmma_accumulator_k_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select0)
model.Add(dense_wmma_accumulator_shared_pos == 0).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select0)
model.Add(A_shared_ax1 == dense_wmma_accumulator_k_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select1)
model.Add(dense_wmma_accumulator_shared_pos == 1).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select1)
model.Add(A_shared_ax1 == dense_wmma_accumulator_k_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select2)
model.Add(dense_wmma_accumulator_shared_pos == 2).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select2)
model.Add(A_shared_ax1 == dense_wmma_accumulator_k_inner_inner_inner_inner).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select3)
model.Add(dense_wmma_accumulator_shared_pos == 3).OnlyEnforceIf(dense_wmma_accumulator_shared_pos_select3)
model.Add(sum([dense_wmma_accumulator_shared_pos_select0, dense_wmma_accumulator_shared_pos_select1, dense_wmma_accumulator_shared_pos_select2, dense_wmma_accumulator_shared_pos_select3]) == 1)
model.Add(A_shared_offset == 0).OnlyEnforceIf(A_shared_offset_cand0)
model.Add(A_shared_offset == 8).OnlyEnforceIf(A_shared_offset_cand1)
model.Add(A_shared_offset == 16).OnlyEnforceIf(A_shared_offset_cand2)
model.Add(A_shared_offset == 24).OnlyEnforceIf(A_shared_offset_cand3)
model.Add(A_shared_offset == 32).OnlyEnforceIf(A_shared_offset_cand4)
model.Add(A_shared_offset == 48).OnlyEnforceIf(A_shared_offset_cand5)
model.Add(sum([A_shared_offset_cand0, A_shared_offset_cand1, A_shared_offset_cand2, A_shared_offset_cand3, A_shared_offset_cand4, A_shared_offset_cand5]) == 1)
model.Add(sum([A_shared_ax1, A_shared_offset]) == A_shared_align_size)
model.AddMultiplicationEquality(A_shared_ax0_ax1_fused, [A_shared_ax0, A_shared_ax1])
model.Add(A_shared_vectorize == 1).OnlyEnforceIf(A_shared_vectorize_cand0)
model.Add(A_shared_vectorize == 2).OnlyEnforceIf(A_shared_vectorize_cand1)
model.Add(A_shared_vectorize == 4).OnlyEnforceIf(A_shared_vectorize_cand2)
model.Add(A_shared_vectorize == 8).OnlyEnforceIf(A_shared_vectorize_cand3)
model.Add(sum([A_shared_vectorize_cand0, A_shared_vectorize_cand1, A_shared_vectorize_cand2, A_shared_vectorize_cand3]) == 1)
model.AddMultiplicationEquality(threads, [threadIdx_x, threadIdx_y])
model.AddMultiplicationEquality(A_shared_shared_mem_size, [A_shared_ax0, A_shared_align_size])
model.AddMultiplicationEquality(BPad_shared_shared_mem_size, [BPad_shared_ax0, BPad_shared_align_size])
model.AddMultiplicationEquality(dense_wmma_accumulator_shared_shared_mem_size, [dense_wmma_accumulator_shared_ax0, dense_wmma_accumulator_shared_align_size])
solver = cp_model.CpSolver()
status = solver.Solve(model)
