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
C_wmma_accumulator_shared_pos = model.NewIntVar(0, 3, 'C_wmma_accumulator_shared_pos')
C_wmma_accumulator_local_pos = model.NewIntVar(0, 3, 'C_wmma_accumulator_local_pos')
C_shared_pos = model.NewIntVar(1, 5, 'C_shared_pos')
C_wmma_accumulator_shared_local_pos = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_local_pos')
output_vectorize = model.NewIntVar(1, 4, 'output_vectorize')
output_vectorize_cand0 = model.NewIntVar(0, 1, 'output_vectorize_cand0')
output_vectorize_cand1 = model.NewIntVar(0, 1, 'output_vectorize_cand1')
output_vectorize_cand2 = model.NewIntVar(0, 1, 'output_vectorize_cand2')
threadIdx_x = model.NewIntVar(32, 32, 'threadIdx_x')
threadIdx_y = model.NewIntVar(1, 1024, 'threadIdx_y')
blockIdx_x = model.NewIntVar(1, 8888888, 'blockIdx_x')
C_unroll_pragma = model.NewIntVar(0, 5, 'C_unroll_pragma')
C_i_outer = model.NewIntVar(1, 256, 'C_i_outer')
C_i_inner = model.NewIntVar(1, 256, 'C_i_inner')
C_j_outer = model.NewIntVar(1, 3136, 'C_j_outer')
C_j_inner = model.NewIntVar(1, 3136, 'C_j_inner')
Ci_innertileSpatial = model.NewIntVar(1, 256, 'Ci_innertileSpatial')
C_i_inner_outer = model.NewIntVar(1, 256, 'C_i_inner_outer')
C_i_inner_inner = model.NewIntVar(1, 256, 'C_i_inner_inner')
Cj_innertileSpatial = model.NewIntVar(1, 3136, 'Cj_innertileSpatial')
C_j_inner_outer = model.NewIntVar(1, 3136, 'C_j_inner_outer')
C_j_inner_inner = model.NewIntVar(1, 3136, 'C_j_inner_inner')
C_i_inner_outer_j_inner_outer_fused = model.NewIntVar(1, 1000000, 'C_i_inner_outer_j_inner_outer_fused')
Ci_inner_innertileSpatial = model.NewIntVar(1, 256, 'Ci_inner_innertileSpatial')
C_i_inner_inner_outer = model.NewIntVar(1, 256, 'C_i_inner_inner_outer')
C_i_inner_inner_inner = model.NewIntVar(1, 256, 'C_i_inner_inner_inner')
Cj_inner_innertileSpatial = model.NewIntVar(1, 3136, 'Cj_inner_innertileSpatial')
C_j_inner_inner_outer = model.NewIntVar(1, 3136, 'C_j_inner_inner_outer')
C_j_inner_inner_inner = model.NewIntVar(1, 3136, 'C_j_inner_inner_inner')
C_i_inner_inner_outer_j_inner_inner_outer_fused = model.NewIntVar(1, 1000000, 'C_i_inner_inner_outer_j_inner_inner_outer_fused')
Ci_inner_inner_innertileSpatial = model.NewIntVar(1, 256, 'Ci_inner_inner_innertileSpatial')
C_i_inner_inner_inner_outer = model.NewIntVar(1, 256, 'C_i_inner_inner_inner_outer')
C_i_inner_inner_inner_inner = model.NewIntVar(1, 256, 'C_i_inner_inner_inner_inner')
Cj_inner_inner_innertileSpatial = model.NewIntVar(1, 3136, 'Cj_inner_inner_innertileSpatial')
C_j_inner_inner_inner_outer = model.NewIntVar(1, 3136, 'C_j_inner_inner_inner_outer')
C_j_inner_inner_inner_inner = model.NewIntVar(1, 3136, 'C_j_inner_inner_inner_inner')
C_i_inner_inner_inner_outer_j_inner_inner_inner_outer_fused = model.NewIntVar(1, 1000000, 'C_i_inner_inner_inner_outer_j_inner_inner_inner_outer_fused')
C_vectorize = model.NewIntVar(1, 8, 'C_vectorize')
C_vectorize_cand0 = model.NewIntVar(0, 1, 'C_vectorize_cand0')
C_vectorize_cand1 = model.NewIntVar(0, 1, 'C_vectorize_cand1')
C_vectorize_cand2 = model.NewIntVar(0, 1, 'C_vectorize_cand2')
C_vectorize_cand3 = model.NewIntVar(0, 1, 'C_vectorize_cand3')
Ci_inner_inner_inner_innertileSpatial = model.NewIntVar(1, 256, 'Ci_inner_inner_inner_innertileSpatial')
C_i_inner_inner_inner_inner_outer = model.NewIntVar(1, 256, 'C_i_inner_inner_inner_inner_outer')
C_i_inner_inner_inner_inner_inner = model.NewIntVar(1, 256, 'C_i_inner_inner_inner_inner_inner')
Cj_inner_inner_inner_innertileSpatial = model.NewIntVar(1, 3136, 'Cj_inner_inner_inner_innertileSpatial')
C_j_inner_inner_inner_inner_outer = model.NewIntVar(1, 3136, 'C_j_inner_inner_inner_inner_outer')
C_j_inner_inner_inner_inner_inner = model.NewIntVar(1, 3136, 'C_j_inner_inner_inner_inner_inner')
C_i_inner_inner_inner_inner_inner_j_inner_inner_inner_inner_inner_fused = model.NewIntVar(1, 1000000, 'C_i_inner_inner_inner_inner_inner_j_inner_inner_inner_inner_inner_fused')
C_wmma_accumulator_shared_ax0 = model.NewIntVar(1, 256, 'C_wmma_accumulator_shared_ax0')
C_shared_pos_select0 = model.NewIntVar(0, 1, 'C_shared_pos_select0')
C_shared_pos_select1 = model.NewIntVar(0, 1, 'C_shared_pos_select1')
C_shared_pos_select2 = model.NewIntVar(0, 1, 'C_shared_pos_select2')
C_shared_pos_select3 = model.NewIntVar(0, 1, 'C_shared_pos_select3')
C_shared_pos_select4 = model.NewIntVar(0, 1, 'C_shared_pos_select4')
C_shared_pos_select5 = model.NewIntVar(0, 1, 'C_shared_pos_select5')
C_wmma_accumulator_shared_ax1 = model.NewIntVar(1, 3136, 'C_wmma_accumulator_shared_ax1')
C_wmma_accumulator_shared_offset = model.NewIntVar(0, 48, 'C_wmma_accumulator_shared_offset')
C_wmma_accumulator_shared_offset_cand0 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_offset_cand0')
C_wmma_accumulator_shared_offset_cand1 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_offset_cand1')
C_wmma_accumulator_shared_offset_cand2 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_offset_cand2')
C_wmma_accumulator_shared_offset_cand3 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_offset_cand3')
C_wmma_accumulator_shared_offset_cand4 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_offset_cand4')
C_wmma_accumulator_shared_offset_cand5 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_offset_cand5')
C_wmma_accumulator_shared_align_size = model.NewIntVar(1, 88888888, 'C_wmma_accumulator_shared_align_size')
C_wmma_accumulator_sharedax0tileSpatial = model.NewIntVar(1, 256, 'C_wmma_accumulator_sharedax0tileSpatial')
C_wmma_accumulator_shared_ax0_outer = model.NewIntVar(1, 256, 'C_wmma_accumulator_shared_ax0_outer')
C_wmma_accumulator_shared_ax0_inner = model.NewIntVar(1, 256, 'C_wmma_accumulator_shared_ax0_inner')
C_wmma_accumulator_sharedax1tileSpatial = model.NewIntVar(1, 3136, 'C_wmma_accumulator_sharedax1tileSpatial')
C_wmma_accumulator_shared_ax1_outer = model.NewIntVar(1, 3136, 'C_wmma_accumulator_shared_ax1_outer')
C_wmma_accumulator_shared_ax1_inner = model.NewIntVar(1, 3136, 'C_wmma_accumulator_shared_ax1_inner')
C_wmma_accumulator_shared_ax0_outer_ax1_outer_fused = model.NewIntVar(1, 1000000, 'C_wmma_accumulator_shared_ax0_outer_ax1_outer_fused')
C_wmma_accumulator_shared_ax0_inner_outer = model.NewIntVar(1, 256, 'C_wmma_accumulator_shared_ax0_inner_outer')
C_wmma_accumulator_shared_ax0_inner_inner = model.NewIntVar(1, 256, 'C_wmma_accumulator_shared_ax0_inner_inner')
C_wmma_accumulator_shared_ax1_inner_outer = model.NewIntVar(1, 3136, 'C_wmma_accumulator_shared_ax1_inner_outer')
C_wmma_accumulator_shared_ax1_inner_inner = model.NewIntVar(1, 3136, 'C_wmma_accumulator_shared_ax1_inner_inner')
C_wmma_accumulator_i_c = model.NewIntVar(1, 256, 'C_wmma_accumulator_i_c')
C_wmma_accumulator_shared_local_pos_select0 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_local_pos_select0')
C_wmma_accumulator_shared_local_pos_select1 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_local_pos_select1')
C_wmma_accumulator_j_c = model.NewIntVar(1, 3136, 'C_wmma_accumulator_j_c')
C_wmma_accumulatori_ctileAll = model.NewIntVar(1, 256, 'C_wmma_accumulatori_ctileAll')
C_wmma_accumulator_i_c_outer = model.NewIntVar(1, 256, 'C_wmma_accumulator_i_c_outer')
C_wmma_accumulator_i_c_inner = model.NewIntVar(1, 256, 'C_wmma_accumulator_i_c_inner')
C_wmma_accumulatorj_ctileAll = model.NewIntVar(1, 3136, 'C_wmma_accumulatorj_ctileAll')
C_wmma_accumulator_j_c_outer = model.NewIntVar(1, 3136, 'C_wmma_accumulator_j_c_outer')
C_wmma_accumulator_j_c_inner = model.NewIntVar(1, 3136, 'C_wmma_accumulator_j_c_inner')
C_wmma_accumulatorktileAll = model.NewIntVar(1, 2304, 'C_wmma_accumulatorktileAll')
C_wmma_accumulator_k_outer = model.NewIntVar(1, 2304, 'C_wmma_accumulator_k_outer')
C_wmma_accumulator_k_inner = model.NewIntVar(1, 2304, 'C_wmma_accumulator_k_inner')
C_wmma_accumulatori_c_innertileAll = model.NewIntVar(1, 256, 'C_wmma_accumulatori_c_innertileAll')
C_wmma_accumulator_i_c_inner_outer = model.NewIntVar(1, 256, 'C_wmma_accumulator_i_c_inner_outer')
C_wmma_accumulator_i_c_inner_inner = model.NewIntVar(1, 256, 'C_wmma_accumulator_i_c_inner_inner')
C_wmma_accumulatorj_c_innertileAll = model.NewIntVar(1, 3136, 'C_wmma_accumulatorj_c_innertileAll')
C_wmma_accumulator_j_c_inner_outer = model.NewIntVar(1, 3136, 'C_wmma_accumulator_j_c_inner_outer')
C_wmma_accumulator_j_c_inner_inner = model.NewIntVar(1, 3136, 'C_wmma_accumulator_j_c_inner_inner')
C_wmma_accumulatork_innertileAll = model.NewIntVar(1, 2304, 'C_wmma_accumulatork_innertileAll')
C_wmma_accumulator_k_inner_outer = model.NewIntVar(1, 2304, 'C_wmma_accumulator_k_inner_outer')
C_wmma_accumulator_k_inner_inner = model.NewIntVar(1, 2304, 'C_wmma_accumulator_k_inner_inner')
C_wmma_accumulatori_c_inner_innertileAll = model.NewIntVar(1, 256, 'C_wmma_accumulatori_c_inner_innertileAll')
C_wmma_accumulator_i_c_inner_inner_outer = model.NewIntVar(1, 256, 'C_wmma_accumulator_i_c_inner_inner_outer')
C_wmma_accumulator_i_c_inner_inner_inner = model.NewIntVar(1, 256, 'C_wmma_accumulator_i_c_inner_inner_inner')
C_wmma_accumulatorj_c_inner_innertileAll = model.NewIntVar(1, 3136, 'C_wmma_accumulatorj_c_inner_innertileAll')
C_wmma_accumulator_j_c_inner_inner_outer = model.NewIntVar(1, 3136, 'C_wmma_accumulator_j_c_inner_inner_outer')
C_wmma_accumulator_j_c_inner_inner_inner = model.NewIntVar(1, 3136, 'C_wmma_accumulator_j_c_inner_inner_inner')
C_wmma_accumulatork_inner_innertileAll = model.NewIntVar(1, 2304, 'C_wmma_accumulatork_inner_innertileAll')
C_wmma_accumulator_k_inner_inner_outer = model.NewIntVar(1, 2304, 'C_wmma_accumulator_k_inner_inner_outer')
C_wmma_accumulator_k_inner_inner_inner = model.NewIntVar(1, 2304, 'C_wmma_accumulator_k_inner_inner_inner')
C_wmma_accumulator_i_c_inner_inner_inner_outer = model.NewIntVar(1, 256, 'C_wmma_accumulator_i_c_inner_inner_inner_outer')
C_wmma_accumulator_i_c_inner_inner_inner_inner = model.NewIntVar(1, 256, 'C_wmma_accumulator_i_c_inner_inner_inner_inner')
C_wmma_accumulator_j_c_inner_inner_inner_outer = model.NewIntVar(1, 3136, 'C_wmma_accumulator_j_c_inner_inner_inner_outer')
C_wmma_accumulator_j_c_inner_inner_inner_inner = model.NewIntVar(1, 3136, 'C_wmma_accumulator_j_c_inner_inner_inner_inner')
C_wmma_accumulator_k_inner_inner_inner_outer = model.NewIntVar(1, 2304, 'C_wmma_accumulator_k_inner_inner_inner_outer')
C_wmma_accumulator_k_inner_inner_inner_inner = model.NewIntVar(1, 2304, 'C_wmma_accumulator_k_inner_inner_inner_inner')
A_shared_wmma_matrix_b_ax0 = model.NewIntVar(1, 3136, 'A_shared_wmma_matrix_b_ax0')
C_wmma_accumulator_local_pos_select0 = model.NewIntVar(0, 1, 'C_wmma_accumulator_local_pos_select0')
C_wmma_accumulator_local_pos_select1 = model.NewIntVar(0, 1, 'C_wmma_accumulator_local_pos_select1')
C_wmma_accumulator_local_pos_select2 = model.NewIntVar(0, 1, 'C_wmma_accumulator_local_pos_select2')
C_wmma_accumulator_local_pos_select3 = model.NewIntVar(0, 1, 'C_wmma_accumulator_local_pos_select3')
A_shared_wmma_matrix_b_ax1 = model.NewIntVar(1, 2304, 'A_shared_wmma_matrix_b_ax1')
A_shared_wmma_matrix_b_ax0_outer = model.NewIntVar(1, 3136, 'A_shared_wmma_matrix_b_ax0_outer')
A_shared_wmma_matrix_b_ax0_inner = model.NewIntVar(1, 3136, 'A_shared_wmma_matrix_b_ax0_inner')
A_shared_wmma_matrix_b_ax1_outer = model.NewIntVar(1, 2304, 'A_shared_wmma_matrix_b_ax1_outer')
A_shared_wmma_matrix_b_ax1_inner = model.NewIntVar(1, 2304, 'A_shared_wmma_matrix_b_ax1_inner')
A_shared_ax0 = model.NewIntVar(1, 3136, 'A_shared_ax0')
A_shared_tmp_ax0 = model.NewIntVar(1, 3136, 'A_shared_tmp_ax0')
C_wmma_accumulator_shared_pos_select0 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_pos_select0')
C_wmma_accumulator_shared_pos_select1 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_pos_select1')
C_wmma_accumulator_shared_pos_select2 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_pos_select2')
C_wmma_accumulator_shared_pos_select3 = model.NewIntVar(0, 1, 'C_wmma_accumulator_shared_pos_select3')
A_shared_ax1 = model.NewIntVar(1, 2304, 'A_shared_ax1')
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
B_shared_wmma_matrix_a_ax0 = model.NewIntVar(1, 256, 'B_shared_wmma_matrix_a_ax0')
B_shared_wmma_matrix_a_ax1 = model.NewIntVar(1, 2304, 'B_shared_wmma_matrix_a_ax1')
B_shared_wmma_matrix_a_ax0_outer = model.NewIntVar(1, 256, 'B_shared_wmma_matrix_a_ax0_outer')
B_shared_wmma_matrix_a_ax0_inner = model.NewIntVar(1, 256, 'B_shared_wmma_matrix_a_ax0_inner')
B_shared_wmma_matrix_a_ax1_outer = model.NewIntVar(1, 2304, 'B_shared_wmma_matrix_a_ax1_outer')
B_shared_wmma_matrix_a_ax1_inner = model.NewIntVar(1, 2304, 'B_shared_wmma_matrix_a_ax1_inner')
B_shared_ax0 = model.NewIntVar(1, 256, 'B_shared_ax0')
B_shared_tmp_ax0 = model.NewIntVar(1, 256, 'B_shared_tmp_ax0')
B_shared_ax1 = model.NewIntVar(1, 2304, 'B_shared_ax1')
B_shared_offset = model.NewIntVar(0, 48, 'B_shared_offset')
B_shared_offset_cand0 = model.NewIntVar(0, 1, 'B_shared_offset_cand0')
B_shared_offset_cand1 = model.NewIntVar(0, 1, 'B_shared_offset_cand1')
B_shared_offset_cand2 = model.NewIntVar(0, 1, 'B_shared_offset_cand2')
B_shared_offset_cand3 = model.NewIntVar(0, 1, 'B_shared_offset_cand3')
B_shared_offset_cand4 = model.NewIntVar(0, 1, 'B_shared_offset_cand4')
B_shared_offset_cand5 = model.NewIntVar(0, 1, 'B_shared_offset_cand5')
B_shared_align_size = model.NewIntVar(1, 88888888, 'B_shared_align_size')
B_shared_ax0_ax1_fused = model.NewIntVar(1, 1000000, 'B_shared_ax0_ax1_fused')
B_shared_vectorize = model.NewIntVar(1, 8, 'B_shared_vectorize')
B_shared_vectorize_cand0 = model.NewIntVar(0, 1, 'B_shared_vectorize_cand0')
B_shared_vectorize_cand1 = model.NewIntVar(0, 1, 'B_shared_vectorize_cand1')
B_shared_vectorize_cand2 = model.NewIntVar(0, 1, 'B_shared_vectorize_cand2')
B_shared_vectorize_cand3 = model.NewIntVar(0, 1, 'B_shared_vectorize_cand3')
threads = model.NewIntVar(1, 1024, 'threads')
B_shared_shared_mem_size = model.NewIntVar(1, 24576, 'B_shared_shared_mem_size')
A_shared_shared_mem_size = model.NewIntVar(1, 24576, 'A_shared_shared_mem_size')
C_wmma_accumulator_shared_shared_mem_size = model.NewIntVar(1, 24576, 'C_wmma_accumulator_shared_shared_mem_size')
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
model.Add(output_vectorize == 1).OnlyEnforceIf(output_vectorize_cand0)
model.Add(output_vectorize == 2).OnlyEnforceIf(output_vectorize_cand1)
model.Add(output_vectorize == 4).OnlyEnforceIf(output_vectorize_cand2)
model.Add(sum([output_vectorize_cand0, output_vectorize_cand1, output_vectorize_cand2]) == 1)
model.AddMultiplicationEquality(256, [C_i_outer, C_i_inner])
model.Add(C_i_outer == 1)
model.AddMultiplicationEquality(3136, [C_j_outer, C_j_inner])
model.Add(C_j_outer == 1)
model.AddMultiplicationEquality(C_i_inner, [C_i_inner_outer, C_i_inner_inner])
model.Add(C_i_inner_outer == Ci_innertileSpatial)
model.AddMultiplicationEquality(C_j_inner, [C_j_inner_outer, C_j_inner_inner])
model.Add(C_j_inner_outer == Cj_innertileSpatial)
model.AddMultiplicationEquality(blockIdx_x, [Ci_innertileSpatial, Cj_innertileSpatial])
model.AddMultiplicationEquality(C_i_inner_outer_j_inner_outer_fused, [C_i_inner_outer, C_j_inner_outer])
model.AddMultiplicationEquality(C_i_inner_inner, [C_i_inner_inner_outer, C_i_inner_inner_inner])
model.Add(C_i_inner_inner_outer == Ci_inner_innertileSpatial)
model.AddMultiplicationEquality(C_j_inner_inner, [C_j_inner_inner_outer, C_j_inner_inner_inner])
model.Add(C_j_inner_inner_outer == Cj_inner_innertileSpatial)
model.AddMultiplicationEquality(threadIdx_y, [Ci_inner_innertileSpatial, Cj_inner_innertileSpatial])
model.AddMultiplicationEquality(C_i_inner_inner_outer_j_inner_inner_outer_fused, [C_i_inner_inner_outer, C_j_inner_inner_outer])
model.AddMultiplicationEquality(C_i_inner_inner_inner, [C_i_inner_inner_inner_outer, C_i_inner_inner_inner_inner])
model.Add(C_i_inner_inner_inner_outer == Ci_inner_inner_innertileSpatial)
model.AddMultiplicationEquality(C_j_inner_inner_inner, [C_j_inner_inner_inner_outer, C_j_inner_inner_inner_inner])
model.Add(C_j_inner_inner_inner_outer == Cj_inner_inner_innertileSpatial)
model.AddMultiplicationEquality(threadIdx_x, [Ci_inner_inner_innertileSpatial, Cj_inner_inner_innertileSpatial])
model.AddMultiplicationEquality(C_i_inner_inner_inner_outer_j_inner_inner_inner_outer_fused, [C_i_inner_inner_inner_outer, C_j_inner_inner_inner_outer])
model.Add(C_vectorize == 1).OnlyEnforceIf(C_vectorize_cand0)
model.Add(C_vectorize == 2).OnlyEnforceIf(C_vectorize_cand1)
model.Add(C_vectorize == 4).OnlyEnforceIf(C_vectorize_cand2)
model.Add(C_vectorize == 8).OnlyEnforceIf(C_vectorize_cand3)
model.Add(sum([C_vectorize_cand0, C_vectorize_cand1, C_vectorize_cand2, C_vectorize_cand3]) == 1)
model.AddMultiplicationEquality(C_i_inner_inner_inner_inner, [C_i_inner_inner_inner_inner_outer, C_i_inner_inner_inner_inner_inner])
model.Add(C_i_inner_inner_inner_inner_outer == Ci_inner_inner_inner_innertileSpatial)
model.AddMultiplicationEquality(C_j_inner_inner_inner_inner, [C_j_inner_inner_inner_inner_outer, C_j_inner_inner_inner_inner_inner])
model.Add(C_j_inner_inner_inner_inner_outer == Cj_inner_inner_inner_innertileSpatial)
model.AddMultiplicationEquality(C_i_inner_inner_inner_inner_inner_j_inner_inner_inner_inner_inner_fused, [C_i_inner_inner_inner_inner_inner, C_j_inner_inner_inner_inner_inner])
model.Add(C_i_inner_inner_inner_inner_inner_j_inner_inner_inner_inner_inner_fused == C_vectorize)
model.Add(C_shared_pos != 0)
model.Add(C_wmma_accumulator_shared_ax0 == C_i_inner).OnlyEnforceIf(C_shared_pos_select0)
model.Add(C_shared_pos == 0).OnlyEnforceIf(C_shared_pos_select0)
model.Add(C_wmma_accumulator_shared_ax0 == C_i_inner_inner).OnlyEnforceIf(C_shared_pos_select1)
model.Add(C_shared_pos == 1).OnlyEnforceIf(C_shared_pos_select1)
model.Add(C_wmma_accumulator_shared_ax0 == C_i_inner_inner_inner).OnlyEnforceIf(C_shared_pos_select2)
model.Add(C_shared_pos == 2).OnlyEnforceIf(C_shared_pos_select2)
model.Add(C_wmma_accumulator_shared_ax0 == C_i_inner_inner_inner_inner).OnlyEnforceIf(C_shared_pos_select3)
model.Add(C_shared_pos == 3).OnlyEnforceIf(C_shared_pos_select3)
model.Add(C_wmma_accumulator_shared_ax0 == C_i_inner_inner_inner_inner_inner).OnlyEnforceIf(C_shared_pos_select4)
model.Add(C_shared_pos == 4).OnlyEnforceIf(C_shared_pos_select4)
model.Add(C_wmma_accumulator_shared_ax0 == 1).OnlyEnforceIf(C_shared_pos_select5)
model.Add(C_shared_pos == 5).OnlyEnforceIf(C_shared_pos_select5)
model.Add(sum([C_shared_pos_select0, C_shared_pos_select1, C_shared_pos_select2, C_shared_pos_select3, C_shared_pos_select4, C_shared_pos_select5]) == 1)
model.Add(C_wmma_accumulator_shared_ax1 == C_j_inner).OnlyEnforceIf(C_shared_pos_select0)
model.Add(C_shared_pos == 0).OnlyEnforceIf(C_shared_pos_select0)
model.Add(C_wmma_accumulator_shared_ax1 == C_j_inner_inner).OnlyEnforceIf(C_shared_pos_select1)
model.Add(C_shared_pos == 1).OnlyEnforceIf(C_shared_pos_select1)
model.Add(C_wmma_accumulator_shared_ax1 == C_j_inner_inner_inner).OnlyEnforceIf(C_shared_pos_select2)
model.Add(C_shared_pos == 2).OnlyEnforceIf(C_shared_pos_select2)
model.Add(C_wmma_accumulator_shared_ax1 == C_j_inner_inner_inner_inner).OnlyEnforceIf(C_shared_pos_select3)
model.Add(C_shared_pos == 3).OnlyEnforceIf(C_shared_pos_select3)
model.Add(C_wmma_accumulator_shared_ax1 == C_j_inner_inner_inner_inner_inner).OnlyEnforceIf(C_shared_pos_select4)
model.Add(C_shared_pos == 4).OnlyEnforceIf(C_shared_pos_select4)
model.Add(C_wmma_accumulator_shared_ax1 == 1).OnlyEnforceIf(C_shared_pos_select5)
model.Add(C_shared_pos == 5).OnlyEnforceIf(C_shared_pos_select5)
model.Add(sum([C_shared_pos_select0, C_shared_pos_select1, C_shared_pos_select2, C_shared_pos_select3, C_shared_pos_select4, C_shared_pos_select5]) == 1)
model.Add(C_wmma_accumulator_shared_offset == 0).OnlyEnforceIf(C_wmma_accumulator_shared_offset_cand0)
model.Add(C_wmma_accumulator_shared_offset == 8).OnlyEnforceIf(C_wmma_accumulator_shared_offset_cand1)
model.Add(C_wmma_accumulator_shared_offset == 16).OnlyEnforceIf(C_wmma_accumulator_shared_offset_cand2)
model.Add(C_wmma_accumulator_shared_offset == 24).OnlyEnforceIf(C_wmma_accumulator_shared_offset_cand3)
model.Add(C_wmma_accumulator_shared_offset == 32).OnlyEnforceIf(C_wmma_accumulator_shared_offset_cand4)
model.Add(C_wmma_accumulator_shared_offset == 48).OnlyEnforceIf(C_wmma_accumulator_shared_offset_cand5)
model.Add(sum([C_wmma_accumulator_shared_offset_cand0, C_wmma_accumulator_shared_offset_cand1, C_wmma_accumulator_shared_offset_cand2, C_wmma_accumulator_shared_offset_cand3, C_wmma_accumulator_shared_offset_cand4, C_wmma_accumulator_shared_offset_cand5]) == 1)
model.Add(sum([C_wmma_accumulator_shared_ax1, C_wmma_accumulator_shared_offset]) == C_wmma_accumulator_shared_align_size)
model.AddMultiplicationEquality(C_wmma_accumulator_shared_ax0, [C_wmma_accumulator_shared_ax0_outer, C_wmma_accumulator_shared_ax0_inner])
model.Add(C_wmma_accumulator_shared_ax0_outer == C_wmma_accumulator_sharedax0tileSpatial)
model.AddMultiplicationEquality(C_wmma_accumulator_shared_ax1, [C_wmma_accumulator_shared_ax1_outer, C_wmma_accumulator_shared_ax1_inner])
model.Add(C_wmma_accumulator_shared_ax1_outer == C_wmma_accumulator_sharedax1tileSpatial)
model.AddMultiplicationEquality(threadIdx_y, [C_wmma_accumulator_sharedax0tileSpatial, C_wmma_accumulator_sharedax1tileSpatial])
model.AddMultiplicationEquality(C_wmma_accumulator_shared_ax0_outer_ax1_outer_fused, [C_wmma_accumulator_shared_ax0_outer, C_wmma_accumulator_shared_ax1_outer])
model.AddMultiplicationEquality(C_wmma_accumulator_shared_ax0_inner, [C_wmma_accumulator_shared_ax0_inner_outer, C_wmma_accumulator_shared_ax0_inner_inner])
model.Add(C_wmma_accumulator_shared_ax0_inner_inner == wmma_m)
model.AddMultiplicationEquality(C_wmma_accumulator_shared_ax1_inner, [C_wmma_accumulator_shared_ax1_inner_outer, C_wmma_accumulator_shared_ax1_inner_inner])
model.Add(C_wmma_accumulator_shared_ax1_inner_inner == wmma_n)
model.Add(C_wmma_accumulator_i_c == C_wmma_accumulator_shared_ax0_inner).OnlyEnforceIf(C_wmma_accumulator_shared_local_pos_select0)
model.Add(C_wmma_accumulator_shared_local_pos == 0).OnlyEnforceIf(C_wmma_accumulator_shared_local_pos_select0)
model.Add(C_wmma_accumulator_i_c == C_wmma_accumulator_shared_ax0_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_local_pos_select1)
model.Add(C_wmma_accumulator_shared_local_pos == 1).OnlyEnforceIf(C_wmma_accumulator_shared_local_pos_select1)
model.Add(sum([C_wmma_accumulator_shared_local_pos_select0, C_wmma_accumulator_shared_local_pos_select1]) == 1)
model.Add(C_wmma_accumulator_j_c == C_wmma_accumulator_shared_ax1_inner).OnlyEnforceIf(C_wmma_accumulator_shared_local_pos_select0)
model.Add(C_wmma_accumulator_shared_local_pos == 0).OnlyEnforceIf(C_wmma_accumulator_shared_local_pos_select0)
model.Add(C_wmma_accumulator_j_c == C_wmma_accumulator_shared_ax1_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_local_pos_select1)
model.Add(C_wmma_accumulator_shared_local_pos == 1).OnlyEnforceIf(C_wmma_accumulator_shared_local_pos_select1)
model.Add(sum([C_wmma_accumulator_shared_local_pos_select0, C_wmma_accumulator_shared_local_pos_select1]) == 1)
model.AddMultiplicationEquality(C_wmma_accumulator_i_c, [C_wmma_accumulator_i_c_outer, C_wmma_accumulator_i_c_inner])
model.Add(C_wmma_accumulator_i_c_outer == C_wmma_accumulatori_ctileAll)
model.AddMultiplicationEquality(C_wmma_accumulator_j_c, [C_wmma_accumulator_j_c_outer, C_wmma_accumulator_j_c_inner])
model.Add(C_wmma_accumulator_j_c_outer == C_wmma_accumulatorj_ctileAll)
model.AddMultiplicationEquality(2304, [C_wmma_accumulator_k_outer, C_wmma_accumulator_k_inner])
model.Add(C_wmma_accumulator_k_outer == C_wmma_accumulatorktileAll)
model.AddMultiplicationEquality(C_wmma_accumulator_i_c_inner, [C_wmma_accumulator_i_c_inner_outer, C_wmma_accumulator_i_c_inner_inner])
model.Add(C_wmma_accumulator_i_c_inner_outer == C_wmma_accumulatori_c_innertileAll)
model.AddMultiplicationEquality(C_wmma_accumulator_j_c_inner, [C_wmma_accumulator_j_c_inner_outer, C_wmma_accumulator_j_c_inner_inner])
model.Add(C_wmma_accumulator_j_c_inner_outer == C_wmma_accumulatorj_c_innertileAll)
model.AddMultiplicationEquality(C_wmma_accumulator_k_inner, [C_wmma_accumulator_k_inner_outer, C_wmma_accumulator_k_inner_inner])
model.Add(C_wmma_accumulator_k_inner_outer == C_wmma_accumulatork_innertileAll)
model.AddMultiplicationEquality(C_wmma_accumulator_i_c_inner_inner, [C_wmma_accumulator_i_c_inner_inner_outer, C_wmma_accumulator_i_c_inner_inner_inner])
model.Add(C_wmma_accumulator_i_c_inner_inner_outer == C_wmma_accumulatori_c_inner_innertileAll)
model.AddMultiplicationEquality(C_wmma_accumulator_j_c_inner_inner, [C_wmma_accumulator_j_c_inner_inner_outer, C_wmma_accumulator_j_c_inner_inner_inner])
model.Add(C_wmma_accumulator_j_c_inner_inner_outer == C_wmma_accumulatorj_c_inner_innertileAll)
model.AddMultiplicationEquality(C_wmma_accumulator_k_inner_inner, [C_wmma_accumulator_k_inner_inner_outer, C_wmma_accumulator_k_inner_inner_inner])
model.Add(C_wmma_accumulator_k_inner_inner_outer == C_wmma_accumulatork_inner_innertileAll)
model.AddMultiplicationEquality(C_wmma_accumulator_i_c_inner_inner_inner, [C_wmma_accumulator_i_c_inner_inner_inner_outer, C_wmma_accumulator_i_c_inner_inner_inner_inner])
model.Add(C_wmma_accumulator_i_c_inner_inner_inner_inner == wmma_m)
model.AddMultiplicationEquality(C_wmma_accumulator_j_c_inner_inner_inner, [C_wmma_accumulator_j_c_inner_inner_inner_outer, C_wmma_accumulator_j_c_inner_inner_inner_inner])
model.Add(C_wmma_accumulator_j_c_inner_inner_inner_inner == wmma_n)
model.AddMultiplicationEquality(C_wmma_accumulator_k_inner_inner_inner, [C_wmma_accumulator_k_inner_inner_inner_outer, C_wmma_accumulator_k_inner_inner_inner_inner])
model.Add(C_wmma_accumulator_k_inner_inner_inner_inner == wmma_k)
model.Add(C_wmma_accumulator_shared_pos <= C_wmma_accumulator_local_pos)
model.Add(A_shared_wmma_matrix_b_ax0 == C_wmma_accumulator_j_c_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select0)
model.Add(C_wmma_accumulator_local_pos == 0).OnlyEnforceIf(C_wmma_accumulator_local_pos_select0)
model.Add(A_shared_wmma_matrix_b_ax0 == C_wmma_accumulator_j_c_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select1)
model.Add(C_wmma_accumulator_local_pos == 1).OnlyEnforceIf(C_wmma_accumulator_local_pos_select1)
model.Add(A_shared_wmma_matrix_b_ax0 == C_wmma_accumulator_j_c_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select2)
model.Add(C_wmma_accumulator_local_pos == 2).OnlyEnforceIf(C_wmma_accumulator_local_pos_select2)
model.Add(A_shared_wmma_matrix_b_ax0 == C_wmma_accumulator_j_c_inner_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select3)
model.Add(C_wmma_accumulator_local_pos == 3).OnlyEnforceIf(C_wmma_accumulator_local_pos_select3)
model.Add(sum([C_wmma_accumulator_local_pos_select0, C_wmma_accumulator_local_pos_select1, C_wmma_accumulator_local_pos_select2, C_wmma_accumulator_local_pos_select3]) == 1)
model.Add(A_shared_wmma_matrix_b_ax1 == C_wmma_accumulator_k_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select0)
model.Add(C_wmma_accumulator_local_pos == 0).OnlyEnforceIf(C_wmma_accumulator_local_pos_select0)
model.Add(A_shared_wmma_matrix_b_ax1 == C_wmma_accumulator_k_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select1)
model.Add(C_wmma_accumulator_local_pos == 1).OnlyEnforceIf(C_wmma_accumulator_local_pos_select1)
model.Add(A_shared_wmma_matrix_b_ax1 == C_wmma_accumulator_k_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select2)
model.Add(C_wmma_accumulator_local_pos == 2).OnlyEnforceIf(C_wmma_accumulator_local_pos_select2)
model.Add(A_shared_wmma_matrix_b_ax1 == C_wmma_accumulator_k_inner_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select3)
model.Add(C_wmma_accumulator_local_pos == 3).OnlyEnforceIf(C_wmma_accumulator_local_pos_select3)
model.Add(sum([C_wmma_accumulator_local_pos_select0, C_wmma_accumulator_local_pos_select1, C_wmma_accumulator_local_pos_select2, C_wmma_accumulator_local_pos_select3]) == 1)
model.AddMultiplicationEquality(A_shared_wmma_matrix_b_ax0, [A_shared_wmma_matrix_b_ax0_outer, A_shared_wmma_matrix_b_ax0_inner])
model.Add(A_shared_wmma_matrix_b_ax0_inner == wmma_n)
model.AddMultiplicationEquality(A_shared_wmma_matrix_b_ax1, [A_shared_wmma_matrix_b_ax1_outer, A_shared_wmma_matrix_b_ax1_inner])
model.Add(A_shared_wmma_matrix_b_ax1_inner == wmma_k)
model.Add(A_shared_tmp_ax0 == C_wmma_accumulator_j_c_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select0)
model.Add(C_wmma_accumulator_shared_pos == 0).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select0)
model.Add(A_shared_tmp_ax0 == C_wmma_accumulator_j_c_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select1)
model.Add(C_wmma_accumulator_shared_pos == 1).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select1)
model.Add(A_shared_tmp_ax0 == C_wmma_accumulator_j_c_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select2)
model.Add(C_wmma_accumulator_shared_pos == 2).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select2)
model.Add(A_shared_tmp_ax0 == C_wmma_accumulator_j_c_inner_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select3)
model.Add(C_wmma_accumulator_shared_pos == 3).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select3)
model.Add(sum([C_wmma_accumulator_shared_pos_select0, C_wmma_accumulator_shared_pos_select1, C_wmma_accumulator_shared_pos_select2, C_wmma_accumulator_shared_pos_select3]) == 1)
model.AddMultiplicationEquality(A_shared_ax0, [A_shared_tmp_ax0, C_wmma_accumulator_sharedax1tileSpatial])
model.Add(A_shared_ax1 == C_wmma_accumulator_k_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select0)
model.Add(C_wmma_accumulator_shared_pos == 0).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select0)
model.Add(A_shared_ax1 == C_wmma_accumulator_k_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select1)
model.Add(C_wmma_accumulator_shared_pos == 1).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select1)
model.Add(A_shared_ax1 == C_wmma_accumulator_k_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select2)
model.Add(C_wmma_accumulator_shared_pos == 2).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select2)
model.Add(A_shared_ax1 == C_wmma_accumulator_k_inner_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select3)
model.Add(C_wmma_accumulator_shared_pos == 3).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select3)
model.Add(sum([C_wmma_accumulator_shared_pos_select0, C_wmma_accumulator_shared_pos_select1, C_wmma_accumulator_shared_pos_select2, C_wmma_accumulator_shared_pos_select3]) == 1)
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
model.Add(B_shared_wmma_matrix_a_ax0 == C_wmma_accumulator_i_c_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select0)
model.Add(C_wmma_accumulator_local_pos == 0).OnlyEnforceIf(C_wmma_accumulator_local_pos_select0)
model.Add(B_shared_wmma_matrix_a_ax0 == C_wmma_accumulator_i_c_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select1)
model.Add(C_wmma_accumulator_local_pos == 1).OnlyEnforceIf(C_wmma_accumulator_local_pos_select1)
model.Add(B_shared_wmma_matrix_a_ax0 == C_wmma_accumulator_i_c_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select2)
model.Add(C_wmma_accumulator_local_pos == 2).OnlyEnforceIf(C_wmma_accumulator_local_pos_select2)
model.Add(B_shared_wmma_matrix_a_ax0 == C_wmma_accumulator_i_c_inner_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select3)
model.Add(C_wmma_accumulator_local_pos == 3).OnlyEnforceIf(C_wmma_accumulator_local_pos_select3)
model.Add(sum([C_wmma_accumulator_local_pos_select0, C_wmma_accumulator_local_pos_select1, C_wmma_accumulator_local_pos_select2, C_wmma_accumulator_local_pos_select3]) == 1)
model.Add(B_shared_wmma_matrix_a_ax1 == C_wmma_accumulator_k_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select0)
model.Add(C_wmma_accumulator_local_pos == 0).OnlyEnforceIf(C_wmma_accumulator_local_pos_select0)
model.Add(B_shared_wmma_matrix_a_ax1 == C_wmma_accumulator_k_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select1)
model.Add(C_wmma_accumulator_local_pos == 1).OnlyEnforceIf(C_wmma_accumulator_local_pos_select1)
model.Add(B_shared_wmma_matrix_a_ax1 == C_wmma_accumulator_k_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select2)
model.Add(C_wmma_accumulator_local_pos == 2).OnlyEnforceIf(C_wmma_accumulator_local_pos_select2)
model.Add(B_shared_wmma_matrix_a_ax1 == C_wmma_accumulator_k_inner_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_local_pos_select3)
model.Add(C_wmma_accumulator_local_pos == 3).OnlyEnforceIf(C_wmma_accumulator_local_pos_select3)
model.Add(sum([C_wmma_accumulator_local_pos_select0, C_wmma_accumulator_local_pos_select1, C_wmma_accumulator_local_pos_select2, C_wmma_accumulator_local_pos_select3]) == 1)
model.AddMultiplicationEquality(B_shared_wmma_matrix_a_ax0, [B_shared_wmma_matrix_a_ax0_outer, B_shared_wmma_matrix_a_ax0_inner])
model.Add(B_shared_wmma_matrix_a_ax0_inner == wmma_m)
model.AddMultiplicationEquality(B_shared_wmma_matrix_a_ax1, [B_shared_wmma_matrix_a_ax1_outer, B_shared_wmma_matrix_a_ax1_inner])
model.Add(B_shared_wmma_matrix_a_ax1_inner == wmma_k)
model.Add(B_shared_tmp_ax0 == C_wmma_accumulator_i_c_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select0)
model.Add(C_wmma_accumulator_shared_pos == 0).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select0)
model.Add(B_shared_tmp_ax0 == C_wmma_accumulator_i_c_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select1)
model.Add(C_wmma_accumulator_shared_pos == 1).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select1)
model.Add(B_shared_tmp_ax0 == C_wmma_accumulator_i_c_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select2)
model.Add(C_wmma_accumulator_shared_pos == 2).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select2)
model.Add(B_shared_tmp_ax0 == C_wmma_accumulator_i_c_inner_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select3)
model.Add(C_wmma_accumulator_shared_pos == 3).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select3)
model.Add(sum([C_wmma_accumulator_shared_pos_select0, C_wmma_accumulator_shared_pos_select1, C_wmma_accumulator_shared_pos_select2, C_wmma_accumulator_shared_pos_select3]) == 1)
model.AddMultiplicationEquality(B_shared_ax0, [B_shared_tmp_ax0, C_wmma_accumulator_sharedax0tileSpatial])
model.Add(B_shared_ax1 == C_wmma_accumulator_k_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select0)
model.Add(C_wmma_accumulator_shared_pos == 0).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select0)
model.Add(B_shared_ax1 == C_wmma_accumulator_k_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select1)
model.Add(C_wmma_accumulator_shared_pos == 1).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select1)
model.Add(B_shared_ax1 == C_wmma_accumulator_k_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select2)
model.Add(C_wmma_accumulator_shared_pos == 2).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select2)
model.Add(B_shared_ax1 == C_wmma_accumulator_k_inner_inner_inner_inner).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select3)
model.Add(C_wmma_accumulator_shared_pos == 3).OnlyEnforceIf(C_wmma_accumulator_shared_pos_select3)
model.Add(sum([C_wmma_accumulator_shared_pos_select0, C_wmma_accumulator_shared_pos_select1, C_wmma_accumulator_shared_pos_select2, C_wmma_accumulator_shared_pos_select3]) == 1)
model.Add(B_shared_offset == 0).OnlyEnforceIf(B_shared_offset_cand0)
model.Add(B_shared_offset == 8).OnlyEnforceIf(B_shared_offset_cand1)
model.Add(B_shared_offset == 16).OnlyEnforceIf(B_shared_offset_cand2)
model.Add(B_shared_offset == 24).OnlyEnforceIf(B_shared_offset_cand3)
model.Add(B_shared_offset == 32).OnlyEnforceIf(B_shared_offset_cand4)
model.Add(B_shared_offset == 48).OnlyEnforceIf(B_shared_offset_cand5)
model.Add(sum([B_shared_offset_cand0, B_shared_offset_cand1, B_shared_offset_cand2, B_shared_offset_cand3, B_shared_offset_cand4, B_shared_offset_cand5]) == 1)
model.Add(sum([B_shared_ax1, B_shared_offset]) == B_shared_align_size)
model.AddMultiplicationEquality(B_shared_ax0_ax1_fused, [B_shared_ax0, B_shared_ax1])
model.Add(B_shared_vectorize == 1).OnlyEnforceIf(B_shared_vectorize_cand0)
model.Add(B_shared_vectorize == 2).OnlyEnforceIf(B_shared_vectorize_cand1)
model.Add(B_shared_vectorize == 4).OnlyEnforceIf(B_shared_vectorize_cand2)
model.Add(B_shared_vectorize == 8).OnlyEnforceIf(B_shared_vectorize_cand3)
model.Add(sum([B_shared_vectorize_cand0, B_shared_vectorize_cand1, B_shared_vectorize_cand2, B_shared_vectorize_cand3]) == 1)
model.AddMultiplicationEquality(threads, [threadIdx_x, threadIdx_y])
model.AddMultiplicationEquality(B_shared_shared_mem_size, [B_shared_ax0, B_shared_align_size])
model.AddMultiplicationEquality(A_shared_shared_mem_size, [A_shared_ax0, A_shared_align_size])
model.AddMultiplicationEquality(C_wmma_accumulator_shared_shared_mem_size, [C_wmma_accumulator_shared_ax0, C_wmma_accumulator_shared_align_size])
solver = cp_model.CpSolver()
status = solver.Solve(model)
