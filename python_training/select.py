import json

path = "records.txt"

def apply_best(path):
    to_sort = []
    for row in open(path):
        perf={}
        param={}
        json_dict = json.loads(row)
        perf['perf'] = json_dict['perf']
        param['densei.innertileSpatial'] = json_dict['param']['densei.innertileSpatial']
        param['densej.innertileSpatial'] = json_dict['param']['densej.innertileSpatial']
        param['densei.inner.innertileSpatial'] = json_dict['param']['densei.inner.innertileSpatial']
        param['densej.inner.innertileSpatial'] = json_dict['param']['densej.inner.innertileSpatial']
        param['densei.inner.inner.innertileSpatial'] = json_dict['param']['densei.inner.inner.innertileSpatial']
        param['densej.inner.inner.innertileSpatial'] = json_dict['param']['densej.inner.inner.innertileSpatial']
        param['densei.inner.inner.inner.innertileSpatial'] = json_dict['param']['densei.inner.inner.inner.innertileSpatial']
        param['densej.inner.inner.inner.innertileSpatial'] = json_dict['param']['densej.inner.inner.inner.innertileSpatial']
        param['dense_shared_pos'] = json_dict['param']['dense_shared_pos']
        param['dense.wmma.accumulator.shared_ax1'] = json_dict['param']['dense.wmma.accumulator.shared_ax1']
        param['dense.wmma.accumulator.shared_offset'] = json_dict['param']['dense.wmma.accumulator.shared_offset']
        param['dense.wmma.accumulator.sharedax0tileSpatial'] = json_dict['param']['dense.wmma.accumulator.sharedax0tileSpatial']
        param['dense.wmma.accumulator.sharedax1tileSpatial'] = json_dict['param']['dense.wmma.accumulator.sharedax1tileSpatial']
        param['wmma_m'] = json_dict['param']['wmma_m']
        param['wmma_k'] = json_dict['param']['wmma_k']
        param['wmma_n'] = json_dict['param']['wmma_n']
        param['dense.wmma.accumulator.shared_local_pos'] = json_dict['param']['dense.wmma.accumulator.shared_local_pos']
        param['dense.wmma.accumulatori.ctileAll'] = json_dict['param']['dense.wmma.accumulatori.ctileAll']
        param['dense.wmma.accumulatorj.ctileAll'] = json_dict['param']['dense.wmma.accumulatorj.ctileAll']
        param['dense.wmma.accumulatorktileAll'] = json_dict['param']['dense.wmma.accumulatorktileAll']
        param['dense.wmma.accumulatori.c.innertileAll'] = json_dict['param']['dense.wmma.accumulatori.c.innertileAll']
        param['dense.wmma.accumulatorj.c.innertileAll'] = json_dict['param']['dense.wmma.accumulatorj.c.innertileAll']
        param['dense.wmma.accumulatork.innertileAll'] = json_dict['param']['dense.wmma.accumulatork.innertileAll']
        param['dense.wmma.accumulatori.c.inner.innertileAll'] = json_dict['param']['dense.wmma.accumulatori.c.inner.innertileAll']
        param['dense.wmma.accumulatorj.c.inner.innertileAll'] = json_dict['param']['dense.wmma.accumulatorj.c.inner.innertileAll']
        param['dense.wmma.accumulatork.inner.innertileAll'] = json_dict['param']['dense.wmma.accumulatork.inner.innertileAll']
        param['dense.wmma.accumulator_local_pos'] = json_dict['param']['dense.wmma.accumulator_local_pos']
        param['dense.wmma.accumulator_shared_pos'] = json_dict['param']['dense.wmma.accumulator_shared_pos']
        param['B.shared_ax1'] = json_dict['param']['B.shared_ax1']
        param['B.shared_offset'] = json_dict['param']['B.shared_offset']
        param['B.shared_vectorize'] = json_dict['param']['B.shared_vectorize']
        param['threadIdx.x'] = json_dict['param']['threadIdx.x']
        param['threadIdx.y'] = json_dict['param']['threadIdx.y']
        param['A.shared_ax1'] = json_dict['param']['A.shared_ax1']
        param['A.shared_offset'] = json_dict['param']['A.shared_offset']
        param['A.shared_vectorize'] = json_dict['param']['A.shared_vectorize']
        param['dense_unroll_pragma'] = json_dict['param']['dense_unroll_pragma']
        to_sort.append((perf, param))
    return to_sort

to_sort = apply_best(path)
for perf, param in to_sort:
    # 将格式化的键值对列表连接成一个字符串，用逗号分隔
    formatted_string = ", ".join([f"{key}: {value}" for key, value in param.items()])
    print(formatted_string)
    print(perf)