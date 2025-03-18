import tvm
from tvm import te
import numpy as np

def demonstrate_schedule_impact():
    n = 1024
    A = te.placeholder((n, n), name='A')
    B = te.placeholder((n, n), name='B')
    C = te.compute((n, n), lambda i, j: A[i, j] + 1, name='C')

    k = te.reduce_axis((0, n), name='k')
    D = te.compute((n, n), lambda i, j: te.sum(C[i, k] * B[k, j], axis=k), name='D')
    E = te.compute((n, n), lambda i, j: D[i, j] * 2, name='E')

    s_original = te.create_schedule(E.op)
    print("===== 原始计算图 =====")
    #print(tvm.lower(s_original, [A, B, E], simple_mode=True))
    print("\n计算图中的阶段数:", len(s_original.stages))
    for stage in s_original.stages:
        print(f"s_original.stages: {stage}")

    print("\n\n===== 应用内联后的计算图 =====")
    s_inline = te.create_schedule(E.op)
    s_inline[C].compute_inline()
    #print(tvm.lower(s_inline, [A, B, E], simple_mode=True))
    print("\n计算图中的阶段数:", len(s_inline.stages), "（减少了C阶段）")
    for stage in s_inline.stages:
        print(f"s_inline.stage: {stage}")

    print("\n\n===== 应用cache_read/cache_write后的计算图 =====")
    s_cache = te.create_schedule(E.op)
    B_shared = s_cache.cache_read(B, "shared", [D])

    D_local = s_cache.cache_write(D, "local")
    #print(tvm.lower(s_cache, [A, B, E], simple_mode=True))
    print("\n计算图中的阶段数:", len(s_cache.stages), "(增加了缓存阶段)")
    for stage in s_cache.stages:
        print(f"s_cache.stage: {stage}")

    print("\n\n===== 应用循环变换后的计算图 =====")
    s_transform = te.create_schedule(E.op)

    i, j = s_transform[D].op.axis
    io, ii = s_transform[D].split(i, factor=32)
    jo, ji = s_transform[D].split(j, factor=32)
    s_transform[D].reorder(io, jo, ii, ji)
    #print(tvm.lower(s_transform, [A, B, E], simple_mode=True))
    print("\n计算图中的阶段数:", len(s_transform.stages), "(阶段数不变)")
    for stage in s_transform.stages:
        print(f"s_transform.stage: {stage}")

    print("\n\n===== 应用线程绑定后的计算图 =====")
    s_bind = te.create_schedule(E.op)
    i, j = s_bind[D].op.axis
    bx, tx = s_bind[D].split(i, factor=32)
    by, ty = s_bind[D].split(j, factor=32)
    s_bind[D].bind(bx, te.thread_axis("blockIdx.x"))
    s_bind[D].bind(by, te.thread_axis("blockIdx.y"))
    s_bind[D].bind(tx, te.thread_axis("threadIdx.x"))
    s_bind[D].bind(ty, te.thread_axis("threadIdx.y"))
    #print(tvm.lower(s_bind, [A, B, E], simple_mode=True))
    print("\n计算图中的阶段数:", len(s_bind.stages), "（阶段数不变）")
    for stage in s_bind.stages:
        print(f"s_bind.stage: {stage}")


if __name__ == "__main__":
    demonstrate_schedule_impact()