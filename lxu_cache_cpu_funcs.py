import os

os.environ["TVM_NUM_THREADS"] = os.environ.get("TVM_NUM_THREADS", "40")

FCACHE, BCACHE = {}, {}


def get_forward_cpu_handle(B, E, D):
    global FCACHE
    key = (B, E, D)
    if key in FCACHE:
        return FCACHE[key].handle.value

    import tvm

    n = tvm.te.var("N")
    W = tvm.te.placeholder((E, D), dtype="float32", name="W")
    I = tvm.te.placeholder((n,), dtype="int32", name="I")
    O = tvm.te.placeholder((B + 1,), dtype="int32", name="O")
    MASK = tvm.te.placeholder((n,), dtype="int8", name="MASK")

    # TODO: tune?
    BLOCK = 3
    ACC = 2

    def sls(W, I, O, MASK):
        def ir(Y, W, I, O, MASK):
            ib = tvm.tir.ir_builder.create()
            Yb = ib.buffer_ptr(Y)
            Wb = ib.buffer_ptr(W)
            Ib = ib.buffer_ptr(I)
            Ob = ib.buffer_ptr(O)
            MASKb = ib.buffer_ptr(MASK)
            with ib.for_range(0, B, name="b", for_type="parallel") as b:
                block = ib.allocate(
                    "float32", 2 * BLOCK * D, name="block", scope="local"
                )
                indices_start, indices_end = Ob[b], Ob[b + 1]
                L = indices_end - indices_start
                Lblock = tvm.tir.floordiv(L, BLOCK)
                acc = ib.allocate("float32", ACC * D, name="acc", scope="local")
                with ib.for_range(
                    0, ACC * D, name="d", for_type="vectorize"
                ) as d:
                    acc[d] = 0.0
                # fetch block 0
                with ib.for_range(0, Lblock, name="l_block") as l_block:
                    for lb in range(BLOCK):
                        i = l_block * BLOCK + lb
                        ind = Ib[indices_start + i]
                        mask = MASKb[indices_start + i]
                        with ib.if_scope(mask != tvm.tir.IntImm("int8", 0)):
                            with ib.for_range(
                                0, D, name="d", for_type="vectorize"
                            ) as d:
                                acc[(lb % ACC) * D + d] += Wb[ind * D + d]
                with ib.for_range(0, L - Lblock * BLOCK, name="l_rem") as l_rem:
                    i = l_rem + Lblock * BLOCK
                    ind = Ib[indices_start + i]
                    mask = MASKb[indices_start + i]
                    with ib.if_scope(mask != tvm.tir.IntImm("int8", 0)):
                        with ib.for_range(
                            0, D, name="d", for_type="vectorize"
                        ) as d:
                            acc[d] += Wb[ind * D + d]
                for pacc in range(1, ACC):
                    with ib.for_range(
                        0, D, name="d", for_type="vectorize"
                    ) as d:
                        acc[d] += acc[pacc * D + d]
                with ib.for_range(0, D, name="d", for_type="vectorize") as d:
                    Yb[b * D + d] = acc[d]
            return ib.get()

        return tvm.te.extern(
            (B, D),
            [W, I, O, MASK],
            lambda ins, outs: ir(outs[0], ins[0], ins[1], ins[2], ins[3]),
            name="sls_fwd",
            dtype="float32",
        )

    Y = sls(W, I, O, MASK)
    s = tvm.te.create_schedule(Y.op)
    # print(tvm.lower(s, [W, I, O, MASK, Y]))
    target = tvm.target.create("llvm -mcpu=core-avx2")
    with target:
        f = tvm.build(s, [W, I, O, MASK, Y])
    FCACHE[key] = f.get_function(f.entry_name)
    return get_forward_cpu_handle(B, E, D)


def get_backward_cpu_handle(B, E, D):
    global BCACHE
    key = (B, E, D)
    if key in BCACHE:
        return BCACHE[key].handle.value

    import tvm

    NT = int(os.environ["TVM_NUM_THREADS"])
    n = tvm.te.var("N")
    W = tvm.te.placeholder((E, D), dtype="float32", name="W")
    I = tvm.te.placeholder((n,), dtype="int32", name="I")
    O = tvm.te.placeholder((B + 1,), dtype="int32", name="O")
    MASK = tvm.te.placeholder((n,), dtype="int8", name="MASK")

    learning_rate = tvm.tir.Var("lr", dtype="float32")

    def sls_bwd(GO, W, I, O, MASK, learning_rate):
        def ir(GO, W, I, O, MASK):
            ib = tvm.tir.ir_builder.create()
            Wb = ib.buffer_ptr(W)
            GOb = ib.buffer_ptr(GO)
            Ib = ib.buffer_ptr(I)
            Ob = ib.buffer_ptr(O)
            MASKb = ib.buffer_ptr(MASK)

            with ib.for_range(0, NT, name="nt", for_type="parallel") as nt:
                M_start = tvm.tir.floordiv(E * nt, NT)
                M_end = tvm.tir.floordiv(E * (nt + 1), NT)
                with ib.for_range(0, B, name="b") as b:
                    indices_start, indices_end = Ob[b], Ob[b + 1]
                    with ib.for_range(
                        0, indices_end - indices_start, name="i"
                    ) as i:
                        ind = Ib[i + indices_start]
                        with ib.if_scope(ind >= M_start):
                            with ib.if_scope(ind < M_end):
                                mask = MASKb[i + indices_start]
                                with ib.if_scope(
                                    mask != tvm.tir.IntImm("int8", 0)
                                ):
                                    with ib.for_range(
                                        0, D, name="d", for_type="vectorize"
                                    ) as d:
                                        Wb[ind * D + d] -= (
                                            learning_rate * GOb[b * D + d]
                                        )
            return ib.get()

        return tvm.te.extern(
            # alignment stuff
            (32,),
            [GO, W, I, O, MASK],
            lambda ins, outs: ir(ins[0], ins[1], ins[2], ins[3], ins[4]),
            name="sls_bwd",
            dtype="float32",
        )

    GO = tvm.te.placeholder((B, D), "float32", name="GO")
    GW = sls_bwd(GO, W, I, O, MASK, learning_rate=learning_rate)
    s = tvm.te.create_schedule(GW.op)
    # print(tvm.lower(s, [GO, W, I, O, MASK, learning_rate]))
    target = tvm.target.create("llvm -mcpu=core-avx2")
    with target:
        f = tvm.build(s, [GO, W, I, O, MASK, learning_rate])
    BCACHE[key] = f.get_function(f.entry_name)
    return get_backward_cpu_handle(B, E, D)


def lxu_cache_forward_mixed_cpu_cuda(
    weights,
    indices,
    offsets,
    lxu_cache_locations,
    lxu_cache_weights,
    B_block_size,
    indices_cpu,
    offsets_cpu,
    mask_cpu,
    output_cpu,
    handle,
    cpu_stream,
    cpu_event_start,
    cpu_event_finish,
):
    import torch
    import table_batched_embeddings

    cpu_event_start.record()
    gpu_output = table_batched_embeddings.lxu_cache_forward_mixed_cuda(
        weights,
        indices,
        offsets,
        None,
        lxu_cache_locations,
        lxu_cache_weights,
        B_block_size,
    )
    with torch.cuda.stream(cpu_stream):
        cpu_event_start.wait()
        table_batched_embeddings.lxu_cache_forward_cpu(
            weights,
            indices_cpu,
            offsets_cpu,
            None,
            mask_cpu,
            output_cpu,
            handle,
        )
        output_cpu_gpu = output_cpu.to(gpu_output.device, non_blocking=True)
        cpu_event_finish.record()
    cpu_event_finish.wait()
    return gpu_output + output_cpu_gpu

# def lxu_cache_forward_mixed_cpu_cuda(
#     weights,
#     indices,
#     offsets,
#     lxu_cache_locations,
#     lxu_cache_weights,
#     B_block_size,
#     indices_cpu,
#     offsets_cpu,
#     mask_cpu,
#     output_cpu,
#     handle,
#     cpu_stream,
#     cpu_event_start,
#     cpu_event_finish,
# ):
#     import torch
#     import table_batched_embeddings

#     gpu_output = table_batched_embeddings.lxu_cache_forward_mixed_cuda(
#         weights,
#         indices,
#         offsets,
#         None,
#         lxu_cache_locations,
#         lxu_cache_weights,
#         B_block_size,
#     )
#     table_batched_embeddings.lxu_cache_forward_cpu(
#         weights,
#         indices_cpu,
#         offsets_cpu,
#         None,
#         mask_cpu,
#         output_cpu,
#         handle,
#     )
#     return gpu_output + output_cpu.to(gpu_output.device)    