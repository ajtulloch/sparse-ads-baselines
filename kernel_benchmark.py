import numpy as np
import torch, logging, click, functools
import table_batched_embeddings, table_batched_embeddings_ops
logging.basicConfig(level=logging.DEBUG)
np.random.seed(42)

def div_round_up(a, b):
    return int((a + b - 1) // b) * b


def get_table_batched_offsets_from_dense(merged_indices):
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.int().contiguous().view(-1).cuda(),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).int().cuda(),
    )


def benchmark_torch_function(iters, warmup_iters, f, *args, **kwargs):
    for _ in range(warmup_iters): # Warmup
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        start_event.record()
        f(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event) * 1.0e-3
    return total_time / iters


def benchmark_conv(batch_size, H, W, IC, OC, stride, dilation, FHW, is_dw, iters, warmup_iters, backward):
    input_feature = torch.randn(batch_size, IC, H, W).cuda()
    conv = torch.nn.Conv2d(IC, OC, FHW, stride=stride, dilation=dilation, groups=(IC if is_dw else 1)).cuda()

    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            conv,
            input_feature
        )
        logging.info(
            f"Conv, input size: ({batch_size}, {IC}, {H}, {W}), filter size ({OC}, {IC}, {FHW}, {FHW}) \
                BW: {(batch_size * H * W * IC + FHW * FHW * IC * OC + batch_size * H * W * OC) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )
    else:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            conv.mean().backward,
            retain_graph=True
        )
        logging.info(
            f"Conv backward, input size: ({batch_size}, {IC}, {H}, {W}), filter size ({OC}, {IC}, {FHW}, {FHW}) \
                BW: {(batch_size * H * W * IC + FHW * FHW * IC * OC + batch_size * H * W * OC) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )


def benchmark_linear(M, N, K, iters, warmup_iters, backward):
    A = torch.randn(M, K, requires_grad=True).cuda()
    linear = torch.nn.Linear(K, N).cuda()
    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            linear,
            A
        )
        logging.info(
            f"Linear forward, input size: ({M}, {K}), linear in size ({K}), linear out size ({N}),\
                BW: {(M * K + N * K + M * N) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )
    else:
        out = linear(A)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            out.mean().backward,
            retain_graph=True,
        )
        logging.info(
            f"Linear backward, input size: ({M}, {K}), linear in size ({K}), linear out size ({N}),\
                BW: {(M * K + N * K + M * N) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )


def benchmark_fc(batch_size, M, N, K, iters, warmup_iters, backward):
    if batch_size == 1:
        A = torch.randn(M, K, requires_grad=True).cuda()
        B = torch.randn(N, K, requires_grad=True).cuda()
        C = torch.randn(M, N, requires_grad=True).cuda()
        if not backward:
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                torch.addmm,
                C, A, B.T,
            )
            logging.info(
                f"Addmm, tensor A size: ({M}, {K}), tensor B size: ({K}, {N}),\
                    BW: {(M * K + N * K + M * N) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
            )
        else:
            torch.addmm(C, A, B.T)
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                C.mean().backward,
                retain_graph=True,
            )
            logging.info(
                f"AddmmBackward, tensor A size: ({M}, {K}), tensor B size: ({K}, {N}),\
                    BW: {(M * K + N * K + M * N) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
            )

    else:
        A = torch.randn(batch_size, M, K, requires_grad=True).cuda()
        B = torch.randn(batch_size, N, K, requires_grad=True).cuda()
        if not backward:
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                torch.bmm,
                A, torch.transpose(B, 1, 2),
            )
            logging.info(
                f"Bmm, tensor A size: ({batch_size}, {M}, {K}), tensor B size: ({batch_size}, {K}, {N}),\
                    BW: {batch_size * (M * K + N * K + M * N) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
            )
        else:
            C = torch.bmm(A, torch.transpose(B, 1, 2))
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                C.mean().backward,
                retain_graph=True,
            )
            logging.info(
                f"BmmBackward, tensor A size: ({batch_size}, {M}, {K}), tensor B size: ({batch_size}, {K}, {N}),\
                    BW: {batch_size * (M * K + N * K + M * N) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
            )

def benchmark_tril(batch_size, M, N, diag, iters, warmup_iters, backward):
    assert M == N, "Input tensor should be square!"
    Z = torch.randn(batch_size, M, N, requires_grad=True).cuda()
    li = torch.tensor([i for i in range(M) for j in range(i + diag)])
    lj = torch.tensor([j for i in range(N) for j in range(i + diag)])
    def zflat_wrapper(Z, i, j):
        return Z[:, i, j]

    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            zflat_wrapper,
            Z,
            li,
            lj
        )
        logging.info(
            f"Index forward, tensor size: ({batch_size}, {M}, {N}),\
                BW: unknown, Time: {time_per_iter * 1.0e6:.0f}us"
        )
    else:
        out = zflat_wrapper(Z, li, lj)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            out.mean().backward,
            retain_graph=True,
        )
        logging.info(
            f"Index forward, tensor size: ({batch_size}, {M}, {N}),\
                BW: unknown, Time: {time_per_iter * 1.0e6:.0f}us"
        )


def benchmark_bn(batch_size, H, W, OC, iters, warmup_iters, backward):
    out_feature = torch.randn(batch_size, OC, H, W, requires_grad=True).cuda()
    bn = torch.nn.BatchNorm2d(OC).cuda()

    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            bn,
            out_feature
        )
        logging.info(
            f"BN forward, tensor size: ({batch_size}, {OC}, {H}, {W}),\
                BW: {batch_size * H * W * OC * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )
    else:
        output = bn(out_feature)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            output.mean().backward,
            retain_graph=True,
        )
        logging.info(
            f"BN forward, tensor size: ({batch_size}, {OC}, {H}, {W}),\
                BW: {batch_size * H * W * OC * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )


def benchmark_concat(batch_size, M, N, K, iters, warmup_iters):
    A = torch.randn(batch_size, M, K).cuda()
    B = torch.randn(batch_size, N, K).cuda()

    time_per_iter = benchmark_torch_function(
        iters,
        warmup_iters,
        torch.cat,
        (A, B),
        dim=1
    )

    logging.info(
        f"Concat, tensor A size: ({batch_size}, {M}, {K}), tensor B size: ({batch_size}, {N}, {K}),\
            BW: {2 * (batch_size * M * K + batch_size * N * K) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
    )


def benchmark_memcpy(batch_size, M, N, iters, warmup_iters):
    A = torch.randn(batch_size, M, N)

    time_per_iter = benchmark_torch_function(
        iters,
        warmup_iters,
        A.to,
        device="cuda"
    )

    logging.info(
        f"Memcpy, size: ({batch_size}, {M}, {N}), \
            BW: {(batch_size * M * N) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
    )


def benchmark_transpose(batch_size, M, N, trans_type, iters, warmup_iters):
    A = torch.randn(batch_size, M, N).cuda()
    if trans_type == 0:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            A.permute(0, 2, 1).contiguous
        )
    elif trans_type == 1:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            A.permute(2, 1, 0).contiguous
        )
    else: # 2
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            A.permute(1, 0, 2).contiguous
    )

    logging.info(
        f"Transpose, size: ({batch_size}, {M}, {N}), trans_type: {trans_type}, \
            BW: {(batch_size * M * N) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
    )


def benchmark_relu(batch_size, M, N, iters, warmup_iters):
    A = torch.randn(batch_size, M, N).cuda()
    time_per_iter = benchmark_torch_function(
        iters,
        warmup_iters,
        torch.relu,
        A
    )

    logging.info(
        f"ReLU, size: ({batch_size}, {M}, {N}), \
            BW: {(batch_size * M * N) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
    )


def benchmark_embedding_lookup(B, E, T, L, D, BT_block_size, iters, warmup_iters, backward, shmem, sgd, fp16, managed, mixed):
    Es = [int(x) for x in E.split('-')]
    if len(Es) == 1:
        Es = Es * T
    assert len(Es) == T

    if mixed:
        mixed_D = [
            div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(mixed_D)
    cc = (
        table_batched_embeddings_ops.TableBatchedEmbeddingBags(
            T,
            Es,
            D,
            optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
            learning_rate=0.1,
            managed=table_batched_embeddings_ops.EmbeddingLocation.DEVICE
            if not managed
            else table_batched_embeddings_ops.EmbeddingLocation.HOST_MAPPED,
            eps=0.1,
            stochastic_rounding=False,
            fp16=fp16,
        ).cuda()
        if not mixed
        else table_batched_embeddings_ops.MixedDimTableBatchedEmbeddingBags(
            [(Es, d) for d in mixed_D],
            optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
            learning_rate=0.1,
            managed=table_batched_embeddings_ops.EmbeddingLocation.DEVICE
            if not managed
            else table_batched_embeddings_ops.EmbeddingLocation.HOST_MAPPED,
            eps=0.1,
            stochastic_rounding=False,
            fp16=fp16,
        ).cuda()
    )

    logging.info(
        f"Embedding parameters: {cc.embedding_weights.numel() / 1.0e9:.2f}GParam"
    )

    R = False

    def w2(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(w, o, x, *args):
            c(w, o, x.random_(0, E - 1), *args)

        return z

    def w3(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(g, w, o, x, *args):
            c(g, w, o, x.random_(0, E - 1), *args)

        return z

    def w4(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(g, w, o, a, x, *args):
            c(g, w, o, a, x.random_(0, E - 1), *args)

        return z

    def w6(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(g, w, o, a, b, d, x, *args):
            c(g, w, o, a, b, d, x.random_(0, E - 1), *args)

        return z

    idxs = []
    for x in range(T):
        idxs.append(torch.randint(low=0, high=Es[x] - 1, size=(B, L)).int().cuda())
    merged_indices = torch.stack(idxs, dim=0)

    (indices, offsets) = get_table_batched_offsets_from_dense(merged_indices)

    assert indices.shape[0] == B * T * L
    assert all(
        l == L for l in (offsets[1:] - offsets[:-1]).detach().cpu().numpy().tolist()
    )
    per_sample_weights = None
    stochastic = False # TODO: Fix this
    exact = 1
    y0 = (
        table_batched_embeddings.forward(
            cc.embedding_weights,
            cc.table_offsets,
            indices,
            offsets,
            per_sample_weights,
            L,
            1,
            shmem,
        )
        if not mixed
        else table_batched_embeddings.forward_mixed_D(
            cc.embedding_weights,
            cc.table_offsets,
            cc.dim_offsets,
            cc.total_D,
            indices,
            offsets,
            per_sample_weights,
            L,
            1,
            shmem,
        )
    )

    y = (
        table_batched_embeddings.forward(
            cc.embedding_weights,
            cc.table_offsets,
            indices,
            offsets,
            per_sample_weights,
            L,
            BT_block_size,
            shmem,
        )
        if not mixed
        else table_batched_embeddings.forward_mixed_D(
            cc.embedding_weights,
            cc.table_offsets,
            cc.dim_offsets,
            cc.total_D,
            indices,
            offsets,
            per_sample_weights,
            L,
            BT_block_size,
            False,
        )
    )
    torch.testing.assert_allclose(y, y0)

    if not backward:
        time_per_iter = (
            benchmark_torch_function(
                iters,
                warmup_iters,
                w2(table_batched_embeddings.forward),
                cc.embedding_weights,
                cc.table_offsets,
                indices,
                offsets,
                per_sample_weights,
                L,
                BT_block_size,
                shmem,
            )
            if not mixed
            else benchmark_torch_function(
                iters,
                warmup_iters,
                w4(table_batched_embeddings.forward_mixed_D),
                cc.embedding_weights,
                cc.table_offsets,
                cc.dim_offsets,
                cc.total_D,
                indices,
                offsets,
                per_sample_weights,
                L,
                BT_block_size,
                shmem,
            )
        )
        logging.info(
            f"Embedding Lookup Forward, B: {B} {(BT_block_size, shmem)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )

    else: # backward
        go = torch.randn_like(y0)

        learning_rate = 0.05
        eps = 0.01

        if sgd:
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                w3(table_batched_embeddings.backward_sgd),
                go,
                cc.embedding_weights,
                cc.table_offsets,
                indices,
                offsets,
                learning_rate,
                L,
                BT_block_size,
                shmem,
            )

            logging.info(
                f"Embedding Lookup Backward-SGD, B: {B} {(BT_block_size, shmem)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {2 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
            )
        else: # adagrad
            if not exact:
                time_per_iter = (
                    benchmark_torch_function(
                        iters,
                        warmup_iters,
                        w3(table_batched_embeddings.backward_approx_adagrad),
                        go,
                        cc.embedding_weights,
                        cc.table_offsets,
                        indices,
                        offsets,
                        per_sample_weights,
                        cc.optimizer_state,
                        learning_rate,
                        eps,
                        L,
                        stochastic,
                        BT_block_size,
                    )
                    if not mixed
                    else benchmark_torch_function(
                        iters,
                        warmup_iters,
                        w6(
                            table_batched_embeddings.backward_approx_adagrad_mixed_D
                        ),
                        go,
                        cc.embedding_weights,
                        cc.table_offsets,
                        cc.table_dim_offsets,
                        cc.dim_offsets,
                        cc.total_D,
                        indices,
                        offsets,
                        per_sample_weights,
                        cc.optimizer_state,
                        learning_rate,
                        eps,
                        L,
                        stochastic,
                        BT_block_size,
                    )
                )
            else:
                time_per_iter = (
                    benchmark_torch_function(
                        iters,
                        warmup_iters,
                        w3(table_batched_embeddings.backward_exact_adagrad),
                        go,
                        cc.embedding_weights,
                        cc.table_offsets,
                        indices,
                        offsets,
                        per_sample_weights,
                        cc.optimizer_state,
                        learning_rate,
                        eps,
                        stochastic,
                        BT_block_size,
                    )
                    if not mixed
                    else benchmark_torch_function(
                        iters,
                        warmup_iters,
                        w6(table_batched_embeddings.backward_exact_adagrad_mixed_D),
                        go,
                        cc.embedding_weights,
                        cc.table_offsets,
                        cc.table_dim_offsets,
                        cc.dim_offsets,
                        cc.total_D,
                        indices,
                        offsets,
                        per_sample_weights,
                        cc.optimizer_state,
                        learning_rate,
                        eps,
                        stochastic,
                        BT_block_size,
                    )
                )

            logging.info(
                f"Embedding Lookup Backward-ADAGRAD-{'nonstochastic' if not stochastic else 'stochastic'}-{'EXACT' if exact else 'APPROX'}-{'R' if R else 'NR'}, B: {B} ({BT_block_size}), E: {E}, T: {T}, D: {D}, L: {L}, BW: {2 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
            )


@click.command()
@click.option("--op-type", default="embedding_lookup")
@click.option("--iters", default=100)
@click.option("--warmup-iters", default=5)
@click.option("--backward", is_flag=True, default=False)
@click.option("--batch-size", default=128)
# Embedding lookup
@click.option("--num-embeddings", default="1000") # str by default in case 'x-y-z'
@click.option("--num-tables", default=64)
@click.option("--bag-size", default=38)
@click.option("--embedding-dim", default=32)
@click.option("--rows-per-block", default=32)
@click.option("--shmem", is_flag=True, default=False)
@click.option("--sgd", is_flag=True, default=False)
@click.option("--fp16", is_flag=True, default=False)
@click.option("--managed", is_flag=True, default=False)
@click.option("--mixed", is_flag=True, default=False)
# GEMM and transpose tril and more
@click.option("--M", default=512)
@click.option("--N", default=512)
@click.option("--K", default=512)
@click.option("--trans-type", default=0)
@click.option("--diag", default=0)
# Conv
@click.option("--H", default=64)
@click.option("--W", default=64)
@click.option("--IC", default=64)
@click.option("--OC", default=64)
@click.option("--stride", default=1)
@click.option("--dilation", default=1)
@click.option("--FHW", default=3)
@click.option("--is-dw", is_flag=True, default=False)
def cli(
    op_type,
    iters,
    warmup_iters,
    backward,
    batch_size,
    num_embeddings,
    num_tables,
    bag_size,
    embedding_dim,
    rows_per_block,
    shmem,
    sgd,
    fp16,
    managed,
    mixed,
    m,
    n,
    k,
    trans_type,
    diag,
    h,
    w,
    ic,
    oc,
    stride,
    dilation,
    fhw,
    is_dw
):
    if op_type == "embedding_lookup":
        benchmark_embedding_lookup(
            batch_size,
            num_embeddings,
            num_tables,
            bag_size,
            embedding_dim,
            rows_per_block,
            iters,
            warmup_iters,
            backward,
            shmem,
            sgd,
            fp16,
            managed,
            mixed,
        )
    elif op_type == "fully_connected":
        benchmark_fc(
            batch_size,
            m,
            n,
            k,
            iters,
            warmup_iters,
            backward,
        )
    elif op_type == "linear":
        benchmark_linear(
            m,
            n,
            k,
            iters,
            warmup_iters,
            backward,
        )
    elif op_type == "conv":
        benchmark_conv(
            batch_size,
            h,
            w,
            ic,
            oc,
            stride,
            dilation,
            fhw,
            is_dw,
            iters,
            warmup_iters,
            backward,
        )
    elif op_type == "concat":
        benchmark_concat(
            batch_size,
            m,
            n,
            k,
            iters,
            warmup_iters,
        )
    elif op_type == "memcpy":
        benchmark_memcpy(
            batch_size,
            m,
            n,
            iters,
            warmup_iters,
        )
    elif op_type == "transpose":
            benchmark_transpose(
            batch_size,
            m,
            n,
            trans_type,
            iters,
            warmup_iters,
        )
    elif op_type == "relu":
        benchmark_relu(
            batch_size,
            m,
            n,
            iters,
            warmup_iters,
        )
    elif op_type == "tril":
        benchmark_tril(
            batch_size,
            m,
            n,
            diag,
            iters,
            warmup_iters,
            backward,
        )
    elif op_type == "bn":
        benchmark_bn(
            batch_size,
            h,
            w,
            oc,
            iters,
            warmup_iters,
            backward,
        )
    else:
        raise Exception("Op type not supported!")


if __name__ == "__main__":
    cli()
