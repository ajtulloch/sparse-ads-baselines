import numpy as np
import torch, logging, click, functools
import table_batched_embeddings, table_batched_embeddings_ops
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    ComputeDevice,
    EmbeddingLocation,
    OptimType,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
)
logging.basicConfig(level=logging.DEBUG)
np.random.seed(42)
torch.manual_seed(42)

def div_round_up(a, b):
    return int((a + b - 1) // b) * b


def get_table_batched_offsets_from_dense(merged_indices):
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    indices = merged_indices.contiguous().view(-1).long()
    offsets = torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long()

    assert indices.shape[0] == B * T * L
    assert all(
        l == L for l in (offsets[1:] - offsets[:-1]).numpy().tolist()
    )
    return (indices.cuda(), offsets.cuda())


def get_reuse_factor(indices):
    _, indices_counts = torch.unique(indices, return_counts=True)
    unique_counts, counts_counts = torch.unique(indices_counts, return_counts=True)
    total_counts = counts_counts.sum().item()

    if total_counts == 0:
        bin_counts = [0.0 for _ in range(17)]
    else:
        start, end = 0, 1
        bin_counts = []
        for _ in range(16):
            bin_counts.append(counts_counts[(unique_counts > start) & (unique_counts <= end)].sum().item())
            start = end
            end *= 2
        bin_counts.append(counts_counts[unique_counts > start].sum().item())
        bin_counts = [x/total_counts for x in bin_counts]
    return bin_counts


def generate_emb_data(B, Es, T, Ls):
    idxs = []
    for i in range(T):
        idxs.append(torch.randint(low=0, high=Es[i] - 1, size=(B, Ls[i])).int().cuda())
    merged_indices = torch.stack(idxs, dim=0)

    (indices, offsets) = get_table_batched_offsets_from_dense(merged_indices)
    return indices, offsets, get_reuse_factor(indices)


def get_emb_data_from_dataset(dataset_path, B, TIDs):
    # e.g. (dlrm_datasets) offsets: [856 * 65536, 1], lengths: [856, 65536]
    indices, offsets, lengths = torch.load(dataset_path)
    _, total_batch_size = lengths.shape # L per table per batch

    lS_indices, lS_offsets, lS_bin_counts, Es, Ls = [], [], [], [], []

    for t in TIDs:
        start = t * total_batch_size
        end = (t + 1) * total_batch_size + 1
        table_offsets = offsets[start:end]
        table_indices = indices[table_offsets[0]:table_offsets[-1]] # length = total number of lookups
        table_offsets = table_offsets - offsets[start] # length = total_batch_size

        # # All non-zero lookup batches
        # Ls_nonzero = torch.nonzero(lengths[t])

        # # B batches that have non-zero lookups
        # b_idx = np.random.choice(len(Ls_nonzero), B, replace=False)
        # Ls = lengths[t][Ls_nonzero[b_idx]]

        b_idx = np.sort(np.random.choice(total_batch_size, B, replace=False))

        ind = []
        os = []
        o = 0
        for x, L in zip(table_offsets[b_idx], lengths[t][b_idx]):
            ind.append(table_indices[x:(x+L)])
            os.append(o)
            o += L.item()
        os.append(o)
        batch_indices = torch.cat(ind, dim=0).long()
        lS_indices.append(batch_indices)
        lS_offsets.append(torch.tensor(os))
        lS_bin_counts.append(get_reuse_factor(batch_indices))

        E = table_indices.max().int().item() + 1 if len(table_indices) > 0 else 1
        Es.append(max(100, E)) # If less than 100 rows, make it 100
        Ls.append(sum(lengths[t][b_idx]).item() / B) # Average L per table

    args_indices = torch.cat([x.view(-1) for x in lS_indices], dim=0).int() # Concatenated indices
    E_offsets = [0] + np.cumsum([x.view(-1).shape[0] for x in lS_indices]).tolist() # Cumsum of number-of-lookups per table
    args_offsets = torch.cat([torch.tensor([0])] + [x[1:] + y for x, y in zip(lS_offsets, E_offsets[:-1])], dim=0).int() # Offset of each lookup when lookups are concatenated

    return [args_indices.cuda(), args_offsets.cuda(), lS_bin_counts, Es, Ls]


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


def benchmark_fbgemm(iters, warmup_iters, op, input_data, grads_tensor, fwbw, *args, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for idx in range(warmup_iters): # Warmup
        indices, offsets = input_data[idx]
        op(indices, offsets, *args, **kwargs)
    torch.cuda.synchronize()
    total_time = 0.0
    for idx in range(warmup_iters, warmup_iters + iters):
        indices, offsets = input_data[idx]
        if fwbw == "fw":
            start_event.record()
            op(indices, offsets, *args, **kwargs)
            end_event.record()
        else: # bw
            tmp = op(indices, offsets, *args, **kwargs)
            torch.cuda.synchronize()
            start_event.record()
            tmp.backward(grads_tensor)
            end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event) * 1.0e-3
    return total_time / iters


def benchmark_conv2d(batch_size, H, W, IC, OC, stride, dilation, FH, FW, is_dw, iters, warmup_iters, backward):
    input_feature = torch.randn(batch_size, IC, H, W, requires_grad=True).cuda()
    padding = []
    for f in [FH, FW]:
        padding.append((f - 1) // 2) # Only consider SAME with dilation = 1 for now
    conv = torch.nn.Conv2d(IC, OC, (FH, FW), stride=stride, dilation=dilation, padding=padding, groups=(IC if is_dw else 1)).cuda()

    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            conv,
            input_feature
        )
        logging.info(
            f"Conv2d, input size: ({batch_size}, {IC}, {H}, {W}), filter size ({OC}, {IC}, {FH}, {FW}) \
                BW: {(batch_size * H * W * IC + FH * FW * IC * OC + batch_size * H * W * OC) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )
    else:
        out = conv(input_feature)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            out.mean().backward,
            retain_graph=True
        )
        logging.info(
            f"Conv2d backward, input size: ({batch_size}, {IC}, {H}, {W}), filter size ({OC}, {IC}, {FH}, {FW}) \
                BW: {(batch_size * H * W * IC + FH * FW * IC * OC + batch_size * H * W * OC) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )


# For parallel multi-head mm
def benchmark_conv1d(batch_size, L, IC, OC, groups, iters, warmup_iters, backward):
    input_feature = torch.randn(batch_size, IC * groups, L, requires_grad=True).cuda()
    conv = torch.nn.Conv1d(IC * groups, OC * groups, kernel_size=1, groups=groups).cuda()

    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            conv,
            input_feature
        )
        logging.info(
            f"Conv1d, input size: ({batch_size}, {IC * groups}, {L}), filter size ({IC * groups}, {OC * groups}) \
                BW: {(batch_size * IC * L + IC * OC * groups + batch_size * OC * L * groups) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
        )
    else:
        out = conv(input_feature)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            out.mean().backward,
            retain_graph=True
        )
        logging.info(
            f"Conv1d backward, ({batch_size}, {IC * groups}, {L}), filter size ({IC * groups}, {OC * groups}) \
                BW: {(batch_size * IC * L + IC * OC * groups + batch_size * OC * L * groups) * 4 / time_per_iter / 1.0e9: .2f}GB/s, Time: {time_per_iter * 1.0e6:.0f}us"
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
            f"BN backward, tensor size: ({batch_size}, {OC}, {H}, {W}),\
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
    D = int(D)

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
            optimizer=table_batched_embeddings_ops.Optimizer.SGD
            if sgd
            else table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
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
            optimizer=table_batched_embeddings_ops.Optimizer.SGD
            if sgd
            else table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
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

    (indices, offsets) = generate_emb_data(B, Es, T, L)

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
    torch.testing.assert_close(y, y0)

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


# Following the fashion of the old benchmark. Should probably change to the style of FBGEMM's new benchmark 
# for diverse indices distribution per requests.
def benchmark_embedding_lookup_fbgemm(B, E, T, L, D, iters, warmup_iters, backward, sgd, fp16, managed, caching, dataset):
    assert torch.cuda.is_available(), "GPU not found!"
    assert not (L == 0 and dataset is None) # When L = 0 there should be dataset

    Es = [int(x) for x in E.split('-')] # Borrow E for TIDs for dataset benchmark
    if len(Es) == 1:
        Es = Es * T
    assert len(Es) == T
    Ds = [int(x) for x in D.split('-')]
    if len(Ds) == 1:
        Ds = Ds * T
    assert len(Ds) == T

    data = []
    if dataset is not None:
        TIDs = Es
        indices, offsets, reuse_factors, Es, Ls = get_emb_data_from_dataset(dataset, B, TIDs)
        data = [(indices, offsets)] * (warmup_iters + iters)
    else:
        for _ in range(warmup_iters + iters):
            indices, offsets, reuse_factors = generate_emb_data(B, Es, T, L)
        data.append((indices, offsets))
    grads_tensor = torch.randn(B, sum(Ds)).cuda()
    fwbw = "bw" if backward else "fw"

    placement = EmbeddingLocation.DEVICE
    if caching:
        placement = EmbeddingLocation.MANAGED_CACHING
    elif managed:
        placement = EmbeddingLocation.MANAGED
    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                e,
                d,
                placement,
                ComputeDevice.CUDA
            )
            for e, d in zip(Es, Ds)
        ],
        optimizer=OptimType.EXACT_SGD if sgd else OptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        eps=0.1,
        weights_precision=SparseType.FP16 if fp16 else SparseType.FP32,
        stochastic_rounding=False,
        output_dtype=SparseType.FP32,
    ).to(torch.cuda.current_device())

    time_per_iter = benchmark_fbgemm(iters, warmup_iters, emb, data, grads_tensor, fwbw)

    logging.info(
        "Reuse factors: {}".format('_'.join(['-'.join([f'{rf:.8f}' for rf in t_rf]) for t_rf in reuse_factors]))
    )
    if not backward:
        logging.info(
            f"Embedding Lookup Forward (FBGEMM), B: {B}, E: {'-'.join([str(e) for e in Es])}, T: {T}, D: {'-'.join([str(d) for d in Ds])}, L: {'-'.join([str(l) for l in Ls])}, Time: {time_per_iter * 1.0e6:.0f}us"
        )
    else: # backward
        logging.info(
            f"Embedding Lookup Backward-SGD (FBGEMM), B: {B}, E: {'-'.join([str(e) for e in Es])}, T: {T}, D: {'-'.join([str(d) for d in Ds])}, L: {'-'.join([str(l) for l in Ls])}, Time: {time_per_iter * 1.0e6:.0f}us"
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
@click.option("--embedding-dim", default="32")
@click.option("--rows-per-block", default=32)
@click.option("--shmem", is_flag=True, default=False)
@click.option("--sgd", is_flag=True, default=False)
@click.option("--fp16", is_flag=True, default=False)
@click.option("--managed", is_flag=True, default=False)
@click.option("--mixed", is_flag=True, default=False)
@click.option("--fbgemm", is_flag=True, default=False)
@click.option("--caching", is_flag=True, default=False)
@click.option("--dataset", default=None)
# GEMM and transpose tril and more
@click.option("--M", default=512)
@click.option("--N", default=512)
@click.option("--K", default=512)
@click.option("--trans-type", default=0)
@click.option("--diag", default=0)
# Convs
@click.option("--IC", default=64)
@click.option("--OC", default=64)
# Conv2d and BN
@click.option("--H", default=64)
@click.option("--W", default=64)
@click.option("--stride", default=1)
@click.option("--dilation", default=1)
@click.option("--FH", default=3)
@click.option("--FW", default=3)
@click.option("--is-dw", is_flag=True, default=False)
# Conv1d
@click.option("--L", default=64)
@click.option("--groups", default=16)
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
    fbgemm,
    caching,
    dataset,
    m,
    n,
    k,
    trans_type,
    diag,
    ic,
    oc,
    h,
    w,
    stride,
    dilation,
    fh,
    fw,
    is_dw,
    l,
    groups,
):
    if op_type == "embedding_lookup":
        if fbgemm:
            benchmark_embedding_lookup_fbgemm(
                batch_size,
                num_embeddings,
                num_tables,
                bag_size,
                embedding_dim,
                iters,
                warmup_iters,
                backward,
                sgd,
                fp16,
                managed,
                caching,
                dataset,
            )
        else:
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
    elif op_type == "conv2d":
        benchmark_conv2d(
            batch_size,
            h,
            w,
            ic,
            oc,
            stride,
            dilation,
            fh,
            fw,
            is_dw,
            iters,
            warmup_iters,
            backward,
        )
    elif op_type == "conv1d":
        benchmark_conv1d(
            batch_size,
            l,
            ic,
            oc,
            groups,
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

