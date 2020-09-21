import numpy as np
import torch, logging, click, sys, functools
import table_batched_embeddings_ops
logging.basicConfig(level=logging.DEBUG)


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


def benchmark_torch_function(iters, f, *args):
    f(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        f(*args)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters


def div_round_up(a, b):
    return int(((a + b - 1) // b) * b)


def benchmark_forward(B, E, T, L, D, iters, fp16, managed, mixed):
    logging.basicConfig(level=logging.DEBUG)
    import torch
    import table_batched_embeddings

    np.random.seed(42)
    if mixed:
        mixed_D = [
            div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(mixed_D)
    cc = (
        table_batched_embeddings_ops.TableBatchedEmbeddingBags(
            T,
            E,
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
            [(E, d) for d in mixed_D],
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

    zs = [
        torch.tensor(np.random.zipf(a=1.2, size=(B, L))).int().cuda()
        % E
        # torch.randint(low=0, high=E - 1, size=(T, B, L)).int().cuda()
    ]

    print(
        f"Duplicate proportion: {1.0 - np.unique(zs[0].detach().cpu().numpy()).size / zs[0].detach().cpu().numpy().size}"
    )
    merged_indices = torch.stack(zs, dim=0)

    merged_indices = torch.randint(low=0, high=E - 1, size=(T, B, L)).int().cuda()

    (indices, offsets) = get_table_batched_offsets_from_dense(merged_indices)
    assert indices.shape[0] == B * T * L
    assert all(
        l == L for l in (offsets[1:] - offsets[:-1]).detach().cpu().numpy().tolist()
    )
    per_sample_weights = None
    print(indices.shape, indices.min(), indices.max(), indices)
    y0 = (
        table_batched_embeddings.forward(
            cc.embedding_weights,
            cc.table_offsets,
            indices,
            offsets,
            per_sample_weights,
            L,
            1,
            False,
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
            False,
        )
    )

    for BT_block_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for shmem in [True, False]:
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

    for BT_block_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for shmem in [True, False]:
            time_per_iter = (
                benchmark_torch_function(
                    iters,
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
                f"Forward, B: {B} {(BT_block_size, shmem)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
            )

    go = torch.randn_like(y0)

    learning_rate = 0.05
    eps = 0.01
    # for BT_block_size in [1, 2, 4, 8, 16, 32]:
    #     for shmem in [True, False]:
    #         time_per_iter = benchmark_torch_function(
    #             iters,
    #             w3(table_batched_embeddings.backward_sgd),
    #             go,
    #             cc.embedding_weights,
    #             cc.table_offsets,
    #             indices,
    #             offsets,
    #             learning_rate,
    #             L,
    #             BT_block_size,
    #             shmem,
    #         )

    #         logging.info(
    #             f"Backward-SGD, B: {B} {(BT_block_size, shmem)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {2 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
    #         )
    for BT_block_size in [
        1,
        2,
        4,
        8,
        16,
        32,
    ]:
        for exact in [0, 1]:
            for stochastic in [0, 1] if fp16 else [0]:
                if not exact:
                    time_per_iter = (
                        benchmark_torch_function(
                            iters,
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
                    f"Backward-ADAGRAD-{'nonstochastic' if not stochastic else 'stochastic'}-{'EXACT' if exact else 'APPROX'}-{'R' if R else 'NR'}, B: {B} ({BT_block_size}), E: {E}, T: {T}, D: {D}, L: {L}, BW: {2 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
                )


@click.command()
@click.option("--num-tables", default=64)
@click.option("--num-embeddings", default=int(1e4))
@click.option("--embedding-dim", default=32)
@click.option("--batch-size", default=128)
@click.option("--bag-size", default=32)
@click.option("--iters", default=100)
@click.option("--fp16", is_flag=True, default=False)
@click.option("--managed", is_flag=True, default=False)
@click.option("--mixed", is_flag=True, default=False)
def cli(
    num_tables,
    num_embeddings,
    embedding_dim,
    batch_size,
    bag_size,
    iters,
    fp16,
    managed,
    mixed,
):
    def f():
        benchmark_forward(
            batch_size,
            num_embeddings,
            num_tables,
            bag_size,
            embedding_dim,
            iters,
            fp16,
            managed,
            mixed,
        )

    f()


if __name__ == "__main__":
    cli()
