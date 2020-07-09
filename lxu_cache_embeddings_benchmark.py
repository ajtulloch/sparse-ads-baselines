import functools
import click
import numpy as np
import logging

import sys
import torch
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
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        f(*args)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters


def benchmark_microbenchmark(B, E, L, D, C, iters, fp16, managed, mixed):
    logging.basicConfig(level=logging.DEBUG)
    import torch
    import table_batched_embeddings

    T = 1
    np.random.seed(42)
    cc = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
        T,
        E,
        D,
        optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        managed=table_batched_embeddings_ops.EmbeddingLocation.HOST_MAPPED,
        eps=0.1,
        stochastic_rounding=False,
        fp16=fp16,
    ).cuda()
    ccd = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
        T,
        E,
        D,
        optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        managed=table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
        eps=0.1,
        stochastic_rounding=False,
        fp16=fp16,
    ).cuda()

    logging.info(
        f"Embedding parameters: {cc.embedding_weights.numel() / 1.0e9:.2f}GParam"
    )

    R = True

    def w1(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(w, x, *args):
            c(w, x.random_(0, E - 1), *args)

        return z

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
        # torch.tensor(np.random.zipf(a=1.2, size=(B, L))).int().cuda()
        # % E
        torch.randint(low=0, high=E - 1, size=(T, B, L))
        .int()
        .cuda()
    ]

    print(
        f"Duplicate proportion: {1.0 - np.unique(zs[0].detach().cpu().numpy()).size / zs[0].detach().cpu().numpy().size}"
    )
    merged_indices = torch.stack(zs, dim=0)

    merged_indices = (
        torch.randint(low=0, high=E - 1, size=(T, B, L)).int().cuda()
    )

    (indices, offsets) = get_table_batched_offsets_from_dense(merged_indices)
    assert indices.shape[0] == B * T * L
    assert all(
        ls == L
        for ls in (offsets[1:] - offsets[:-1]).detach().cpu().numpy().tolist()
    )
    per_sample_weights = None
    print(indices.shape, indices.min(), indices.max(), indices)
    ASSOC = 32
    lxu_cache_state = torch.zeros(3, C, ASSOC).int().cuda()
    lxu_cache_weights = torch.zeros(C * ASSOC, D).float().cuda()

    for N_block_size in [32]:
        lxu_cache_state = torch.zeros(3, C, ASSOC).int().cuda()
        time_per_iter = benchmark_torch_function(
            iters,
            table_batched_embeddings.lxu_cache_populate,
            cc.embedding_weights.view(E, D),
            indices,
            lxu_cache_state,
            lxu_cache_weights,
            1,
            N_block_size,
        )
        logging.info(
            f"LRU Populate, random values, B: {B} {(N_block_size,)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
        )
        time_per_iter = benchmark_torch_function(
            iters,
            table_batched_embeddings.lxu_cache_populate,
            cc.embedding_weights.view(E, D),
            indices,
            lxu_cache_state,
            lxu_cache_weights,
            2,
            N_block_size,
        )
        logging.info(
            f"LRU Populate, same values, B: {B} {(N_block_size,)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
        )
        for N_block_size_ in [1, 4, 8, 16, 32]:
            time_per_iter = benchmark_torch_function(
                iters,
                table_batched_embeddings.lxu_cache_lookup,
                indices,
                lxu_cache_state,
                3,
                N_block_size_,
            )
            logging.info(
                f"LRU Lookup, same values, B: {B} {(N_block_size_,)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {4 * B * T * L * 32 / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
            )
        lxu_cache_locations = table_batched_embeddings.lxu_cache_lookup(
            indices, lxu_cache_state, 3, N_block_size
        )
        time_per_iter = benchmark_torch_function(
            iters,
            table_batched_embeddings.lxu_cache_forward,
            cc.embedding_weights.view(E, D),
            indices,
            offsets,
            None,
            lxu_cache_locations,
            lxu_cache_weights,
            N_block_size,
        )
        logging.info(
            f"Forward (LRU), same values, B: {B} {(N_block_size,)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
        )
        time_per_iter = benchmark_torch_function(
            iters,
            table_batched_embeddings.forward,
            cc.embedding_weights,
            cc.table_offsets,
            indices,
            offsets,
            per_sample_weights,
            L,
            N_block_size,
            False,
        )
        logging.info(
            f"Forward (Managed), B: {B} {(N_block_size, False)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
        )
        time_per_iter = benchmark_torch_function(
            iters,
            table_batched_embeddings.forward,
            ccd.embedding_weights,
            ccd.table_offsets,
            indices,
            offsets,
            per_sample_weights,
            L,
            N_block_size,
            False,
        )
        logging.info(
            f"Forward (GPU), B: {B} {(N_block_size, False)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
        )
        go = torch.randn(B, D).cuda()
        time_per_iter = benchmark_torch_function(
            iters,
            table_batched_embeddings.lxu_cache_backward_sgd,
            go,
            cc.embedding_weights,
            indices,
            offsets,
            lxu_cache_locations,
            lxu_cache_weights,
            0.05,
            32,
        )
        logging.info(
            f"Backward (LRU), same values, B: {B} {(N_block_size,)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {2 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
        )
        time_per_iter = benchmark_torch_function(
            iters,
            table_batched_embeddings.backward_sgd,
            go.view(B, 1, D),
            cc.embedding_weights,
            cc.table_offsets,
            indices,
            offsets,
            0.05,
            L,
            N_block_size,
            False,
        )
        logging.info(
            f"Backward (Managed), same values, B: {B} {(N_block_size,)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {2 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
        )
        time_per_iter = benchmark_torch_function(
            iters,
            table_batched_embeddings.backward_sgd,
            go.view(B, 1, D),
            ccd.embedding_weights,
            ccd.table_offsets,
            indices,
            offsets,
            0.05,
            L,
            N_block_size,
            False,
        )
        logging.info(
            f"Backward (GPU), same values, B: {B} {(N_block_size,)}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {2 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
        )


def generate_requests(iters, B, L, E, inter_request_reuse, alpha):
    if alpha <= 1.0:
        all_indices = np.random.randint(low=0, high=E, size=(iters, B * L))
    else:
        all_indices = np.random.zipf(a=alpha, size=(iters, B * L)) % E

    for it in range(iters - 1):
        reused_indices = np.random.choice(
            range(B * L), size=int(B * L * inter_request_reuse), replace=False
        )
        all_indices[it + 1, reused_indices] = all_indices[it, reused_indices]

    average_intra_request_shared = np.average([
        1 - (np.unique(all_indices[it]).size / all_indices[it].size) for it in range(iters)
    ])
    average_inter_request_shared = np.average([
        np.intersect1d(all_indices[it], all_indices[it+1]).size / np.unique(np.concatenate([all_indices[it+1], all_indices[it]])).size for it in range(iters-1)
    ])
    print(
        f"intra-request shared: {average_intra_request_shared * 100:.1f}%, inter-request shared: {average_inter_request_shared * 100:.1f}%"
    )
    all_indices = torch.tensor(all_indices).cuda().int()
    rs = [
        get_table_batched_offsets_from_dense(all_indices[it].view(1, B, L))
        for it in range(iters)
    ]
    return rs


def benchmark_e2e(B, E, L, D, C, iters, inter_request_reuse, alpha):
    logging.basicConfig(level=logging.DEBUG)
    import torch
    import table_batched_embeddings

    T = 1
    fp16 = 0
    np.random.seed(42)
    emb = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
        T,
        E,
        D,
        optimizer=table_batched_embeddings_ops.Optimizer.SGD,
        learning_rate=0.1,
        managed=table_batched_embeddings_ops.EmbeddingLocation.HOST_MAPPED,
        eps=0.1,
    ).cuda()

    # emb_gpu = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
    #     T,
    #     E,
    #     D,
    #     optimizer=table_batched_embeddings_ops.Optimizer.SGD,
    #     learning_rate=0.1,
    #     managed=table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
    #     eps=0.1,
    # ).cuda()
    emb_gpu = None
    lxu_emb = table_batched_embeddings_ops.LXUCacheEmbeddingBag(1, D, C,).cuda()
    lxu_emb.embedding_weights = emb.embedding_weights

    logging.info(
        f"Embedding parameters: {emb.embedding_weights.numel() / 1.0e9:.2f}GParam"
    )

    requests = generate_requests(2 * iters, B, L, E, inter_request_reuse, alpha)
    warmup_requests, requests = requests[:iters], requests[iters:]
    for indices, offsets in warmup_requests:
        lxu_emb(indices, offsets)

    def benchmark_requests(f):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for (indices, offsets) in requests:
            f(indices, offsets)
        end_event.record()
        torch.cuda.synchronize()
        return (start_event.elapsed_time(end_event) * 1.0e-3) / len(requests)

    # time_per_iter = benchmark_requests(
    #     lambda indices, offsets: emb(indices, offsets)
    # )
    # logging.info(
    #     f"Forward (Managed), irr: {inter_request_reuse}, alpha: {alpha}, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
    # )
    # time_per_iter = benchmark_requests(
    #     lambda indices, offsets: emb_gpu(indices, offsets)
    # )
    # logging.info(
    #     f"Forward (GPU), irr: {inter_request_reuse}, alpha: {alpha}, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
    # )
    # time_per_iter = benchmark_requests(
    #     lambda indices, offsets: lxu_emb(indices, offsets)
    # )
    # logging.info(
    #     f"Forward (LRU), irr: {inter_request_reuse}, alpha: {alpha}, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {(2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
    # )
    grad_out = torch.randn(B, D).cuda()
    time_per_iter = benchmark_requests(
        lambda indices, offsets: emb(indices, offsets).backward(
            grad_out.view(B, 1, D)
        )
    )
    logging.info(
        f"ForwardBackward (Managed), irr: {inter_request_reuse}, alpha: {alpha}, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {3 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
    )
    if emb_gpu:
        time_per_iter = benchmark_requests(
            lambda indices, offsets: emb_gpu(indices, offsets).backward(
                grad_out.view(B, 1, D)
            )
        )
        logging.info(
            f"ForwardBackward (GPU), irr: {inter_request_reuse}, alpha: {alpha}, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {3 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
        )
    time_per_iter = benchmark_requests(
        lambda indices, offsets: lxu_emb(indices, offsets).backward(grad_out)
    )
    logging.info(
        f"ForwardBackward (LRU), irr: {inter_request_reuse}, alpha: {alpha}, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, BW: {3 * (2 if fp16 else 4) * B * T * L * D / time_per_iter / 1.0e9: .2f}GB/s, T: {time_per_iter * 1.0e6:.0f}us"
    )


@click.group()
def cli():
    pass


@cli.command()
@click.option("--num-embeddings", default=int(1e7))
@click.option("--cache-sets", default=int(1e5))
@click.option("--embedding-dim", default=32)
@click.option("--batch-size", default=128)
@click.option("--bag-size", default=32)
@click.option("--iters", default=100)
@click.option("--remote", is_flag=True, default=False)
@click.option("--fp16", is_flag=True, default=False)
@click.option("--managed", is_flag=True, default=False)
@click.option("--mixed", is_flag=True, default=False)
def microbenchmark(
    num_embeddings,
    cache_sets,
    embedding_dim,
    batch_size,
    bag_size,
    iters,
    remote,
    fp16,
    managed,
    mixed,
):
    def f():
        import torch

        benchmark_microbenchmark(
            batch_size,
            num_embeddings,
            bag_size,
            embedding_dim,
            cache_sets,
            iters,
            fp16,
            managed,
            mixed,
        )

    if remote:
        import submitit

        executor = submitit.AutoExecutor(folder="sparse_embedding_perf")
        executor.update_parameters(
            timeout_min=10,
            partition="dev",
            constraint="volta32gb",
            gpus_per_node=1,
        )
        job = executor.submit(f)
        job.wait()
        job.result()
        logging.info("Finished")
        import time

        time.sleep(1)
        print(job.stdout())
        print(job.stderr(), file=sys.stderr)
        logging.info("Finished")
    else:
        f()


@cli.command()
@click.option("--num-embeddings", default=int(1e7))
@click.option("--cache-sets", default=int(1e5))
@click.option("--embedding-dim", default=32)
@click.option("--batch-size", default=128)
@click.option("--bag-size", default=32)
@click.option("--iters", default=100)
@click.option("--inter-request-reuse", type=float, default=0.1)
@click.option("--alpha", type=float, default=1.000000001)
@click.option("--remote", is_flag=True, default=False)
def e2e(
    num_embeddings,
    cache_sets,
    embedding_dim,
    batch_size,
    bag_size,
    iters,
    inter_request_reuse,
    alpha,
    remote,
):
    def f():
        import torch

        benchmark_e2e(
            batch_size,
            num_embeddings,
            bag_size,
            embedding_dim,
            cache_sets,
            iters,
            inter_request_reuse,
            alpha,
        )

    if remote:
        import submitit

        executor = submitit.AutoExecutor(folder="sparse_embedding_perf")
        executor.update_parameters(
            timeout_min=10,
            partition="dev",
            constraint="volta32gb",
            gpus_per_node=1,
        )
        job = executor.submit(f)
        job.wait()
        job.result()
        logging.info("Finished")
        import time

        time.sleep(1)
        print(job.stdout())
        print(job.stderr(), file=sys.stderr)
        logging.info("Finished")
    else:
        f()


if __name__ == "__main__":
    cli()
