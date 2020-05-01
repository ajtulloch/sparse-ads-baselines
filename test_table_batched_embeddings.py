from hypothesis import given, settings, Verbosity
import hypothesis.strategies as st
import torch
import numpy as np

import table_batched_embeddings_ops


def div_round_up(a, b):
    return int((a + b - 1) // b) * b


def get_offsets_from_dense(indices):
    (B, L) = indices.size()
    return (
        indices.int().contiguous().view(-1),
        torch.tensor(
            np.cumsum(np.asarray([0] + [L for _ in range(B)])[:-1]).astype(np.int32)
        ).cuda(),
    )


def get_table_batched_offsets_from_dense(merged_indices):
    (B, T, L) = merged_indices.size()
    lengths = np.ones((B, T)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.int().contiguous().view(-1).cuda(),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).int().cuda(),
    )


@given(
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=512),
    st.integers(min_value=1, max_value=32),
    st.integers(min_value=1, max_value=32),
    st.booleans(),
    st.booleans(),
)
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=20)
def test_forward(T, D, B, L, fp16, weighted):
    D = D * 4
    E = int(1e4)
    bs = [torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda() for _ in range(T)]
    if fp16:
        bs = [b.half() for b in bs]

    xs = [torch.randint(low=0, high=E, size=(B, L)).cuda() for _ in range(T)]
    xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]

    if fp16:
        xws = [xw.half() for xw in xws]

    fs = (
        [b(x) for (b, x) in zip(bs, xs)]
        if not weighted
        else [b(x, per_sample_weights=xw) for (b, x, xw) in zip(bs, xs, xws)]
    )

    f = torch.cat([f.view(B, 1, D) for f in fs], dim=1)

    cc = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
        T, E, D, fp16=fp16
    ).cuda()

    for t in range(T):
        cc.embedding_weights.data.view(T, E, D)[t, :, :] = bs[t].weight
    x = torch.cat([x.view(B, 1, L) for x in xs], dim=1)
    xw = torch.cat([xw.view(B, 1, L) for xw in xws], dim=1)

    (indices, offsets) = get_table_batched_offsets_from_dense(x)
    fc2 = (
        cc(indices, offsets)
        if not weighted
        else cc(indices, offsets, xw.contiguous().view(-1).cuda())
    )
    torch.testing.assert_allclose(f, fc2)


@given(
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=512),
    st.integers(min_value=1, max_value=32),
    st.integers(min_value=1, max_value=32),
    st.booleans(),
)
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=20)
def test_backward_sgd(T, D, B, L, fp16):
    D = D * 4
    E = int(1e4)
    bs = [torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda() for _ in range(T)]
    if fp16:
        bs = [b.half() for b in bs]
    xs = [
        torch.from_numpy(
            np.random.choice(range(E), size=(B, L), replace=False).astype(np.int64)
        ).cuda()
        for _ in range(T)
    ]

    def b_indices(b, x):
        (indices, offsets) = get_offsets_from_dense(x)
        return b(indices.long(), offsets.to(torch.int64))

    fs = [b_indices(b, x) for (b, x) in zip(bs, xs)]
    gos = [torch.randn_like(f) for f in fs]
    [f.backward(go) for (f, go) in zip(fs, gos)]
    # do SGD update
    lr = 0.05
    new_weights = [(b.weight - b.weight.grad * lr) for b in bs]

    f = torch.cat([f.view(B, 1, D) for f in fs], dim=1)

    cc = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
        T,
        E,
        D,
        optimizer=table_batched_embeddings_ops.Optimizer.SGD,
        learning_rate=0.05,
        fp16=fp16,
    ).cuda()

    for t in range(T):
        cc.embedding_weights.data.view(T, E, D)[t, :, :] = bs[t].weight

    x = torch.cat([x.view(B, 1, L) for x in xs], dim=1)
    (indices, offsets) = get_table_batched_offsets_from_dense(x)
    fc2 = cc(indices, offsets)
    fc2.backward(torch.cat([go.view(B, 1, D) for go in gos], dim=1))
    for t in range(T):
        torch.testing.assert_allclose(
            cc.embedding_weights.view(T, E, D)[t, :, :], new_weights[t]
        )


@given(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=512),
    st.integers(min_value=1, max_value=256),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=3),
    st.booleans(),
    st.booleans(),
    st.booleans(),
)
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=20)
def test_backward_adagrad(T, D, B, L, D_gradcheck, fp16, stochastic_rounding, weighted):
    E = int(1e4)
    D_gradcheck = D_gradcheck * 4

    D = D * 4
    bs = [torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda() for _ in range(T)]
    if fp16:
        bs = [b.half() for b in bs]

    xs = [
        torch.from_numpy(
            np.random.choice(range(E), size=(B, L), replace=False).astype(np.int64)
        ).cuda()
        for _ in range(T)
    ]
    xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]
    # xws = [torch.ones(size=(B, L)).cuda().float() * 2 for _ in range(T)]

    if fp16:
        xws = [xw.half() for xw in xws]

    def b_indices(b, x):
        (indices, offsets) = get_offsets_from_dense(x)
        return b(indices.long(), offsets.to(torch.int64))

    fs = (
        [b_indices(b, x) for (b, x) in zip(bs, xs)]
        if not weighted
        else [b(x, per_sample_weights=xw) for (b, x, xw) in zip(bs, xs, xws)]
    )
    gos = [torch.randn_like(f) for f in fs]
    [f.backward(go) for (f, go) in zip(fs, gos)]
    # do SGD update
    lr = 0.5
    eps = 0.2

    new_weights = [b.weight for b in bs]
    f = torch.cat([f.view(B, 1, D) for f in fs], dim=1)
    cc = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
        T,
        E,
        D,
        optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
        learning_rate=lr,
        eps=eps,
        fp16=fp16,
        stochastic_rounding=stochastic_rounding,
    ).cuda()

    for t in range(T):
        cc.embedding_weights.data.view(T, E, D)[t, :, :] = bs[t].weight

    x = torch.cat([x.view(B, 1, L) for x in xs], dim=1)
    xw = torch.cat([xw.view(B, 1, L) for xw in xws], dim=1)

    (indices, offsets) = get_table_batched_offsets_from_dense(x)
    fc2 = (
        cc(indices, offsets)
        if not weighted
        else cc(indices, offsets, xw.contiguous().view(-1).cuda())
    )
    fc2.backward(torch.cat([go.view(B, 1, D) for go in gos], dim=1))

    # optimizer state is sum_square_grads.
    for t in range(T):
        torch.testing.assert_allclose(
            cc.optimizer_state.view(T, E)[t],
            bs[t].weight.grad.float().to_dense().pow(2).sum(dim=1),
            atol=1.0e-3 if fp16 else 1.0e-4,
            rtol=1.0e-3 if fp16 else 1.0e-4,
        )

    for t in range(T):
        torch.testing.assert_allclose(
            cc.embedding_weights.view(T, E, D)[t, :, :].float(),
            torch.addcdiv(
                bs[t].weight.float(),
                value=-lr,
                tensor1=bs[t].weight.grad.float().to_dense(),
                tensor2=cc.optimizer_state.view(T, E)[t, :]
                .sqrt_()
                .add_(eps)
                .view(E, 1),
            ),
            atol=1.0e-3 if fp16 else 1.0e-4,
            rtol=1.0e-3 if fp16 else 1.0e-4,
        )

    if weighted:
        cc = (
            table_batched_embeddings_ops.TableBatchedEmbeddingBags(
                T,
                E,
                D_gradcheck,
                optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
                learning_rate=0.0,
                eps=eps,
                fp16=fp16,
                stochastic_rounding=stochastic_rounding,
            )
            .cuda()
            .double()
        )
        per_sample_weights = xw.contiguous().view(-1).cuda().double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        cc.embedding_weights.requires_grad = False
        torch.autograd.gradcheck(cc, (indices, offsets, per_sample_weights))


@given(
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=2, max_value=512),
    st.integers(min_value=1, max_value=32),
    st.integers(min_value=1, max_value=32),
    st.booleans(),
    st.booleans(),
)
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=20)
def test_forward_mixed(T, D_max, B, L, fp16, weighted):
    Ds = [np.random.randint(low=1, high=D_max) * 4 for _ in range(T)]
    Es = [np.random.randint(low=int(0.5 * 1e4), high=int(2 * 1.0e4)) for _ in range(T)]
    bs = [
        torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
        for (E, D) in zip(Es, Ds)
    ]
    if fp16:
        bs = [b.half() for b in bs]

    xs = [torch.randint(low=0, high=E, size=(B, L)).cuda() for E in Es]
    xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]

    if fp16:
        xws = [xw.half() for xw in xws]

    fs = (
        [b(x) for (b, x) in zip(bs, xs)]
        if not weighted
        else [b(x, per_sample_weights=xw) for (b, x, xw) in zip(bs, xs, xws)]
    )

    f = torch.cat([f.view(B, -1) for f in fs], dim=1)

    cc = table_batched_embeddings_ops.MixedDimTableBatchedEmbeddingBags(
        [(E, D) for (E, D) in zip(Es, Ds)], fp16=fp16
    ).cuda()

    for t, weights in enumerate(cc.split_embedding_weights()):
        weights[:, :] = bs[t].weight
    x = torch.cat([x.view(B, 1, L) for x in xs], dim=1)
    xw = torch.cat([xw.view(B, 1, L) for xw in xws], dim=1)

    (indices, offsets) = get_table_batched_offsets_from_dense(x)
    fc2 = (
        cc(indices, offsets)
        if not weighted
        else cc(indices, offsets, xw.contiguous().view(-1).cuda())
    )
    torch.testing.assert_allclose(f, fc2)


@given(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=2, max_value=512),
    st.integers(min_value=1, max_value=256),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=3),
    st.booleans(),
    st.booleans(),
    st.booleans(),
)
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=20)
def test_backward_adagrad_mixed(
    T, D_max, B, L, D_gradcheck, fp16, stochastic_rounding, weighted
):
    Ds = [np.random.randint(low=1, high=D_max) * 4 for _ in range(T)]
    Es = [np.random.randint(low=int(0.5 * 1e4), high=int(2 * 1.0e4)) for _ in range(T)]
    print(list(zip(Es, Ds)))
    bs = [
        torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda()
        for (E, D) in zip(Es, Ds)
    ]

    if fp16:
        bs = [b.half() for b in bs]

    xs = [
        torch.from_numpy(
            np.random.choice(range(E), size=(B, L), replace=False).astype(np.int64)
        ).cuda()
        for E in Es
    ]
    xws = [torch.randn(size=(B, L)).cuda() for _ in range(T)]
    # xws = [torch.ones(size=(B, L)).cuda().float() * 2 for _ in range(T)]

    if fp16:
        xws = [xw.half() for xw in xws]

    def b_indices(b, x):
        (indices, offsets) = get_offsets_from_dense(x)
        return b(indices.long(), offsets.to(torch.int64))

    fs = (
        [b_indices(b, x) for (b, x) in zip(bs, xs)]
        if not weighted
        else [b(x, per_sample_weights=xw) for (b, x, xw) in zip(bs, xs, xws)]
    )
    gos = [torch.randn_like(f) for f in fs]
    [f.backward(go) for (f, go) in zip(fs, gos)]
    # do SGD update
    lr = 0.5
    eps = 0.2

    new_weights = [b.weight for b in bs]
    f = torch.cat([f.view(B, -1) for f in fs], dim=1)

    cc = table_batched_embeddings_ops.MixedDimTableBatchedEmbeddingBags(
        [(E, D) for (E, D) in zip(Es, Ds)],
        optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
        learning_rate=lr,
        eps=eps,
        fp16=fp16,
        stochastic_rounding=stochastic_rounding,
    ).cuda()

    for t, weights in enumerate(cc.split_embedding_weights()):
        weights[:, :] = bs[t].weight

    x = torch.cat([x.view(B, 1, L) for x in xs], dim=1)
    xw = torch.cat([xw.view(B, 1, L) for xw in xws], dim=1)

    (indices, offsets) = get_table_batched_offsets_from_dense(x)
    fc2 = (
        cc(indices, offsets)
        if not weighted
        else cc(indices, offsets, xw.contiguous().view(-1).cuda())
    )
    goc = torch.cat([go.view(B, -1) for go in gos], dim=1).contiguous()
    fc2.backward(goc)

    # optimizer state is sum_square_grads.
    for t in range(T):
        torch.testing.assert_allclose(
            cc.split_optimizer_state()[t],
            bs[t].weight.grad.float().to_dense().pow(2).sum(dim=1),
            atol=1.0e-3 if fp16 else 1.0e-4,
            rtol=1.0e-3 if fp16 else 1.0e-4,
        )

    for t in range(T):
        torch.testing.assert_allclose(
            cc.split_embedding_weights()[t].float(),
            torch.addcdiv(
                bs[t].weight.float(),
                value=-lr,
                tensor1=bs[t].weight.grad.float().to_dense(),
                tensor2=cc.split_optimizer_state()[t]
                .sqrt_()
                .add_(eps)
                .view(-1, 1),
            ),
            atol=1.0e-3 if fp16 else 1.0e-4,
            rtol=1.0e-3 if fp16 else 1.0e-4,
        )

    if weighted:
        D_gradcheck = D_gradcheck * 4
        cc = table_batched_embeddings_ops.MixedDimTableBatchedEmbeddingBags(
            [(E, D_gradcheck) for (E, D) in zip(Es, Ds)],
            optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
            learning_rate=0.0,
            eps=eps,
            fp16=fp16,
            stochastic_rounding=stochastic_rounding,
        ).cuda().double()

        per_sample_weights = xw.contiguous().view(-1).cuda().double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        cc.embedding_weights.requires_grad = False
        torch.autograd.gradcheck(cc, (indices, offsets, per_sample_weights))