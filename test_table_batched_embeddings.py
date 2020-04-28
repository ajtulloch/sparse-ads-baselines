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
)
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=10)
def test_forward(T, D, B, L):
    D = D * 4
    E = int(1e4)
    bs = [torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda() for _ in range(T)]

    xs = [torch.randint(low=0, high=E, size=(B, L)).cuda() for _ in range(T)]
    fs = [b(x) for (b, x) in zip(bs, xs)]

    def b_indices(b, x):
        (indices, offsets) = get_offsets_from_dense(x)
        return b(indices.long(), offsets.to(torch.int64))

    fs2 = [b_indices(b, x) for (b, x) in zip(bs, xs)]

    for t in range(T):
        torch.testing.assert_allclose(fs[t], fs2[t])

    f = torch.cat([f.view(B, 1, D) for f in fs], dim=1)

    cc = table_batched_embeddings_ops.TableBatchedEmbeddingBags(T, E, D).cuda()

    for t in range(T):
        cc.embedding_weights.data.view(T, E, D)[t, :, :] = bs[t].weight
    x = torch.cat([x.view(B, 1, L) for x in xs], dim=1)
    (indices, offsets) = get_table_batched_offsets_from_dense(x)
    fc2 = cc(indices, offsets)
    torch.testing.assert_allclose(f, fc2)


@given(
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=512),
    st.integers(min_value=1, max_value=32),
    st.integers(min_value=1, max_value=32),
)
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=10)
def test_backward_sgd(T, D, B, L):
    D = D * 4
    E = int(1e4)
    bs = [torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda() for _ in range(T)]

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
)
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=10)
def test_backward_adagrad(T, D, B, L):
    E = int(1e4)
    D = D * 4
    bs = [torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True).cuda() for _ in range(T)]
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
    gos = [torch.ones_like(f) for f in fs]
    [f.backward(go) for (f, go) in zip(fs, gos)]
    # do SGD update
    lr = 0.05
    eps = 0.2

    # new_weights = [b.weight.addcdiv(b.weight.grad, b.weight.grad.to_dense().pow(2).sum(dim=1, keepdim=True).sqrt().add(eps).expand(*b.weight.grad.shape), -lr) for b in bs]
    new_weights = [b.weight for b in bs]
    f = torch.cat([f.view(B, 1, D) for f in fs], dim=1)
    cc = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
        T,
        E,
        D,
        optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
        learning_rate=lr,
        eps=eps,
    ).cuda()

    for t in range(T):
        cc.embedding_weights.data.view(T, E, D)[t, :, :] = bs[t].weight

    x = torch.cat([x.view(B, 1, L) for x in xs], dim=1)
    (indices, offsets) = get_table_batched_offsets_from_dense(x)
    fc2 = cc(indices, offsets)
    fc2.backward(torch.cat([go.view(B, 1, D) for go in gos], dim=1))
    # optimizer state is sum_square_grads.
    for t in range(T):
        torch.testing.assert_allclose(
            cc.optimizer_state.view(T, E)[t],
            bs[t].weight.grad.to_dense().pow(2).sum(dim=1),
        )

    for t in range(T):
        torch.testing.assert_allclose(
            cc.embedding_weights.view(T, E, D)[t, :, :],
            torch.addcdiv(
                bs[t].weight,
                value=-lr,
                tensor1=bs[t].weight.grad.to_dense(),
                tensor2=cc.optimizer_state.view(T, E)[t, :]
                .sqrt_()
                .add_(eps)
                .view(E, 1),
            ),
        )
