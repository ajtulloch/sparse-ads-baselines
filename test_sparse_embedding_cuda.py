import torch
from sparse_embedding_cuda_ops import ReduceScatterFunction, UniformShardedEmbeddingBags, sparse_embedding_cuda, All2AllFunction
import horovod.torch as hvd
import numpy as np
hvd.init()

from parameterized import parameterized
from hypothesis import given, settings, Verbosity
import hypothesis.strategies as st


@given(st.integers(min_value=1, max_value=4),
       st.integers(min_value=1, max_value=256),
       st.integers(min_value=1, max_value=32),
       st.integers(min_value=1, max_value=32))
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=10)
def test_forward(T, D, B, L):
    E = int(1e5)
    bs = [
        torch.nn.EmbeddingBag(E, D, mode='sum', sparse=True).cuda()
        for _ in range(T)
    ]
    xs = [torch.randint(low=0, high=E, size=(B, L)).cuda() for _ in range(T)]
    fs = [b(x) for (b, x) in zip(bs, xs)]
    f = torch.cat([f.view(B, 1, D) for f in fs], dim=1)
    cc = UniformShardedEmbeddingBags(T, E, D).cuda()
    for t in range(T):
        cc.embedding_weights.data[:, t, :] = bs[t].weight
    x = torch.cat([x.view(B, 1, L) for x in xs], dim=1)
    fc = cc(x)
    torch.testing.assert_allclose(f, fc)


@given(st.integers(min_value=1, max_value=4),
       st.integers(min_value=1, max_value=256),
       st.integers(min_value=1, max_value=32),
       st.integers(min_value=1, max_value=32))
@settings(verbosity=Verbosity.verbose, deadline=None, max_examples=10)
def test_backward(T, D, B, L):
    E = int(1e5)
    bs = [
        torch.nn.EmbeddingBag(E, D, mode='sum', sparse=True).cuda()
        for _ in range(T)
    ]
    xs = [
        torch.from_numpy(
            np.random.choice(range(E), size=(B, L),
                             replace=False).astype(np.int64)).cuda()
        for _ in range(T)
    ]
    fs = [b(x) for (b, x) in zip(bs, xs)]
    gos = [torch.randn_like(f) for f in fs]
    [f.backward(go) for (f, go) in zip(fs, gos)]

    # do SGD update
    lr = 0.05
    new_weights = [(b.weight - b.weight.grad * lr) for b in bs]

    cc = UniformShardedEmbeddingBags(T, E, D).cuda()
    for t in range(T):
        cc.embedding_weights.data[:, t, :] = bs[t].weight

    x = torch.cat([x.view(B, 1, L) for x in xs], dim=1)

    fc = cc(x)
    fc.backward(torch.cat([go.view(B, 1, D) for go in gos], dim=1))

    for t in range(T):
        torch.testing.assert_allclose(cc.embedding_weights[:, t, :],
                                      new_weights[t])


@parameterized([
    (1, 1, 1),
    (2, 8, 7),
    (4, 3, 284),
])
def test_reduce_scatter_forward(B, T, D):
    torch.cuda.set_device(hvd.rank())
    torch.manual_seed(42)
    E_ranks = torch.randn(hvd.size(), B * hvd.size(), T, D).cuda()
    R = ReduceScatterFunction.apply(E_ranks[hvd.rank()])
    torch.testing.assert_allclose(
        R,
        torch.sum(E_ranks, dim=0)[B * hvd.rank():B * (hvd.rank() + 1)])


@parameterized([
    (1, 1, 1),
    (2, 5, 7),
    (4, 3, 284),
])
def test_reduce_scatter_backward(B, T, D):
    torch.cuda.set_device(hvd.rank())
    torch.manual_seed(42)
    E_ranks = torch.randn(hvd.size(), B * hvd.size(), T, D).cuda()
    E_ranks.requires_grad = True
    Y_ranks = torch.randn(hvd.size(), B, T, D).cuda()
    R = ReduceScatterFunction.apply(E_ranks[hvd.rank()])
    R.backward(Y_ranks[hvd.rank()])
    torch.testing.assert_allclose(
        E_ranks.grad[hvd.rank()].view(hvd.size(), B, T, D), Y_ranks)

    if hvd.size() == 1:
        torch.autograd.gradcheck(ReduceScatterFunction.apply,
                                 E_ranks[hvd.rank()].double())


@parameterized([
    (1, 1, 1),
    (2, 8, 7),
    (4, 3, 284),
])
def test_all_to_all_forward(B, T, D):
    torch.cuda.set_device(hvd.rank())
    torch.manual_seed(42)
    E_ranks = torch.randn(hvd.size(), B * hvd.size(), T, D).cuda()
    R = All2AllFunction.apply(E_ranks[hvd.rank()])
    # TODO
