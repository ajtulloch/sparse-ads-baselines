import table_batched_embeddings
import torch
from torch import nn
import numpy as np
import enum


class LookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, weights, table_offsets, indices, offsets, optimizer, optimizer_state, learning_rate, eps
    ):
        ctx.optimizer = optimizer
        ctx.learning_rate = learning_rate
        ctx.eps = eps
        ctx.save_for_backward(weights, table_offsets, indices, offsets, optimizer_state)
        T_block_size = 0
        L_max = 200 
        return table_batched_embeddings.forward(
            weights, table_offsets, indices, offsets, L_max, T_block_size
        )

    @staticmethod
    def backward(ctx, grad_output):
        (weights, table_offsets, indices, offsets, optimizer_state) = ctx.saved_tensors
        T_block_size = 0
        L_max = 200
        assert ctx.optimizer in (Optimizer.SGD, Optimizer.APPROX_ROWWISE_ADAGRAD)
        if ctx.optimizer == Optimizer.SGD:
            table_batched_embeddings.backward_sgd(
                grad_output,
                weights,
                table_offsets,
                indices,
                offsets,
                ctx.learning_rate,
                L_max,
                T_block_size,
            )
        elif ctx.optimizer == Optimizer.APPROX_ROWWISE_ADAGRAD:
            table_batched_embeddings.backward_approx_adagrad(
                grad_output,
                weights,
                table_offsets,
                indices,
                offsets,
                optimizer_state,
                ctx.learning_rate,
                ctx.eps, 
                L_max
            )
        return (torch.cuda.sparse.FloatTensor(*weights.size()), None, None, None, None, None, None, None)

class Optimizer(enum.Enum):
    SGD = 1
    APPROX_ROWWISE_ADAGRAD = 2

class TableBatchedEmbeddingBags(nn.Module):
    def __init__(
        self, num_tables, num_embeddings, embedding_dim, optimizer=Optimizer.SGD, learning_rate=0.01, eps=None
    ):
        super(TableBatchedEmbeddingBags, self).__init__()
        self.embedding_weights = nn.Parameter(
            torch.randn(num_tables * num_embeddings, embedding_dim)
        )
        self.register_buffer(
            "table_offsets",
            torch.tensor(
                [0]
                + np.cumsum([num_embeddings for _ in range(num_tables - 1)]).tolist()
            ).int(),
        )
        # TODO: unused by SGD
        self.register_buffer(
            "optimizer_state", torch.zeros(num_tables * num_embeddings).float(),
        )

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.eps = eps

    def forward(self, sharded_sparse_features, sharded_offsets):
        return LookupFunction.apply(
            self.embedding_weights,
            self.table_offsets,
            sharded_sparse_features,
            sharded_offsets,
            self.optimizer,
            self.optimizer_state,
            self.learning_rate,
            self.eps
        )
