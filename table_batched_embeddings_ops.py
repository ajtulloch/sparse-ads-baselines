import torch
from torch import nn
import numpy as np
import enum

import logging


class LookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weights,
        table_offsets,
        indices,
        offsets,
        per_sample_weights,
        optimizer,
        optimizer_state,
        learning_rate,
        eps,
        stochastic_rounding,
    ):
        import table_batched_embeddings

        ctx.optimizer = optimizer
        ctx.learning_rate = learning_rate
        ctx.eps = eps
        ctx.stochastic_rounding = stochastic_rounding
        ctx.save_for_backward(
            weights,
            table_offsets,
            indices,
            offsets,
            per_sample_weights,
            optimizer_state,
        )
        BT_block_size = max(1024 / weights.shape[0], 1) * 4
        L_max = 200  # TODO: pass this in correctly.
        return table_batched_embeddings.forward(
            weights,
            table_offsets,
            indices,
            offsets,
            per_sample_weights,
            L_max,
            BT_block_size,
            False,
        )

    @staticmethod
    def backward(ctx, grad_output):
        import table_batched_embeddings

        (
            weights,
            table_offsets,
            indices,
            offsets,
            per_sample_weights,
            optimizer_state,
        ) = ctx.saved_tensors
        L_max = 200
        assert ctx.optimizer in (Optimizer.SGD, Optimizer.APPROX_ROWWISE_ADAGRAD)
        if ctx.optimizer == Optimizer.SGD:
            BT_block_size = max(1024 / weights.shape[0], 1) * 4
            assert per_sample_weights is None
            grad_per_sample_weight = table_batched_embeddings.backward_sgd(
                grad_output,
                weights,
                table_offsets,
                indices,
                offsets,
                ctx.learning_rate,
                L_max,
                BT_block_size,
                False,  # shared mem
            )
        elif ctx.optimizer == Optimizer.APPROX_ROWWISE_ADAGRAD:
            BT_block_size = 1
            grad_per_sample_weight = table_batched_embeddings.backward_approx_adagrad(
                grad_output,
                weights,
                table_offsets,
                indices,
                offsets,
                per_sample_weights,
                optimizer_state,
                ctx.learning_rate,
                ctx.eps,
                L_max,
                ctx.stochastic_rounding,
                BT_block_size,
            )
        return (
            torch.cuda.sparse.FloatTensor(*weights.size()),
            None,
            None,
            None,
            grad_per_sample_weight,
            None,
            None,
            None,
            None,
            None,
        )


class Optimizer(enum.Enum):
    SGD = 1
    APPROX_ROWWISE_ADAGRAD = 2


class EmbeddingLocation(enum.Enum):
    DEVICE = 0
    HOST_MAPPED = 1


class TableBatchedEmbeddingBags(nn.Module):
    def __init__(
        self,
        num_tables,
        num_embeddings,
        embedding_dim,
        optimizer=Optimizer.SGD,
        learning_rate=0.01,
        eps=None,
        stochastic_rounding=False,
        fp16=False,
        managed=EmbeddingLocation.DEVICE,
    ):
        import table_batched_embeddings

        super(TableBatchedEmbeddingBags, self).__init__()
        assert managed in (EmbeddingLocation.DEVICE, EmbeddingLocation.HOST_MAPPED)
        if managed == EmbeddingLocation.DEVICE:
            logging.info("Allocating device embedding bag")
            embedding_data = torch.randn(
                num_tables * num_embeddings,
                embedding_dim,
                device=torch.cuda.current_device(),
                dtype=torch.float16 if fp16 else torch.float32,
            )
        elif managed == EmbeddingLocation.HOST_MAPPED:
            logging.info("Allocating host-mapped embedding bag")
            embedding_data = torch.randn(
                size=(num_tables * num_embeddings, embedding_dim),
                out=table_batched_embeddings.new_managed_tensor(
                    torch.randn(1).cuda() if not fp16 else torch.randn(1).cuda().half(),
                    (num_tables * num_embeddings, embedding_dim),
                ),
            )

        self.embedding_weights = nn.Parameter(embedding_data)
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
        self.stochastic_rounding = stochastic_rounding

    def forward(
        self, sharded_sparse_features, sharded_offsets, per_sample_weights=None
    ):
        return LookupFunction.apply(
            self.embedding_weights,
            self.table_offsets,
            sharded_sparse_features,
            sharded_offsets,
            per_sample_weights,
            self.optimizer,
            self.optimizer_state,
            self.learning_rate,
            self.eps,
            self.stochastic_rounding,
        )
