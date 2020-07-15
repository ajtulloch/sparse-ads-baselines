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
        managed
    ):
        import table_batched_embeddings

        ctx.optimizer = optimizer
        ctx.learning_rate = learning_rate
        ctx.eps = eps
        ctx.stochastic_rounding = stochastic_rounding
        ctx.managed = managed
        ctx.save_for_backward(
            weights,
            table_offsets,
            indices,
            offsets,
            per_sample_weights,
            optimizer_state,
        )
        BT_block_size = int(max(256 / weights.shape[1], 1))
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
        assert ctx.optimizer in (
            Optimizer.SGD,
            Optimizer.APPROX_ROWWISE_ADAGRAD,
            Optimizer.EXACT_ROWWISE_ADAGRAD,
        )
        if ctx.optimizer == Optimizer.SGD:
            BT_block_size = int(max(256 / weights.shape[1], 1))
            assert per_sample_weights is None
            if True: #ctx.managed == EmbeddingLocation.DEVICE:
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
            else:
                grad_per_sample_weight = table_batched_embeddings.backward_exact_sgd(
                    grad_output,
                    weights,
                    table_offsets,
                    indices,
                    offsets,
                    None,
                    ctx.learning_rate,
                    #L_max,
                    BT_block_size,
                    #False,  # shared mem
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
        elif ctx.optimizer == Optimizer.EXACT_ROWWISE_ADAGRAD:
            BT_block_size = 1
            grad_per_sample_weight = table_batched_embeddings.backward_exact_adagrad(
                grad_output,
                weights,
                table_offsets,
                indices,
                offsets,
                per_sample_weights,
                optimizer_state,
                ctx.learning_rate,
                ctx.eps,
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
            None,
        )


class Optimizer(enum.Enum):
    SGD = 1
    APPROX_ROWWISE_ADAGRAD = 2
    EXACT_ROWWISE_ADAGRAD = 3


class EmbeddingLocation(enum.Enum):
    DEVICE = 0
    HOST_MAPPED = 1
    MANAGED = 2


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
        self.managed = managed
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
                out=table_batched_embeddings.new_host_mapped_tensor(
                    torch.randn(1).cuda()
                    if not fp16
                    else torch.randn(1).cuda().half(),
                    (num_tables * num_embeddings, embedding_dim),
                ),
            )
        elif managed == EmbeddingLocation.MANAGED:
            logging.info("Allocating managed embedding bag")
            embedding_data = torch.randn(
                size=(num_tables * num_embeddings, embedding_dim),
                out=table_batched_embeddings.new_managed_tensor(
                    torch.randn(1).cuda()
                    if not fp16
                    else torch.randn(1).cuda().half(),
                    (num_tables * num_embeddings, embedding_dim),
                ),
            )

        self.embedding_weights = nn.Parameter(embedding_data)
        self.register_buffer(
            "table_offsets",
            torch.tensor(
                [0]
                + np.cumsum(
                    [num_embeddings for _ in range(num_tables - 1)]
                ).tolist()
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
            self.managed
        )


class MixedDimLookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weights,
        table_offsets,
        table_dim_offsets,
        dim_offsets,
        total_D,
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
        ctx.total_D = total_D
        ctx.save_for_backward(
            weights,
            table_offsets,
            table_dim_offsets,
            dim_offsets,
            indices,
            offsets,
            per_sample_weights,
            optimizer_state,
        )
        BT_block_size = 1  # TODO - fix this.
        L_max = 200  # TODO: pass this in correctly.
        return table_batched_embeddings.forward_mixed_D(
            weights,
            table_offsets,
            dim_offsets,
            total_D,
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
            table_dim_offsets,
            dim_offsets,
            indices,
            offsets,
            per_sample_weights,
            optimizer_state,
        ) = ctx.saved_tensors
        L_max = 200
        assert ctx.optimizer in (
            Optimizer.APPROX_ROWWISE_ADAGRAD,
            Optimizer.EXACT_ROWWISE_ADAGRAD,
        )
        BT_block_size = 1
        if ctx.optimizer == Optimizer.APPROX_ROWWISE_ADAGRAD:
            grad_per_sample_weight = table_batched_embeddings.backward_approx_adagrad_mixed_D(
                grad_output,
                weights,
                table_offsets,
                table_dim_offsets,
                dim_offsets,
                ctx.total_D,
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
        else:
            grad_per_sample_weight = table_batched_embeddings.backward_exact_adagrad_mixed_D(
                grad_output,
                weights,
                table_offsets,
                table_dim_offsets,
                dim_offsets,
                ctx.total_D,
                indices,
                offsets,
                per_sample_weights,
                optimizer_state,
                ctx.learning_rate,
                ctx.eps,
                ctx.stochastic_rounding,
                BT_block_size,
            )

        return (
            torch.cuda.sparse.FloatTensor(*weights.size()),  # weights
            None,  # table_offsets
            None,  # table_dim_offsets
            None,  # dim_offsets
            None,  # total_D,
            None,  # indices
            None,  # offsets
            grad_per_sample_weight,
            None,
            None,
            None,
            None,
            None,
        )


class MixedDimTableBatchedEmbeddingBags(nn.Module):
    def __init__(
        self,
        embeddings,
        optimizer=Optimizer.SGD,
        learning_rate=0.01,
        eps=None,
        stochastic_rounding=False,
        fp16=False,
        managed=EmbeddingLocation.DEVICE,
    ):
        import table_batched_embeddings

        super(MixedDimTableBatchedEmbeddingBags, self).__init__()
        assert managed in (
            EmbeddingLocation.DEVICE,
            EmbeddingLocation.HOST_MAPPED,
        )
        rows, dims = zip(*embeddings)
        if managed == EmbeddingLocation.DEVICE:
            logging.info("Allocating device embedding bag")
            embedding_data = torch.randn(
                size=(sum(row * dim for (row, dim) in embeddings),),
                device=torch.cuda.current_device(),
                dtype=torch.float16 if fp16 else torch.float32,
            )
        elif managed == EmbeddingLocation.HOST_MAPPED:
            logging.info("Allocating host-mapped embedding bag")
            embedding_data = torch.randn(
                size=(sum(row * dim for (row, dim) in embeddings),),
                out=table_batched_embeddings.new_managed_tensor(
                    torch.randn(1).cuda()
                    if not fp16
                    else torch.randn(1).cuda().half(),
                    (sum(row * dim for (row, dim) in embeddings),),
                ),
            )
        T = len(embeddings)
        self.embedding_weights = nn.Parameter(embedding_data)
        self.register_buffer(
            "table_dim_offsets",
            torch.tensor([0] + np.cumsum(rows).tolist()).int(),
        )
        self.register_buffer(
            "table_offsets",
            torch.tensor(
                [0] + np.cumsum([r * d for r, d in zip(rows, dims)]).tolist()
            ).long(),
        )
        self.register_buffer(
            "dim_offsets", torch.tensor([0] + np.cumsum(dims).tolist()).int(),
        )

        # TODO: unused by SGD
        self.register_buffer(
            "optimizer_state", torch.zeros(sum(rows)).float(),
        )
        self.total_D = sum(dims)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.eps = eps
        self.stochastic_rounding = stochastic_rounding

    def forward(
        self, sharded_sparse_features, sharded_offsets, per_sample_weights=None
    ):
        return MixedDimLookupFunction.apply(
            self.embedding_weights,
            self.table_offsets,
            self.table_dim_offsets,
            self.dim_offsets,
            self.total_D,
            sharded_sparse_features,
            sharded_offsets,
            per_sample_weights,
            self.optimizer,
            self.optimizer_state,
            self.learning_rate,
            self.eps,
            self.stochastic_rounding,
        )

    def split_embedding_weights(self):
        """
        Returns a list of weights, split by table 
        """
        T = self.table_offsets.size(0) - 1
        return [
            self.embedding_weights.detach()[
                self.table_offsets[i] : self.table_offsets[i + 1]
            ].view(
                self.table_dim_offsets[i + 1] - self.table_dim_offsets[i],
                self.dim_offsets[i + 1] - self.dim_offsets[i],
            )
            for i in range(T)
        ]

    def split_optimizer_state(self):
        """
        Returns a list of optimizer states, split by table
        """
        T = self.table_offsets.size(0) - 1
        return [
            self.optimizer_state.detach()[
                self.table_dim_offsets[i] : self.table_dim_offsets[i + 1]
            ]
            for i in range(T)
        ]

    def split_output(self, output):
        """
        Returns a list of outputs, split by table.
        """
        T = self.table_offsets.size(0) - 1
        return [
            output[:, self.dim_offsets[i] : self.dim_offsets[i + 1]]
            for i in range(T)
        ]


NOT_FOUND = np.iinfo(np.int32).max

import os

class LXUCacheLookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weights,
        lxu_cache_weights,
        lxu_cache_state,
        indices,
        offsets,
        learning_rate,
        t,
        pipelined,
    ):
        import table_batched_embeddings
        ctx.pipelined = pipelined
        ctx.learning_rate = learning_rate
        ctx.t = t
        if os.getenv('LXU_CACHE_PROFILE'):
            lxu_cache_locations = table_batched_embeddings.lxu_cache_lookup(
                indices, lxu_cache_state, t, 32
            )
            print(
                f"Cache hit rate pre-population on iteration {t}: {np.count_nonzero(lxu_cache_locations.cpu().numpy() != NOT_FOUND) / lxu_cache_locations.numel() * 100:.2f}%, {np.count_nonzero(lxu_cache_locations.cpu().numpy() == NOT_FOUND)} misses"
            )
        if not pipelined:
            table_batched_embeddings.lxu_cache_populate(
                weights, indices, lxu_cache_state, lxu_cache_weights, t, 32,
            )
        lxu_cache_locations = table_batched_embeddings.lxu_cache_lookup(
            indices, lxu_cache_state, t, 32
        )
        # print(
        #     f"Cache hit rate post-population on iteration {t}: {np.count_nonzero(lxu_cache_locations.cpu().numpy() != NOT_FOUND) / lxu_cache_locations.numel() * 100:.2f}%, {np.count_nonzero(lxu_cache_locations.cpu().numpy() == NOT_FOUND)} misses"
        # )
        ctx.save_for_backward(
            weights,
            lxu_cache_weights,
            lxu_cache_state,
            indices,
            offsets,
            lxu_cache_locations,
        )
        return table_batched_embeddings.lxu_cache_forward(
            weights,
            indices,
            offsets,
            None,
            lxu_cache_locations,
            lxu_cache_weights,
            32,
        )

    @staticmethod
    def backward(ctx, grad_output):
        import table_batched_embeddings

        (
            weights,
            lxu_cache_weights,
            lxu_cache_state,
            indices,
            offsets,
            lxu_cache_locations,
        ) = ctx.saved_tensors
        if ctx.pipelined:
            # birthday paradox #guarantee's if we do single-stage pipelining we won't get regressions.
            pass
            # lxu_cache_locations = table_batched_embeddings.lxu_cache_lookup(
            #     indices, lxu_cache_state, ctx.t, 32
            # )
        table_batched_embeddings.lxu_cache_backward_sgd(
            grad_output,
            weights,
            indices,
            offsets,
            lxu_cache_locations,
            lxu_cache_weights,
            ctx.learning_rate,
            32,
        )
        return (
            torch.cuda.sparse.FloatTensor(*weights.size()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LXUCacheEmbeddingBag(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        cache_sets,
        learning_rate=0.01,
        fp16=False,
    ):
        import table_batched_embeddings

        super(LXUCacheEmbeddingBag, self).__init__()
        logging.info("Allocating host-mapped lru-backed embedding bag")
        embedding_data = torch.randn(
            size=(num_embeddings, embedding_dim),
            out=table_batched_embeddings.new_host_mapped_tensor(
                torch.randn(1).cuda()
                if not fp16
                else torch.randn(1).cuda().half(),
                (num_embeddings, embedding_dim),
            ),
        )
        self.embedding_weights = nn.Parameter(embedding_data)
        self.register_buffer(
            "lxu_cache_state", torch.zeros(4, cache_sets, 32).int()
        )
        self.lxu_cache_state[2, :, :] = -1
        self.register_buffer(
            "lxu_cache_weights",
            torch.zeros(cache_sets * 32, embedding_dim).to(
                embedding_data.dtype
            ),
        )
        self.t = 1
        self.learning_rate = learning_rate

    def prefetch(self, indices):
        import table_batched_embeddings
        self.t += 1
        table_batched_embeddings.lxu_cache_populate(
            self.embedding_weights, indices, self.lxu_cache_state, self.lxu_cache_weights, self.t, 32,
        )

    def forward(self, indices, offsets, pipelined=False):
        self.t += 1
        return LXUCacheLookupFunction.apply(
            self.embedding_weights,
            self.lxu_cache_weights,
            self.lxu_cache_state,
            indices,
            offsets,
            self.learning_rate,
            self.t,
            pipelined,
        )

