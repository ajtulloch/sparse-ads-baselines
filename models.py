import click

import torch
from torch import nn, Tensor
from typing import List

import horovod.torch as hvd
import apex
from sparse_embedding_cuda_ops import UniformShardedEmbeddingBags, FastZeroFusedSGD, LookupFunction, All2AllFunction, ReduceScatterFunction


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.relu(out)


class SNN(nn.Module):
    def __init__(self, num_tables, num_embeddings, embedding_dim,
                 dense_features_dim):
        super(SNN, self).__init__()
        self.dense_mlp = MLP(dense_features_dim, dense_features_dim,
                             embedding_dim).cuda()
        self.embedding_tables = nn.ModuleList()
        for _ in range(num_tables):
            self.embedding_tables.append(
                nn.EmbeddingBag(num_embeddings,
                                embedding_dim,
                                mode="sum",
                                sparse=True).cuda())
        self.output_mlp = MLP(embedding_dim * num_tables + embedding_dim, 512,
                              1).cuda()

    def forward(self, dense_features, sparse_features):
        dense_x = self.dense_mlp(dense_features)
        embedding_x = [
            embedding(indices)
            for (indices,
                 embedding) in zip(sparse_features, self.embedding_tables)
        ]
        features = torch.cat([dense_x] + embedding_x, dim=1)
        logits = self.output_mlp(features)
        return logits


class UniformShardedSNN(nn.Module):
    def __init__(self, num_tables, num_embeddings, embedding_dim,
                 dense_features_dim):
        super(UniformShardedSNN, self).__init__()
        self.dense_mlp = MLP(dense_features_dim, dense_features_dim,
                             embedding_dim)
        self.embedding_tables = UniformShardedEmbeddingBags(
            num_tables, num_embeddings, embedding_dim)
        self.output_mlp = MLP(embedding_dim * num_tables + embedding_dim, 512,
                              1)

    def forward(self, dense_features, sharded_sparse_features):
        dense_x = self.dense_mlp(dense_features)
        embeddings_x = self.embedding_tables(sharded_sparse_features)
        features = torch.cat(
            [dense_x, embeddings_x.flatten(start_dim=1)], dim=1)
        logits = self.output_mlp(features)
        return logits


class DistributedUniformShardedSNN(nn.Module):
    def __init__(self, num_tables, num_embeddings, embedding_dim,
                 dense_features_dim):
        super(DistributedUniformShardedSNN, self).__init__()
        self.dense_mlp = MLP(dense_features_dim, dense_features_dim,
                             embedding_dim)
        assert num_embeddings % hvd.size() == 0
        self.embedding_tables = UniformShardedEmbeddingBags(
            num_tables, num_embeddings // hvd.size(), embedding_dim)
        self.output_mlp = MLP(embedding_dim * num_tables + embedding_dim, 512,
                              1)

    def forward(self, dense_features, sharded_sparse_features):
        dense_x = self.dense_mlp(dense_features)
        embeddings_x = ReduceScatterFunction.apply(
            self.embedding_tables(sharded_sparse_features))
        features = torch.cat(
            [dense_x, embeddings_x.flatten(start_dim=1)], dim=1)
        logits = self.output_mlp(features)
        return logits


class DistributedPartitionShardedSNN(nn.Module):
    def __init__(self, num_tables, num_embeddings, embedding_dim,
                 dense_features_dim):
        super(DistributedPartitionShardedSNN, self).__init__()
        self.dense_mlp = MLP(dense_features_dim, embedding_dim, embedding_dim)
        assert num_tables % hvd.size() == 0
        self.embedding_tables = UniformShardedEmbeddingBags(
            num_tables // hvd.size(), num_embeddings, embedding_dim)
        self.output_mlp = MLP(embedding_dim * num_tables + embedding_dim, 512,
                              1)

    def forward(self, dense_features, partitioned_sparse_features):
        dense_x = self.dense_mlp(dense_features)
        embeddings_x = All2AllFunction.apply(
            self.embedding_tables(partitioned_sparse_features))
        features = torch.cat(
            [dense_x, embeddings_x.flatten(start_dim=1)], dim=1)
        logits = self.output_mlp(features)
        return logits
