#include <torch/extension.h>
#include <ATen/ATen.h>

using namespace at;

Tensor batched_embedding_forward_cuda(Tensor weights, Tensor table_offsets,
                                      Tensor indices, Tensor offsets,
                                      c10::optional<Tensor> indice_weights,
                                      int64_t L_max, int64_t BT_block_size,
                                      bool shmem);

void batched_embedding_backward_sgd_cuda(Tensor grad_output, Tensor weights,
                                         Tensor table_offsets, Tensor indices,
                                         Tensor offsets, float learning_rate,
                                         int64_t L_max, int64_t T_block_size, bool shmem);

c10::optional<Tensor> batched_embedding_backward_sgd_exact_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, c10::optional<Tensor> indice_weights, float learning_rate,
    int64_t BT_block_size);

c10::optional<Tensor> batched_embedding_backward_adagrad_approx_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps, int64_t L_max,
    bool stochastic_rounding, int64_t BT_block_size);

c10::optional<Tensor> batched_embedding_backward_adagrad_exact_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps,
    bool stochastic_rounding, int64_t BT_block_size);

Tensor new_managed_tensor(Tensor self, std::vector<std::int64_t> sizes);
Tensor new_host_mapped_tensor(Tensor self, std::vector<std::int64_t> sizes);

Tensor batched_embedding_forward_mixed_D_cuda(
    Tensor weights, Tensor table_offsets, Tensor dim_offsets, int64_t total_D,
    Tensor indices, Tensor offsets, c10::optional<Tensor> indice_weights,
    int64_t L_max, int64_t BT_block_size, bool shmem);

c10::optional<Tensor> batched_embedding_backward_adagrad_approx_mixed_D_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets,
    Tensor table_dim_offsets, Tensor dim_offsets, int64_t total_D, Tensor indices,
    Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps, int64_t L_max,
    bool stochastic_rounding, int64_t BT_block_size);

c10::optional<Tensor> batched_embedding_backward_adagrad_exact_mixed_D_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets,
    Tensor table_dim_offsets, Tensor dim_offsets, int64_t total_D,
    Tensor indices, Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps,
    bool stochastic_rounding, int64_t BT_block_size);

Tensor construct_offsets(Tensor batch_offsets_per_table, // [T][B]
                         Tensor total_indices_per_table  // [T]
);

void lxu_cache_populate_cuda(Tensor weights, Tensor indices,
                             Tensor lxu_cache_state, Tensor lxu_cache_weights,
                             int64_t t, int64_t N_block_size);

Tensor lxu_cache_lookup_cuda(Tensor indices, Tensor lxu_cache_state, int64_t t,
                             int64_t N_block_size);

Tensor lxu_cache_forward_cuda(Tensor weights,
                              Tensor indices, Tensor offsets,
                              c10::optional<Tensor> indice_weights,
                              Tensor lxu_cache_locations,
                              Tensor lxu_cache_weights, int64_t B_block_size);

void lxu_cache_backward_sgd_cuda(Tensor grad_output, Tensor weights,
                                 Tensor indices,
                                 Tensor offsets,
                                 Tensor lxu_cache_locations,
                                 Tensor lxu_cache_weights,
                                 float learning_rate,
                                 int64_t B_block_size);

void lxu_cache_flush_cuda(Tensor weights, Tensor lxu_cache_state,
                          Tensor lxu_cache_weights, int64_t B_block_size);

void lxu_cache_backward_sgd_exact_cuda(Tensor grad_output, Tensor weights,
                                       Tensor indices, Tensor offsets,
                                       Tensor lxu_cache_locations,
                                       Tensor lxu_cache_weights,
                                       float learning_rate,
                                       int64_t B_block_size);

std::pair<Tensor, Tensor> lxu_cache_unique_indices_cuda(Tensor indices);

Tensor lxu_cache_forward_cpu(Tensor weights, Tensor indices_cpu,
                             Tensor offsets_cpu,
                             c10::optional<Tensor> indice_weights_cpu,
                             Tensor mask_cpu, Tensor output_cpu, int64_t handle);


void lxu_cache_backward_sgd_cpu(Tensor grad_output_cpu, Tensor weights,
                                Tensor indices_cpu, Tensor offsets_cpu,
                                Tensor mask_cpu, float learning_rate,
                                int64_t handle);


Tensor lxu_cache_forward_mixed_cuda(Tensor weights, Tensor indices,
                                    Tensor offsets,
                                    c10::optional<Tensor> indice_weights,
                                    Tensor lxu_cache_locations,
                                    Tensor lxu_cache_weights,
                                    int64_t B_block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &batched_embedding_forward_cuda);
    m.def("backward_sgd", &batched_embedding_backward_sgd_cuda);
    m.def("backward_exact_sgd", &batched_embedding_backward_sgd_exact_cuda);

    m.def("backward_approx_adagrad", &batched_embedding_backward_adagrad_approx_cuda);
    m.def("backward_exact_adagrad", &batched_embedding_backward_adagrad_exact_cuda);

    m.def("backward_approx_adagrad_mixed_D", &batched_embedding_backward_adagrad_approx_mixed_D_cuda);
    m.def("backward_exact_adagrad_mixed_D", &batched_embedding_backward_adagrad_exact_mixed_D_cuda);

    m.def("forward_mixed_D", &batched_embedding_forward_mixed_D_cuda);
    m.def("construct_offsets", &construct_offsets);
    m.def("new_managed_tensor", &new_managed_tensor);
    m.def("new_host_mapped_tensor", &new_host_mapped_tensor);

    m.def("lxu_cache_populate", &lxu_cache_populate_cuda);
    m.def("lxu_cache_lookup", &lxu_cache_lookup_cuda);
    m.def("lxu_cache_forward", &lxu_cache_forward_cuda);
    m.def("lxu_cache_backward_sgd", &lxu_cache_backward_sgd_cuda);
    m.def("lxu_cache_backward_sgd_exact", &lxu_cache_backward_sgd_exact_cuda);
    m.def("lxu_cache_flush", &lxu_cache_flush_cuda);
    m.def("lxu_cache_unique_indices", &lxu_cache_unique_indices_cuda);
    m.def("lxu_cache_forward_cpu", &lxu_cache_forward_cpu);
    m.def("lxu_cache_backward_sgd_cpu", &lxu_cache_backward_sgd_cpu);

    m.def("lxu_cache_forward_mixed_cuda", &lxu_cache_forward_mixed_cuda);
}
