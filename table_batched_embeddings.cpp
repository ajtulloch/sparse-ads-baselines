#include <torch/extension.h>
#include <ATen/ATen.h>

using namespace at;

Tensor batched_embedding_forward_cuda(
    Tensor weights, Tensor table_offsets, Tensor indices, Tensor offsets, int64_t L_max, int64_t B_block_size, bool shmem);

void batched_embedding_backward_sgd_cuda(Tensor grad_output, Tensor weights,
                                         Tensor table_offsets, Tensor indices,
                                         Tensor offsets, float learning_rate,
                                         int64_t L_max, int64_t T_block_size, bool shmem);

void batched_embedding_backward_adagrad_approx_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, Tensor optimizer_state, float learning_rate, float eps,
    int64_t L_max, bool stochastic_rounding, int64_t BT_block_size);

Tensor new_managed_tensor(Tensor self, std::vector<std::int64_t> sizes);
Tensor new_host_mapped_tensor(Tensor self, std::vector<std::int64_t> sizes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &batched_embedding_forward_cuda);
  m.def("backward_sgd", &batched_embedding_backward_sgd_cuda);
  m.def("backward_approx_adagrad", &batched_embedding_backward_adagrad_approx_cuda);
  m.def("new_managed_tensor", &new_managed_tensor);
  m.def("new_host_mapped_tensor", &new_host_mapped_tensor);
}
