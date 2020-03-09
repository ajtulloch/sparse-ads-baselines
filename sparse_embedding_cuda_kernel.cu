#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

static inline __device__ void gpuAtomicAdd(double* address, double val) {
  atomicAdd(address, val);
}

static inline __device__ void gpuAtomicAdd(float* address, float val) {
  atomicAdd(address, val);
}

static inline __device__ void gpuAtomicAdd(at::Half* address, at::Half val) {
#if ((CUDA_VERSION < 10000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    at::Half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
#else
  atomicAdd(reinterpret_cast<__half*>(address), val);
#endif
}

template <typename scalar_t>
__global__ void sparse_embedding_cuda_forward_kernel_impl(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits>
        indices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        output) {
  const int B = indices.size(0);
  const int T = indices.size(1);
  const int D = weights.size(2);
  const int L = indices.size(2);
  const int E = weights.size(0);

  int t_d = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  int d = t_d % D;
  int t = t_d / D;
  if (t < T) {
    at::acc_type<scalar_t, true> sum = 0.0f;
    for (int l = 0; l < L; ++l) {
      // TODO: does ldg work better?
      const int64_t ind = __ldg(&indices[b][t][l]);
      // TODO: enable asserts.
      // assert(ind >= 0);
      // assert(ind < E);
      sum += __ldg(&weights[ind][t][d]);
    }
    output[b][t][d] = sum;
  }
}

template <typename scalar_t, bool T_blocked>
__global__ void sparse_embedding_cuda_forward_fast_kernel_impl(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits>
        indices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        output) {
  extern __shared__ int shmem_indices[];
  if (!T_blocked) {
    const int L = indices.size(2);
    int d = threadIdx.x;
    int b = blockIdx.x;
    int t = blockIdx.y;

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[i] = __ldg((&indices[b][t][0]) + i);
    }
    __syncthreads();

    at::acc_type<scalar_t, true> sum = 0.0f;
#pragma unroll 8
    for (int l = 0; l < L; ++l) {
      sum += __ldg((&weights[shmem_indices[l]][t][0]) + d);
    }
    output[b][t][d] = sum;
  } else {
    const int L = indices.size(2);
    const int T = indices.size(1);

    int d = threadIdx.x;
    int t_t = threadIdx.y;
    int b = blockIdx.x;
    int t_b = blockIdx.y;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t < T) {
      for (int i = threadIdx.x; i < L; i += blockDim.x) {
        shmem_indices[t_t * L + i] = __ldg((&indices[b][t][0]) + i);
      }
    }
    __syncthreads();
    if (t < T) {
      at::acc_type<scalar_t, true> sum = 0.0f;
#pragma unroll 8
      for (int l = 0; l < L; ++l) {
        sum += __ldg((&weights[shmem_indices[t_t * L + l]][t][0]) + d);
      }
      output[b][t][d] = sum;
    }
  }
}

template <typename scalar_t>
__global__ void sparse_embedding_cuda_fused_backward_update_kernel_impl(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits>
        indices,
    float lr) {
  const int B = indices.size(0);
  const int T = indices.size(1);
  const int D = weights.size(2);
  const int L = indices.size(2);
  const int E = weights.size(0);
  // grad_output: [B][T][D]
  // weights: [E][T][D]
  // indices: [B][T][L]
  // launch grid of size [B][T][L x D]

  // what if we launch a kernel for every !grad_output - collaboratively load
  // indices L for each index, and then scatter the update to each l?
  // That is, [d] threads per block

  int l_d = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  int t = blockIdx.z;
  int d = l_d % D;
  int l = l_d / D;
  if (l < L) {
    const auto g = __ldg(&grad_output[b][t][d]);
    const int64_t ind = __ldg(&indices[b][t][l]);
    // TODO: need atomicadd?
    // gpuAtomicAdd(&weights[ind][t][d], -lr * g);
    weights[ind][t][d] -= lr * g;
  }
}

template <typename scalar_t, bool T_blocked>
__global__ void sparse_embedding_cuda_fused_backward_update_fast_kernel_impl(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits>
        indices,
    float lr) {
  extern __shared__ int shmem_indices[];

  if (!T_blocked) {
    const int L = indices.size(2);
    const int T = indices.size(1);

    int d = threadIdx.x;
    int b = blockIdx.x;
    int t = blockIdx.y;
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[i] = __ldg((&indices[b][t][0]) + i);
    }
    __syncthreads();

#pragma unroll 8
    for (int l = 0; l < L; ++l) {
      const auto g = __ldg((&grad_output[b][t][0]) + d);
      *((&weights[shmem_indices[l]][t][0]) + d) -= lr * g;
    }
  } else {
    const int L = indices.size(2);
    const int T = indices.size(1);

    int d = threadIdx.x;
    int t_t = threadIdx.y;
    int b = blockIdx.x;
    int t_b = blockIdx.y;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    // [T_t][L] tile.
    // need T_t * L <= 32
    if (t < T) {
      for (int i = threadIdx.x; i < L; i += blockDim.x) {
        shmem_indices[t_t * L + i] = __ldg((&indices[b][t][0]) + i);
      }
    }
    __syncthreads();

    if (t < T) {
#pragma unroll 8
      for (int l = 0; l < L; ++l) {
        const auto g = __ldg((&grad_output[b][t][0]) + d);
        *((&weights[shmem_indices[t_t * L + l]][t][0]) + d) -= lr * g;
      }
    }
  }
}

torch::Tensor sparse_embedding_cuda_forward_kernel(
    // [E][T][D]
    torch::Tensor weights,
    // [B][T][L // #device]
    torch::Tensor indices) {
  const auto B = indices.size(0);
  const auto T = indices.size(1);
  const auto D = weights.size(2);

  auto output = at::empty({B, T, D}, weights.options());

  const int threads = 1024;
  const dim3 blocks((T * D + threads - 1) / threads, B);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "sparse_embedding_cuda_forward", ([&] {
        sparse_embedding_cuda_forward_kernel_impl<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            weights.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());

  return output;
}

torch::Tensor sparse_embedding_cuda_forward_fast_kernel(
    // [E][T][D]
    torch::Tensor weights,
    // [B][T][L // #device]
    torch::Tensor indices) {
  const auto B = indices.size(0);
  const auto T = indices.size(1);
  const auto D = weights.size(2);
  const auto L = indices.size(2);
  auto output = at::empty({B, T, D}, weights.options());

  // Launch D threads.
  if (D < 64) {
    const int T_t = 64 / D;
    const int T_b = (T + T_t - 1) / T_t;
    const dim3 threads(D, T_t);
    const dim3 blocks(B, T_b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_forward", ([&] {
          sparse_embedding_cuda_forward_fast_kernel_impl<
              scalar_t, true><<<blocks, threads, T_t * L * sizeof(int),
                                at::cuda::getCurrentCUDAStream()>>>(
              weights
                  .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              indices.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
              output
                  .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  } else {
    const int threads = D;
    const dim3 blocks(B, T);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_forward", ([&] {
          sparse_embedding_cuda_forward_fast_kernel_impl<scalar_t, false><<<
              blocks, threads, L * sizeof(int), at::cuda::getCurrentCUDAStream()>>>(
              weights
                  .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              indices.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
              output
                  .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  }
  return output;
}

void sparse_embedding_cuda_backward_update_kernel(torch::Tensor grad_output,
                                                  torch::Tensor weights,
                                                  torch::Tensor indices,
                                                  float lr) {
  const auto B = indices.size(0);
  const auto T = indices.size(1);
  const auto D = weights.size(2);
  const int L = indices.size(2);

  const int threads = 1024;
  const dim3 blocks((L * D + threads - 1) / threads, B, T);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "sparse_embedding_cuda_backward_update", ([&] {
        sparse_embedding_cuda_fused_backward_update_kernel_impl<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output
                .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            lr);
      }));
  AT_CUDA_CHECK(cudaGetLastError());

  return;
}

void sparse_embedding_cuda_backward_update_fast_kernel(
    torch::Tensor grad_output, torch::Tensor weights, torch::Tensor indices,
    float lr) {
  const auto B = indices.size(0);
  const auto T = indices.size(1);
  const auto D = weights.size(2);
  const auto L = indices.size(2);

  if (D < 64) {
    const int T_t = 64 / D;
    const int T_b = (T + T_t - 1) / T_t;
    const dim3 threads(D, T_t);
    const dim3 blocks(B, T_b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_backward_update_fast", ([&] {
          sparse_embedding_cuda_fused_backward_update_fast_kernel_impl<
              scalar_t, true><<<blocks, threads, T_t * L * sizeof(int),
                                at::cuda::getCurrentCUDAStream()>>>(
              grad_output
                  .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
              weights
                  .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              indices.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
              lr);
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  } else {
    const int threads = D;
    const dim3 blocks(B, T);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_backward_update_fast", ([&] {
          sparse_embedding_cuda_fused_backward_update_fast_kernel_impl<
              scalar_t, false><<<blocks, threads, L * sizeof(int),
                                 at::cuda::getCurrentCUDAStream()>>>(
              grad_output
                  .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
              weights
                  .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              indices.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
              lr);
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  }
  return;
}