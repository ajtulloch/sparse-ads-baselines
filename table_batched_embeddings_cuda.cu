#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace at;

namespace {

static constexpr int32_t kWarpSize = 32;

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

template <typename T> struct Sum {
  __device__ inline T operator()(T a, T b) const { return a + b; }

  inline __device__ T identity() const { return static_cast<T>(0); }
};

template <typename T, typename Op, int ReduceWidth = kWarpSize>
__device__ inline T warpReduceAll(T val, Op op) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val = op(val, shfl_xor(val, mask));
  }

  return val;
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
__device__ inline T warpReduceAllSum(T val) {
  return warpReduceAll<T, Sum<T>, ReduceWidth>(val, Sum<T>());
}

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

/// Performs a block-wide reduction
template <typename T, typename Op, bool BroadcastAll, bool KillWARDependency>
__device__ inline T blockReduceAll(T val, Op op, T *smem) {
  int laneId = threadIdx.x % kWarpSize;
  int warpId = threadIdx.x / kWarpSize;

  val = warpReduceAll<T, Op>(val, op);
  if (laneId == 0) {
    smem[warpId] = val;
  }
  __syncthreads();

  if (warpId == 0) {
    val = laneId < divUp(blockDim.x, kWarpSize) ? smem[laneId] : op.identity();
    val = warpReduceAll<T, Op>(val, op);

    if (BroadcastAll) {
      __threadfence_block();

      if (laneId == 0) {
        smem[0] = val;
      }
    }
  }

  if (BroadcastAll) {
    __syncthreads();
    val = smem[0];
  }

  if (KillWARDependency) {
    __syncthreads();
  }

  return val;
}

/// Sums a register value across the entire block
template <typename T, bool BroadcastAll, bool KillWARDependency>
__device__ inline T blockReduceAllSum(T val, T *smem) {
  return blockReduceAll<T, Sum<T>, BroadcastAll, KillWARDependency>(
      val, Sum<T>(), smem);
}

static inline __device__ double gpuAtomicAdd(double *address, double val) {
  return atomicAdd(address, val);
}

static inline __device__ float gpuAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}

static inline __device__ at::Half gpuAtomicAdd(at::Half *address,
                                               at::Half val) {
#if ((CUDA_VERSION < 10000) ||                                                 \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  unsigned int *address_as_ui =
      (unsigned int *)((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  at::Half hsum;
  do {
    assumed = old;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
  hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return hsum;
#else
  return atomicAdd(reinterpret_cast<__half *>(address), val);
#endif
}
}

template <typename scalar_t, bool T_blocked>
__global__ void batched_embedding_forward_kernel(
    // [\sum_t E_t][D]
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices, // [N = \sum_{b,t} L_{b,t} total indices, i.e. flattened
                 // [B][T][L]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        offsets, // [B x T + 1]
    // offsets = cumsum([0] + lengths.contiguous()), where lengths L is [B][T].
    PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits>
        output, // [B][T][D],
    int32_t L_max) {

  extern __shared__ int32_t shmem_indices[];

  const int32_t B = output.size(0);
  const int32_t T = output.size(1);
  const int32_t D = output.size(2);

  if (!T_blocked) {
    int32_t d = threadIdx.x;
    int32_t b = blockIdx.x;
    int32_t t = blockIdx.y;
    const int32_t table_offset = table_offsets[t];
    const int32_t indices_start = offsets[b * T + t];
    const int32_t indices_end = offsets[b * T + t + 1];
    int32_t L = indices_end - indices_start;

    for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();

    acc_type<scalar_t, true> sum = 0.0f;
#pragma unroll 8
    for (int32_t l = 0; l < L; ++l) {
      sum += __ldg((&weights[table_offset + shmem_indices[l]][0]) + d);
    }
    *((&output[b][t][0]) + d) = sum;
  } else {
    int32_t d = threadIdx.x;
    int32_t t_t = threadIdx.y;
    int32_t b = blockIdx.x;
    int32_t t = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t L = 0;
    if (t < T) {
      const int32_t indices_start = offsets[b * T + t];
      const int32_t indices_end = offsets[b * T + t + 1];
      L = indices_end - indices_start;
      for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
        shmem_indices[t_t * L_max + i] = __ldg(&indices[indices_start + i]);
      }
    }
    __syncthreads();
    if (t < T) {
      const int32_t table_offset = table_offsets[t];
      acc_type<scalar_t, true> sum = 0.0f;
#pragma unroll 8
      for (int32_t l = 0; l < L; ++l) {
        sum += __ldg(
            (&weights[table_offset + shmem_indices[t_t * L_max + l]][0]) + d);
      }
      *((&output[b][t][0]) + d) = sum;
    }
  }
}

template <typename scalar_t, bool T_blocked, typename F>
__global__ void batched_embedding_backward_sgd_kernel(
    const PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    int32_t L_max, F f) {
  extern __shared__ int32_t shmem_indices[];

  const int32_t B = grad_output.size(0);
  const int32_t T = grad_output.size(1);
  const int32_t D = grad_output.size(2);

  if (!T_blocked) {
    int32_t d = threadIdx.x;
    int32_t b = blockIdx.x;
    int32_t t = blockIdx.y;

    const int32_t table_offset = table_offsets[t];
    const int32_t indices_start = offsets[b * T + t];
    const int32_t indices_end = offsets[b * T + t + 1];
    int32_t L = indices_end - indices_start;

    for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();
#pragma unroll 8
    for (int32_t l = 0; l < L; ++l) {
      const auto g = __ldg(&grad_output[b][t][0] + d);
      f(&weights[table_offset + shmem_indices[l]][0] + d, g);
    }
  } else {
    int32_t d = threadIdx.x;
    int32_t t_t = threadIdx.y;
    int32_t b = blockIdx.x;
    int32_t t = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t L = 0;
    if (t < T) {
      const int32_t indices_start = offsets[b * T + t];
      const int32_t indices_end = offsets[b * T + t + 1];
      L = indices_end - indices_start;
      for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
        shmem_indices[t_t * L_max + i] = __ldg(&indices[indices_start + i]);
      }
    }
    __syncthreads();
    if (t < T) {
      const int32_t table_offset = table_offsets[t];
#pragma unroll 8
      for (int32_t l = 0; l < L; ++l) {
        const auto g = __ldg(&grad_output[b][t][0] + d);
        f(&weights[table_offset + shmem_indices[t_t * L_max + l]][0] + d, g);
      }
    }
  }
}

Tensor batched_embedding_forward_cuda(Tensor weights, Tensor table_offsets,
                                      Tensor indices, Tensor offsets,
                                      int64_t L_max, int64_t T_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  auto output = empty({B, T, D}, weights.options());
  if (T_block_size > 0) {
    const int32_t T_t = T_block_size;
    const int32_t T_b = (T + T_t - 1) / T_t;
    const dim3 threads(D, T_t);
    const dim3 blocks(B, T_b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "batched_embedding_forward_kernel", ([&] {
          batched_embedding_forward_kernel<
              scalar_t, true><<<blocks, threads, T_t * L_max * sizeof(int32_t),
                                at::cuda::getCurrentCUDAStream()>>>(
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max));
        }));
  } else {
    const int32_t threads = D;
    const dim3 blocks(B, T);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "batched_embedding_forward_kernel", ([&] {
          batched_embedding_forward_kernel<
              scalar_t, false><<<blocks, threads, L_max * sizeof(int),
                                 at::cuda::getCurrentCUDAStream()>>>(
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max));
        }));
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

template <typename scalar_t, bool use_atomics> struct SGDFunctor {
  SGDFunctor(float learning_rate) : learning_rate_(learning_rate) {}
  float learning_rate_;
  inline void __device__ operator()(scalar_t *weight, scalar_t grad) {
    if (use_atomics) {
      // TODO: stochastic rounding for fp16?
      gpuAtomicAdd(weight, -learning_rate_ * grad);
    } else {
      *weight -= grad * learning_rate_;
    }
  }
};

void batched_embedding_backward_sgd_cuda(Tensor grad_output, Tensor weights,
                                         Tensor table_offsets, Tensor indices,
                                         Tensor offsets, float learning_rate,
                                         int64_t L_max, int64_t T_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  if (T_block_size > 0) {
    const int32_t T_t = T_block_size;
    const int32_t T_b = (T + T_t - 1) / T_t;
    const dim3 threads(D, T_t);
    const dim3 blocks(B, T_b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "batched_embedding_backward_kernel", ([&] {
          batched_embedding_backward_sgd_kernel<
              scalar_t, true><<<blocks, threads, T_t * L_max * sizeof(int32_t),
                                at::cuda::getCurrentCUDAStream()>>>(
              grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max),
              SGDFunctor<scalar_t, false>(learning_rate));
        }));
  } else {
    const int32_t threads = D;
    const dim3 blocks(B, T);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "batched_embedding_backward_kernel", ([&] {
          batched_embedding_backward_sgd_kernel<
              scalar_t, false><<<blocks, threads, L_max * sizeof(int),
                                 at::cuda::getCurrentCUDAStream()>>>(
              grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max),
              SGDFunctor<scalar_t, false>(learning_rate));
        }));
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return;
}

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_adagrad_approx_kernel(
    const PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<scalar_t, true>, 1, RestrictPtrTraits>
        optimizer_state,
    int32_t L_max, F f) {
  extern __shared__ int32_t shmem[];

  const int32_t B = grad_output.size(0);
  const int32_t T = grad_output.size(1);
  const int32_t D = grad_output.size(2);

  int32_t *shmem_indices = &shmem[0];
  auto *shmem_reduction =
      reinterpret_cast<acc_type<scalar_t, true> *>(&shmem_indices[L_max]);
  auto *shmem_multipliers = &shmem_reduction[D / kWarpSize];

  int32_t d = threadIdx.x;
  int32_t b = blockIdx.x;
  int32_t t = blockIdx.y;

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[b * T + t];
  const int32_t indices_end = offsets[b * T + t + 1];
  int32_t L = indices_end - indices_start;
  const acc_type<scalar_t, true> g = __ldg(&grad_output[b][t][0] + d);
  const acc_type<scalar_t, true> g_sum_square =
      blockReduceAllSum<acc_type<scalar_t, true>, true, false>(
          g * g,
          reinterpret_cast<acc_type<scalar_t, true> *>(&shmem_reduction[0]));

  // thread per D, over each L, means each thread needs all $L multipliers.
  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    shmem_indices[i] = idx;
    shmem_multipliers[i] =
        f.update_momentum(g_sum_square, &optimizer_state[table_offset + idx]);
  }
  __syncthreads();
  for (int32_t l = 0; l < L; ++l) {
    const auto g = __ldg(&grad_output[b][t][0] + d);
    f.update_weight(&weights[table_offset + shmem_indices[l]][0] + d,
                    shmem_multipliers[l], g);
  }
}

template <typename scalar_t, bool use_atomics> struct AdaGradFunctor {
  AdaGradFunctor(float learning_rate, float eps)
      : learning_rate_(learning_rate), eps_(eps) {}
  float learning_rate_;
  float eps_;
  __device__ inline __attribute__((always_inline)) acc_type<scalar_t, true>
  update_momentum(acc_type<scalar_t, true> sum_square_grads,
                  acc_type<scalar_t, true> *optimizer_state) {
    acc_type<scalar_t, true> old_sum_square_grads;
    if (use_atomics) {
      old_sum_square_grads = gpuAtomicAdd(optimizer_state, sum_square_grads);
    } else {
      old_sum_square_grads = __ldg(optimizer_state);
      *optimizer_state = old_sum_square_grads + sum_square_grads;
    }
    return learning_rate_ *
           (1.0 / (sqrt(old_sum_square_grads + sum_square_grads) + eps_));
  }

  __device__ inline __attribute__((always_inline)) void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                scalar_t grad) {
    // TODO: stochastic rounding for fp16?
    if (use_atomics) {
      gpuAtomicAdd(weight, -multiplier * grad);
    } else {
      *weight -= grad * multiplier;
    }
  }
};

void batched_embedding_backward_adagrad_approx_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, Tensor optimizer_state, float learning_rate, float eps,
    int64_t L_max) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);
  // TODO: relax this constraint. It is due to the BlockReduce assuming a full
  // warp.
  AT_ASSERT(D % kWarpSize == 0);

  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  const int32_t threads = D;
  const dim3 blocks(B, T);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        batched_embedding_backward_adagrad_approx_kernel<
            scalar_t><<<blocks, threads,
                        L_max * sizeof(int) +
                            sizeof(acc_type<scalar_t, true>) * D / kWarpSize +
                            L_max * sizeof(acc_type<scalar_t, true>),
                        at::cuda::getCurrentCUDAStream()>>>(
            grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
            table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,
                                              RestrictPtrTraits>(),
            static_cast<int32_t>(L_max),
            AdaGradFunctor<scalar_t, false>(learning_rate, eps));
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return;
}

static void CUDAManagedDeleter(void *ptr) { AT_CUDA_CHECK(cudaFree(ptr)); }

struct CUDAManagedAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void *ptr;
    AT_CUDA_CHECK(cudaMallocManaged(&ptr, size));
    AT_CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation,
                                cudaCpuDeviceId));
    AT_CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy,
                                at::cuda::current_device()));
    return {ptr,
            ptr,
            &CUDAManagedDeleter,
            {at::DeviceType::CUDA, at::cuda::current_device()}};
  }

  at::DeleterFnPtr raw_deleter() const override { return &CUDAManagedDeleter; }
};

static CUDAManagedAllocator g_managed_allocator;

static void CUDAHostMappedDeleter(void *ptr) {
  AT_CUDA_CHECK(cudaFreeHost(ptr));
}

struct CUDAHostMappedAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void *ptr;
    AT_CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined |
                                                cudaHostAllocMapped));
    return {ptr,
            ptr,
            &CUDAHostMappedDeleter,
            {at::DeviceType::CUDA, at::cuda::current_device()}};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &CUDAHostMappedDeleter;
  }
};

static CUDAHostMappedAllocator g_host_mapped_allocator;

std::vector<int64_t> defaultStrides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (size_t i = sizes.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= sizes[i - 1];
  }
  return strides;
}

int64_t computeStorageSize(IntArrayRef sizes, IntArrayRef strides) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  int64_t size = 1;
  for (size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] == 0) {
      return 0;
    }
    size += strides[i] * (sizes[i] - 1);
  }
  return size;
}

Tensor new_managed_tensor(Tensor self, std::vector<std::int64_t> sizes) {
  auto strides = defaultStrides(sizes);
  auto storage = Storage(self.dtype(), computeStorageSize(sizes, strides),
                         &g_managed_allocator,
                         /*resizable=*/false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, sizes, strides);
  return tensor;
}

Tensor new_host_mapped_tensor(Tensor self, std::vector<std::int64_t> sizes) {
  auto strides = defaultStrides(sizes);
  auto storage = Storage(self.dtype(), computeStorageSize(sizes, strides),
                         &g_host_mapped_allocator,
                         /*resizable=*/false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, sizes, strides);
  return tensor;
}