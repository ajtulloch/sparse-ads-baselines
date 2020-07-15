#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGenerator.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_select.cuh"

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mutex>
#define DMLC_GLOG_DEFINED 1
#include <tvm/runtime/packed_func.h>

using namespace at;

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

namespace {

static constexpr int32_t kWarpSize = 32;
static constexpr int32_t kMaxThreads = 1024;

template <typename T> struct Vec4T {};

struct Half4 {
  half2 a;
  half2 b;

  __device__ inline void store(Half *p) {
#if CUDA_VERSION >= 9000
    asm("st.v2.u32 [%0], {%1, %2};"
        :
        : "l"(p), "r"(__HALF2_TO_UI(a)), "r"(__HALF2_TO_UI(b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(a.x), "r"(b.x));
#endif
  }
};

template <> struct Vec4T<Half> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const Half *p) {
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
        : "l"(p));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(out.a.x), "=r"(out.b.x)
        : "l"(p));
#endif

    float2 a = __half22float2(out.a);
    float2 b = __half22float2(out.b);

    acc.x = a.x;
    acc.y = a.y;
    acc.z = b.x;
    acc.w = b.y;
  }

  DEVICE_INLINE void store(Half *p) {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    float2 b;
    b.x = acc.z;
    b.y = acc.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
    out.store(p);
  }

  DEVICE_INLINE static void copy(const Half *src, Half *dst) {
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
        : "l"(src));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(out.a.x), "=r"(out.b.x)
        : "l"(src));
#endif
#if CUDA_VERSION >= 9000
    asm("st.v2.u32 [%0], {%1, %2};"
        :
        : "l"(dst), "r"(__HALF2_TO_UI(out.a)), "r"(__HALF2_TO_UI(out.b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(dst), "r"(out.a.x), "r"(out.b.x));
#endif
  }
};

template <> struct Vec4T<float> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const float *p) { acc = *((const float4 *)p); }
  DEVICE_INLINE void store(float *p) { *((float4 *)p) = acc; }
  DEVICE_INLINE static void copy(const float *src, float *dst) {
    *((float4 *)dst) = *((const float4 *)src);
  }
};

template <> struct Vec4T<double> {
  double4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const double *p) { acc = *((const double4 *)p); }
  DEVICE_INLINE void store(double *p) { *((double4 *)p) = acc; }
  DEVICE_INLINE static void copy(const double *src, double *dst) {
    *((double4 *)dst) = *((const double4 *)src);
  }
};

template <typename T>
DEVICE_INLINE T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xFFFFFFFF, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warpReduceAllSum(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val += shfl_xor(val, mask);
  }
  return val;
}

static DEVICE_INLINE uint64_t gpuAtomicOr(uint64_t *p, uint64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long),
                "expected uint64_t to be unsigned long long");
  return static_cast<unsigned long long>(
      atomicOr(reinterpret_cast<unsigned long long int *>(p),
               static_cast<unsigned long long int>(val)));
}

static DEVICE_INLINE double gpuAtomicAdd(int32_t *address, int32_t val) {
  return atomicAdd(address, val);
}

static DEVICE_INLINE double gpuAtomicAdd(double *address, double val) {
  return atomicAdd(address, val);
}

static DEVICE_INLINE float gpuAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}

static DEVICE_INLINE at::Half gpuAtomicAdd(at::Half *address, at::Half val) {
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

template <typename scalar_t> struct UnweightedForward {
  DEVICE_INLINE void accumulate(Vec4T<scalar_t> &sum, Vec4T<scalar_t> weight,
                                int32_t indices_offset) {
    sum.acc.x += weight.acc.x;
    sum.acc.y += weight.acc.y;
    sum.acc.z += weight.acc.z;
    sum.acc.w += weight.acc.w;
  }
};

template <typename scalar_t> struct WeightedForward {
  const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits> indice_weights_;

  WeightedForward(const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
                      indice_weights)
      : indice_weights_(indice_weights) {}

  DEVICE_INLINE void accumulate(Vec4T<scalar_t> &sum, Vec4T<scalar_t> weight,
                                int32_t indices_offset) {
    acc_type<scalar_t, true> element_weight = indice_weights_[indices_offset];
    sum.acc.x += weight.acc.x * element_weight;
    sum.acc.y += weight.acc.y * element_weight;
    sum.acc.z += weight.acc.z * element_weight;
    sum.acc.w += weight.acc.w * element_weight;
  }
};

template <typename scalar_t, bool shared_indices, typename F>
__global__ void batched_embedding_forward_kernel_1(
    // [\sum_t E_t][D]
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices, // [N = \sum_{b,t} L_{b,t} total indices, i.e. flattened
                 // [T][B][L]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        offsets, // [T x B + 1]
    // offsets = cumsum([0] + lengths.contiguous()), where lengths L is [T][.
    PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits>
        output, // [B][T][D],
    int32_t L_max,
    F f) {

  extern __shared__ int32_t shmem_indices[];

  const int32_t B = output.size(0);
  const int32_t T = output.size(1);
  const int32_t D = output.size(2);

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t t = b_t / B;
  int32_t b = b_t % B;

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  if (shared_indices) {
    int32_t shmem_offset = threadIdx.y * L_max;
    for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[shmem_offset + i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = shmem_indices[shmem_offset + l];
        Vec4T<scalar_t> weight((&weights[table_offset + idx][0]) + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][t][0]) + d * 4);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = __ldg(&indices[indices_start + l]);
        Vec4T<scalar_t> weight((&weights[table_offset + idx][0]) + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][t][0]) + d * 4);
    }
  }
}

Tensor batched_embedding_forward_cuda(Tensor weights, Tensor table_offsets,
                                      Tensor indices, Tensor offsets,
                                      c10::optional<Tensor> indice_weights,
                                      int64_t L_max, int64_t BT_block_size,
                                      bool shmem) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size != 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT((B * T) % BT_block_size == 0);
  AT_ASSERT(D % 4 == 0);
  const dim3 threads(std::min(D / 4, kMaxThreads / BT_block_size),
                     BT_block_size);
  const dim3 blocks((B * T) / BT_block_size);

#define X(shmem_param, shmem_size, functor)                                    \
  batched_embedding_forward_kernel_1<scalar_t, (shmem_param)><<<               \
      blocks, threads, (shmem_size), at::cuda::getCurrentCUDAStream()>>>(      \
      weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),        \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),              \
      static_cast<int32_t>(L_max), (functor))

  auto output = empty({B, T, D}, weights.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_forward_kernel", ([&] {
        if (shmem) {
          if (indice_weights) {
            X(true, BT_block_size * L_max * sizeof(int32_t),
              WeightedForward<scalar_t>(
                  indice_weights
                      ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>()));
          } else {
            X(true, BT_block_size * L_max * sizeof(int32_t),
              UnweightedForward<scalar_t>());
          }
        } else {
          if (indice_weights) {
            X(false, 0,
              WeightedForward<scalar_t>(
                  indice_weights
                      ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>()));
          } else {
            X(false, 0, UnweightedForward<scalar_t>());
          }
        }
      }));

#undef X
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

template <typename scalar_t, bool use_atomics> struct SGDFunctor {
  SGDFunctor(float learning_rate) : learning_rate_(learning_rate) {}
  float learning_rate_;

  inline void __device__ operator()(scalar_t *weight, Vec4T<scalar_t> grad) {
    // don't handle atomic case.
    // if (use_atomics) {
    if (false) {
      gpuAtomicAdd(&weight[0], -grad.acc.x * learning_rate_);
      gpuAtomicAdd(&weight[1], -grad.acc.y * learning_rate_);
      gpuAtomicAdd(&weight[2], -grad.acc.z * learning_rate_);
      gpuAtomicAdd(&weight[3], -grad.acc.w * learning_rate_);
      // Vec4T<scalar_t> weight_new(weight);
      // weight_new.acc.x -= grad.acc.x * learning_rate_;
      // weight_new.acc.y -= grad.acc.y * learning_rate_;
      // weight_new.acc.z -= grad.acc.z * learning_rate_;
      // weight_new.acc.w -= grad.acc.w * learning_rate_;
    } else {
      Vec4T<scalar_t> weight_new(weight);
      weight_new.acc.x -= grad.acc.x * learning_rate_;
      weight_new.acc.y -= grad.acc.y * learning_rate_;
      weight_new.acc.z -= grad.acc.z * learning_rate_;
      weight_new.acc.w -= grad.acc.w * learning_rate_;
      weight_new.store(weight);
    }
  }
};

template <typename scalar_t, bool shared_mem, typename F>
__global__ void batched_embedding_backward_sgd_kernel_1(
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

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  if (shared_mem) {
    int32_t shmem_offset = threadIdx.y * L_max;

    for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[shmem_offset + i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();

    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> g(&grad_output[b][t][0] + d * 4);
      for (int32_t l = 0; l < L; ++l) {
        auto idx = shmem_indices[shmem_offset + l];
        f(&weights[table_offset + idx][0] + d * 4, g);
      }
    }
  } else {
    for (int32_t l = 0; l < L; ++l) {
      auto idx = __ldg(&indices[indices_start + l]);
      for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
        Vec4T<scalar_t> g(&grad_output[b][t][0] + d * 4);
        f(&weights[table_offset + idx][0] + d * 4, g);
      }
    }
  }
}

void batched_embedding_backward_sgd_cuda(Tensor grad_output, Tensor weights,
                                         Tensor table_offsets, Tensor indices,
                                         Tensor offsets, float learning_rate,
                                         int64_t L_max, int64_t BT_block_size,
                                         bool shmem) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(BT_block_size > 0);
  AT_ASSERT(D % 4 == 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_kernel", ([&] {
        const dim3 threads(std::min(D / 4, kMaxThreads / BT_block_size),
                           BT_block_size);
        const dim3 blocks((B * T) / BT_block_size);

        if (shmem) {
          batched_embedding_backward_sgd_kernel_1<
              scalar_t,
              true><<<blocks, threads, BT_block_size * L_max * sizeof(int32_t),
                      at::cuda::getCurrentCUDAStream()>>>(
              grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max),
              SGDFunctor<scalar_t, true>(learning_rate));
        } else {
          batched_embedding_backward_sgd_kernel_1<
              scalar_t,
              false><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max),
              SGDFunctor<scalar_t, true>(learning_rate));
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return;
}

template <typename scalar_t, bool use_atomics> struct AdaGradFunctor {
  AdaGradFunctor(float learning_rate, float eps)
      : learning_rate_(learning_rate), eps_(eps) {}
  float learning_rate_;
  float eps_;
  DEVICE_INLINE acc_type<scalar_t, true>
  update_momentum(acc_type<scalar_t, true> sum_square_grads,
                  acc_type<scalar_t, true> *optimizer_state,
                  int32_t indices_offset) {
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

  struct State {};
  DEVICE_INLINE State init_update(int thread_id) { return State{}; }

  DEVICE_INLINE void update_weight(scalar_t *weight,
                                   acc_type<scalar_t, true> multiplier,
                                   Vec4T<scalar_t> grad, State &state,
                                   State &sample_weight_state) {
    // can't use atomics.
    Vec4T<scalar_t> weight_new(weight);
    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    weight_new.store(weight);
  }

  DEVICE_INLINE State init_sample_weight(int32_t indices_offset) {
    return State{};
  }

  DEVICE_INLINE void update_sample_weight(State &state,
                                          int32_t indices_offset) {
    return;
  }
};

template <typename scalar_t, bool use_atomics> struct WeightedAdaGradFunctor {
  WeightedAdaGradFunctor(
      float learning_rate, float eps,
      const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
          indice_weights,
      PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
          grad_indice_weights)
      : learning_rate_(learning_rate), eps_(eps),
        indice_weights_(indice_weights),
        grad_indice_weights_(grad_indice_weights) {}

  float learning_rate_;
  float eps_;
  const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits> indice_weights_;
  PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits> grad_indice_weights_;

  DEVICE_INLINE acc_type<scalar_t, true>
  update_momentum(acc_type<scalar_t, true> sum_square_grads,
                  acc_type<scalar_t, true> *optimizer_state,
                  int32_t indices_offset) {
    acc_type<scalar_t, true> element_weight = indice_weights_[indices_offset];
    acc_type<scalar_t, true> weighted_sum_square_grads =
        sum_square_grads * element_weight * element_weight;
    acc_type<scalar_t, true> old_sum_square_grads;
    if (use_atomics) {
      old_sum_square_grads =
          gpuAtomicAdd(optimizer_state, weighted_sum_square_grads);
    } else {
      old_sum_square_grads = __ldg(optimizer_state);
      *optimizer_state = old_sum_square_grads + weighted_sum_square_grads;
    }
    return element_weight * learning_rate_ *
           (1.0 /
            (sqrt(old_sum_square_grads + weighted_sum_square_grads) + eps_));
  }

  struct State {};
  DEVICE_INLINE State init_update(int thread_id) { return State{}; }

  DEVICE_INLINE void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                Vec4T<scalar_t> grad, State &state,
                acc_type<scalar_t, true> &sample_weight_state) {
    // can't use atomics.
    Vec4T<scalar_t> weight_new(weight);

    sample_weight_state +=
        weight_new.acc.x * grad.acc.x + weight_new.acc.y * grad.acc.y +
        weight_new.acc.z * grad.acc.z + weight_new.acc.w * grad.acc.w;

    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    weight_new.store(weight);
  }
  DEVICE_INLINE acc_type<scalar_t, true>
  init_sample_weight(int32_t indices_offset) {
    return 0.0;
  }

  DEVICE_INLINE void
  update_sample_weight(acc_type<scalar_t, true> &sample_weight_state,
                       int32_t indices_offset) {
    auto accumulated_sample_weight_state =
        warpReduceAllSum<acc_type<scalar_t, true>>(sample_weight_state);
    // one thread per warp responsible for updating parameter.
    // TODO: ugly?
    if (threadIdx.x == 0) {
      grad_indice_weights_[indices_offset] = accumulated_sample_weight_state;
    }
  }
};

template <typename scalar_t>
DEVICE_INLINE void stochastic_rounding_vector(scalar_t *output,
                                              Vec4T<scalar_t> value,
                                              uint4 random_bits) {
  value.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(Half *output, Vec4T<Half> value,
                                              uint4 random_bits) {
  Half4 v;

  v.a.x = __float2half_rz(
      __uint_as_float(__float_as_uint(value.acc.x) + (random_bits.x >> 19)));
  v.a.y = __float2half_rz(
      __uint_as_float(__float_as_uint(value.acc.y) + (random_bits.y >> 19)));
  v.b.x = __float2half_rz(
      __uint_as_float(__float_as_uint(value.acc.z) + (random_bits.z >> 19)));
  v.b.y = __float2half_rz(
      __uint_as_float(__float_as_uint(value.acc.w) + (random_bits.w >> 19)));
  v.store(output);
}

template <typename scalar_t, bool use_atomics>
struct StochasticRoundingWeightedAdaGradFunctor
    : public WeightedAdaGradFunctor<scalar_t, use_atomics> {
  StochasticRoundingWeightedAdaGradFunctor(
      float learning_rate, float eps,
      const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
          indice_weights,
      PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
          grad_indice_weights,
      std::pair<uint64_t, uint64_t> seeds)
      : WeightedAdaGradFunctor<scalar_t, use_atomics>(
            learning_rate, eps, indice_weights, grad_indice_weights),
        seeds_(seeds) {}
  std::pair<uint64_t, uint64_t> seeds_;

  DEVICE_INLINE curandStatePhilox4_32_10_t init_update(int thread_id) {
    curandStatePhilox4_32_10_t state;
    curand_init(seeds_.first, thread_id, seeds_.second, &state);
    return state;
  }

  DEVICE_INLINE void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                Vec4T<scalar_t> grad, curandStatePhilox4_32_10_t &state,
                acc_type<scalar_t, true> &sample_weight_state) {
    Vec4T<scalar_t> weight_new(weight);
    sample_weight_state +=
        weight_new.acc.x * grad.acc.x + weight_new.acc.y * grad.acc.y +
        weight_new.acc.z * grad.acc.z + weight_new.acc.w * grad.acc.w;
    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    uint4 bits = curand4(&state);
    stochastic_rounding_vector(weight, weight_new, bits);
  }
};

template <typename scalar_t, bool use_atomics>
struct StochasticRoundingAdaGradFunctor
    : public AdaGradFunctor<scalar_t, use_atomics> {
  StochasticRoundingAdaGradFunctor(float learning_rate, float eps,
                                   std::pair<uint64_t, uint64_t> seeds)
      : AdaGradFunctor<scalar_t, use_atomics>(learning_rate, eps),
        seeds_(seeds) {}
  std::pair<uint64_t, uint64_t> seeds_;

  DEVICE_INLINE curandStatePhilox4_32_10_t init_update(int thread_id) {
    curandStatePhilox4_32_10_t state;
    curand_init(seeds_.first, thread_id, seeds_.second, &state);
    return state;
  }

  DEVICE_INLINE void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                Vec4T<scalar_t> grad, curandStatePhilox4_32_10_t &state,
                typename AdaGradFunctor<scalar_t, use_atomics>::State
                    &sample_weight_state) {
    Vec4T<scalar_t> weight_new(weight);
    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    uint4 bits = curand4(&state);
    stochastic_rounding_vector(weight, weight_new, bits);
  }
};

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_adagrad_approx_kernel_1(
    const PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<scalar_t, true>, 1, RestrictPtrTraits>
        optimizer_state,
    int32_t L_max, F f) {
  extern __shared__ int32_t shmem[];
  auto *shmem_multipliers = (acc_type<scalar_t, true> *)(&shmem[0]);
  const int32_t B = grad_output.size(0);
  const int32_t T = grad_output.size(1);
  const int32_t D = grad_output.size(2);

  // do warp-per-D (so only need warp reduction)
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  acc_type<scalar_t, true> g_local_sum_square = 0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> g(&grad_output[b][t][0] + d * 4);
    g_local_sum_square += g.acc.x * g.acc.x + g.acc.y * g.acc.y +
                          g.acc.z * g.acc.z + g.acc.w * g.acc.w;
  }
  const acc_type<scalar_t, true> g_sum_square =
      warpReduceAllSum<acc_type<scalar_t, true>>(g_local_sum_square);

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  auto state = f.init_update(blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x);
  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    shmem_multipliers[threadIdx.y * L_max + i] = f.update_momentum(
        g_sum_square, &optimizer_state[table_offset + idx], indices_start + i);
  }
  __syncthreads();
  for (int32_t l = 0; l < L; ++l) {
    auto idx = indices[indices_start + l];
    acc_type<scalar_t, true> multiplier =
        shmem_multipliers[threadIdx.y * L_max + l];

    auto sample_weight_state = f.init_sample_weight(indices_start + l);
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> grad_out(&grad_output[b][t][0] + d * 4);
      f.update_weight(&weights[table_offset + idx][0] + d * 4, multiplier,
                      grad_out, state, sample_weight_state);
    }
    f.update_sample_weight(sample_weight_state, indices_start + l);
  }
}

c10::optional<Tensor> batched_embedding_backward_adagrad_approx_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps, int64_t L_max,
    bool stochastic_rounding, int64_t BT_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);

  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size > 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT(D % 4 == 0);
  AT_ASSERT(BT_block_size * kWarpSize <= kMaxThreads);
  const dim3 threads(kWarpSize, BT_block_size);
  const dim3 blocks((B * T) / BT_block_size);
  c10::optional<Tensor> grad_indice_weights = c10::nullopt;
  if (indice_weights) {
    grad_indice_weights = at::empty(indices.sizes(), grad_output.options());
  }

#define X(functor)                                                             \
  batched_embedding_backward_adagrad_approx_kernel_1<                          \
      scalar_t><<<blocks, threads,                                             \
                  BT_block_size * L_max * sizeof(acc_type<scalar_t, true>),    \
                  at::cuda::getCurrentCUDAStream()>>>(                         \
      grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),         \
      weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),        \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,           \
                                        RestrictPtrTraits>(),                  \
      static_cast<int32_t>(L_max), (functor))

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        if (!stochastic_rounding) {
          if (indice_weights) {
            auto f = WeightedAdaGradFunctor<scalar_t, false>(
                learning_rate, eps,
                indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                grad_indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>());
            X(f);
          } else {
            auto f = AdaGradFunctor<scalar_t, false>(learning_rate, eps);
            X(f);
          }
        } else {
          std::pair<uint64_t, uint64_t> rng_engine_inputs;
          {
            auto gen = at::cuda::detail::getDefaultCUDAGenerator();
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs = gen->philox_engine_inputs(
                L_max * ((D + kWarpSize - 1) / kWarpSize));
          }
          if (indice_weights) {
            auto f = StochasticRoundingWeightedAdaGradFunctor<scalar_t, false>(
                learning_rate, eps,
                indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                grad_indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                rng_engine_inputs);
            X(f);
          } else {
            auto f = StochasticRoundingAdaGradFunctor<scalar_t, false>(
                learning_rate, eps, rng_engine_inputs);
            X(f);
          }
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());

#undef X

  return grad_indice_weights;
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
    AT_CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocMapped));
    // AT_CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined |
    //                                               cudaHostAllocMapped));

    void *dev_ptr;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, ptr, 0));
    return {dev_ptr,
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

template <typename scalar_t, bool shared_indices, typename F>
__global__ void batched_embedding_forward_kernel_mixed_D_1(
    // [\sum_t E_t x D_t]
    const PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
        table_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        dim_offsets, // [T+1]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices, // [N = \sum_{b,t} L_{b,t} total indices, i.e. flattened
                 // [B][T][L]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        offsets, // [B x T + 1]
    // offsets = cumsum([0] + lengths.contiguous()), where lengths L is [B][T].
    PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits>
        output, // [B][\sum_t D_t],
    int32_t L_max,
    F f) {

  extern __shared__ int32_t shmem_indices[];
  const int32_t T = table_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t t = b_t / B;
  int32_t b = b_t % B;
  const int32_t dim_start = dim_offsets[t];
  const int32_t dim_end = dim_offsets[t + 1];
  const int32_t D = dim_end - dim_start;

  const int64_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  if (shared_indices) {
    int32_t shmem_offset = threadIdx.y * L_max;
    for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[shmem_offset + i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = shmem_indices[shmem_offset + l];
        Vec4T<scalar_t> weight((&weights[0]) + table_offset + idx * D + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][0]) + dim_start + d * 4);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = __ldg(&indices[indices_start + l]);
        Vec4T<scalar_t> weight((&weights[0]) + table_offset + idx * D + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][0]) + dim_start + d * 4);
    }
  }
}

Tensor batched_embedding_forward_mixed_D_cuda(
    Tensor weights, Tensor table_offsets, Tensor dim_offsets, int64_t total_D,
    Tensor indices, Tensor offsets, c10::optional<Tensor> indice_weights,
    int64_t L_max, int64_t BT_block_size, bool shmem) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0) - 1;
  AT_ASSERT(T > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size != 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT((B * T) % BT_block_size == 0);
  const dim3 threads(kMaxThreads / BT_block_size, BT_block_size);
  const dim3 blocks((B * T) / BT_block_size);

#define X(shmem_param, shmem_size, functor)                                    \
  batched_embedding_forward_kernel_mixed_D_1<scalar_t, (shmem_param)><<<       \
      blocks, threads, (shmem_size), at::cuda::getCurrentCUDAStream()>>>(      \
      weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),        \
      dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),          \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      output.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),              \
      static_cast<int32_t>(L_max), (functor))

  auto output = empty({B, total_D}, weights.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_forward_kernel", ([&] {
        if (shmem) {
          if (indice_weights) {
            X(true, BT_block_size * L_max * sizeof(int32_t),
              WeightedForward<scalar_t>(
                  indice_weights
                      ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>()));
          } else {
            X(true, BT_block_size * L_max * sizeof(int32_t),
              UnweightedForward<scalar_t>());
          }
        } else {
          if (indice_weights) {
            X(false, 0,
              WeightedForward<scalar_t>(
                  indice_weights
                      ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>()));
          } else {
            X(false, 0, UnweightedForward<scalar_t>());
          }
        }
      }));

#undef X
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_adagrad_approx_kernel_mixed_D_1(
    const PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_dim_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        dim_offsets, // [T+1]

    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<scalar_t, true>, 1, RestrictPtrTraits>
        optimizer_state,
    int32_t L_max, F f) {
  extern __shared__ int32_t shmem[];
  auto *shmem_multipliers = (acc_type<scalar_t, true> *)(&shmem[0]);
  const int32_t T = table_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;

  // do warp-per-D (so only need warp reduction)
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  const int32_t dim_start = dim_offsets[t];
  const int32_t dim_end = dim_offsets[t + 1];
  const int32_t D = dim_end - dim_start;

  acc_type<scalar_t, true> g_local_sum_square = 0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> g(&grad_output[b][0] + dim_start + d * 4);
    g_local_sum_square += g.acc.x * g.acc.x + g.acc.y * g.acc.y +
                          g.acc.z * g.acc.z + g.acc.w * g.acc.w;
  }
  const acc_type<scalar_t, true> g_sum_square =
      warpReduceAllSum<acc_type<scalar_t, true>>(g_local_sum_square);

  const int32_t table_dim_offset = table_dim_offsets[t];

  const int64_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  auto state = f.init_update(blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x);
  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    shmem_multipliers[threadIdx.y * L_max + i] = f.update_momentum(
        g_sum_square, &optimizer_state[table_dim_offset + idx],
        indices_start + i);
  }
  __syncthreads();
  for (int32_t l = 0; l < L; ++l) {
    auto idx = indices[indices_start + l];
    acc_type<scalar_t, true> multiplier =
        shmem_multipliers[threadIdx.y * L_max + l];

    auto sample_weight_state = f.init_sample_weight(indices_start + l);
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> grad_out(&grad_output[b][0] + dim_start + d * 4);
      f.update_weight((&weights[0]) + table_offset + idx * D + d * 4,
                      multiplier, grad_out, state, sample_weight_state);
    }
    f.update_sample_weight(sample_weight_state, indices_start + l);
  }
}

c10::optional<Tensor> batched_embedding_backward_adagrad_approx_mixed_D_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets,
    Tensor table_dim_offsets, Tensor dim_offsets, int64_t total_D,
    Tensor indices, Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps, int64_t L_max,
    bool stochastic_rounding, int64_t BT_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0) - 1;
  AT_ASSERT(T > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size > 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT(BT_block_size * kWarpSize <= kMaxThreads);
  const dim3 threads(kWarpSize, BT_block_size);
  const dim3 blocks((B * T) / BT_block_size);
  c10::optional<Tensor> grad_indice_weights = c10::nullopt;
  if (indice_weights) {
    grad_indice_weights =
        at::empty(indice_weights->sizes(), grad_output.options());
  }

#define X(functor)                                                             \
  batched_embedding_backward_adagrad_approx_kernel_mixed_D_1<                  \
      scalar_t><<<blocks, threads,                                             \
                  BT_block_size * L_max * sizeof(acc_type<scalar_t, true>),    \
                  at::cuda::getCurrentCUDAStream()>>>(                         \
      grad_output.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),         \
      weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),        \
      table_dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),    \
      dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),          \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,           \
                                        RestrictPtrTraits>(),                  \
      static_cast<int32_t>(L_max), (functor))

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        if (!stochastic_rounding) {
          if (indice_weights) {
            auto f = WeightedAdaGradFunctor<scalar_t, false>(
                learning_rate, eps,
                indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                grad_indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>());
            X(f);
          } else {
            auto f = AdaGradFunctor<scalar_t, false>(learning_rate, eps);
            X(f);
          }
        } else {
          std::pair<uint64_t, uint64_t> rng_engine_inputs;
          {
            auto gen = at::cuda::detail::getDefaultCUDAGenerator();
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs = gen->philox_engine_inputs(
                L_max * ((total_D + kWarpSize - 1) / kWarpSize));
          }
          if (indice_weights) {
            auto f = StochasticRoundingWeightedAdaGradFunctor<scalar_t, false>(
                learning_rate, eps,
                indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                grad_indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                rng_engine_inputs);
            X(f);
          } else {
            auto f = StochasticRoundingAdaGradFunctor<scalar_t, false>(
                learning_rate, eps, rng_engine_inputs);
            X(f);
          }
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());

#undef X

  return grad_indice_weights;
}

__global__ void construct_offsets_kernel(
    const PackedTensorAccessor32<int32_t, 2, RestrictPtrTraits>
        batch_offsets_per_table,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        total_indices_per_table,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> output) {
  const int32_t T = batch_offsets_per_table.size(0);
  const int32_t B = batch_offsets_per_table.size(1);

  // do warp-per-D (so only need warp reduction)
  int32_t b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b = b_t % B;
  int32_t t = b_t / B;
  if (t < T) {
    int32_t upper = 0;
    if (b != B - 1) {
      upper = batch_offsets_per_table[t][b + 1];
    } else {
      upper = total_indices_per_table[t];
    }
    int32_t lower = batch_offsets_per_table[t][b];
    output[1 + t * B + b] = upper - lower;
  }
}

Tensor construct_offsets(Tensor batch_offsets_per_table, // [T][B]
                         Tensor total_indices_per_table  // [T]
                         ) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(batch_offsets_per_table.get_device());

  const auto T = batch_offsets_per_table.size(0);
  const auto B = batch_offsets_per_table.size(1);
  const dim3 threads(kMaxThreads);
  const dim3 blocks((B * T + 1 + kMaxThreads - 1) / kMaxThreads);

  Tensor output =
      at::zeros({1 + B * T}, batch_offsets_per_table.options().dtype(kInt));

  construct_offsets_kernel<<<blocks, threads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      batch_offsets_per_table
          .packed_accessor32<int32_t, 2, RestrictPtrTraits>(),
      total_indices_per_table
          .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      output.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

struct IndexInfo {
  int32_t b_t;
  int32_t indices_idx;
};

__global__ void linear_index_weight_offsets_mixed_D_kernel(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_dim_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    IndexInfo *__restrict__ infos,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> linear_indices) {
  const int32_t T = table_dim_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  const int32_t table_dim_offset = table_dim_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    IndexInfo info;
    info.b_t = b_t;
    info.indices_idx = indices_start + i;
    infos[indices_start + i] = info;
    linear_indices[indices_start + i] = table_dim_offset + idx;
  }
}

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_adagrad_exact_kernel_mixed_D_1(
    const PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_dim_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        dim_offsets, // [T+1]

    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<scalar_t, true>, 1, RestrictPtrTraits>
        optimizer_state,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices,
    const IndexInfo *__restrict__ infos,

    int32_t indices_numel, F f) {
  const int32_t T = table_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;

  int32_t sorted_linear_indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (sorted_linear_indice_id >= indices_numel) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  // check if this warp is responsible for this whole segment.
  bool segment_start = (sorted_linear_indice_id == 0 ||
                        sorted_linear_indices[sorted_linear_indice_id - 1] !=
                            sorted_linear_indices[sorted_linear_indice_id]);

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  int32_t linear_index = sorted_linear_indices[sorted_linear_indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (sorted_linear_indice_id + SL < indices_numel &&
         sorted_linear_indices[sorted_linear_indice_id + SL] == linear_index) {
    SL += 1;
  }
  // now, each segment corresponds to exactly one table `t` and row in that
  // table (`idx`). Thus, we can hoist out some of the book-keeping.
  auto info_0 = infos[sorted_linear_indice_id];
  const int32_t t = info_0.b_t / B;
  const int64_t table_offset = table_offsets[t];
  const int32_t table_dim_offset = table_dim_offsets[t];
  const int32_t idx = linear_index - table_dim_offset;
  const int32_t dim_start = dim_offsets[t];
  const int32_t dim_end = dim_offsets[t + 1];
  const int32_t D = dim_end - dim_start;

  acc_type<scalar_t, true> g_local_sum_square = 0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      int32_t b = infos[sorted_linear_indice_id + sl].b_t % B;
      // TODO: apply per-sample weighting.
      Vec4T<scalar_t> grad_out(&grad_output[b][0] + dim_start + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    g_local_sum_square += sum.acc.x * sum.acc.x + sum.acc.y * sum.acc.y +
                          sum.acc.z * sum.acc.z + sum.acc.w * sum.acc.w;
  }

  const acc_type<scalar_t, true> g_sum_square =
      warpReduceAllSum<acc_type<scalar_t, true>>(g_local_sum_square);

  auto state = f.init_update(blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x);

  acc_type<scalar_t, true> multiplier = f.update_momentum(
      g_sum_square, &optimizer_state[table_dim_offset + idx], -1);

  auto sample_weight_state = f.init_sample_weight(-1);
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      auto b = infos[sorted_linear_indice_id + sl].b_t % B;
      Vec4T<scalar_t> grad_out(&grad_output[b][0] + dim_start + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    f.update_weight((&weights[0]) + table_offset + idx * D + d * 4, multiplier,
                    sum, state, sample_weight_state);
  }
  f.update_sample_weight(sample_weight_state, -1);
}

c10::optional<Tensor> batched_embedding_backward_adagrad_exact_mixed_D_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets,
    Tensor table_dim_offsets, Tensor dim_offsets, int64_t total_D,
    Tensor indices, Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps,
    bool stochastic_rounding, int64_t BT_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0) - 1;
  AT_ASSERT(T > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size > 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT(BT_block_size * kWarpSize <= kMaxThreads);

  const dim3 threads(kWarpSize, BT_block_size);

  c10::optional<Tensor> grad_indice_weights = c10::nullopt;
  if (indice_weights) {
    grad_indice_weights =
        at::empty(indice_weights->sizes(), grad_output.options());
  }

  auto infos =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto infos_sorted =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto linear_indices = at::empty_like(indices);
  auto linear_indices_sorted = at::empty_like(indices);

  linear_index_weight_offsets_mixed_D_kernel<<<
      dim3((B * T) / BT_block_size), threads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      table_dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      (IndexInfo *)infos.data_ptr(),
      linear_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());

  size_t temp_storage_bytes = 0;
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(optimizer_state.numel()) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  auto temp_storage = at::empty({static_cast<int64_t>(temp_storage_bytes)},
                                indices.options().dtype(kByte));

  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp_storage.data_ptr(), temp_storage_bytes,
      (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(optimizer_state.numel()) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  const dim3 blocks((linear_indices.numel() + BT_block_size - 1) /
                    BT_block_size);

#define X(functor)                                                             \
  batched_embedding_backward_adagrad_exact_kernel_mixed_D_1<                   \
      scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(     \
      grad_output.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),         \
      weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),        \
      table_dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),    \
      dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),          \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,           \
                                        RestrictPtrTraits>(),                  \
      linear_indices_sorted                                                    \
          .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),                 \
      (const IndexInfo *)infos_sorted.data_ptr(),                              \
      static_cast<int32_t>(linear_indices_sorted.numel()), (functor))

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        if (!stochastic_rounding) {
          if (indice_weights) {
            AT_ASSERT(false);
          } else {
            auto f = AdaGradFunctor<scalar_t, false>(learning_rate, eps);
            X(f);
          }
        } else {
          std::pair<uint64_t, uint64_t> rng_engine_inputs;
          {
            auto gen = at::cuda::detail::getDefaultCUDAGenerator();
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs = gen->philox_engine_inputs(
                ((total_D + kWarpSize - 1) / kWarpSize));
          }
          if (indice_weights) {
            AT_ASSERT(false);
          } else {
            auto f = StochasticRoundingAdaGradFunctor<scalar_t, false>(
                learning_rate, eps, rng_engine_inputs);
            X(f);
          }
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());

#undef X

  return grad_indice_weights;
}

__global__ void linear_index_weight_offsets_kernel(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    IndexInfo *__restrict__ infos,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> linear_indices) {
  const int32_t T = table_offsets.size(0);
  const int32_t B = (offsets.size(0) - 1) / T;

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    IndexInfo info;
    info.b_t = b_t;
    info.indices_idx = indices_start + i;
    infos[indices_start + i] = info;
    linear_indices[indices_start + i] = table_offset + idx;
  }
}

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_adagrad_exact_kernel_1(
    const PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<scalar_t, true>, 1, RestrictPtrTraits>
        optimizer_state,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices,
    const IndexInfo *__restrict__ infos, int32_t indices_numel, F f) {
  const int32_t B = grad_output.size(0);
  const int32_t T = grad_output.size(1);
  const int32_t D = grad_output.size(2);

  int32_t sorted_linear_indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (sorted_linear_indice_id >= indices_numel) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  // check if this warp is responsible for this whole segment.
  bool segment_start = (sorted_linear_indice_id == 0 ||
                        sorted_linear_indices[sorted_linear_indice_id - 1] !=
                            sorted_linear_indices[sorted_linear_indice_id]);

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  int32_t linear_index = sorted_linear_indices[sorted_linear_indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (sorted_linear_indice_id + SL < indices_numel &&
         sorted_linear_indices[sorted_linear_indice_id + SL] == linear_index) {
    SL += 1;
  }
  // now, each segment corresponds to exactly one table `t` and row in that
  // table (`idx`). Thus, we can hoist out some of the book-keeping.
  auto info_0 = infos[sorted_linear_indice_id];
  const int32_t t = info_0.b_t / B;
  const int64_t table_offset = table_offsets[t];
  const int32_t idx = linear_index - table_offset;

  acc_type<scalar_t, true> g_local_sum_square = 0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      int32_t b = infos[sorted_linear_indice_id + sl].b_t % B;
      // TODO: apply per-sample weighting.
      Vec4T<scalar_t> grad_out(&grad_output[b][t][0] + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    g_local_sum_square += sum.acc.x * sum.acc.x + sum.acc.y * sum.acc.y +
                          sum.acc.z * sum.acc.z + sum.acc.w * sum.acc.w;
  }

  const acc_type<scalar_t, true> g_sum_square =
      warpReduceAllSum<acc_type<scalar_t, true>>(g_local_sum_square);

  auto state = f.init_update(blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x);

  acc_type<scalar_t, true> multiplier =
      f.update_momentum(g_sum_square, &optimizer_state[table_offset + idx], -1);

  auto sample_weight_state = f.init_sample_weight(-1);
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      auto b = infos[sorted_linear_indice_id + sl].b_t % B;
      Vec4T<scalar_t> grad_out(&grad_output[b][t][0] + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    f.update_weight(&weights[table_offset + idx][0] + d * 4, multiplier, sum,
                    state, sample_weight_state);
  }
  f.update_sample_weight(sample_weight_state, -1);
}

c10::optional<Tensor> batched_embedding_backward_adagrad_exact_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps,
    bool stochastic_rounding, int64_t BT_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);

  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size > 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT(D % 4 == 0);
  AT_ASSERT(BT_block_size * kWarpSize <= kMaxThreads);
  const dim3 threads(kWarpSize, BT_block_size);

  c10::optional<Tensor> grad_indice_weights = c10::nullopt;
  if (indice_weights) {
    grad_indice_weights = at::empty(indices.sizes(), grad_output.options());
  }

  auto infos =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto infos_sorted =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto linear_indices = at::empty_like(indices);
  auto linear_indices_sorted = at::empty_like(indices);

  linear_index_weight_offsets_kernel<<<dim3((B * T) / BT_block_size), threads,
                                       0, at::cuda::getCurrentCUDAStream()>>>(
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      (IndexInfo *)infos.data_ptr(),
      linear_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());

  size_t temp_storage_bytes = 0;
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(optimizer_state.numel()) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  auto temp_storage = at::empty({static_cast<int64_t>(temp_storage_bytes)},
                                indices.options().dtype(kByte));

  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp_storage.data_ptr(), temp_storage_bytes,
      (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(optimizer_state.numel()) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  const dim3 blocks((linear_indices.numel() + BT_block_size - 1) /
                    BT_block_size);

#define X(functor)                                                             \
  batched_embedding_backward_adagrad_exact_kernel_1<                           \
      scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(     \
      grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),         \
      weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),        \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,           \
                                        RestrictPtrTraits>(),                  \
      linear_indices_sorted                                                    \
          .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),                 \
      (const IndexInfo *)infos_sorted.data_ptr(),                              \
      static_cast<int32_t>(linear_indices_sorted.numel()), (functor))

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        if (!stochastic_rounding) {
          if (indice_weights) {
            AT_ASSERT(false);
          } else {
            auto f = AdaGradFunctor<scalar_t, false>(learning_rate, eps);
            X(f);
          }
        } else {
          std::pair<uint64_t, uint64_t> rng_engine_inputs;
          {
            auto gen = at::cuda::detail::getDefaultCUDAGenerator();
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs =
                gen->philox_engine_inputs(((D + kWarpSize - 1) / kWarpSize));
          }
          if (indice_weights) {
            AT_ASSERT(false);
          } else {
            auto f = StochasticRoundingAdaGradFunctor<scalar_t, false>(
                learning_rate, eps, rng_engine_inputs);
            X(f);
          }
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());

#undef X

  return grad_indice_weights;
}

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_sgd_exact_kernel_1(
    const PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices,
    const IndexInfo *__restrict__ infos, int32_t indices_numel, F f) {
  const int32_t B = grad_output.size(0);
  const int32_t T = grad_output.size(1);
  const int32_t D = grad_output.size(2);

  int32_t sorted_linear_indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (sorted_linear_indice_id >= indices_numel) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  // check if this warp is responsible for this whole segment.
  bool segment_start = (sorted_linear_indice_id == 0 ||
                        sorted_linear_indices[sorted_linear_indice_id - 1] !=
                            sorted_linear_indices[sorted_linear_indice_id]);

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  int32_t linear_index = sorted_linear_indices[sorted_linear_indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (sorted_linear_indice_id + SL < indices_numel &&
         sorted_linear_indices[sorted_linear_indice_id + SL] == linear_index) {
    SL += 1;
  }
  // now, each segment corresponds to exactly one table `t` and row in that
  // table (`idx`). Thus, we can hoist out some of the book-keeping.
  auto info_0 = infos[sorted_linear_indice_id];
  const int32_t t = info_0.b_t / B;
  const int64_t table_offset = table_offsets[t];
  const int32_t idx = linear_index - table_offset;

  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      int32_t b = infos[sorted_linear_indice_id + sl].b_t % B;
      // TODO: apply per-sample weighting.
      Vec4T<scalar_t> grad_out(&grad_output[b][t][0] + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    f(&weights[table_offset + idx][0] + d * 4, sum);
  }
}

c10::optional<Tensor> batched_embedding_backward_sgd_exact_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, c10::optional<Tensor> indice_weights, float learning_rate,
    int64_t BT_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);

  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size > 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT(D % 4 == 0);
  AT_ASSERT(BT_block_size * kWarpSize <= kMaxThreads);
  const dim3 threads(kWarpSize, BT_block_size);

  c10::optional<Tensor> grad_indice_weights = c10::nullopt;
  if (indice_weights) {
    grad_indice_weights = at::empty(indices.sizes(), grad_output.options());
  }

  auto infos =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto infos_sorted =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto linear_indices = at::empty_like(indices);
  auto linear_indices_sorted = at::empty_like(indices);

  linear_index_weight_offsets_kernel<<<dim3((B * T) / BT_block_size), threads,
                                       0, at::cuda::getCurrentCUDAStream()>>>(
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      (IndexInfo *)infos.data_ptr(),
      linear_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());

  size_t temp_storage_bytes = 0;
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(weights.size(0)) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  auto temp_storage = at::empty({static_cast<int64_t>(temp_storage_bytes)},
                                indices.options().dtype(kByte));

  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp_storage.data_ptr(), temp_storage_bytes,
      (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(weights.size(0)) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  const dim3 blocks((linear_indices.numel() + BT_block_size - 1) /
                    BT_block_size);

#define X(functor)                                                             \
  batched_embedding_backward_sgd_exact_kernel_1<                               \
      scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(     \
      grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),         \
      weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),        \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      linear_indices_sorted                                                    \
          .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),                 \
      (const IndexInfo *)infos_sorted.data_ptr(),                              \
      static_cast<int32_t>(linear_indices_sorted.numel()), (functor))

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_sgd_exact_kernel", ([&] {
        AT_ASSERT(!indice_weights);
        auto f = SGDFunctor<scalar_t, false>(learning_rate);
        X(f);
      }));
  AT_CUDA_CHECK(cudaGetLastError());

#undef X

  return grad_indice_weights;
}

__global__ void restack_offsets_kernel(
    const PackedTensorAccessor32<int32_t, 3, RestrictPtrTraits>
        stack_offset_output,
    const PackedTensorAccessor32<int32_t, 2, RestrictPtrTraits>
        indice_lengths_output,
    PackedTensorAccessor32<int32_t, 2, RestrictPtrTraits> output) {
  const auto W = stack_offset_output.size(0);
  const auto T = stack_offset_output.size(1);
  const auto B = stack_offset_output.size(2);

  int32_t w_b = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t t = blockIdx.y;
  int32_t b = w_b % B;
  int32_t w = w_b / B;
  if (w < W) {
    int32_t upper = 0;
    if (b != B - 1) {
      upper = stack_offset_output[w][t][b + 1];
    } else {
      upper = indice_lengths_output[w][t];
    }
    int32_t lower = stack_offset_output[w][t][b];
    output[t][1 + w * B + b] = upper - lower;
  }
}

Tensor restack_offsets(Tensor stack_offset_output,  // [W][T][B]
                       Tensor indice_lengths_output // [W][T]
                       ) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(stack_offset_output.get_device());

  const auto W = stack_offset_output.size(0);
  const auto T = stack_offset_output.size(1);
  const auto B = stack_offset_output.size(2);

  const dim3 threads(kMaxThreads);
  const dim3 blocks((W * B + kMaxThreads - 1) / kMaxThreads, T);

  Tensor output = at::zeros({T, 1 + W * B}, stack_offset_output.options());

  restack_offsets_kernel<<<blocks, threads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      stack_offset_output.packed_accessor32<int32_t, 3, RestrictPtrTraits>(),
      indice_lengths_output.packed_accessor32<int32_t, 2, RestrictPtrTraits>(),
      output.packed_accessor32<int32_t, 2, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

inline int32_t div_round_up(int32_t a, int32_t b) { return (a + b - 1) / b; }

// 32 bit Murmur3 hash
DEVICE_INLINE uint32_t Murmur3(uint32_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k;
}
constexpr int32_t kCuckooMissing = -1;

__global__ void
open_insert(const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
            PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> cuckoo_keys) {
  unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= indices.size(0)) {
    return;
  }
  int32_t key = indices[n];
  uint32_t slot = Murmur3(key) % cuckoo_keys.size(0);
  while (true) {
    int32_t prev = atomicCAS(&cuckoo_keys[slot], kCuckooMissing, key);
    if (prev == kCuckooMissing || prev == key) {
      return;
    }
    slot = (slot + 1) % cuckoo_keys.size(0);
  }
}

__global__ void cuckoo_unique_keys(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> cuckoo_keys,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> unique_indices,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        unique_indices_count) {
  unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= cuckoo_keys.size(0)) {
    return;
  }

  int32_t key = cuckoo_keys[n];
  if (key != kCuckooMissing) {
    // CUDA warp aggregation should make this not insanely slow
    unique_indices[atomicAdd(&unique_indices_count[0], 1)] = key;
  }
}

static inline uint32_t next_pow2(uint32_t x) {
  x -= 1;
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return x + 1;
}

constexpr float kLoadFactor = 1.25;

std::pair<Tensor, Tensor> lxu_cache_unique_indices_cuda(Tensor indices) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());

  auto cuckoo_keys =
      at::empty({next_pow2(indices.size(0) * kLoadFactor)}, indices.options())
          .fill_(kCuckooMissing);
  auto unique_indices = at::empty_like(indices);
  open_insert<<<dim3(div_round_up(indices.numel(), kMaxThreads)), kMaxThreads,
                0, at::cuda::getCurrentCUDAStream()>>>(
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      cuckoo_keys.packed_accessor32<int32_t, 1, RestrictPtrTraits>());

  auto unique_indices_count = at::zeros({1}, indices.options().dtype(kInt));

  cuckoo_unique_keys<<<dim3(div_round_up(cuckoo_keys.numel(), kMaxThreads)),
                       kMaxThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      cuckoo_keys.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      unique_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      unique_indices_count.packed_accessor32<int32_t, 1, RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_pair(unique_indices, unique_indices_count);
}

constexpr int32_t kCacheSlotTime = 0;
constexpr int32_t kCacheSlotFreq = 1;
constexpr int32_t kCacheSlotCurrentIndex = 2;
// constexpr int32_t kCacheSlotBloomFilter = 3;

// TODO: decouple these?
constexpr int32_t kAssoc = kWarpSize;

DEVICE_INLINE int32_t hash_function(int32_t x) { return x; }

DEVICE_INLINE uint32_t hash1(uint32_t a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

DEVICE_INLINE uint32_t hash2(uint32_t key) {
  key += ~(key << 15);
  key ^= (key >> 10);
  key += (key << 3);
  key ^= (key >> 6);
  key += ~(key << 11);
  key ^= (key >> 16);
  return key;
}

// simple blocked bloom filter
constexpr int32_t kQueries = 4;

DEVICE_INLINE void bloom_filter_insert(int32_t key, uint32_t storage_blocks,
                                       uint64_t *storage) {
  const uint32_t h0 = hash1(key);
  const uint32_t h1 = hash2(key);

  const auto block_idx = (h0 + 0 * h1) % storage_blocks;
  uint64_t block = 0;

#pragma unroll
  for (auto i = 0; i < kQueries; ++i) {
    const uint32_t bit = uint32_t(h0 + (i + 1) * h1) % 64;
    block |= (uint64_t(1) << bit);
  }
  gpuAtomicOr(&storage[block_idx], block);
}

DEVICE_INLINE bool bloom_filter_query(int32_t key, uint32_t storage_blocks,
                                      const uint64_t *storage) {
  const uint32_t h0 = hash1(key);
  const uint32_t h1 = hash2(key);

  const auto block_idx = (h0 + 0 * h1) % storage_blocks;
  const uint64_t block = storage[block_idx];

#pragma unroll
  for (auto i = 0; i < kQueries; ++i) {
    const uint32_t bit = uint32_t(h0 + (i + 1) * h1) % 64;
    if ((block & (uint64_t(1) << bit)) == 0) {
      return false;
    }
  }
  return true;
}

__global__ void batched_embedding_forward_kernel_lxu_cache_find_uncached(
    const PackedTensorAccessor32<int32_t, 3, RestrictPtrTraits> lxu_cache_state,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        unique_indices, // [N = \sum_{b} L_{b} total indices, i.e. flattened
                        // [B][L]
    int32_t *N_unique,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> cache_sets) {
  const int32_t N = unique_indices.size(0);
  const int32_t C = lxu_cache_state.size(1);

  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }
  if (n >= *N_unique) {
    if (threadIdx.x == 0) {
      cache_sets[n] = C; // invalid index! used as sentinel
    }
    return;
  }
  // insert into the bloom filter.
  auto idx = unique_indices[n];
  int32_t cache_set = hash_function(idx) % C;
  auto slot = threadIdx.x;
  bool found = lxu_cache_state[kCacheSlotCurrentIndex][cache_set][slot] == idx;
  // pre-condition - only one thread can satisfy this check.
  if (__any_sync(0xFFFFFFFF, found)) {
    if (threadIdx.x == 0) {
      cache_sets[n] = C; // invalid index! used as sentinel
    }
  } else {
    if (threadIdx.x == 0) {
      cache_sets[n] = cache_set;
    }
  }
}

// Warp bitonic K/V sorting code from @jhj
template <typename T> struct Comparator {
  __device__ static inline bool lt(T a, T b) { return a < b; }
  __device__ static inline bool gt(T a, T b) { return a > b; }
};

template <typename T> inline __device__ void assign(bool assign, T &x, T y) {
  x = assign ? y : x;
}

template <typename K, typename V, int L, bool Dir, typename Comp,
          bool IsBitonic>
inline __device__ void warpBitonicMergeLE16(K &k, V &v) {
  // static_assert(utils::isPowerOf2(L), "L must be a power-of-2");
  // static_assert(L <= kWarpSize / 2, "merge list size must be <= 16");

  int laneId = threadIdx.x;

  if (!IsBitonic) {
    // Reverse the first comparison stage.
    // For example, merging a list of size 8 has the exchanges:
    // 0 <-> 15, 1 <-> 14, ...
    K otherK = shfl_xor(k, 2 * L - 1);
    V otherV = shfl_xor(v, 2 * L - 1);

    // Whether we are the lesser thread in the exchange
    bool small = !(laneId & L);

    if (Dir) {
      // See the comment above how performing both of these
      // comparisons in the warp seems to win out over the
      // alternatives in practice
      bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
      assign(s, k, otherK);
      assign(s, v, otherV);

    } else {
      bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
      assign(s, k, otherK);
      assign(s, v, otherV);
    }
  }

#pragma unroll
  for (int stride = IsBitonic ? L : L / 2; stride > 0; stride /= 2) {
    K otherK = shfl_xor(k, stride);
    V otherV = shfl_xor(v, stride);

    // Whether we are the lesser thread in the exchange
    bool small = !(laneId & stride);

    if (Dir) {
      bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
      assign(s, k, otherK);
      assign(s, v, otherV);

    } else {
      bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
      assign(s, k, otherK);
      assign(s, v, otherV);
    }
  }
}

template <typename K, typename V, bool Dir, typename Comp> struct BitonicSort {
  static inline __device__ void sort(K k[1], V v[1]) {
    static_assert(kWarpSize == 32, "unexpected warp size");
    warpBitonicMergeLE16<K, V, 1, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16<K, V, 2, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16<K, V, 4, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16<K, V, 8, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16<K, V, 16, Dir, Comp, false>(k[0], v[0]);
  }
};

template <typename scalar_t>
__global__ void batched_embedding_forward_kernel_lxu_cache_insert_uncached(
    PackedTensorAccessor32<int32_t, 3, RestrictPtrTraits> lxu_cache_state,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> lxu_cache_weights,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_cache_sets, // [N = \sum_{b} L_{b} total indices, i.e.
                           // flattened
                           // [B][L]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        cache_set_sorted_indices, // [N = \sum_{b} L_{b} total indices, i.e.
                                  // flattened [B][L]
    int32_t t) {
  const int32_t C = lxu_cache_state.size(1);

  const int32_t N = sorted_cache_sets.size(0);
  const int32_t D = weights.size(1);
  assert(lxu_cache_weights.size(0) == C * kAssoc);
  assert(lxu_cache_weights.size(1) == D);
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (n == 0 || sorted_cache_sets[n - 1] != sorted_cache_sets[n]);

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so
    // we can just exit this warp entirely.
    return;
  }
  int32_t cache_set = sorted_cache_sets[n];
  if (cache_set >= C) {
    // ignore the already-existing elements
    return;
  }

  int32_t SL = 1;
  while (n + SL < N && sorted_cache_sets[n + SL] == cache_set) {
    SL += 1;
  }

  // now, we need to insert the (unique!) values in indices[n:n + SL] into
  // our slots.
  const int32_t slot = threadIdx.x;
  const int32_t slot_time = lxu_cache_state[kCacheSlotTime][cache_set][slot];
  const int32_t slot_freq = lxu_cache_state[kCacheSlotFreq][cache_set][slot];
  // TODO: merge LRU + LFU? Just use LFU? Configurabe?
  int32_t costs[1] = {slot_time};
  int32_t slots[1] = {slot};
  // int32_t slots[1] = {slot_freq};

  BitonicSort<int32_t, int32_t, 1, Comparator<int32_t>>::sort(costs, slots);
  const int32_t sorted_slot = slots[0];

#pragma unroll
  for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
    const int32_t insert_slot = __shfl_sync(0xFFFFFFFF, sorted_slot, l);
    const int32_t insert_slot_time =
        __shfl_sync(0xFFFFFFFF, slot_time, insert_slot);
    const int32_t insert_slot_freq =
        __shfl_sync(0xFFFFFFFF, slot_freq, insert_slot);
    // minor optimization - check insert_slot_freq to determine if any updates
    // were applied to this slot while it was cached.
    if (insert_slot_time > 0 && insert_slot_freq > 0) {
      // evict from slot to backing storage
      const int32_t current_idx =
          lxu_cache_state[kCacheSlotCurrentIndex][cache_set][insert_slot];
      for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
        Vec4T<scalar_t>::copy(
            (&lxu_cache_weights[cache_set * kAssoc + insert_slot][0]) + d * 4,
            (&weights[current_idx][0]) + d * 4);
      }
    }
    const int32_t insert_idx = cache_set_sorted_indices[n + l];
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t>::copy(
          (&weights[insert_idx][0]) + d * 4,
          (&lxu_cache_weights[cache_set * kAssoc + insert_slot][0]) + d * 4);
    }
    if (threadIdx.x == 0) {
      lxu_cache_state[kCacheSlotCurrentIndex][cache_set][insert_slot] =
          insert_idx;
    }
  }
  if (slot < min(SL, kWarpSize)) {
    lxu_cache_state[kCacheSlotTime][cache_set][sorted_slot] = t;
    lxu_cache_state[kCacheSlotFreq][cache_set][sorted_slot] = 0;
  }
}

constexpr int32_t kCacheLocationMissing = std::numeric_limits<int32_t>::max();

template <int32_t ILP>
__global__ void batched_embedding_forward_kernel_lxu_cache_lookup_cached(
    PackedTensorAccessor32<int32_t, 3, RestrictPtrTraits> lxu_cache_state,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices, // [N = \sum_{b} L_{b} total indices, i.e. flattened
                 // [B][L]
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        lxu_cache_locations,
    int32_t t) {
  const int32_t C = lxu_cache_state.size(1);
  const int32_t N = indices.size(0);

  int32_t n_ilp = blockIdx.x * blockDim.y + threadIdx.y;
  if (n_ilp * ILP >= N) {
    return;
  }
  int32_t idx[ILP];
#pragma unroll
  for (auto ilp = 0; ilp < ILP; ++ilp) {
    idx[ilp] = indices[n_ilp * ILP + ilp];
  }
  int32_t cache_set[ILP];
#pragma unroll
  for (auto ilp = 0; ilp < ILP; ++ilp) {
    cache_set[ilp] = hash_function(idx[ilp]) % C; //[n_ilp * ILP + ilp];
  }
  auto slot = threadIdx.x;

  // int32_t cache_set = hash_function(idx) % C;
  bool found[ILP];
#pragma unroll
  for (auto ilp = 0; ilp < ILP; ++ilp) {
    found[ilp] = __ldg(&lxu_cache_state[kCacheSlotCurrentIndex][0][0] +
                       cache_set[ilp] * kAssoc + slot) == idx[ilp];
  }
// pre-condition - at most one thread will return found.
#pragma unroll
  for (auto ilp = 0; ilp < ILP; ++ilp) {
    if (found[ilp]) {
      lxu_cache_state[kCacheSlotTime][cache_set[ilp]][slot] = t;
      lxu_cache_locations[n_ilp * ILP + ilp] = cache_set[ilp] * kAssoc + slot;
      gpuAtomicAdd(&lxu_cache_state[kCacheSlotFreq][cache_set[ilp]][slot], 1);
    }
  }
#pragma unroll
  for (auto ilp = 0; ilp < ILP; ++ilp) {
    if (!__any_sync(0xFFFFFFFF, found[ilp])) {
      // should never actually be executed.
      if (threadIdx.x == 0) {
        lxu_cache_locations[n_ilp * ILP + ilp] = kCacheLocationMissing;
      }
    }
  }
}

// START: PopulateCache phase:
// Sort(ids) -> ensure idx_ids are in sorted order in the cache_loc list.
// Unique(ids) -> ensure idx_ids are unique in cache_lot list
// FindUncached(ids) -> if (i >= numUnique) { kMaxInt : cache_loc} else { if
// lookup? then kMaxInt (and set LRU counter??) else cache_set }
// Sort(FindUncached) -> sorted order of cache slots
// Launch warp per item cache_set
// Find segment length for current cache set (bail out early if not current
// segment)
// Warp Bitonic sort slots by LRU/LFU state
// for (i in lower_bound: upper_bound) {
// evict current entry in slot
// insert new slot, update LRU counter.
// }
// FINISH: PopulateCache

// START: LookupCache phase:
// Lookup(ids)
// FINISH

// START: ForwardPass phase:
//
// Lookup(ids): lookup ids in sorted list
//
//
// algorithm:

void lxu_cache_populate_cuda(Tensor weights, Tensor indices,
                             Tensor lxu_cache_state, Tensor lxu_cache_weights,
                             int64_t t, int64_t N_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());
  const int32_t N = indices.numel();
  auto unique_indices_and_count = lxu_cache_unique_indices_cuda(indices);
  auto unique_indices = unique_indices_and_count.first;
  auto unique_indices_count = unique_indices_and_count.second;
  // Step: find uncached indices
  auto cache_sets = empty_like(indices);
  {
    batched_embedding_forward_kernel_lxu_cache_find_uncached<<<
        div_round_up(N, N_block_size), dim3(kWarpSize, N_block_size), 0,
        at::cuda::getCurrentCUDAStream()>>>(
        lxu_cache_state.packed_accessor32<int32_t, 3, RestrictPtrTraits>(),
        unique_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
        (int32_t *)unique_indices_count.data_ptr(),
        cache_sets.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
    AT_CUDA_CHECK(cudaGetLastError());
  }
  // Step: sort the cache sets and ids
  auto sorted_cache_sets = empty_like(cache_sets);
  auto cache_set_sorted_unique_indices = empty_like(unique_indices);
  {
    size_t temp_storage_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, (uint32_t *)cache_sets.data_ptr(),
        (uint32_t *)sorted_cache_sets.data_ptr(),
        (int32_t *)unique_indices.data_ptr(),
        (int32_t *)cache_set_sorted_unique_indices.data_ptr(), N, 0,
        int(log2(lxu_cache_state.size(1) + 1) + 1),
        at::cuda::getCurrentCUDAStream(), false));

    auto temp_storage = at::empty({static_cast<int64_t>(temp_storage_bytes)},
                                  indices.options().dtype(kByte));
    AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp_storage.data_ptr(), temp_storage_bytes,
        (uint32_t *)cache_sets.data_ptr(),
        (uint32_t *)sorted_cache_sets.data_ptr(),
        (int32_t *)unique_indices.data_ptr(),
        (int32_t *)cache_set_sorted_unique_indices.data_ptr(), N, 0,
        int(log2(lxu_cache_state.size(1) + 1) + 1),
        at::cuda::getCurrentCUDAStream(), false));
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(),
      "batched_embedding_forward_kernel_lxu_cache_insert_uncached", ([&] {
        batched_embedding_forward_kernel_lxu_cache_insert_uncached<<<
            div_round_up(N, N_block_size), dim3(kWarpSize, N_block_size), 0,
            at::cuda::getCurrentCUDAStream()>>>(
            lxu_cache_state.packed_accessor32<int32_t, 3, RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
            sorted_cache_sets
                .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            cache_set_sorted_unique_indices
                .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            t);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
  return;
}

Tensor lxu_cache_lookup_cuda(Tensor indices, Tensor lxu_cache_state, int64_t t,
                             int64_t N_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());
  const int32_t N = indices.numel();
  auto lxu_cache_locations = empty_like(indices);
  batched_embedding_forward_kernel_lxu_cache_lookup_cached<4><<<
      div_round_up(div_round_up(N, 4) * 4, N_block_size),
      dim3(kWarpSize, N_block_size), 0, at::cuda::getCurrentCUDAStream()>>>(
      lxu_cache_state.packed_accessor32<int32_t, 3, RestrictPtrTraits>(),
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      lxu_cache_locations.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      t);
  AT_CUDA_CHECK(cudaGetLastError());
  return lxu_cache_locations;
}

enum class LXUCacheMaskAction {
  USE_BACKING_STORE,
  SKIP,
};

template <typename scalar_t, LXUCacheMaskAction Action, typename F>
__global__ void lxu_cache_forward_kernel_1(
    // [\sum_t E_t][D]
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices, // [N = \sum_{b} L_{b} total indices, i.e. flattened
                 // [B][L]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        offsets, // [B + 1]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        lxu_cache_locations, // [N]
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits>
        lxu_cache_weights, // [N]

    PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits>
        output, // [B][D],
    F f) {

  const int32_t B = output.size(0);
  const int32_t D = output.size(1);

  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }

  const int32_t indices_start = offsets[b];
  const int32_t indices_end = offsets[b + 1];
  int32_t L = indices_end - indices_start;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t l = 0; l < L; ++l) {
      auto cache_idx = __ldg(&lxu_cache_locations[indices_start + l]);
      // common case.
      if (cache_idx != kCacheLocationMissing) {
        Vec4T<scalar_t> weight((&lxu_cache_weights[cache_idx][0]) + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      } else {
        if (Action == LXUCacheMaskAction::USE_BACKING_STORE) {
          auto idx = __ldg(&indices[indices_start + l]);
          Vec4T<scalar_t> weight((&weights[idx][0]) + d * 4);
          f.accumulate(sum, weight, indices_start + l);
        }
      }
    }
    sum.store((&output[b][0]) + d * 4);
  }
}

DLDataType getDLDataType(const Tensor &t) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = t.element_size() * 8;
  switch (t.scalar_type()) {
  case ScalarType::Byte:
    dtype.code = DLDataTypeCode::kDLUInt;
    break;
  case ScalarType::Char:
    dtype.code = DLDataTypeCode::kDLInt;
    break;
  case ScalarType::Double:
    dtype.code = DLDataTypeCode::kDLFloat;
    break;
  case ScalarType::Float:
    dtype.code = DLDataTypeCode::kDLFloat;
    break;
  case ScalarType::Int:
    dtype.code = DLDataTypeCode::kDLInt;
    break;
  case ScalarType::Long:
    dtype.code = DLDataTypeCode::kDLInt;
    break;
  case ScalarType::Short:
    dtype.code = DLDataTypeCode::kDLInt;
    break;
  case ScalarType::Half:
    dtype.code = DLDataTypeCode::kDLFloat;
    break;
  case ScalarType::Bool:
    dtype.code = DLDataTypeCode::kDLUInt;
    break;
  }
  return dtype;
}

DLTensor toDLTensor(Tensor src, bool host_mapped = false) {
  DLTensor rv;
  rv.data =
      host_mapped ? src.storage().data_ptr().get_context() : src.data_ptr();
  rv.ctx.device_type = DLDeviceType::kDLCPU;
  rv.ctx.device_id = 0;
  rv.ndim = src.dim();
  rv.dtype = getDLDataType(src);
  rv.shape = const_cast<int64_t *>(src.sizes().data());
  rv.strides = const_cast<int64_t *>(src.strides().data());
  rv.byte_offset = 0;
  return rv;
}

Tensor lxu_cache_forward_cpu(Tensor weights, Tensor indices_cpu,
                             Tensor offsets_cpu,
                             c10::optional<Tensor> indice_weights_cpu,
                             Tensor mask_cpu, Tensor output_cpu,
                             int64_t handle) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());
  const auto D = weights.size(1);
  const auto B = (offsets_cpu.size(0) - 1);
  // auto output_cpu = empty({B, D}, indices_cpu.options()
  //                                     .device(kCPU)
  //                                     .dtype(weights.options().dtype())
  //                                     .pinned_memory(true));
  std::function<void()> *functor = new std::function<void()>(
      [weights, indices_cpu, offsets_cpu, mask_cpu, output_cpu, handle]() {
        auto pf = *static_cast<tvm::runtime::PackedFunc *>(
            reinterpret_cast<TVMFunctionHandle>(handle));
        auto weights_dl = toDLTensor(weights, true);
        auto indices_dl = toDLTensor(indices_cpu, false);
        auto offsets_dl = toDLTensor(offsets_cpu, false);
        auto mask_dl = toDLTensor(mask_cpu, false);
        auto output_dl = toDLTensor(output_cpu, false);
        // std::chrono::steady_clock::time_point begin =
        // std::chrono::steady_clock::now();
        pf(&weights_dl, &indices_dl, &offsets_dl, &mask_dl, &output_dl);
        // std::chrono::steady_clock::time_point end =
        // std::chrono::steady_clock::now();
        // std::cout << "Inside cudaStreamAddCallback t = " <<
        // std::chrono::duration_cast<std::chrono::microseconds>(end -
        // begin).count() << "us" << std::endl;
      });
  // auto callFunctor = [](void *userData) -> void {
  //   auto *f = reinterpret_cast<std::function<void()> *>(userData);
  //   (*f)();
  //   delete f; // TODO: will this invoke destructors that call any CUDA API
  //             // functions?
  // };
  auto callFunctor = [](cudaStream_t stream, cudaError_t status,
                        void *userData) -> void {
    auto *f = reinterpret_cast<std::function<void()> *>(userData);
    (*f)();
    delete f; // TODO: will this invoke destructors that call any CUDA API
    // functions?
  };

  // auto pf = *static_cast<tvm::runtime::PackedFunc *>(
  //     reinterpret_cast<TVMFunctionHandle>(handle));
  // auto weights_dl = toDLTensor(weights, true);
  // auto indices_dl = toDLTensor(indices_cpu, false);
  // auto offsets_dl = toDLTensor(offsets_cpu, false);
  // auto mask_dl = toDLTensor(mask_cpu, false);
  // auto output_dl = toDLTensor(output_cpu, false);
  // std::chrono::steady_clock::time_point begin =
  // std::chrono::steady_clock::now();
  // pf(&weights_dl, &indices_dl, &offsets_dl, &mask_dl, &output_dl);
  // std::chrono::steady_clock::time_point end =
  // std::chrono::steady_clock::now();
  // std::cout << "Outside cudaStreamAddCallback t = " <<
  // std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
  // << "[us]" << std::endl;

  // std::chrono::steady_clock::time_point begin =
  // std::chrono::steady_clock::now();
  // (*functor)();
  // std::chrono::steady_clock::time_point end =
  // std::chrono::steady_clock::now();
  // std::cout << "Outside cudaStreamAddCallback(..) = " <<
  // std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
  // << "[us]" << std::endl;
  AT_CUDA_CHECK(cudaStreamAddCallback(at::cuda::getCurrentCUDAStream(),
                                      callFunctor, functor, 0));
  return output_cpu;
}

void lxu_cache_backward_sgd_cpu(Tensor grad_output_cpu, Tensor weights,
                                Tensor indices_cpu, Tensor offsets_cpu,
                                Tensor mask_cpu, float learning_rate,
                                int64_t handle) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());
  const auto D = weights.size(1);
  const auto B = (offsets_cpu.size(0) - 1);
  std::function<void()> *functor = new std::function<void()>(
      [grad_output_cpu, weights, indices_cpu, offsets_cpu, mask_cpu,
       learning_rate, handle]() {
        auto pf = *static_cast<tvm::runtime::PackedFunc *>(
            reinterpret_cast<TVMFunctionHandle>(handle));
        auto grad_output_dl = toDLTensor(grad_output_cpu, false);
        auto weights_dl = toDLTensor(weights, true);
        auto indices_dl = toDLTensor(indices_cpu, false);
        auto offsets_dl = toDLTensor(offsets_cpu, false);
        auto mask_dl = toDLTensor(mask_cpu, false);
        pf(&grad_output_dl, &weights_dl, &indices_dl, &offsets_dl, &mask_dl,
           learning_rate);
      });
  auto callFunctor = [](void *userData) -> void {
    auto *f = reinterpret_cast<std::function<void()> *>(userData);
    (*f)();
    delete f; // TODO: will this invoke destructors that call any CUDA API
    // functions?
  };
  AT_CUDA_CHECK(cudaLaunchHostFunc(at::cuda::getCurrentCUDAStream(),
                                   callFunctor, functor));
  return;
}

template <LXUCacheMaskAction Action>
Tensor
lxu_cache_forward_cuda_impl(Tensor weights, Tensor indices, Tensor offsets,
                            c10::optional<Tensor> indice_weights,
                            Tensor lxu_cache_locations,
                            Tensor lxu_cache_weights, int64_t B_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lxu_cache_weights.get_device());
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1);
  AT_ASSERT(B > 0);
  AT_ASSERT(D % 4 == 0);
  const dim3 threads(std::min(D / 4, kMaxThreads / B_block_size), B_block_size);
  const dim3 blocks(div_round_up(B, B_block_size));

#define X(functor)                                                             \
  lxu_cache_forward_kernel_1<                                                  \
      scalar_t,                                                                \
      Action><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(       \
      weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),             \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      lxu_cache_locations.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),  \
      lxu_cache_weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),   \
      output.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(), (functor))

  auto output = empty({B, D}, lxu_cache_weights.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "lxu_cache_forward_kernel", ([&] {
        if (indice_weights) {
          auto f = WeightedForward<scalar_t>(
              indice_weights
                  ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>());
          X(f);
        } else {
          auto f = UnweightedForward<scalar_t>();
          X(f);
        }
      }));

#undef X
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

Tensor lxu_cache_forward_cuda(Tensor weights, Tensor indices, Tensor offsets,
                              c10::optional<Tensor> indice_weights,
                              Tensor lxu_cache_locations,
                              Tensor lxu_cache_weights, int64_t B_block_size) {
  return lxu_cache_forward_cuda_impl<LXUCacheMaskAction::USE_BACKING_STORE>(
      weights, indices, offsets, indice_weights, lxu_cache_locations,
      lxu_cache_weights, B_block_size);
}

Tensor lxu_cache_forward_mixed_cuda(Tensor weights, Tensor indices,
                                    Tensor offsets,
                                    c10::optional<Tensor> indice_weights,
                                    Tensor lxu_cache_locations,
                                    Tensor lxu_cache_weights,
                                    int64_t B_block_size) {
  return lxu_cache_forward_cuda_impl<LXUCacheMaskAction::SKIP>(
      weights, indices, offsets, indice_weights, lxu_cache_locations,
      lxu_cache_weights, B_block_size);
}

template <typename scalar_t>
__global__ void lxu_cache_flush_kernel_1(
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    PackedTensorAccessor32<int32_t, 3, RestrictPtrTraits> lxu_cache_state,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> lxu_cache_weights) {
  const auto D = lxu_cache_weights.size(1);
  const auto B = lxu_cache_weights.size(0);

  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  const int32_t slot = b % kAssoc;
  const int32_t cache_set = b / kAssoc;
  const int32_t slot_time = lxu_cache_state[kCacheSlotTime][cache_set][slot];
  const int32_t slot_freq = lxu_cache_state[kCacheSlotFreq][cache_set][slot];
  if (slot_time > 0 && slot_freq > 0) {
    // evict from slot to backing storage
    const int32_t current_idx =
        lxu_cache_state[kCacheSlotCurrentIndex][cache_set][slot];
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t>::copy(&lxu_cache_weights[b][d * 4],
                            &weights[current_idx][d * 4]);
    }
  }
}

void lxu_cache_flush_cuda(Tensor weights, Tensor lxu_cache_state,
                          Tensor lxu_cache_weights, int64_t B_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lxu_cache_weights.get_device());
  const auto D = lxu_cache_weights.size(1);
  const auto B = lxu_cache_weights.size(0);
  AT_ASSERT(D % 4 == 0);
  const dim3 threads(std::min(D / 4, kMaxThreads / B_block_size), B_block_size);
  const dim3 blocks(div_round_up(B, B_block_size));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      lxu_cache_weights.type(), "lxu_cache_flush_kernel", ([&] {
        lxu_cache_flush_kernel_1<
            scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
            lxu_cache_state.packed_accessor32<int32_t, 3, RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<scalar_t, 2, RestrictPtrTraits>());
      }));

#undef X
  AT_CUDA_CHECK(cudaGetLastError());
  return;
}

template <typename scalar_t, typename F>
__global__ void lxu_cache_backward_sgd_kernel_1(
    const PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        lxu_cache_locations,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> lxu_cache_weights,
    F f) {

  const int32_t B = grad_output.size(0);
  const int32_t D = grad_output.size(2);

  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  const int32_t indices_start = offsets[b];
  const int32_t indices_end = offsets[b + 1];
  int32_t L = indices_end - indices_start;

  for (int32_t l = 0; l < L; ++l) {
    auto cache_idx = __ldg(&lxu_cache_locations[indices_start + l]);
    if (cache_idx != kCacheLocationMissing) {
      for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
        Vec4T<scalar_t> g(&grad_output[b][0] + d * 4);
        f(&lxu_cache_weights[cache_idx][0] + d * 4, g);
      }
    } else {
      auto idx = __ldg(&indices[indices_start + l]);
      for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
        Vec4T<scalar_t> g(&grad_output[b][0] + d * 4);
        f(&weights[idx][0] + d * 4, g);
      }
    }
  }
}

void lxu_cache_backward_sgd_cuda(Tensor grad_output, Tensor weights,
                                 Tensor indices, Tensor offsets,
                                 Tensor lxu_cache_locations,
                                 Tensor lxu_cache_weights, float learning_rate,
                                 int64_t B_block_size) {

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lxu_cache_weights.get_device());
  const auto D = weights.size(1);
  const auto B = (offsets.size(0) - 1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "lxu_cache_backward_sgd_kernel", ([&] {
        const dim3 threads(std::min(D / 4, kMaxThreads / B_block_size),
                           B_block_size);
        const dim3 blocks(div_round_up(B, B_block_size));
        lxu_cache_backward_sgd_kernel_1<
            scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
            indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            lxu_cache_locations
                .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
            SGDFunctor<scalar_t, true>(learning_rate));
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return;
}

template <typename scalar_t, typename F>
__global__ void lxu_cache_backward_sgd_exact_kernel_1(
    const PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices,
    const IndexInfo *__restrict__ infos,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        lxu_cache_locations,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> lxu_cache_weights,
    F f) {
  const int32_t B = grad_output.size(0);
  // const int32_t T = 1; // grad_output.size(1);
  const int32_t D = grad_output.size(1);
  const int32_t indices_numel = indices.size(0);
  int32_t sorted_linear_indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (sorted_linear_indice_id >= indices_numel) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so
    // we can just exit this warp entirely.
    return;
  }

  // check if this warp is responsible for this whole segment.
  bool segment_start = (sorted_linear_indice_id == 0 ||
                        sorted_linear_indices[sorted_linear_indice_id - 1] !=
                            sorted_linear_indices[sorted_linear_indice_id]);

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so
    // we can just exit this warp entirely.
    return;
  }

  int32_t linear_index = sorted_linear_indices[sorted_linear_indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (sorted_linear_indice_id + SL < indices_numel &&
         sorted_linear_indices[sorted_linear_indice_id + SL] == linear_index) {
    SL += 1;
  }
  // now, each segment corresponds to exactly one table `t` and row in that
  // table (`idx`). Thus, we can hoist out some of the book-keeping.
  auto info_0 = infos[sorted_linear_indice_id];
  // const int32_t t = info_0.b_t / B;
  const int64_t table_offset = 0; // table_offsets[t];
  const int32_t idx = linear_index - table_offset;
  const int32_t cache_idx = lxu_cache_locations[info_0.indices_idx];

  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      int32_t b = infos[sorted_linear_indice_id + sl].b_t % B;
      // TODO: apply per-sample weighting.
      Vec4T<scalar_t> grad_out(&grad_output[b][0] + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    auto cache_idx = __ldg(&lxu_cache_locations[info_0.indices_idx]);
    // common case.
    if (cache_idx != kCacheLocationMissing) {
      f((&lxu_cache_weights[cache_idx][0]) + d * 4, sum);
    } else {
      f((&weights[idx][0]) + d * 4, sum);
    }
  }
}

void lxu_cache_backward_sgd_exact_cuda(Tensor grad_output, Tensor weights,
                                       Tensor indices, Tensor offsets,
                                       Tensor lxu_cache_locations,
                                       Tensor lxu_cache_weights,
                                       float learning_rate,
                                       int64_t B_block_size) {

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lxu_cache_weights.get_device());
  const auto D = weights.size(1);
  const auto B = (offsets.size(0) - 1);

  auto infos =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto infos_sorted =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto linear_indices = at::empty_like(indices);
  auto linear_indices_sorted = at::empty_like(indices);
  auto table_offsets = at::zeros({1}, indices.options());
  const dim3 threads(kWarpSize, B_block_size);
  AT_ASSERT(B % B_block_size == 0);

  linear_index_weight_offsets_kernel<<<dim3(B / B_block_size), threads, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      (IndexInfo *)infos.data_ptr(),
      linear_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());

  size_t temp_storage_bytes = 0;
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(weights.size(0)) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  auto temp_storage = at::empty({static_cast<int64_t>(temp_storage_bytes)},
                                indices.options().dtype(kByte));

  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp_storage.data_ptr(), temp_storage_bytes,
      (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(weights.size(0)) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  const dim3 blocks(div_round_up(indices.numel(), B_block_size));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "lxu_cache_backward_sgd_kernel", ([&] {
        const dim3 threads(std::min(D / 4, kMaxThreads / B_block_size),
                           B_block_size);
        const dim3 blocks(div_round_up(B, B_block_size));
        lxu_cache_backward_sgd_exact_kernel_1<
            scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
            indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            linear_indices_sorted
                .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            (const IndexInfo *)infos_sorted.data_ptr(),
            lxu_cache_locations
                .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
            SGDFunctor<scalar_t, false>(learning_rate));
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return;
}
