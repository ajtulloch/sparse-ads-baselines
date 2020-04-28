#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGenerator.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mutex>

using namespace at;

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
  inline __device__ Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  inline __device__ Vec4T(const Half *p) {
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

  inline __device__ void accumulate(const Half *p) {
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

    acc.x += a.x;
    acc.y += a.y;
    acc.z += b.x;
    acc.w += b.y;
  }

  inline __device__ void store(Half *p) {
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
};

template <> struct Vec4T<float> {
  float4 acc;
  inline __device__ Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  inline __device__ Vec4T(const float *p) { acc = *((const float4 *)p); }

  inline __device__ void accumulate(const float *p) {
    auto val = *((const float4 *)p);
    acc.x += val.x;
    acc.y += val.y;
    acc.z += val.z;
    acc.w += val.w;
  }

  inline __device__ void store(float *p) { *((float4 *)p) = acc; }
};

template <> struct Vec4T<double> {
  double4 acc;
  inline __device__ Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  inline __device__ Vec4T(const double *p) { acc = *((const double4 *)p); }

  inline __device__ void accumulate(const double *p) {
    auto val = *((const double4 *)p);
    acc.x += val.x;
    acc.y += val.y;
    acc.z += val.z;
    acc.w += val.w;
  }

  inline __device__ void store(double *p) { *((double4 *)p) = acc; }
};

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

template <typename T> struct Sum {
  inline __device__ T operator()(T a, T b) const { return a + b; }
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

template <typename scalar_t, bool shared_indices>
__global__ void batched_embedding_forward_kernel_1(
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

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t t = b_t / B;
  int32_t b = b_t % B;

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[b * T + t];
  const int32_t indices_end = offsets[b * T + t + 1];
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
        sum.accumulate((&weights[table_offset + idx][0]) + d * 4);
      }
      sum.store((&output[b][t][0]) + d * 4);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = __ldg(&indices[indices_start + l]);
        sum.accumulate((&weights[table_offset + idx][0]) + d * 4);
      }
      sum.store((&output[b][t][0]) + d * 4);
    }
  }
}

Tensor batched_embedding_forward_cuda(Tensor weights, Tensor table_offsets,
                                      Tensor indices, Tensor offsets,
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

  auto output = empty({B, T, D}, weights.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_forward_kernel", ([&] {
        const dim3 threads(std::min(D / 4, kMaxThreads / BT_block_size),
                           BT_block_size);
        const dim3 blocks((B * T) / BT_block_size);
        if (shmem) {
          batched_embedding_forward_kernel_1<
              scalar_t,
              true><<<blocks, threads, BT_block_size * L_max * sizeof(int32_t),
                      at::cuda::getCurrentCUDAStream()>>>(
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max));
        } else {
          batched_embedding_forward_kernel_1<
              scalar_t,
              false><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max));
        }
      }));
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

  inline void __device__ operator()(scalar_t *weight, Vec4T<scalar_t> grad) {
    // don't handle atomic case.
    Vec4T<scalar_t> weight_new(weight);
    weight_new.acc.x -= grad.acc.x * learning_rate_;
    weight_new.acc.y -= grad.acc.y * learning_rate_;
    weight_new.acc.z -= grad.acc.z * learning_rate_;
    weight_new.acc.w -= grad.acc.w * learning_rate_;
    weight_new.store(weight);
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
  const int32_t indices_start = offsets[b * T + t];
  const int32_t indices_end = offsets[b * T + t + 1];
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
              SGDFunctor<scalar_t, false>(learning_rate));
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
              SGDFunctor<scalar_t, false>(learning_rate));
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return;
}

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
  // assume valid.

  acc_type<scalar_t, true> g_local_sum_square = 0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> g(&grad_output[b][t][0] + d * 4);
    g_local_sum_square += g.acc.x * g.acc.x + g.acc.y * g.acc.y +
                          g.acc.z * g.acc.z + g.acc.w * g.acc.w;
  }
  const acc_type<scalar_t, true> g_sum_square =
      warpReduceAllSum<acc_type<scalar_t, true>>(g_local_sum_square);

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[b * T + t];
  const int32_t indices_end = offsets[b * T + t + 1];
  int32_t L = indices_end - indices_start;

  auto state = f.init_update(blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x);
  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    shmem_multipliers[threadIdx.y * L_max + i] =
        f.update_momentum(g_sum_square, &optimizer_state[table_offset + idx]);
  }
  __syncthreads();
  for (int32_t l = 0; l < L; ++l) {
    auto idx = indices[indices_start + l];
    acc_type<scalar_t, true> multiplier =
        shmem_multipliers[threadIdx.y * L_max + l];
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      f.update_weight(&weights[table_offset + idx][0] + d * 4, multiplier,
                      Vec4T<scalar_t>(&grad_output[b][t][0] + d * 4), state);
    }
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
  struct State {};

  __device__ inline __attribute__((always_inline)) State
  init_update(int thread_id) {
    return State{};
  }

  __device__ inline __attribute__((always_inline)) void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                scalar_t grad, State &state) {
    if (use_atomics) {
      gpuAtomicAdd(weight, -multiplier * grad);
    } else {
      *weight -= grad * multiplier;
    }
  }

  __device__ inline __attribute__((always_inline)) void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                Vec4T<scalar_t> grad, State &state) {
    // can't use atomics.
    Vec4T<scalar_t> weight_new(weight);
    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    weight_new.store(weight);
  }
};

template <typename scalar_t>
__device__ inline __attribute__((always_inline)) void
stochastic_rounding_scalar(scalar_t *output, acc_type<scalar_t, true> value,
                           uint32_t random_bits) {
  *output = value;
}

template <>
__device__ inline __attribute__((always_inline)) void
stochastic_rounding_scalar(Half *output, float value, uint32_t random_bits) {
  *output = __float2half_rz(
      __uint_as_float(__float_as_uint(value) + (random_bits >> 19)));
}

template <typename scalar_t>
__device__ inline __attribute__((always_inline)) void
stochastic_rounding_vector(scalar_t *output, Vec4T<scalar_t> value,
                           uint4 random_bits) {
  value.store(output);
}

template <>
__device__ inline __attribute__((always_inline)) void
stochastic_rounding_vector(Half *output, Vec4T<Half> value, uint4 random_bits) {
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
struct StochasticRoundingAdaGradFunctor
    : public AdaGradFunctor<scalar_t, use_atomics> {
  StochasticRoundingAdaGradFunctor(float learning_rate, float eps,
                                   std::pair<uint64_t, uint64_t> seeds)
      : AdaGradFunctor<scalar_t, use_atomics>(learning_rate, eps),
        seeds_(seeds) {}
  std::pair<uint64_t, uint64_t> seeds_;

  __device__ inline __attribute__((always_inline)) curandStatePhilox4_32_10_t
  init_update(int thread_id) {
    curandStatePhilox4_32_10_t state;
    curand_init(seeds_.first, thread_id, seeds_.second, &state);
    return state;
  }

  __device__ inline __attribute__((always_inline)) void
  stochastic_rounding(scalar_t *output, acc_type<scalar_t, true> value,
                      uint32_t bits) {
    stochastic_rounding_scalar(output, value, bits);
  }

  __device__ inline __attribute__((always_inline)) void
  stochastic_rounding(scalar_t *output, Vec4T<scalar_t> value, uint4 bits) {
    stochastic_rounding_vector(output, value, bits);
  }

  __device__ inline __attribute__((always_inline)) void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                scalar_t grad, curandStatePhilox4_32_10_t &state) {
    acc_type<scalar_t, true> w_new =
        *weight - multiplier * static_cast<acc_type<scalar_t, true>>(grad);
    uint32_t bits = curand(&state);
    stochastic_rounding(weight, w_new, bits);
  }

  __device__ inline __attribute__((always_inline)) void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                Vec4T<scalar_t> grad, curandStatePhilox4_32_10_t &state) {
    Vec4T<scalar_t> weight_new(weight);
    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    uint4 bits = curand4(&state);
    stochastic_rounding(weight, weight_new, bits);
  }
};

void batched_embedding_backward_adagrad_approx_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, Tensor optimizer_state, float learning_rate, float eps,
    int64_t L_max, bool stochastic_rounding, int64_t BT_block_size) {
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        const dim3 threads(kWarpSize, BT_block_size);
        const dim3 blocks((B * T) / BT_block_size);
        if (!stochastic_rounding) {
          batched_embedding_backward_adagrad_approx_kernel_1<
              scalar_t><<<blocks, threads, BT_block_size * L_max *
                                               sizeof(acc_type<scalar_t, true>),
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
        } else {
          std::pair<uint64_t, uint64_t> rng_engine_inputs;
          {
            auto gen = at::cuda::detail::getDefaultCUDAGenerator();
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs = gen->philox_engine_inputs(
                L_max * ((D + kWarpSize - 1) / kWarpSize));
          }
          batched_embedding_backward_adagrad_approx_kernel_1<
              scalar_t><<<blocks, threads, BT_block_size * L_max *
                                               sizeof(acc_type<scalar_t, true>),
                          at::cuda::getCurrentCUDAStream()>>>(
              grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,
                                                RestrictPtrTraits>(),
              static_cast<int32_t>(L_max),
              StochasticRoundingAdaGradFunctor<scalar_t, false>(
                  learning_rate, eps, rng_engine_inputs));
        }
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