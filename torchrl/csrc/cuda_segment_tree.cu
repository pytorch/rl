// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "cuda_segment_tree_ops.h"

namespace torchrl {
namespace {

constexpr int kThreads = 256;

template <typename T, bool IsMin>
__device__ __forceinline__ T ApplyOp(T lhs, T rhs) {
  if constexpr (IsMin) {
    return lhs < rhs ? lhs : rhs;
  } else {
    return lhs + rhs;
  }
}

template <typename T>
__global__ void SetLeavesKernel(T* tree, int64_t capacity, const int64_t* index,
                                const T* value, int64_t n, bool scalar_value) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  // Apply updates in input order to match the CPU tree when an index appears
  // more than once in the same batch.
  for (int64_t i = 0; i < n; ++i) {
    int64_t idx = index[i];
    tree[capacity + idx] = scalar_value ? value[0] : value[i];
  }
}

template <typename T, bool IsMin>
__global__ void RecomputeLevelKernel(T* tree, int64_t level_start,
                                     int64_t level_count) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= level_count) {
    return;
  }
  int64_t node = level_start + i;
  tree[node] = ApplyOp<T, IsMin>(tree[node << 1], tree[(node << 1) | 1]);
}

template <typename T, bool IsMin>
__global__ void QueryKernel(const T* tree, int64_t capacity, const int64_t* l,
                            const int64_t* r, T* out, int64_t n,
                            T identity_element) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  int64_t left = l[i] | capacity;
  int64_t right = r[i] | capacity;
  T result = identity_element;
  while (left < right) {
    if (left & 1) {
      result = ApplyOp<T, IsMin>(result, tree[left++]);
    }
    if (right & 1) {
      result = ApplyOp<T, IsMin>(result, tree[--right]);
    }
    left >>= 1;
    right >>= 1;
  }
  out[i] = result;
}

template <typename T>
__global__ void ScanLowerBoundKernel(const T* tree, int64_t size,
                                     int64_t capacity, const T* value,
                                     int64_t* out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  T current_value = value[i];
  if (current_value > tree[1]) {
    out[i] = size;
    return;
  }
  int64_t index = 1;
  while (index < capacity) {
    index <<= 1;
    T left_value = tree[index];
    if (current_value > left_value) {
      current_value -= left_value;
      index |= 1;
    }
  }
  out[i] = index ^ capacity;
}

template <typename T, bool IsMin>
void RecomputeTree(torch::Tensor values, int64_t capacity) {
  auto stream = at::cuda::getCurrentCUDAStream(values.get_device());
  T* tree = values.data_ptr<T>();
  for (int64_t level_count = capacity >> 1; level_count >= 1;
       level_count >>= 1) {
    int64_t blocks = (level_count + kThreads - 1) / kThreads;
    RecomputeLevelKernel<T, IsMin>
        <<<blocks, kThreads, 0, stream>>>(tree, level_count, level_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (level_count == 1) {
      break;
    }
  }
}

template <typename T, bool IsMin>
void CudaSegmentTreeUpdateImpl(torch::Tensor values, int64_t capacity,
                               torch::Tensor index, torch::Tensor value) {
  const c10::cuda::CUDAGuard device_guard(values.device());
  auto index_contiguous = index.contiguous().to(values.device());
  auto value_contiguous = value.contiguous().to(values.device());
  int64_t n = index_contiguous.numel();
  if (n == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream(values.get_device());
  SetLeavesKernel<T><<<1, 1, 0, stream>>>(
      values.data_ptr<T>(), capacity, index_contiguous.data_ptr<int64_t>(),
      value_contiguous.data_ptr<T>(), n, value_contiguous.numel() == 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  RecomputeTree<T, IsMin>(values, capacity);
}

template <typename T, bool IsMin>
torch::Tensor CudaSegmentTreeQueryImpl(torch::Tensor values, int64_t capacity,
                                       torch::Tensor l, torch::Tensor r,
                                       T identity_element) {
  const c10::cuda::CUDAGuard device_guard(values.device());
  auto l_contiguous = l.contiguous().to(values.device());
  auto r_contiguous = r.contiguous().to(values.device());
  auto out = torch::empty(l_contiguous.sizes(), values.options());
  int64_t n = l_contiguous.numel();
  if (n == 0) {
    return out;
  }
  int64_t blocks = (n + kThreads - 1) / kThreads;
  auto stream = at::cuda::getCurrentCUDAStream(values.get_device());
  QueryKernel<T, IsMin><<<blocks, kThreads, 0, stream>>>(
      values.data_ptr<T>(), capacity, l_contiguous.data_ptr<int64_t>(),
      r_contiguous.data_ptr<int64_t>(), out.data_ptr<T>(), n, identity_element);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.reshape(l.sizes());
}

}  // namespace

template <typename T>
void CudaSegmentTreeUpdate(torch::Tensor values, int64_t capacity,
                           torch::Tensor index, torch::Tensor value,
                           CudaSegmentTreeOp op) {
  TORCH_CHECK(values.is_cuda(), "CUDA segment tree values must live on CUDA");
  TORCH_CHECK(index.dtype() == torch::kInt64, "index must be an int64 tensor");
  TORCH_CHECK(value.dtype() == values.scalar_type(),
              "value dtype must match the tree dtype");
  if (op == CudaSegmentTreeOp::Min) {
    CudaSegmentTreeUpdateImpl<T, true>(values, capacity, index, value);
  } else {
    CudaSegmentTreeUpdateImpl<T, false>(values, capacity, index, value);
  }
}

template <typename T>
torch::Tensor CudaSegmentTreeQuery(torch::Tensor values, int64_t capacity,
                                   torch::Tensor l, torch::Tensor r,
                                   CudaSegmentTreeOp op) {
  TORCH_CHECK(values.is_cuda(), "CUDA segment tree values must live on CUDA");
  TORCH_CHECK(l.dtype() == torch::kInt64, "l must be an int64 tensor");
  TORCH_CHECK(r.dtype() == torch::kInt64, "r must be an int64 tensor");
  if (op == CudaSegmentTreeOp::Min) {
    return CudaSegmentTreeQueryImpl<T, true>(values, capacity, l, r,
                                             std::numeric_limits<T>::max());
  }
  return CudaSegmentTreeQueryImpl<T, false>(values, capacity, l, r, T(0));
}

template <typename T>
torch::Tensor CudaSumSegmentTreeScanLowerBound(torch::Tensor values,
                                               int64_t size, int64_t capacity,
                                               torch::Tensor value) {
  TORCH_CHECK(values.is_cuda(), "CUDA segment tree values must live on CUDA");
  TORCH_CHECK(value.dtype() == values.scalar_type(),
              "value dtype must match the tree dtype");
  const c10::cuda::CUDAGuard device_guard(values.device());
  auto value_contiguous = value.contiguous().to(values.device());
  auto out = torch::empty(value_contiguous.sizes(),
                          value_contiguous.options().dtype(torch::kInt64));
  int64_t n = value_contiguous.numel();
  if (n == 0) {
    return out;
  }
  int64_t blocks = (n + kThreads - 1) / kThreads;
  auto stream = at::cuda::getCurrentCUDAStream(values.get_device());
  ScanLowerBoundKernel<T><<<blocks, kThreads, 0, stream>>>(
      values.data_ptr<T>(), size, capacity, value_contiguous.data_ptr<T>(),
      out.data_ptr<int64_t>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.reshape(value.sizes());
}

template void CudaSegmentTreeUpdate<float>(torch::Tensor, int64_t,
                                           torch::Tensor, torch::Tensor,
                                           CudaSegmentTreeOp);
template void CudaSegmentTreeUpdate<double>(torch::Tensor, int64_t,
                                            torch::Tensor, torch::Tensor,
                                            CudaSegmentTreeOp);

template torch::Tensor CudaSegmentTreeQuery<float>(torch::Tensor, int64_t,
                                                   torch::Tensor, torch::Tensor,
                                                   CudaSegmentTreeOp);
template torch::Tensor CudaSegmentTreeQuery<double>(torch::Tensor, int64_t,
                                                    torch::Tensor,
                                                    torch::Tensor,
                                                    CudaSegmentTreeOp);

template torch::Tensor CudaSumSegmentTreeScanLowerBound<float>(torch::Tensor,
                                                               int64_t, int64_t,
                                                               torch::Tensor);
template torch::Tensor CudaSumSegmentTreeScanLowerBound<double>(torch::Tensor,
                                                                int64_t,
                                                                int64_t,
                                                                torch::Tensor);

}  // namespace torchrl
