#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "torch_data_type.h"

namespace py = pybind11;

namespace torchrl {

// SegmentTree is a tree data structure to maintain statistics of intervals.
// https://en.wikipedia.org/wiki/Segment_tree
// Here is the implementaion of non-recursive SegmentTree for single point
// update and interval query. The time complexities of both Update and Query are
// O(logN).
// One example of a SegmentTree is shown below.
//
//                          1: [0, 8)
//                       /             \
//            2 [0, 4)                       3 [4, 8)
//          /         \                    /          \
//     4: [0, 2)      5: [2, 4)      6: [4, 6)      7: [6, 8)
//     /     \        /      \        /     \        /      \
//   8: 0   9: 1   10: 2   11: 3   12: 4   13: 5   14: 6   15: 7

template <typename T, class Operator> class SegmentTree {
public:
  SegmentTree(int64_t size, const T &identity_element)
      : size_(size), identity_element_(identity_element) {
    for (capacity_ = 1; capacity_ < size; capacity_ <<= 1)
      ;
    values_.assign(2 * capacity_, identity_element_);
  }

  int64_t size() const { return size_; }

  int64_t capacity() const { return capacity_; }

  const T &identity_element() const { return identity_element_; }

  const T &At(int64_t index) const { return values_[index | capacity_]; }

  std::vector<T> At(const std::vector<int64_t> &index) const {
    const int64_t n = index.size();
    std::vector<T> value(n);
    BatchAtImpl(n, index.data(), value.data());
    return value;
  }

  py::array_t<T> At(const py::array_t<int64_t> &index) const {
    assert(index.ndim() == 1);
    const int64_t n = index.size();
    py::array_t<T> value(n);
    BatchAtImpl(n, index.data(), value.mutable_data());
    return value;
  }

  torch::Tensor At(const torch::Tensor &index) const {
    assert(index.dtype() == torch::kInt64);
    const torch::Tensor index_contiguous = index.contiguous();
    const int64_t n = index_contiguous.numel();
    torch::Tensor value =
        torch::empty_like(index_contiguous, utils::TorchDataType<T>::value);
    BatchAtImpl(n, index_contiguous.data_ptr<int64_t>(), value.data_ptr<T>());
    return value;
  }

  // Update the item at index to value.
  // Time complexity: O(logN).
  void Update(int64_t index, const T &value) {
    index |= capacity_;
    for (values_[index] = value; index > 1; index >>= 1) {
      values_[index >> 1] = op_(values_[index], values_[index ^ 1]);
    }
  }

  void Update(const std::vector<int64_t> &index, const T &value) {
    BatchUpdateImpl(index.size(), index.data(), value);
  }

  void Update(const std::vector<int64_t> &index, const std::vector<T> &value) {
    assert(value.size() == 1 || index.size() == value.size());
    const int64_t n = index.size();
    if (value.size() == 1) {
      BatchUpdateImpl(n, index.data(), value[0]);
    } else {
      BatchUpdateImpl(n, index.data(), value.data());
    }
  }

  void Update(const py::array_t<int64_t> &index, const T &value) {
    assert(index.ndim() == 1);
    BatchUpdateImpl(index.size(), index.data(), value);
  }

  void Update(const py::array_t<int64_t> &index, const py::array_t<T> &value) {
    assert(index.ndim() == 1);
    assert(value.ndim() == 1);
    assert(value.size() == 1 || index.size() == value.size());
    const int64_t n = index.size();
    if (value.size() == 1) {
      BatchUpdateImpl(n, index.data(), *(value.data()));
    } else {
      BatchUpdateImpl(n, index.data(), value.data());
    }
  }

  void Update(const torch::Tensor &index, const T &value) {
    assert(index.dtype() == torch::kInt64);
    const torch::Tensor index_contiguous = index.contiguous();
    const int64_t n = index_contiguous.numel();
    BatchUpdateImpl(n, index_contiguous.data_ptr<int64_t>(), value);
  }

  void Update(const torch::Tensor &index, const torch::Tensor &value) {
    assert(index.dtype() == torch::kInt64);
    assert(value.dtype() == utils::TorchDataType<T>::value);
    assert(value.numel() == 1 || index.sizes() == value.sizes());
    const torch::Tensor index_contiguous = index.contiguous();
    const torch::Tensor value_contiguous = value.contiguous();
    const int64_t n = index_contiguous.numel();
    if (value_contiguous.numel() == 1) {
      BatchUpdateImpl(n, index_contiguous.data_ptr<int64_t>(),
                      *(value_contiguous.data_ptr<T>()));
    } else {
      BatchUpdateImpl(n, index_contiguous.data_ptr<int64_t>(),
                      value_contiguous.data_ptr<T>());
    }
  }

  // Reduce the range of [l, r) by Operator.
  // Time complexity: O(logN)
  T Query(int64_t l, int64_t r) const {
    assert(l < r);
    if (l <= 0 && r >= size_) {
      return values_[1];
    }
    T ret = identity_element_;
    l |= capacity_;
    r |= capacity_;
    while (l < r) {
      if (l & 1) {
        ret = op_(ret, values_[l++]);
      }
      if (r & 1) {
        ret = op_(ret, values_[--r]);
      }
      l >>= 1;
      r >>= 1;
    }
    return ret;
  }

  std::vector<T> Query(const std::vector<int64_t> &l,
                       const std::vector<int64_t> &r) const {
    assert(l.size() == r.size());
    std::vector<T> ret(l.size());
    const int64_t n = l.size();
    BatchQueryImpl(n, l.data(), r.data(), ret.data());
    return ret;
  }

  py::array_t<T> Query(const py::array_t<int64_t> &l,
                       const py::array_t<int64_t> &r) const {
    assert(l.ndim() == 1);
    assert(r.ndim() == 1);
    assert(l.size() == r.size());
    const int64_t n = l.size();
    py::array_t<T> ret(n);
    BatchQueryImpl(n, l.data(), r.data(), ret.mutable_data());
    return ret;
  }

  torch::Tensor Query(const torch::Tensor &l, const torch::Tensor &r) const {
    assert(l.dtype() == torch::kInt64);
    assert(r.dtype() == torch::kInt64);
    assert(l.sizes() == r.sizes());
    const torch::Tensor l_contiguous = l.contiguous();
    const torch::Tensor r_contiguous = r.contiguous();
    torch::Tensor ret =
        torch::empty_like(l_contiguous, utils::TorchDataType<T>::value);
    const int64_t n = l_contiguous.numel();
    BatchQueryImpl(n, l_contiguous.data_ptr<int64_t>(),
                   r_contiguous.data_ptr<int64_t>(), ret.data_ptr<T>());
    return ret;
  }

 protected:
  void BatchAtImpl(int64_t n, const int64_t *index, T *value) const {
    for (int64_t i = 0; i < n; ++i) {
      value[i] = values_[index[i] | capacity_];
    }
  }

  void BatchUpdateImpl(int64_t n, const int64_t *index, const T &value) {
    for (int64_t i = 0; i < n; ++i) {
      Update(index[i], value);
    }
  }

  void BatchUpdateImpl(int64_t n, const int64_t *index, const T* value) {
    for (int64_t i = 0; i < n; ++i) {
      Update(index[i], value[i]);
    }
  }

  void BatchQueryImpl(int64_t n, const int64_t *l, const int64_t *r,
                      T *result) const {
    for (int64_t i = 0; i < n; ++i) {
      result[i] = Query(l[i], r[i]);
    }
  }

  const Operator op_{};
  const int64_t size_;
  int64_t capacity_;
  const T identity_element_;
  std::vector<T> values_;
};

template <typename T>
class SumSegmentTree final : public SegmentTree<T, std::plus<T>> {
public:
  SumSegmentTree(int64_t size) : SegmentTree<T, std::plus<T>>(size, T(0)) {}

  // Get the 1st index where the scan (prefix sum) is not less than value.
  // Time complexity: O(logN)
  int64_t ScanLowerBound(const T &value) const {
    if (value > this->values_[1]) {
      return this->size_;
    }
    int64_t index = 1;
    T current_value = value;
    while (index < this->capacity_) {
      index <<= 1;
      const T &lvalue = this->values_[index];
      if (current_value > lvalue) {
        current_value -= lvalue;
        index |= 1;
      }
    }
    return index ^ this->capacity_;
  }

  std::vector<int64_t> ScanLowerBound(const std::vector<T> &value) const {
    std::vector<int64_t> index(value.size());
    BatchScanLowerBoundImpl(value.size(), value.data(), index.data());
    return index;
  }

  py::array_t<int64_t> ScanLowerBound(const py::array_t<T> &value) const {
    assert(value.ndim() == 1);
    const int64_t n = value.size();
    py::array_t<int64_t> index(n);
    BatchScanLowerBoundImpl(n, value.data(), index.mutable_data());
    return index;
  }

  torch::Tensor ScanLowerBound(const torch::Tensor &value) const {
    assert(value.dtype() == utils::TorchDataType<T>::value);
    const torch::Tensor value_contiguous = value.contiguous();
    torch::Tensor index = torch::empty_like(value_contiguous, torch::kInt64);
    const int64_t n = value_contiguous.numel();
    BatchScanLowerBoundImpl(n, value_contiguous.data_ptr<T>(),
                            index.data_ptr<int64_t>());
    return index;
  }

protected:
  void BatchScanLowerBoundImpl(int64_t n, const T *value,
                               int64_t *index) const {
    for (int64_t i = 0; i < n; ++i) {
      index[i] = ScanLowerBound(value[i]);
    }
  }
};

template <typename T> struct MinOp {
  T operator()(const T &lhs, const T &rhs) const { return std::min(lhs, rhs); }
};

template <typename T>
class MinSegmentTree final : public SegmentTree<T, MinOp<T>> {
public:
  MinSegmentTree(int64_t size)
      : SegmentTree<T, MinOp<T>>(size, std::numeric_limits<T>::max()) {}
};

} // namespace torchrl
