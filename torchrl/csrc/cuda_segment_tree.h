// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "cuda_segment_tree_ops.h"
#include "numpy_utils.h"
#include "torch_utils.h"

namespace py = pybind11;

namespace torchrl {

template <typename T, CudaSegmentTreeOp Op>
class CudaSegmentTree {
 public:
  CudaSegmentTree(int64_t size, const T& identity_element,
                  torch::Device device = torch::Device(torch::kCUDA))
      : size_(size), identity_element_(identity_element) {
    for (capacity_ = 1; capacity_ <= size; capacity_ <<= 1) {
    }
    auto options = torch::TensorOptions()
                       .dtype(utils::TorchDataType<T>::value)
                       .device(device);
    values_ = torch::full({2 * capacity_}, identity_element_, options);
  }

  int64_t size() const { return size_; }

  int64_t capacity() const { return capacity_; }

  const T& identity_element() const { return identity_element_; }

  torch::Device device() const { return values_.device(); }

  const torch::Tensor& values() const { return values_; }

  T At(int64_t index) const {
    return values_.index({index | capacity_}).template item<T>();
  }

  torch::Tensor At(const torch::Tensor& index) const {
    TORCH_CHECK(index.dtype() == torch::kInt64,
                "index must be an int64 tensor");
    auto index_contiguous = index.contiguous().to(values_.device());
    auto flat_index = index_contiguous.reshape({-1}) + capacity_;
    return values_.index_select(0, flat_index).reshape(index.sizes());
  }

  py::array_t<T> At(const py::array_t<int64_t>& index) const {
    auto cpu_index =
        torch::from_blob(const_cast<int64_t*>(index.data()), {index.size()},
                         torch::TensorOptions().dtype(torch::kInt64))
            .clone();
    auto cpu_value = At(cpu_index).cpu().contiguous();
    py::array_t<T> value = utils::NumpyEmptyLike<int64_t, T>(index);
    std::memcpy(value.mutable_data(), cpu_value.template data_ptr<T>(),
                index.size() * sizeof(T));
    return value;
  }

  void Update(int64_t index, const T& value) {
    auto index_tensor = torch::tensor(
        {index},
        torch::TensorOptions().dtype(torch::kInt64).device(values_.device()));
    auto value_tensor =
        torch::tensor({value}, torch::TensorOptions()
                                   .dtype(utils::TorchDataType<T>::value)
                                   .device(values_.device()));
    Update(index_tensor, value_tensor);
  }

  void Update(const torch::Tensor& index, const T& value) {
    auto value_tensor =
        torch::tensor({value}, torch::TensorOptions()
                                   .dtype(utils::TorchDataType<T>::value)
                                   .device(values_.device()));
    Update(index, value_tensor);
  }

  void Update(const torch::Tensor& index, const torch::Tensor& value) {
    TORCH_CHECK(index.dtype() == torch::kInt64,
                "index must be an int64 tensor");
    TORCH_CHECK(value.dtype() == utils::TorchDataType<T>::value,
                "value has an unexpected dtype");
    TORCH_CHECK(value.numel() == 1 || index.sizes() == value.sizes(),
                "value must be scalar or have the same shape as index");
    CudaSegmentTreeUpdate<T>(values_, capacity_, index, value, Op);
  }

  void Update(const py::array_t<int64_t>& index, const T& value) {
    auto index_tensor =
        torch::from_blob(const_cast<int64_t*>(index.data()), {index.size()},
                         torch::TensorOptions().dtype(torch::kInt64))
            .clone()
            .to(values_.device());
    Update(index_tensor, value);
  }

  void Update(const py::array_t<int64_t>& index, const py::array_t<T>& value) {
    TORCH_CHECK(value.size() == 1 || value.size() == index.size(),
                "value must be scalar or have the same size as index");
    auto index_tensor =
        torch::from_blob(const_cast<int64_t*>(index.data()), {index.size()},
                         torch::TensorOptions().dtype(torch::kInt64))
            .clone()
            .to(values_.device());
    auto value_tensor =
        torch::from_blob(
            const_cast<T*>(value.data()), {value.size()},
            torch::TensorOptions().dtype(utils::TorchDataType<T>::value))
            .clone()
            .to(values_.device());
    Update(index_tensor, value_tensor);
  }

  T Query(int64_t l, int64_t r) const {
    auto l_tensor = torch::tensor(
        {l},
        torch::TensorOptions().dtype(torch::kInt64).device(values_.device()));
    auto r_tensor = torch::tensor(
        {r},
        torch::TensorOptions().dtype(torch::kInt64).device(values_.device()));
    return Query(l_tensor, r_tensor).cpu().template item<T>();
  }

  torch::Tensor Query(const torch::Tensor& l, const torch::Tensor& r) const {
    TORCH_CHECK(l.dtype() == torch::kInt64, "l must be an int64 tensor");
    TORCH_CHECK(r.dtype() == torch::kInt64, "r must be an int64 tensor");
    TORCH_CHECK(l.sizes() == r.sizes(), "l and r must have the same shape");
    return CudaSegmentTreeQuery<T>(values_, capacity_, l, r, Op);
  }

  py::array_t<T> Query(const py::array_t<int64_t>& l,
                       const py::array_t<int64_t>& r) const {
    TORCH_CHECK(l.size() == r.size(), "l and r must have the same size");
    auto l_tensor =
        torch::from_blob(const_cast<int64_t*>(l.data()), {l.size()},
                         torch::TensorOptions().dtype(torch::kInt64))
            .clone()
            .to(values_.device());
    auto r_tensor =
        torch::from_blob(const_cast<int64_t*>(r.data()), {r.size()},
                         torch::TensorOptions().dtype(torch::kInt64))
            .clone()
            .to(values_.device());
    auto cpu_result = Query(l_tensor, r_tensor).cpu().contiguous();
    py::array_t<T> ret = utils::NumpyEmptyLike<int64_t, T>(l);
    std::memcpy(ret.mutable_data(), cpu_result.template data_ptr<T>(),
                l.size() * sizeof(T));
    return ret;
  }

  py::array_t<T> DumpValues() const {
    auto leaves =
        values_.slice(0, capacity_, capacity_ + size_).cpu().contiguous();
    py::array_t<T> ret(size_);
    std::memcpy(ret.mutable_data(), leaves.template data_ptr<T>(),
                size_ * sizeof(T));
    return ret;
  }

  void LoadValues(const py::array_t<T>& values) {
    TORCH_CHECK(values.size() == size_,
                "loaded values must match the tree size");
    auto value_tensor =
        torch::from_blob(
            const_cast<T*>(values.data()), {values.size()},
            torch::TensorOptions().dtype(utils::TorchDataType<T>::value))
            .clone()
            .to(values_.device());
    auto index_tensor = torch::arange(
        size_,
        torch::TensorOptions().dtype(torch::kInt64).device(values_.device()));
    Update(index_tensor, value_tensor);
  }

 protected:
  const int64_t size_;
  int64_t capacity_;
  const T identity_element_;
  torch::Tensor values_;
};

template <typename T>
class CudaSumSegmentTree final
    : public CudaSegmentTree<T, CudaSegmentTreeOp::Sum> {
 public:
  explicit CudaSumSegmentTree(
      int64_t size, torch::Device device = torch::Device(torch::kCUDA))
      : CudaSegmentTree<T, CudaSegmentTreeOp::Sum>(size, T(0), device) {}

  int64_t ScanLowerBound(const T& value) const {
    auto value_tensor =
        torch::tensor({value}, torch::TensorOptions()
                                   .dtype(utils::TorchDataType<T>::value)
                                   .device(this->values_.device()));
    return ScanLowerBound(value_tensor).cpu().template item<int64_t>();
  }

  torch::Tensor ScanLowerBound(const torch::Tensor& value) const {
    TORCH_CHECK(value.dtype() == utils::TorchDataType<T>::value,
                "value has an unexpected dtype");
    return CudaSumSegmentTreeScanLowerBound<T>(this->values_, this->size_,
                                               this->capacity_, value);
  }

  py::array_t<int64_t> ScanLowerBound(const py::array_t<T>& value) const {
    auto value_tensor =
        torch::from_blob(
            const_cast<T*>(value.data()), {value.size()},
            torch::TensorOptions().dtype(utils::TorchDataType<T>::value))
            .clone()
            .to(this->values_.device());
    auto cpu_index = ScanLowerBound(value_tensor).cpu().contiguous();
    py::array_t<int64_t> index = utils::NumpyEmptyLike<T, int64_t>(value);
    std::memcpy(index.mutable_data(), cpu_index.template data_ptr<int64_t>(),
                value.size() * sizeof(int64_t));
    return index;
  }
};

template <typename T>
class CudaMinSegmentTree final
    : public CudaSegmentTree<T, CudaSegmentTreeOp::Min> {
 public:
  explicit CudaMinSegmentTree(
      int64_t size, torch::Device device = torch::Device(torch::kCUDA))
      : CudaSegmentTree<T, CudaSegmentTreeOp::Min>(
            size, std::numeric_limits<T>::max(), device) {}
};

template <typename T>
void DefineCudaSumSegmentTree(const std::string& type, py::module& m) {
  const std::string pyclass = "CudaSumSegmentTree" + type;
  py::class_<CudaSumSegmentTree<T>, std::shared_ptr<CudaSumSegmentTree<T>>>(
      m, pyclass.c_str())
      .def(py::init<int64_t, torch::Device>(), py::arg("size"),
           py::arg("device") = torch::Device(torch::kCUDA))
      .def_property_readonly("size", &CudaSumSegmentTree<T>::size)
      .def_property_readonly("capacity", &CudaSumSegmentTree<T>::capacity)
      .def_property_readonly("identity_element",
                             &CudaSumSegmentTree<T>::identity_element)
      .def_property_readonly("device", &CudaSumSegmentTree<T>::device)
      .def("__len__", &CudaSumSegmentTree<T>::size)
      .def("__getitem__",
           py::overload_cast<int64_t>(&CudaSumSegmentTree<T>::At, py::const_))
      .def("__getitem__", py::overload_cast<const py::array_t<int64_t>&>(
                              &CudaSumSegmentTree<T>::At, py::const_))
      .def("__getitem__", py::overload_cast<const torch::Tensor&>(
                              &CudaSumSegmentTree<T>::At, py::const_))
      .def("at",
           py::overload_cast<int64_t>(&CudaSumSegmentTree<T>::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(
                     &CudaSumSegmentTree<T>::At, py::const_))
      .def("__setitem__",
           py::overload_cast<int64_t, const T&>(&CudaSumSegmentTree<T>::Update))
      .def("__setitem__", py::overload_cast<const torch::Tensor&, const T&>(
                              &CudaSumSegmentTree<T>::Update))
      .def("__setitem__",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &CudaSumSegmentTree<T>::Update))
      .def("update",
           py::overload_cast<int64_t, const T&>(&CudaSumSegmentTree<T>::Update))
      .def("update", py::overload_cast<const torch::Tensor&, const T&>(
                         &CudaSumSegmentTree<T>::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &CudaSumSegmentTree<T>::Update))
      .def("query", py::overload_cast<int64_t, int64_t>(
                        &CudaSumSegmentTree<T>::Query, py::const_))
      .def("query",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &CudaSumSegmentTree<T>::Query, py::const_))
      .def("scan_lower_bound",
           py::overload_cast<const T&>(&CudaSumSegmentTree<T>::ScanLowerBound,
                                       py::const_))
      .def("scan_lower_bound",
           py::overload_cast<const torch::Tensor&>(
               &CudaSumSegmentTree<T>::ScanLowerBound, py::const_))
      .def(py::pickle(
          [](const CudaSumSegmentTree<T>& s) {
            return py::make_tuple(s.DumpValues(), s.device());
          },
          [](const py::tuple& t) {
            TORCH_CHECK(t.size() == 2,
                        "invalid CudaSumSegmentTree pickle payload");
            const py::array_t<T>& arr = t[0].cast<py::array_t<T>>();
            auto device = t[1].cast<torch::Device>();
            CudaSumSegmentTree<T> s(arr.size(), device);
            s.LoadValues(arr);
            return s;
          }));
}

template <typename T>
void DefineCudaMinSegmentTree(const std::string& type, py::module& m) {
  const std::string pyclass = "CudaMinSegmentTree" + type;
  py::class_<CudaMinSegmentTree<T>, std::shared_ptr<CudaMinSegmentTree<T>>>(
      m, pyclass.c_str())
      .def(py::init<int64_t, torch::Device>(), py::arg("size"),
           py::arg("device") = torch::Device(torch::kCUDA))
      .def_property_readonly("size", &CudaMinSegmentTree<T>::size)
      .def_property_readonly("capacity", &CudaMinSegmentTree<T>::capacity)
      .def_property_readonly("identity_element",
                             &CudaMinSegmentTree<T>::identity_element)
      .def_property_readonly("device", &CudaMinSegmentTree<T>::device)
      .def("__len__", &CudaMinSegmentTree<T>::size)
      .def("__getitem__",
           py::overload_cast<int64_t>(&CudaMinSegmentTree<T>::At, py::const_))
      .def("__getitem__", py::overload_cast<const py::array_t<int64_t>&>(
                              &CudaMinSegmentTree<T>::At, py::const_))
      .def("__getitem__", py::overload_cast<const torch::Tensor&>(
                              &CudaMinSegmentTree<T>::At, py::const_))
      .def("at",
           py::overload_cast<int64_t>(&CudaMinSegmentTree<T>::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(
                     &CudaMinSegmentTree<T>::At, py::const_))
      .def("__setitem__",
           py::overload_cast<int64_t, const T&>(&CudaMinSegmentTree<T>::Update))
      .def("__setitem__", py::overload_cast<const torch::Tensor&, const T&>(
                              &CudaMinSegmentTree<T>::Update))
      .def("__setitem__",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &CudaMinSegmentTree<T>::Update))
      .def("update",
           py::overload_cast<int64_t, const T&>(&CudaMinSegmentTree<T>::Update))
      .def("update", py::overload_cast<const torch::Tensor&, const T&>(
                         &CudaMinSegmentTree<T>::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &CudaMinSegmentTree<T>::Update))
      .def("query", py::overload_cast<int64_t, int64_t>(
                        &CudaMinSegmentTree<T>::Query, py::const_))
      .def("query",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &CudaMinSegmentTree<T>::Query, py::const_))
      .def(py::pickle(
          [](const CudaMinSegmentTree<T>& s) {
            return py::make_tuple(s.DumpValues(), s.device());
          },
          [](const py::tuple& t) {
            TORCH_CHECK(t.size() == 2,
                        "invalid CudaMinSegmentTree pickle payload");
            const py::array_t<T>& arr = t[0].cast<py::array_t<T>>();
            auto device = t[1].cast<torch::Device>();
            CudaMinSegmentTree<T> s(arr.size(), device);
            s.LoadValues(arr);
            return s;
          }));
}

}  // namespace torchrl
