// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>

#include "segment_tree.h"

namespace py = pybind11;

PYBIND11_MODULE(_torchrl, m) {
  py::class_<torchrl::SumSegmentTree<float>,
             std::shared_ptr<torchrl::SumSegmentTree<float>>>(m,
                                                              "SumSegmentTree")
      .def(py::init<int64_t>())
      .def_property_readonly("size", &torchrl::SumSegmentTree<float>::size)
      .def_property_readonly("capacity",
                             &torchrl::SumSegmentTree<float>::capacity)
      .def_property_readonly("identity_element",
                             &torchrl::SumSegmentTree<float>::identity_element)
      .def("__len__", &torchrl::SumSegmentTree<float>::size)
      .def("__getitem__", py::overload_cast<int64_t>(
                              &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("__getitem__", py::overload_cast<const py::array_t<int64_t>&>(
                              &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("__getitem__", py::overload_cast<const torch::Tensor&>(
                              &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("at", py::overload_cast<int64_t>(&torchrl::SumSegmentTree<float>::At,
                                            py::const_))
      .def("at", py::overload_cast<const py::array_t<int64_t>&>(
                     &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(
                     &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("__setitem__", py::overload_cast<int64_t, const float&>(
                              &torchrl::SumSegmentTree<float>::Update))
      .def("__setitem__",
           py::overload_cast<const py::array_t<int64_t>&, const float&>(
               &torchrl::SumSegmentTree<float>::Update))
      .def("__setitem__", py::overload_cast<const py::array_t<int64_t>&,
                                            const py::array_t<float>&>(
                              &torchrl::SumSegmentTree<float>::Update))
      .def("__setitem__", py::overload_cast<const torch::Tensor&, const float&>(
                              &torchrl::SumSegmentTree<float>::Update))
      .def("__setitem__",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::SumSegmentTree<float>::Update))
      .def("update", py::overload_cast<int64_t, const float&>(
                         &torchrl::SumSegmentTree<float>::Update))
      .def("update",
           py::overload_cast<const py::array_t<int64_t>&, const float&>(
               &torchrl::SumSegmentTree<float>::Update))
      .def("update", py::overload_cast<const py::array_t<int64_t>&,
                                       const py::array_t<float>&>(
                         &torchrl::SumSegmentTree<float>::Update))
      .def("update", py::overload_cast<const torch::Tensor&, const float&>(
                         &torchrl::SumSegmentTree<float>::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::SumSegmentTree<float>::Update))
      .def("query", py::overload_cast<int64_t, int64_t>(
                        &torchrl::SumSegmentTree<float>::Query, py::const_))
      .def("query", py::overload_cast<const py::array_t<int64_t>&,
                                      const py::array_t<int64_t>&>(
                        &torchrl::SumSegmentTree<float>::Query, py::const_))
      .def("query",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::SumSegmentTree<float>::Query, py::const_))
      .def("scan_lower_bound",
           py::overload_cast<const float&>(
               &torchrl::SumSegmentTree<float>::ScanLowerBound, py::const_))
      .def("scan_lower_bound",
           py::overload_cast<const py::array_t<float>&>(
               &torchrl::SumSegmentTree<float>::ScanLowerBound, py::const_))
      .def("scan_lower_bound",
           py::overload_cast<const torch::Tensor&>(
               &torchrl::SumSegmentTree<float>::ScanLowerBound, py::const_))
      .def(py::pickle(
          [](const SumSegmentTree<float>& s) {
            return py::make_tuple(s.DumpValues());
          },
          [](const py::tuple& t) {
            assert(t.size() == 1);
            const py::array_t<T>& arr = t[0].cast<py::array_t<T>>();
            SumSegmentTree<T> s(arr.size());
            s.LoadValues(arr);
            return s;
          }));

  py::class_<torchrl::MinSegmentTree<float>,
             std::shared_ptr<torchrl::MinSegmentTree<float>>>(m,
                                                              "MinSegmentTree")
      .def(py::init<int64_t>())
      .def_property_readonly("size", &torchrl::MinSegmentTree<float>::size)
      .def_property_readonly("capacity",
                             &torchrl::MinSegmentTree<float>::capacity)
      .def_property_readonly("identity_element",
                             &torchrl::MinSegmentTree<float>::identity_element)
      .def("__len__", &torchrl::MinSegmentTree<float>::size)
      .def("__getitem__", py::overload_cast<int64_t>(
                              &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("__getitem__", py::overload_cast<const py::array_t<int64_t>&>(
                              &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("__getitem__", py::overload_cast<const torch::Tensor&>(
                              &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("at", py::overload_cast<int64_t>(&torchrl::MinSegmentTree<float>::At,
                                            py::const_))
      .def("at", py::overload_cast<const py::array_t<int64_t>&>(
                     &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(
                     &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("__setitem__", py::overload_cast<int64_t, const float&>(
                              &torchrl::MinSegmentTree<float>::Update))
      .def("__setitem__",
           py::overload_cast<const py::array_t<int64_t>&, const float&>(
               &torchrl::MinSegmentTree<float>::Update))
      .def("__setitem__", py::overload_cast<const py::array_t<int64_t>&,
                                            const py::array_t<float>&>(
                              &torchrl::MinSegmentTree<float>::Update))
      .def("__setitem__", py::overload_cast<const torch::Tensor&, const float&>(
                              &torchrl::MinSegmentTree<float>::Update))
      .def("__setitem__",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::MinSegmentTree<float>::Update))
      .def("update", py::overload_cast<int64_t, const float&>(
                         &torchrl::MinSegmentTree<float>::Update))
      .def("update",
           py::overload_cast<const py::array_t<int64_t>&, const float&>(
               &torchrl::MinSegmentTree<float>::Update))
      .def("update", py::overload_cast<const py::array_t<int64_t>&,
                                       const py::array_t<float>&>(
                         &torchrl::MinSegmentTree<float>::Update))
      .def("update", py::overload_cast<const torch::Tensor&, const float&>(
                         &torchrl::MinSegmentTree<float>::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::MinSegmentTree<float>::Update))
      .def("query", py::overload_cast<int64_t, int64_t>(
                        &torchrl::MinSegmentTree<float>::Query, py::const_))
      .def("query", py::overload_cast<const py::array_t<int64_t>&,
                                      const py::array_t<int64_t>&>(
                        &torchrl::MinSegmentTree<float>::Query, py::const_))
      .def("query",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::MinSegmentTree<float>::Query, py::const_))
      .def(py::pickle(
          [](const MinSegmentTree<float>& s) {
            return py::make_tuple(s.DumpValues());
          },
          [](const py::tuple& t) {
            assert(t.size() == 1);
            const py::array_t<T>& arr = t[0].cast<py::array_t<T>>();
            MinSegmentTree<T> s(arr.size());
            s.LoadValues(arr);
            return s;
          }));
}
