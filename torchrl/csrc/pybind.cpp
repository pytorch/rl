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
      .def("__len__", &torchrl::SumSegmentTree<float>::size)
      .def("size", &torchrl::SumSegmentTree<float>::size)
      .def("capacity", &torchrl::SumSegmentTree<float>::capacity)
      .def("identity_element",
           &torchrl::SumSegmentTree<float>::identity_element)
      .def("__getitem__", py::overload_cast<int64_t>(
                              &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("__getitem__", py::overload_cast<const py::array_t<int64_t> &>(
                              &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("__getitem__", py::overload_cast<const torch::Tensor&>(
                              &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("at", py::overload_cast<int64_t>(&torchrl::SumSegmentTree<float>::At,
                                            py::const_))
      .def("at", py::overload_cast<const py::array_t<int64_t> &>(
                     &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(
                     &torchrl::SumSegmentTree<float>::At, py::const_))
      .def("__setitem__", py::overload_cast<int64_t, const float &>(
                              &torchrl::SumSegmentTree<float>::Update))
      .def("__setitem__",
           py::overload_cast<const py::array_t<int64_t> &, const float &>(
               &torchrl::SumSegmentTree<float>::Update))
      .def("__setitem__", py::overload_cast<const py::array_t<int64_t> &,
                                            const py::array_t<float> &>(
                              &torchrl::SumSegmentTree<float>::Update))
      .def("__setitem__", py::overload_cast<const torch::Tensor&, const float &>(
                              &torchrl::SumSegmentTree<float>::Update))
      .def("__setitem__",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::SumSegmentTree<float>::Update))
      .def("update", py::overload_cast<int64_t, const float &>(
                         &torchrl::SumSegmentTree<float>::Update))
      .def("update",
           py::overload_cast<const py::array_t<int64_t> &, const float &>(
               &torchrl::SumSegmentTree<float>::Update))
      .def("update", py::overload_cast<const py::array_t<int64_t> &,
                                       const py::array_t<float> &>(
                         &torchrl::SumSegmentTree<float>::Update))
      .def("update", py::overload_cast<const torch::Tensor&, const float &>(
                         &torchrl::SumSegmentTree<float>::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::SumSegmentTree<float>::Update))
      .def("query", py::overload_cast<int64_t, int64_t>(
                        &torchrl::SumSegmentTree<float>::Query, py::const_))
      .def("query", py::overload_cast<const py::array_t<int64_t> &,
                                      const py::array_t<int64_t> &>(
                        &torchrl::SumSegmentTree<float>::Query, py::const_))
      .def("query",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::SumSegmentTree<float>::Query, py::const_))
      .def("scan_lower_bound",
           py::overload_cast<const float &>(
               &torchrl::SumSegmentTree<float>::ScanLowerBound, py::const_))
      .def("scan_lower_bound",
           py::overload_cast<const py::array_t<float> &>(
               &torchrl::SumSegmentTree<float>::ScanLowerBound, py::const_))
      .def("scan_lower_bound",
           py::overload_cast<const torch::Tensor&>(
               &torchrl::SumSegmentTree<float>::ScanLowerBound, py::const_));

  py::class_<torchrl::MinSegmentTree<float>,
             std::shared_ptr<torchrl::MinSegmentTree<float>>>(m,
                                                              "MinSegmentTree")
      .def(py::init<int64_t>())
      .def("__len__", &torchrl::MinSegmentTree<float>::size)
      .def("size", &torchrl::MinSegmentTree<float>::size)
      .def("capacity", &torchrl::MinSegmentTree<float>::capacity)
      .def("identity_element",
           &torchrl::MinSegmentTree<float>::identity_element)
      .def("__getitem__", py::overload_cast<int64_t>(
                              &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("__getitem__", py::overload_cast<const py::array_t<int64_t> &>(
                              &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("__getitem__", py::overload_cast<const torch::Tensor&>(
                              &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("at", py::overload_cast<int64_t>(&torchrl::MinSegmentTree<float>::At,
                                            py::const_))
      .def("at", py::overload_cast<const py::array_t<int64_t> &>(
                     &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(
                     &torchrl::MinSegmentTree<float>::At, py::const_))
      .def("__setitem__", py::overload_cast<int64_t, const float &>(
                              &torchrl::MinSegmentTree<float>::Update))
      .def("__setitem__",
           py::overload_cast<const py::array_t<int64_t> &, const float &>(
               &torchrl::MinSegmentTree<float>::Update))
      .def("__setitem__", py::overload_cast<const py::array_t<int64_t> &,
                                            const py::array_t<float> &>(
                              &torchrl::MinSegmentTree<float>::Update))
      .def("__setitem__", py::overload_cast<const torch::Tensor&, const float &>(
                              &torchrl::MinSegmentTree<float>::Update))
      .def("__setitem__",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::MinSegmentTree<float>::Update))
      .def("update", py::overload_cast<int64_t, const float &>(
                         &torchrl::MinSegmentTree<float>::Update))
      .def("update",
           py::overload_cast<const py::array_t<int64_t> &, const float &>(
               &torchrl::MinSegmentTree<float>::Update))
      .def("update", py::overload_cast<const py::array_t<int64_t> &,
                                       const py::array_t<float> &>(
                         &torchrl::MinSegmentTree<float>::Update))
      .def("update", py::overload_cast<const torch::Tensor&, const float &>(
                         &torchrl::MinSegmentTree<float>::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::MinSegmentTree<float>::Update))
      .def("query", py::overload_cast<int64_t, int64_t>(
                        &torchrl::MinSegmentTree<float>::Query, py::const_))
      .def("query", py::overload_cast<const py::array_t<int64_t> &,
                                      const py::array_t<int64_t> &>(
                        &torchrl::MinSegmentTree<float>::Query, py::const_))
      .def("query",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &torchrl::MinSegmentTree<float>::Query, py::const_));
}

