// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("paged_attention_rocm", &paged_attention,
            py::arg("out"), py::arg("exp_sums"),
            py::arg("max_logits"), py::arg("tmp_out"),
            py::arg("query"), py::arg("key_cache"),
            py::arg("value_cache"), py::arg("scale"),
            py::arg("kv_indptr"), py::arg("kv_indices"),
            py::arg("block_size"), py::arg("num_kv_splits"),
            py::arg("alibi_slopes"), py::arg("kv_cache_dtype"),
            py::arg("kv_cache_layout"), py::arg("k_scale"),
            py::arg("v_scale"), py::arg("fp8_out_scale"),
            py::arg("partition_size"), py::arg("logit_cap")=0);
}