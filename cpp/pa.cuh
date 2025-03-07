#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h> 
#include "hip_compat.h"

#include <algorithm>
#include "dtype_fp8.cuh"
#include "quant_utils.cuh"

template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          typename OUTT,
          int BLOCK_SIZE,
          int HEAD_SIZE,
          int NUM_THREADS,
          bool ALIBI_ENABLED,
          bool LOGITS_SOFT_CAP_ENABLED,
          int GQA_RATIO>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,      // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache, // [num_blocks, num_kv_heads,
                                         // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache, // [num_blocks, num_kv_heads,
                                         // head_size, block_size]
    const float scale,
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_page_indices,   // [max_num_blocks]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const float* __restrict__ alibi_slopes,    // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int kv_seq_stride,
    float* __restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits, // [num_seqs, num_heads,
                                    // max_num_partitions]
    scalar_t* __restrict__ out,     // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    OUTT* __restrict__ final_out,   // [num_seqs, num_heads, head_size]
    float logits_soft_cap,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    const float* __restrict__ fp8_out_scale_ptr);


template <typename scalar_t,
          typename OUTT,
          int HEAD_SIZE,
          int NUM_THREADS,
          int PARTITION_SIZE,
          int NPAR_LOOPS, bool ENABLE_LAST_PAGE_LENS>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                    // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,        // [num_seqs, num_heads,
                                               // max_num_partitions]
    const float* __restrict__ max_logits,      // [num_seqs, num_heads,
                                               // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,      // [num_seqs, num_heads,
                                               // max_num_partitions, head_size]
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const int block_size,
    const int max_num_partitions,
    const float* __restrict__ fp8_out_scale_ptr);

    