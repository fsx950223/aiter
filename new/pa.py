from jinja2 import Template
import os


BUILD_DIR="./build"

function_template = Template("""
void paged_attention({{out_dtype}}* out,
                    void* workspace_buffer,
                    {{dtype}}* query_ptr,
                    {{kv_dtype}}* key_cache_ptr,
                    {{kv_dtype}}* value_cache_ptr,
                    float scale,
                    const int num_seqs,
                    const int q_stride,
                    const int kv_block_stride,
                    const int kv_head_stride,
                    const int kv_seq_stride,
                    int* kv_indptr_ptr,
                    int* kv_page_indices_ptr,
                    int* kv_last_page_lens_ptr,
                    const float* alibi_slopes_ptr,
                    float logits_soft_cap,
                    const float* k_scale_ptr,
                    const float* v_scale_ptr,
                    const float* fp8_out_scale_ptr,
                    hipStream_t stream)
""")

header_template = Template("""
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

{{function_define}};
""")


src_template = Template("""
#include <pa.h>

#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))

{{function_define}}
{
    constexpr int num_kv_heads = {{num_kv_heads}};
    constexpr int num_heads       = {{num_heads}};
    constexpr int head_size       = {{head_size}};
    constexpr int max_num_partitions = {{max_num_partitions}};
    constexpr int PARTITION_SIZE = 256;
    constexpr int gqa_ratio = num_heads / num_kv_heads;
    assert(num_heads % num_kv_heads == 0);

    float* exp_sums_ptr   = reinterpret_cast<float*>(workspace_buffer.data_ptr());
    float* max_logits_ptr = exp_sums_ptr + (num_seqs * num_heads * max_num_partitions);
    {{dtype}}* tmp_out_ptr =
        reinterpret_cast<T*>(max_logits_ptr + (num_seqs * num_heads * max_num_partitions));

    constexpr int NTHR = 256;
    dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
    dim3 block(NTHR);
    if(logits_soft_cap>0){
        paged_attention_ll4mi_QKV_mfma16_kernel<{{dtype}},                       
                                            {{kv_dtype}},                            {{fp8_kv_dtype}},                
                                            {{out_dtype}},                    
                                            {{block_size}},              
                                            head_size,               
                                            NTHR,                    
                                            {{alibi_enabled}},           
                                            true, 
                                            gqa_ratio>               
        <<<grid, block, 0, stream>>>(query_ptr,                      
                                     key_cache_ptr,                  
                                     value_cache_ptr,                
                                     scale,                          
                                     kv_indptr_ptr,                  
                                     kv_page_indices_ptr,            
                                     kv_last_page_lens_ptr,          
                                     alibi_slopes_ptr,               
                                     q_stride,                       
                                     kv_block_stride,                
                                     kv_head_stride,                 
                                     kv_seq_stride,                  
                                     exp_sums_ptr,                   
                                     max_logits_ptr,                 
                                     tmp_out_ptr,                    
                                     out_ptr,                        
                                     logits_soft_cap,                
                                     k_scale_ptr,                    
                                     v_scale_ptr,                    
                                     fp8_out_scale_ptr);
    }else{
       paged_attention_ll4mi_QKV_mfma16_kernel<{{dtype}},                       
                                            {{kv_dtype}},                                                        {{fp8_kv_dtype}},                
                                            {{out_dtype}},                    
                                            {{block_size}},              
                                            head_size,               
                                            NTHR,                    
                                            {{alibi_enabled}},           
                                            false, 
                                            gqa_ratio>               
        <<<grid, block, 0, stream>>>(query_ptr,                      
                                     key_cache_ptr,                  
                                     value_cache_ptr,                
                                     scale,                          
                                     kv_indptr_ptr,                  
                                     kv_page_indices_ptr,            
                                     kv_last_page_lens_ptr,          
                                     alibi_slopes_ptr,               
                                     q_stride,                       
                                     kv_block_stride,                
                                     kv_head_stride,                 
                                     kv_seq_stride,                  
                                     exp_sums_ptr,                   
                                     max_logits_ptr,                 
                                     tmp_out_ptr,                    
                                     out_ptr,                        
                                     logits_soft_cap,                
                                     k_scale_ptr,                    
                                     v_scale_ptr,                    
                                     fp8_out_scale_ptr);
    }
 
    dim3 reduce_grid(num_heads, num_seqs);
    dim3 reduce_block(head_size);
    constexpr int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, warpSize);
    paged_attention_ll4mi_reduce_kernel<{{dtype}}, {{out_dtype}}, head_size, head_size, PARTITION_SIZE, npar_loops, {{enable_last_page_lens}}> 
    <<<reduce_grid, reduce_block, 0, stream>>>(out_ptr,                                        
                                                exp_sums_ptr,        
                                                max_logits_ptr,                                 
                                                tmp_out_ptr,                                   
                                                kv_indptr_ptr,                                 
                                                kv_last_page_lens_ptr,                         
                                                {{block_size}},                                    
                                                max_num_partitions,                            
                                                fp8_out_scale_ptr);
}
""")


makefile_template = Template("""
CXX = hipcc
CXXFLAGS = -fPIC -O3 -std=c++17
INCLUDES = -Iater/csrc/include

TARGET = libpa.so
SRCS = pa.cpp pa.cu
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -shared $(OBJS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
""")

def init_build_dir():
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
    os.system(f"cp ./pa.cu {BUILD_DIR}/")

def compile(num_kv_heads, num_seqs, num_heads, head_size, max_num_partitions, dtype, kv_dtype, fp8_kv_dtype, out_dtype, block_size, alibi_enabled):
    # pass
    init_build_dir()
    header_file = header_template.render(function_define=function_template.render(out_dtype=out_dtype, dtype=dtype, kv_dtype=kv_dtype))
    # print(header_file)
    with open(f"{BUILD_DIR}/pa.h", "w") as f:
        f.write(header_file)
    src_file = src_template.render(num_kv_heads=num_kv_heads, num_seqs=num_seqs, num_heads=num_heads, head_size=head_size, max_num_partitions=max_num_partitions, dtype=dtype, kv_dtype=kv_dtype, fp8_kv_dtype=fp8_kv_dtype, out_dtype=out_dtype, block_size=block_size, alibi_enabled=alibi_enabled, enable_last_page_lens="true" if block_size > 1 else "false", function_define=function_template.render(out_dtype=out_dtype, dtype=dtype, kv_dtype=kv_dtype))
    with open(f"{BUILD_DIR}/pa.cpp", "w") as f:
        f.write(src_file)

    makefile_file = makefile_template.render()
    with open(f"{BUILD_DIR}/Makefile", "w") as f:
        f.write(makefile_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_kv_heads", type=int, required=True)
    parser.add_argument("--num_seqs", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--head_size", type=int, required=True)
    parser.add_argument("--max_num_partitions", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--kv_dtype", type=str, required=True)
    parser.add_argument("--fp8_kv_dtype", type=str, required=True)
    parser.add_argument("--out_dtype", type=str, required=True)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--alibi_enabled", type=str, required=True)
    args = parser.parse_args()
    compile(args.num_kv_heads, args.num_seqs, args.num_heads, args.head_size, args.max_num_partitions, args.dtype, args.kv_dtype, args.fp8_kv_dtype, args.out_dtype, args.block_size, args.alibi_enabled)