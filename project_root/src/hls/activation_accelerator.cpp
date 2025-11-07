#include "activation_accelerator.h"
#include <iostream>
#include <cmath>
#include <hls_math.h>
#include <ap_int.h>
#include <hls_stream.h> 
#include "lut_exp.h"
#include "lut_erf.h"
// Packed AXI transfer type: 16 x bf16 (16-bit) per 256-bit word
typedef ap_uint<256> Pack256;
static const int PACK_ELEMS = 16;
static const float INV_FEATS = 1.0f / 768.0f;

// LUT查表函数
static inline float lut_exp(float x) {
    // Handle special values first
    if (x != x) return NAN;           // NaN input -> NaN output
    
    if (x <= EXP_LUT_MIN) return 0.0f;   // exp(<=-16) ~ 0
    if (x >= EXP_LUT_MAX) return 1.0f;   // 仅用于 softmax/silu 的非正区间上界 0

    const float INV_STEP = 31.936217f;
    float idx_f = (x - EXP_LUT_MIN) * INV_STEP;
    int idx = (int)idx_f;
    float ratio = idx_f - idx;


    return exp_lut[idx] + ratio * (exp_lut[idx + 1] - exp_lut[idx]);
}

static inline float lut_erf(float x) {
    // 处理特殊值
    if (x != x) return NAN;           // NaN输入
    
    if (x < ERF_LUT_MIN) return -1.0f;  // erf(<-4) ~ -1
    if (x > ERF_LUT_MAX) return 1.0f;   // erf(>4) ~ 1
    
    const float INV_ERF_LUT_MIN = 31.874541;
    float idx_f = (x - ERF_LUT_MIN) * INV_ERF_LUT_MIN;
    int idx = (int)idx_f;
    float ratio = idx_f - idx;
    
    return erf_lut[idx] + ratio * (erf_lut[idx + 1] - erf_lut[idx]);
}

// 固定上限常量，避免使用 HLS 不支持的变长数组（VLA）
static const int DATA_SIZE_MAX  = DATA_SIZE; // 64*768
static const int BLOCK_SIZE     = 256;
static const int UNROLL_FACTOR  = 16; // elementwise parallel lanes



// bf16 -> float single element
float bf16_to_float(uint16 in) {
    uint32_t x_f32 = ((uint32_t)in) << 16;
    return *(float*)&x_f32;
}

// float -> bf16 RNE (更精确的实现)
static inline uint16 float_to_bf16(float v) {
    // 特殊处理NaN，确保与PyTorch一致
    if (v != v) {  // 检查NaN
        return 0xFFFF;  // 使用与PyTorch一致的NaN表示
    }
    uint32_t u = *(uint32_t*)&v;
    uint32_t lsb  = (u >> 16) & 1u;
    uint32_t bias = 0x00007FFFu + lsb;
    u += bias;
    return (uint16)(u >> 16);
}

void activation_accelerator(uint16* in0, uint16* in1, uint16* out, int32 stage, int32 config) {
#pragma HLS INTERFACE m_axi port=in0 offset=slave bundle=gmem0 depth=49152
#pragma HLS INTERFACE m_axi port=in1 offset=slave bundle=gmem1 depth=49152
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem2 depth=49152
#pragma HLS INTERFACE s_axilite port=stage
#pragma HLS INTERFACE s_axilite port=config
#pragma HLS INTERFACE s_axilite port=return

    // --- Explicit Banking --- 
    // Instead of one large array, create N independent banks. This forces the synthesizer
    // to create parallel memory hardware, bypassing any tool-internal profitability checks.
    static uint16 buf0_banks[16][3072];
    static uint16 buf1_banks[16][3072];
    static uint16 buf2_banks[16][3072];

    // Partition the banks themselves completely, ensuring each bank is a distinct memory resource.
#pragma HLS ARRAY_PARTITION variable=buf0_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=buf1_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=buf2_banks complete dim=1

#pragma HLS bind_storage variable=buf1_banks type=ram_2p impl=uram


//#pragma HLS bind_storage variable=buf1_banks type=ram_2p impl=uram


    if(stage == 0) {
        // Packed LOAD: read 16 bf16 at a time and scatter to banks
        const Pack256* in0_p = (const Pack256*)in0;
        const Pack256* in1_p = (const Pack256*)in1;
        const int NUM_PACK = 3072;
        for (int p = 0; p < NUM_PACK; ++p) {
#pragma HLS PIPELINE II=1
            Pack256 w0 = in0_p[p];
            Pack256 w1 = in1_p[p];
            for (int j = 0; j < PACK_ELEMS; ++j) {
#pragma HLS UNROLL
                int idx = p * PACK_ELEMS + j;
                int bank_id = idx % UNROLL_FACTOR;
                int bank_addr = idx / UNROLL_FACTOR;
                ap_uint<16> e0 = w0.range(16*j + 15, 16*j);
                ap_uint<16> e1 = w1.range(16*j + 15, 16*j);
                buf0_banks[bank_id][bank_addr] = (uint16)e0;
                buf1_banks[bank_id][bank_addr] = (uint16)e1;
            }
        }
    }

    if(stage == 1) {
#pragma HLS DATAFLOW
        if(config == 0) {
            // Fully-optimized Softmax: Parallel reduction for both max_val and sum
            const int BLOCK_1 = 16;
            const int BLOCK = 32;
            const int NUM_REDUCERS = 4; // Use 4 parallel units for reduction

            for (int r = 0; r < ROWS; ++r) {
                int base = r * FEATS;
                
                // --- Pass 1: Find max_val using parallel tree reduction ---
                float partial_maxes[NUM_REDUCERS];
        #pragma HLS ARRAY_PARTITION variable=partial_maxes complete
                for (int a = 0; a < NUM_REDUCERS; ++a) {
        #pragma HLS UNROLL
                    partial_maxes[a] = -INFINITY;
                }

                bool has_nan = false;
                //bool has_inf = false;
                
                for (int i = 0; i < FEATS; i += BLOCK) {
        #pragma HLS PIPELINE II=1
                    float v_block[BLOCK];
        #pragma HLS ARRAY_PARTITION variable=v_block complete
                    // Load block
                    for (int b = 0; b < BLOCK; ++b) {
        #pragma HLS UNROLL
                        int idx = i + b;
                        float v = -INFINITY;
                        if (idx < FEATS) {
                            int global_idx = base + idx;
                            int bank_id = global_idx % UNROLL_FACTOR;
                            int bank_addr = global_idx / UNROLL_FACTOR;
                            v = bf16_to_float(buf0_banks[bank_id][bank_addr]);
                            if (v != v) has_nan = true;
                            //else if (v == INFINITY || v == -INFINITY) has_inf = true;
                        }
                        v_block[b] = v;
                    
                    }
                    // Block-level max reduction，32→8→4→1 比较树
                    if (!has_nan) {
                        // Level 1: 32 -> 8 (each is max over 4 elements)
                        float l8[8];
        #pragma HLS ARRAY_PARTITION variable=l8 complete
                        for (int k = 0; k < 8; ++k) {
        #pragma HLS UNROLL
                            float a0 = v_block[4*k + 0];
                            float a1 = v_block[4*k + 1];
                            float a2 = v_block[4*k + 2];
                            float a3 = v_block[4*k + 3];
                            float m0 = (a0 > a1) ? a0 : a1;
                            float m1 = (a2 > a3) ? a2 : a3;
                            l8[k] = (m0 > m1) ? m0 : m1;
                        }
                        // Level 2: 8 -> 4
                        float l4[4];
        #pragma HLS ARRAY_PARTITION variable=l4 complete
                        for (int k = 0; k < 4; ++k) {
        #pragma HLS UNROLL
                            float left  = l8[2*k + 0];
                            float right = l8[2*k + 1];
                            l4[k] = (left > right) ? left : right;
                        }
                        // Level 3: 4 -> 1
                        float l2_0 = (l4[0] > l4[1]) ? l4[0] : l4[1];
                        float l2_1 = (l4[2] > l4[3]) ? l4[2] : l4[3];
                        float pmax  = (l2_0 > l2_1) ? l2_0 : l2_1;

                        // Distribute to parallel reduction units
                        int reducer_idx = (i / BLOCK) % NUM_REDUCERS;
                        if (pmax > partial_maxes[reducer_idx]) {
                            partial_maxes[reducer_idx] = pmax;
                        }
                    }
                }

                // Final reduction of partial maxes，比较四路中，谁为最大值
                float max_val = -INFINITY;
                for (int a = 0; a < NUM_REDUCERS; ++a) {
        #pragma HLS UNROLL
                    if (partial_maxes[a] > max_val) max_val = partial_maxes[a];
                }
                // --- Pass 2: Compute sum using parallel tree accumulator ---
                float exp_cache[FEATS];
        #pragma HLS ARRAY_PARTITION variable=exp_cache cyclic factor=16 dim=1

                float partial_sums[NUM_REDUCERS];//将累加模块赋0值
        #pragma HLS ARRAY_PARTITION variable=partial_sums complete
                for (int a = 0; a < NUM_REDUCERS; ++a) {
        #pragma HLS UNROLL
                    partial_sums[a] = 0.0f;
                }
                
                for (int i = 0; i < FEATS; i += BLOCK_1) {
        #pragma HLS PIPELINE II=1
                    float v_block[BLOCK_1];
                    float e_block[BLOCK_1];
        #pragma HLS ARRAY_PARTITION variable=v_block complete
        #pragma HLS ARRAY_PARTITION variable=e_block complete

                    for (int b = 0; b < BLOCK_1; ++b) {
        #pragma HLS UNROLL factor=4
                        int idx = i + b;
                        float v = 0.0f;
                        //if (idx < FEATS) {
                            int global_idx = base + idx;
                            int bank_id = global_idx % UNROLL_FACTOR;
                            int bank_addr = global_idx / UNROLL_FACTOR;
                            v = bf16_to_float(buf0_banks[bank_id][bank_addr]);
                        //}
                        v_block[b] = v;
                    }

                    // Use local accumulators to break dependency and reduce hardware complexity
                    // Two-level adder tree: 16 elems -> 8 partials -> 4 partials -> psum
                    float psum_local8[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        #pragma HLS ARRAY_PARTITION variable=psum_local8 complete

                    for (int b = 0; b < BLOCK_1; ++b) {
        #pragma HLS UNROLL factor=8
                        float v = v_block[b];
                        float e = 0.0f;
                        int idx = i + b;
                        e = lut_exp(v - max_val);
                        exp_cache[idx] = e;
                        psum_local8[b % 8] += e; // distribute into 8 local lanes
                        e_block[b] = e;
                    }
                    // Level 1: 8 -> 4
                    float psum_lvl1[4];
        #pragma HLS ARRAY_PARTITION variable=psum_lvl1 complete
                    for (int k = 0; k < 4; ++k) {
#pragma HLS UNROLL
                        psum_lvl1[k] = psum_local8[2*k] + psum_local8[2*k + 1];
                    }
                    // Level 2: 4 -> 1
                    float psum = (psum_lvl1[0] + psum_lvl1[1]) + (psum_lvl1[2] + psum_lvl1[3]);
                    int reducer_idx = (i / BLOCK_1) % NUM_REDUCERS;
                    partial_sums[reducer_idx] += psum;
                }
                
                float sum = 0.0f;
                for (int a = 0; a < NUM_REDUCERS; ++a) {
        #pragma HLS UNROLL
                    sum += partial_sums[a];
                }
                float inv_sum = 1.0f / sum;

                // --- Pass 3: Normalize and write back ---
                for (int i = 0; i < FEATS; i += 32) {
        #pragma HLS PIPELINE II=1
                    for (int b = 0; b < 32; ++b) {
        #pragma HLS UNROLL
                        int idx = i + b;
                        //if (idx < FEATS) {
                            float e = exp_cache[idx];
                            float out_f = e * inv_sum;
                            int global_idx = base + idx;
                            int bank_id = global_idx % UNROLL_FACTOR;
                            int bank_addr = global_idx / UNROLL_FACTOR;
                            buf2_banks[bank_id][bank_addr] = float_to_bf16(out_f);
                        //}
                    }
                }
            }
        }
        else if(config == 1) {
			// Banked parallel SiLU - each unrolled lane accesses its dedicated bank
			const int GELU_UNROLL = 16;
			const int NUM_ITERS = 3072;  
			
			for (int i = 0; i < NUM_ITERS; ++i) {
#pragma HLS PIPELINE II=1
				for (int k = 0; k < GELU_UNROLL; ++k) {
#pragma HLS UNROLL
					// Calculate global index
					int global_idx = i * GELU_UNROLL + k;
					int bank_id = global_idx % UNROLL_FACTOR;
					int bank_addr = global_idx / UNROLL_FACTOR;
					
					
                    float xi = bf16_to_float(buf0_banks[bank_id][bank_addr]);
					
					// SiLU computation (stable piecewise, LUT only on non-positive domain)
					float sig;
					if (xi >= 0.0f) {
						float exp_neg_xi = lut_exp(-xi);
						sig = 1.0f / (1.0f + exp_neg_xi);
					} else {
						float ex = lut_exp(xi);
						sig = ex / (1.0f + ex);
					}
					float val = xi * sig;
					
					// 使用统一的bf16转换函数

                    uint32_t u = *(uint32_t*)&val;
					uint32_t lsb = (u >> 16) & 1u;
					uint32_t bias = 0x00007FFFu + lsb;
					u += bias;
					
					buf2_banks[bank_id][bank_addr] = (uint16)(u >> 16);

					//buf2_banks[bank_id][bank_addr] = float_to_bf16(val);
				}
			}
		}
        
        else if(config == 2) {
            // 优化的RMSNorm: 使用树形累加器并行计算sum_sq
            const int BLOCK = 32;
            const float eps = 1e-5f;

            for (int r = 0; r < ROWS; ++r) {
                int base = r * FEATS;

                // Pass 1: 使用树形累加器并行计算sum_sq
                const int NUM_ACCUMULATORS = 4;  // 4个并行累加器
                float partial_sums[NUM_ACCUMULATORS];
                float partial_sums_sq[NUM_ACCUMULATORS];
#pragma HLS ARRAY_PARTITION variable=partial_sums complete
#pragma HLS ARRAY_PARTITION variable=partial_sums_sq complete
                for (int a = 0; a < NUM_ACCUMULATORS; ++a) {
#pragma HLS UNROLL
                    partial_sums[a] = 0.0f;
                    partial_sums_sq[a] = 0.0f;
                }

                for (int i = 0; i < FEATS; i += BLOCK) {
#pragma HLS PIPELINE II=1
                    float v_block[BLOCK];
#pragma HLS ARRAY_PARTITION variable=v_block complete
                    // 加载块
                    for (int b = 0; b < BLOCK; ++b) {
#pragma HLS UNROLL
                        int idx = i + b;
                        float v = 0.0f;
                        //if (idx < FEATS) {
                            int global_idx = base + idx;
                            int bank_id = global_idx % UNROLL_FACTOR;
                            int bank_addr = global_idx / UNROLL_FACTOR;
                            v = bf16_to_float(buf0_banks[bank_id][bank_addr]);
                        //}
                        v_block[b] = v;
                    }
                    
                    // 块级sum_sq计算: 使用本地累加器打破依赖
                    // Explicit 32 -> 16 -> 8 -> 4 -> 2 -> 1 adder tree for sum of squares
                    float psq_l1[16];
#pragma HLS ARRAY_PARTITION variable=psq_l1 complete
                    for (int k = 0; k < 16; ++k) {
#pragma HLS UNROLL
                        float v1 = v_block[2*k];
                        float v2 = v_block[2*k + 1];
                        psq_l1[k]  = v1*v1 + v2*v2;
                    }

                    float psq_l2[8];
#pragma HLS ARRAY_PARTITION variable=psq_l2 complete
                    for (int k = 0; k < 8; ++k) {
#pragma HLS UNROLL
                        psq_l2[k]  = psq_l1[2*k] + psq_l1[2*k + 1];
                    }

                    float psq_l3[4];
#pragma HLS ARRAY_PARTITION variable=psq_l3 complete
                    for (int k = 0; k < 4; ++k) {
#pragma HLS UNROLL
                        psq_l3[k]  = psq_l2[2*k] + psq_l2[2*k + 1];
                    }

                    float psq_l4[2];
#pragma HLS ARRAY_PARTITION variable=psq_l4 complete
                    psq_l4[0]  = psq_l3[0] + psq_l3[1];
                    psq_l4[1]  = psq_l3[2] + psq_l3[3];

                    float psum_sq = psq_l4[0] + psq_l4[1];
                    
                    // 分配到不同的累加器 (轮询方式)
                    int acc_idx = (i / BLOCK) % NUM_ACCUMULATORS;
                    partial_sums_sq[acc_idx] += psum_sq;
                }

                // 树形归约: 合并部分和
                float sum_sq = 0.0f;
                for (int a = 0; a < NUM_ACCUMULATORS; ++a) {
#pragma HLS UNROLL
                    sum_sq += partial_sums_sq[a];
                }


                // ✅ 优化：预计算倒数，用乘法替代除法
                float mean_sq = sum_sq * INV_FEATS;
                // ✅ 优化: 使用hls::rsqrt替代sqrt+div，延迟从~56 cycles降至~15
                float inv_rms = hls::rsqrt(mean_sq + eps);

                // Pass 2: 归一化并写回
                for (int i = 0; i < FEATS; i += BLOCK) {
#pragma HLS PIPELINE II=1
                    float v_block[BLOCK];
#pragma HLS ARRAY_PARTITION variable=v_block complete
                    // 加载块
                    for (int b = 0; b < BLOCK; ++b) {
#pragma HLS UNROLL
                        int idx = i + b;
                        float v = 0.0f;
                        //if (idx < FEATS) {
                            int global_idx = base + idx;
                            int bank_id = global_idx % UNROLL_FACTOR;
                            int bank_addr = global_idx / UNROLL_FACTOR;
                            v = bf16_to_float(buf0_banks[bank_id][bank_addr]);
                        //}
                        v_block[b] = v;
                    }
                    
                    // 计算并存储
                    for (int b = 0; b < BLOCK; ++b) {
#pragma HLS UNROLL
                        int idx = i + b;
                        if (idx < FEATS) {
                            float out_f = v_block[b] * inv_rms;
                            int global_idx = base + idx;
                            int bank_id = global_idx % UNROLL_FACTOR;
                            int bank_addr = global_idx / UNROLL_FACTOR;

                            buf2_banks[bank_id][bank_addr] = float_to_bf16(out_f);
                        }
                    }
                }
            }
        }
        else if(config == 3) {
            // Optimized LayerNorm with tree accumulator
            const int BLOCK = 32;
            const float eps = 1e-6f;
            const int NUM_ACCUMULATORS = 2;
        
            for (int r = 0; r < ROWS; ++r) {
                int base = r * FEATS;

                const int NUM_ACCUMULATORS = 4;  // 4个并行累加器
                float partial_sums[NUM_ACCUMULATORS];
                float partial_sums_sq[NUM_ACCUMULATORS];
#pragma HLS ARRAY_PARTITION variable=partial_sums complete
#pragma HLS ARRAY_PARTITION variable=partial_sums_sq complete
                for (int a = 0; a < NUM_ACCUMULATORS; ++a) {
#pragma HLS UNROLL
                    partial_sums[a] = 0.0f;
                    partial_sums_sq[a] = 0.0f;
                }

                for (int i = 0; i < FEATS; i += BLOCK) {
#pragma HLS PIPELINE II=1
                    float v_block[BLOCK];
#pragma HLS ARRAY_PARTITION variable=v_block complete
                    // 加载块
                    for (int b = 0; b < BLOCK; ++b) {
#pragma HLS UNROLL
                        int idx = i + b;
                        float v = 0.0f;
                        if (idx < FEATS) {
                            int global_idx = base + idx;
                            int bank_id = global_idx % UNROLL_FACTOR;
                            int bank_addr = global_idx / UNROLL_FACTOR;
                            v = bf16_to_float(buf0_banks[bank_id][bank_addr]);
                        }
                        v_block[b] = v;
                    }
                    
                    // 块级sum_sq计算: 使用本地累加器打破依赖
                    float psum_local[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    float psum_sq_local[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma HLS ARRAY_PARTITION variable=psum_local complete
#pragma HLS ARRAY_PARTITION variable=psum_sq_local complete

                    for (int b = 0; b < 32; ++b) { // BLOCK is 32
#pragma HLS UNROLL
                        float v = v_block[b];
                        int local_idx = b % 8;
                        psum_local[local_idx] += v;
                        psum_sq_local[local_idx] += v * v;
                    }

                    float psum = (psum_local[0] + psum_local[1]) + (psum_local[2] + psum_local[3]) + (psum_local[4] + psum_local[5]) + (psum_local[6] + psum_local[7]);
                    float psum_sq = (psum_sq_local[0] + psum_sq_local[1]) + (psum_sq_local[2] + psum_sq_local[3]) + (psum_sq_local[4] + psum_sq_local[5]) + (psum_sq_local[6] + psum_sq_local[7]);
                    
                    // 分配到不同的累加器 (轮询方式)
                    int acc_idx = (i / BLOCK) % NUM_ACCUMULATORS;
                    partial_sums[acc_idx] += psum;
                    partial_sums_sq[acc_idx] += psum_sq;
                }

                // 树形归约: 合并部分和
                float sum = 0.0f;
                float sum_sq = 0.0f;
                for (int a = 0; a < NUM_ACCUMULATORS; ++a) {
#pragma HLS UNROLL
                    sum += partial_sums[a];
                    sum_sq += partial_sums_sq[a];
                }


                // ✅ 优化：预计算倒数，用乘法替代除法
                //const float INV_FEATS = 1.0f / 768.0f;
                float mean = sum * INV_FEATS;
                float mean_sq = sum_sq * INV_FEATS;
                float var = mean_sq - mean * mean;
                float denom = hls::sqrtf(var + eps);

                        float inv_std = 1.0f / denom;
            
                    // Pass 2: normalize and write back
                    for (int i = 0; i < FEATS; i += BLOCK) {
#pragma HLS PIPELINE II=1
                        float v_block[BLOCK];
#pragma HLS ARRAY_PARTITION variable=v_block complete
                        for (int b = 0; b < BLOCK; ++b) {
#pragma HLS UNROLL
                            int idx = i + b;
                            float v = 0.0f;
                            if (idx < FEATS) {
                                int global_idx = base + idx;
                                int bank_id = global_idx % UNROLL_FACTOR;
                                int bank_addr = global_idx / UNROLL_FACTOR;
                                v = bf16_to_float(buf0_banks[bank_id][bank_addr]);
                            }
                            v_block[b] = v;
                        }
                        for (int b = 0; b < BLOCK; ++b) {
#pragma HLS UNROLL
                            int idx = i + b;
                            if (idx < FEATS) {
                                //float out_f = (v_block[b] - mean) * inv_std;
                                float out_f = (v_block[b] - mean) * inv_std;
                                //if (out_f == -0.0f) out_f = 0.0f;
                                int global_idx = base + idx;
                                int bank_id = global_idx % UNROLL_FACTOR;
                                int bank_addr = global_idx / UNROLL_FACTOR;
                                buf2_banks[bank_id][bank_addr] = float_to_bf16(out_f);
                            }
                        }
                    }
            }
        }
		else if(config == 4) {
			// True 8-way parallel GELU
			const int GELU_UNROLL = 6;
			const int NUM_ITERS = 8192;  
			
			for (int i = 0; i < NUM_ITERS; ++i) {
#pragma HLS PIPELINE II=1
				for (int k = 0; k < GELU_UNROLL; ++k) {
#pragma HLS UNROLL
					// Calculate global index
					int global_idx = i * GELU_UNROLL + k;
					int bank_id = global_idx % UNROLL_FACTOR;
					int bank_addr = global_idx / UNROLL_FACTOR;
					
					// Read from bank
					/*uint16 xi_bf16 = buf0_banks[bank_id][bank_addr];
					
					// Inline bf16_to_float_single
					uint32_t xi_f32 = ((uint32_t)xi_bf16) << 16;
					float xi = *(float*)&xi_f32;*/
                    float xi = bf16_to_float(buf0_banks[bank_id][bank_addr]);
					// GELU computation
					float inv_sqrt2_f = 0.70710678f; // 1/sqrt(2)
					float erf_arg = xi * inv_sqrt2_f;
					float erf_val = lut_erf(erf_arg);
					float gelu_f = 0.5f * xi * (1.0f + erf_val);
				
					
					// Write back to bank
					buf2_banks[bank_id][bank_addr] = float_to_bf16(gelu_f);

				}
			}
		}
        else if(config == 5) {
            // Banked parallel addition - each unrolled lane accesses its dedicated bank
            const int GELU_UNROLL = 32;
			const int NUM_ITERS = 1536;

			//for (int i = 0; i < 3072; i++) {
            for (int i = 0; i < NUM_ITERS; i++) {
#pragma HLS PIPELINE II=1
				for (int k = 0; k < GELU_UNROLL; ++k) {
#pragma HLS UNROLL
                    int global_idx = i * GELU_UNROLL + k;
                    int bank_id = global_idx % UNROLL_FACTOR;
                    int bank_addr = global_idx / UNROLL_FACTOR;

                    float a = bf16_to_float(buf0_banks[bank_id][bank_addr]);
                    float b = bf16_to_float(buf1_banks[bank_id][bank_addr]);

                    // Float addition
                    float sum = a + b;

                    
                    buf2_banks[bank_id][bank_addr] = float_to_bf16(sum);
                }
            }
        }

		else if(config == 6) {
			// Banked parallel multiply - each unrolled lane accesses its dedicated bank
            const int GELU_UNROLL = 16;
			const int NUM_ITERS = 3072;

			//for (int i = 0; i < 3072; i++) {
            for (int i = 0; i < NUM_ITERS; i++) {
#pragma HLS PIPELINE II=1
				for (int k = 0; k < GELU_UNROLL; ++k) {
#pragma HLS UNROLL
                    int global_idx = i * GELU_UNROLL + k;
                    int bank_id = global_idx % UNROLL_FACTOR;
                    int bank_addr = global_idx / UNROLL_FACTOR;

                    float a = bf16_to_float(buf0_banks[bank_id][bank_addr]);
                    float b = bf16_to_float(buf1_banks[bank_id][bank_addr]);

					// Float multiplication
					float mul = a * b;

		
                    buf2_banks[bank_id][bank_addr] = float_to_bf16(mul);
				}
			}
		}
    }
    if(stage == 2) {
        // Packed STORE: gather from banks and write 16 bf16 at a time
        Pack256* out_p = (Pack256*)out;
        //const int NUM_PACK = DATA_SIZE / PACK_ELEMS;
        for (int p = 0; p < 3072; ++p) {
#pragma HLS PIPELINE II=1
            Pack256 w = 0;
            for (int j = 0; j < PACK_ELEMS; ++j) {
#pragma HLS UNROLL
                int idx = p * PACK_ELEMS + j;
                int bank_id = idx % UNROLL_FACTOR;
                int bank_addr = idx / UNROLL_FACTOR;
                ap_uint<16> e = (ap_uint<16>)buf2_banks[bank_id][bank_addr];
                w.range(16*j + 15, 16*j) = e;
            }
            out_p[p] = w;
        }
    }
}
