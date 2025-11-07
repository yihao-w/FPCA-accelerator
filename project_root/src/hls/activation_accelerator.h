#ifndef ACTIVATION_ACCELERATOR_H
#define ACTIVATION_ACCELERATOR_H

#include <cstdint>

// 尝试使用HLS官方的bfloat16支持
#include <cstdint>
// 先尝试包含hls_bfloat16.h，如果不存在则使用自定义实现
// Vitis HLS 2022.2 不支持 hls_bfloat16.h，使用自定义实现

// Data type definitions for C simulation
typedef uint16_t uint16;
typedef int32_t int32;
typedef int64_t int64;

// bf16位运算加法函数声明
uint16 bf16add(uint16 a_bits, uint16 b_bits);

// Function declaration - 使用uint16数据类型
void activation_accelerator(uint16* in0, uint16* in1, uint16* out, int32 stage, int32 config);

// Configuration definitions (7 operators required by contest)
#define CONFIG_ELTWISE_ADD 5     // Element-wise addition
#define CONFIG_SAFE_SOFTMAX 0    // Safe softmax activation function
#define CONFIG_SILU 1            // SiLU (Swish) activation function
#define CONFIG_RMS_NORM 2        // RMS normalization
#define CONFIG_LAYER_NORM 3      // Layer normalization
#define CONFIG_ELTWISE_MUL 6     // Element-wise multiplication
#define CONFIG_GELU 4            // GELU activation function

// Stage definitions
#define STAGE_LOAD 0      // Data loading stage
#define STAGE_COMPUTE 1   // Computation stage
#define STAGE_STORE 2     // Data storage stage

// Data size (64 x 768)
#define ROWS 64
#define FEATS 768
#define DATA_SIZE (ROWS * FEATS)

#endif // ACTIVATION_ACCELERATOR_H
