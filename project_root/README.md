# Activation Accelerator Project / 激活函数加速器

本项目使用 Xilinx Vitis HLS/Vivado 在 FPGA 上实现常见激活函数（如 exp、erf、ReLU 变体）的硬件加速内核，并配套测试用例与自动化构建脚本。

## 特性 Highlights
- C/C++ HLS 设计与自测 testbench
- 查找表（LUT）加速的 exp/erf 等快速数学运算
- 一键式 HLS 构建脚本（run_hls.tcl）
- 清晰的目录组织，便于扩展软件侧与实验结果管理

## 目录结构 Repository Layout
```
project_root/
├─ src/                         Source code directory
│  ├─ hls/                      HLS kernels and testbench
│  │  ├─ activation_accelerator/        Accelerator core (submodules)
│  │  ├─ activation_accelerator.cpp     C source file (top)
│  │  ├─ activation_accelerator.h       C header
│  │  ├─ lut_erf.h                      C header (erf LUT)
│  │  ├─ lut_exp.h                      C header (exp LUT)
│  │  ├─ testbench.cpp                  C++ test file (self-checking)
│  │  ├─ run_hls.tcl                    Build script (Tcl)
│  │  └─ vitis_hls.log                  HLS build log (generated)
│  └─ rtl/                      RTL code (Verilog/VHDL)
│     ├─ vhdl/                  Accelerator core
│     ├─ activation_accelerator.bit                    
│     ├─ activation_accelerator.hwh                    
│     └─ run_vivado.tcl
├─ data/                        Test data
├─ results/                     Experimental results
│  └─ config_6/
│     └─ errors_point_per_row_config_6_round.csv  Example report
└─ README.md                    Project description
```

## 环境要求 Prerequisites
- Xilinx Vitis HLS 2024.2（或兼容版本）
- Xilinx Vivado（用于 RTL 综合/实现，与 Vitis HLS 版本匹配）
- C/C++ 编译器（gcc/clang 或 MSVC）
- Python 3（可选：用于数据/结果分析与可视化）

## 快速开始 Quick Start
1) 在 Windows 的 Vitis HLS 命令行环境中到达项目根目录，执行：
```
vitis_hls -f src\hls\run_hls.tcl
```
Tcl 脚本将自动：
- 创建 HLS 工程与 solution
- 运行 C-sim（使用 src/hls/testbench.cpp）
- 综合顶层 `activation_accelerator`
- （可选）运行 C/RTL co-sim
- （可选）导出 RTL

构建产物位于脚本创建的 HLS 工程目录（如 `hls_prj/`）。构建日志 `src/hls/vitis_hls.log` 会在运行后生成/更新。

2) 在 Vivado 中使用导出的 RTL（可选）
- 将 HLS 导出的 RTL 加入 Vivado 工程
- 根据需要添加约束（constraints/，如有）
- 运行综合/实现，生成比特流

## 数据与结果 Data and Results
- 测试数据放在 `data/`
- 结果与评估报告放在 `results/`，例如：
  - `results/config_6/errors_point_per_row_config_6_round.csv`
- 可在 `scripts/` 中添加数据生成（data generation）与结果分析脚本

## 开发提示 Development Notes
- 顶层 C 源文件：`src/hls/activation_accelerator.cpp`
- 头文件与 LUT：`src/hls/activation_accelerator.h`, `src/hls/lut_exp.h`, `src/hls/lut_erf.h`
- 测试文件：`src/hls/testbench.cpp`
- 构建脚本：`src/hls/run_hls.tcl`
- 顶层 RTL：`src/rtl/top_module.v`，子模块位于 `src/rtl/accelerator/`, `src/rtl/interface/`, `src/rtl/utils/`

## 常见问题 Troubleshooting
- 版本不匹配：确保 Vivado 与 Vitis HLS 版本一致
- Cosim 失败：检查接口 pragma 与 AXI 设置，必要时打开波形调试
- 时序收敛：尝试放宽时钟周期、增加流水/分区 pragma、或优化关键路径

## 代码风格 Coding Style
- C/C++：建议统一风格（如 LLVM/Google），可使用 clang-format
- RTL：模块自描述，接口/时序在注释或文档中说明

## 许可 License
请在根目录添加 `LICENSE`（如 MIT/Apache-2.0）。

## 维护 Maintainers
在此处填写维护者信息或通过 issue 联系
