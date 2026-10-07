# Software Stack & Toolkits for FlashAttention Research

## Core Programming Languages & DSLs
* **CUDA C++** (Foundational API for low-level GPU programming, shared memory allocation, and warp-level synchronization)
* **Triton** (Python-based DSL by OpenAI, used to implement alternative FlashAttention kernels via block-level programming paradigms)
* **CuTe DSL** (NVIDIA's layout-centric Python DSL within CUTLASS, used to write FlashAttention-4 kernels for accelerated compilation)

## Hardware-Level Libraries & Runtimes
* **NVIDIA CUTLASS** (Collection of CUDA C++ template abstractions for high-performance matrix multiplication and tensor layouts)
* **NVIDIA cuDNN** (Deep learning primitive library providing highly optimized vendor-native implementations of multi-head attention)
* **AMD Composable Kernel** (C++ template library used as the structural backend to port FlashAttention onto ROCm AMD architectures)

## Deep Learning Frameworks & Build Tools
* **PyTorch** (Primary ecosystem orchestration framework used to expose and manage `flash_attn` function hooks)
* **Ninja** (Small build system used for rapid Just-In-Time compilation of custom CUDA extensions directly in Python environments)
