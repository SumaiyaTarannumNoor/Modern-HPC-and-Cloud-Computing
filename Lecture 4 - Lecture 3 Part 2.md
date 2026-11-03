## Why Study GPU for understanding HPC
### Every frontier LLM was trained on tens of thousands of GPUs for months.
### Inference for those same models is served on GPUs. 
## A GPU researcher’s goal is to make matrix multiplication and attention operations approach peak hardware performance, which requires understanding the architectural principles behind the hardware’s design.
## The fundamental architectural principles are:
- CPUs minimize latency through large caches, out-of-order execution, and branch prediction.
- GPUs hide latency by keeping thousands of threads in flight, allowing schedulers to execute ready warps while others wait for memory.
- GPU architecture is fundamentally shaped by this emphasis on latency hiding.

