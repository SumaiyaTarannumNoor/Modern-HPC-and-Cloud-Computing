## Fat Binary
- A fat binary (or multiarchitecture binary) is a computer executable program or library which has been expanded (or "fattened") with code native to multiple instruction sets which can consequently be run on multiple processor types. This results in a file larger than a normal one-architecture binary file, thus the name.
## PTX
- PTX (Parallel Thread eXecution) is NVIDIA's intermediate representation (IR) for GPU programs.
 Portable Virtual Instructions
- PTX defines a hierarchy of threads, thread blocks, and grids, corresponding to CUDA's execution model.
## SASS
- Streaming Assembler (SASS) is the lowest-level, human-readable assembly language for programs running on NVIDIA GPUs. Architecture-specific instructions executed by NVIDIA GPUs.
## PTX Instructions
### Declare floating-point registers
- Declares seven virtual 32-bit floating-point registers (%f0–%f6).
- The compiler maps these virtual registers onto physical GPU registers.
### Fused Multiply-Add (FMA)
- Computes %f5 = (%f4 × %f3) + 1.5.
- fma: Multiplication and addition with one final rounding.
- rn: Round to nearest, ties to even.
- f32: Single-precision (32-bit) floating point.
### Access thread and block identifiers
- %ctaid.x: Block index along the x-axis.
- %ntid.x: Number of threads per block along the x-axis.
- %tid.x: Thread index within the block.
### Global thread index along x:
- Global ID= %r1 X %r2 + %r3
## Thread
- Executes its own instruction sequence.
- Maintains private registers and local execution state.
## Thread Block
- Contains multiple threads assigned to one SM.
- Threads cooperate through shared memory and synchronization.
## Grid
- Contains multiple thread blocks executing the same kernel.
- Blocks can execute concurrently or sequentially across SMs.












 
