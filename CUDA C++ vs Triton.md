CUDA C++: You write C++ code that is compiled into GPU machine instructions and executed on the NVIDIA GPU. The CUDA programming model provides direct access to GPU hardware features.



Triton: You write Python-like GPU kernel code, but the kernel itself does not execute in the Python interpreter. Triton's compiler translates it into GPU machine instructions (typically PTX, then GPU machine code) that execute on the NVIDIA GPU.
