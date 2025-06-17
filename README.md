# CUDA Programming Tutorial

## Project Overview

This repository contains CUDA programming exercises, experiments, and mini-projects, structured to build expertise in parallel computing, GPU programming, and real-time graphics acceleration.


## Topics Covered

- CUDA Development Environment (WSL2 / Ubuntu / NVCC)
- Parallel Execution Model (Threads, Blocks, Grids)
- CUDA Memory Hierarchy (Global, Shared, Constant, Registers)
- Image Processing with CUDA Kernels
- Parallel Algorithms (Prefix Sum, Reduction, Sorting)
- Basic Ray Tracing and Volume Rendering on GPU
- Performance Optimization Strategies


## Repository Structure

```
/cuda_practice_basics/
    ├── 01_hello_cuda/            # CUDA kernel launch fundamentals
    ├── 02_vector_addition/       # Grid and block configuration
    ├── 03_memory_hierarchy/      # Memory model experiments
    ├── 04_image_filtering/       # 2D image convolution kernels
    ├── 05_parallel_prefix_sum/   # Parallel scan implementation
    ├── 06_ray_tracing_basics/    # Simple CUDA-based ray tracer
```

Each directory contains:
- Source Code (.cu files)
- Build Instructions
- Experimental Results (logs, images, performance measurements)

## Environment

- OS: WSL2 Ubuntu 22.04
- CUDA Toolkit: 12.4+
- GPU: RTX 5090
- Compiler: nvcc

