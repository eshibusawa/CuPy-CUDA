# Getting Started with CUDA using CuPy

This repository provides a series of Jupyter notebooks designed to help you get started with CUDA programming using CuPy.
Each notebook progressively introduces core CUDA concepts and demonstrates their practical application with CuPy's intuitive interface.

## Before Starting:
Before working through the notebooks, reviewing the following supplemental materials is recommended if one is unfamiliar with CUDA and parallel computation patterns:

* [Supplement: Introduction to CUDA Programming and GPU Architecture](./00_cuda_programming_and_gpu_architecture.md)
  (Offers a comprehensive explanation of GPU and CPU architectural differences, the CUDA programming model—including grids, blocks, threads, and warps—and the GPU memory hierarchy, providing essential context for understanding and optimizing practical CUDA applications.)

* [Supplement: Introduction to Basic and Advanced Parallel Computation Patterns](./00_basic_and_advanced_parallel_computation_patterns.md)
  (Provides an overview of four fundamental parallel patterns—map, stencil, reduction, and scan—and explains their classification as "basic" or "advanced.")

* [Supplement: Why and When to Use JIT vs. AOT](./00_why_and_when_to_use_jit.md)
  (Compares Just-In-Time (JIT) CUDA development with CuPy to Ahead-Of-Time (AOT) development using CUDA/C++ and CMake, helping readers understand CuPy's strengths and decide when each method is most appropriate for their GPU programming tasks.)

These supplements are intended to provide foundational background, helping readers understand the structure and objectives of each notebook more effectively.

# Notebooks
* [01_basic_array_operations.ipynb](./01_basic_array_operations.ipynb)
    This notebook explains basic CUDA kernel execution for the map pattern using CuPy's `RawModule`.
    It covers fundamental array operations, compile-time constants, and leveraging constant memory, setting the stage for more complex GPU patterns.

* [02_basic_texture_sampling.ipynb](./02_basic_texture_sampling.ipynb)
    Dive into texture memory, a powerful GPU tool optimized for 2D data like images. This notebook demonstrates efficient 2D data access and sampling using texture coordinates, built-in interpolation, and flexible addressing modes with CuPy.

* [03_stencil_patterns_with_shared_memory.ipynb](./03_stencil_patterns_with_shared_memory.ipynb)
    Explore the implementation of stencil patterns, focusing on performance-critical GPU memory concepts.
    This notebook details how to use Shared Memory for block-level cooperation and Warp-level cooperation with shuffle instructions for efficient inter-thread communication.

* [04_implementing_advanced_parallel_patterns_with_CUB.ipynb](./04_implementing_advanced_parallel_patterns_with_CUB.ipynb)
    Explore the implementation of advanced parallel patterns using the CUB library.
    This notebook demonstrates Reduction (e.g., vector normalization) and Scan (e.g., prefix sum), highlighting efficient memory access strategies (striped vs. direct loads) essential for GPU performance.

# Appendix: Practical Applications
The knowledge acquired from these notebooks is directly applicable to a wide range of real-world, high-performance computing problems.

Mastering Notebooks 01 (Basic Array Operations) and 02 (Basic Texture Sampling) provides the foundational skills for implementing high-speed stereo vision algorithms, such as [JBF-Stereo](https://github.com/eshibusawa/JBF-Stereo) and [VFS](https://github.com/eshibusawa/VFS-Python). These notebooks lay the groundwork for efficient data processing and precise memory access, which are critical for various real-time computer vision tasks.

Furthermore, an understanding of CUB's powerful prefix sum operations, as demonstrated in Notebook 04, equips users to tackle advanced applications like real-time object detection ([Stixel-World](https://github.com/eshibusawa/Stixel-World-Python)). This is exemplified in systems that derive object boundaries from depth information. The journey through CuPy, CUDA, and CUB significantly enhances the ability to develop high-performance solutions.
