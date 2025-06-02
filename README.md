# Getting Started with CUDA using CuPy

This repository provides a series of Jupyter notebooks designed to help you get started with CUDA programming using CuPy.
Each notebook progressively introduces core CUDA concepts and demonstrates their practical application with CuPy's intuitive interface.

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

You might be wondering what you can actually build with the knowledge gained from these notebooks.
The skills you'll acquire are directly applicable to a wide range of real-world, high-performance computing problems.

By mastering just Notebooks 01 (Basic Array Operations) and 02 (Basic Texture Sampling), you'll gain the foundational skills to implement high-speed stereo vision algorithms ([JBF-Stereo](https://github.com/eshibusawa/JBF-Stereo) and [VFS](https://github.com/eshibusawa/VFS-Python)).
These notebooks lay the groundwork for efficient data processing and precise memory access, which are critical for various real-time computer vision tasks.
Furthermore, once you understand CUB's powerful prefix sum operations from Notebook 04, you'll be well-equipped to tackle advanced applications like real-time object detection ([Stixel-World](https://github.com/eshibusawa/Stixel-World-Python)), as seen in systems that derive object boundaries from depth information.
This journey through CuPy, CUDA, and CUB can significantly enhance your ability to develop high-performance solutions.
