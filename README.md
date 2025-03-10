# **Optimizing Artificial Neural Networks for High Performance**  

## Overview 

This project focuses on the **performance analysis and optimization** of a Python-based artificial neural network (ANN) implementation designed for **multi-class image classification**. The original implementation, sourced from **Philip Mocz’s GitHub repository** ([pmocz/artificialneuralnetwork-python](https://github.com/pmocz/artificialneuralnetwork-python)), classifies **galaxy images** from the **Sloan Digital Sky Survey (SDSS) and the Galaxy Zoo project** datasets.  

My work explores various **performance optimization techniques** for artificial neural networks, applying strategies such as **profiling, memory layout optimization, Cython acceleration, GPU computation, and parallelization**. The goal is to **reduce execution time** and **improve computational efficiency** by leveraging modern hardware capabilities.  

This project is originally designed as part of the *DD2358 - Introduction to High Performance Computing* course at KTH for my master give by **Stefano Markidis**.

## Repository Structure

- `default/` – Original ANN implementation before optimization.  
- `profiling/` – Performance profiling to identify computational bottlenecks.  
- `data_layout_opti/` – Optimization through memory-efficient data structures.  
- `cython_opti/` – Performance acceleration using **Cython** for compiled execution.  
- `gpu_opti/` – GPU acceleration via **CUDA/PyTorch** for faster training.  
- `parallelize_opti/` – Attempted **parallelization** (limited improvements due to Python GIL).  
- `summary/` – Comparison of all optimization techniques.  

Each folder contains a modified `artificialneuralnetwork.py` file with the applied optimizations.  

## Running the Code

To run any optimized version:  

```bash
python [folder_name]/artificialneuralnetwork.py --max_iter 300
```

For benchmarking, uncomment:  

```python
mean_time, std_dev = stdTime(5)  # Runs the program multiple times to compute execution time statistics
```

## Performance Analysis & Results  

### Performance Analysis & Results
- **Profiling** helped identify major bottlenecks in matrix operations and function calls.  
- **Memory layout changes** (e.g., using row-major order) improved cache efficiency.  
- **Cython optimizations** reduced Python overhead by compiling performance-critical code to C.  
- **GPU acceleration** significantly improved execution speed for large-scale computations.  
- **Parallelization efforts** showed limited benefits due to the Global Interpreter Lock (GIL) in Python. 

## Project Report & Visualizations

The **full analysis report** (`DD2358___Project_Report.pdf`) details the optimization strategies, performance benchmarks, and results. Additionally, `profiling.ipynb` contains **profiling visualizations** to illustrate computational bottlenecks.  


