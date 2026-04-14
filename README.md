# EC527 Final Project: High-Performance Sorting

**Course:** EC527 — High Performance Computing with Multicore and GPUs  
**Team:** Cynthia Young & Phyliss Darko  
**Institution:** Boston University  
**Due:** April 30, 2026

---

## Overview

This project implements and optimizes LSD radix sort across multiple architectures, applying performance engineering techniques from EC527: cache optimization, SIMD/AVX intrinsics, OpenMP multicore parallelism, and CUDA GPU programming.

**Central question:** How much can we speed up sorting, and what works best on which architecture?

---

## Algorithm

**LSD Radix Sort** — sorts 32-bit unsigned integers by processing 8 bits at a time across 4 passes (256 buckets per pass). No comparisons are ever made. Each pass follows three steps:

1. **Count** — tally how many elements fall in each bucket
2. **Scan** — exclusive prefix sum to compute starting output positions
3. **Scatter** — place each element at its correct output position

Input distributions tested: random, sorted (ascending), reverse sorted (descending).  
Validation: output compared element-by-element against C standard library `qsort`.

---

## Repository Structure

```
EC527---Project/
├── CPU/
│   └── radix_sort_cpu.c      # CPU serial baseline - Phyliss branches from here
├── GPU/
│   └── radix_sort_gpu.cu     # GPU baseline - Cynthia branches from here
├── Reference/
│   ├── radix_sort_cpu.c      # shared serial reference (do not modify)
│   └── radix_sort_gpu.cu     # shared GPU scaffold (do not modify)
└── README.md
```

---

## Building

**CPU version** (run on lab machines):
```bash
gcc -O1 radix_sort_cpu.c -lrt -o radix_sort_cpu
./radix_sort_cpu
```

**GPU version** (run on BU SCC with A40 GPU):
```bash
module load cuda/12.8
nvcc -arch=compute_86 -code=sm_86 radix_sort_gpu.cu -o radix_sort_gpu
./radix_sort_gpu
```

> Before running, set `CPNS` in both files to match one's machine's clock speed (`lscpu` will show it).

---

## Array Sizes Tested

| Size | Elements |
|------|----------|
| 1M   | 1,000,000 |
| 2M   | 2,000,000 |
| 4M   | 4,000,000 |
| 8M   | 8,000,000 |
| 16M  | 16,000,000 |
| 32M  | 32,000,000 |
| 64M  | 64,000,000 |

---

## Output Format

**CPU:** `dist, size, time_sec, cycles, GB_per_sec, valid`

**GPU:** `dist, size, t_transfer_sec, t_compute_sec, t_total_sec, compute_cycles, compute_GB_per_sec, valid`

Output is comma-separated for direct paste into Google Sheets.

---

## Optimizations (in progress)

| Optimization | Target | Status |
|---|---|---|
| Serial baseline | CPU | done |
| Cache-friendly pass structure | CPU | in progress |
| SIMD/AVX counting | CPU | in progress |
| OpenMP multicore | CPU | in progress |
| CUDA count/scan/scatter kernels | GPU | in progress |
| Block/grid tuning | GPU | in progress |

---

## Responsibilities

| Area | Owner |
|---|---|
| Serial reference code | shared |
| CPU optimizations + benchmarking | Phyliss |
| GPU CUDA kernels + benchmarking | Cynthia |
| Slides + writeup | shared |

---

## Presentation & Demo

| Resource | Link |
|---|---|
| Slides | *(add link when uploaded)* |
| Video demo | *(add link if recorded)* |
