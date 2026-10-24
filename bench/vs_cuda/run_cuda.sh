#!/bin/bash
# run_cuda.sh — build + run the CUDA side on an NVIDIA machine.
# Requires: nvcc, cuBLAS (CUDA toolkit). Fails loudly if absent.
set -euo pipefail
cd "$(dirname "$0")"
command -v nvcc >/dev/null || { echo "nvcc not found — no CUDA toolchain"; exit 1; }
nvcc -O2 -o gemm_cu gemm.cu -lcublas
nvcc -O2 -o attn_cu attn.cu
nvcc -O2 -o ffn_cu  ffn.cu
nvcc -O2 -o sgd_cu  sgd.cu
./gemm_cu && ./attn_cu && ./ffn_cu && ./sgd_cu
