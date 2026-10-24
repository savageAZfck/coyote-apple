// gemm.cu — matched workload vs bench/vs_cuda/gemm.cy
// Coyote semantics: out[m,n] = sum_k a[m*K+k] * b[n*K+k]
// (named rhs reads storage as [N,K] — coyote's convention).
// Naive one-thread-per-element kernel = coyote's emit strategy,
// plus cuBLAS sgemm for "what NVIDIA actually ships" context.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "init.h"

#define N 512

__global__ void gemm_naive(const float *a, const float *b, float *c, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n * n) return;
    int m = gid / n, col = gid % n;
    float s = 0.f;
    for (int k = 0; k < n; k++) s += a[m * n + k] * b[col * n + k];
    c[gid] = s;
}

int main(void) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *ha = coyote_init("io::a", (size_t)N * N);
    float *hb = coyote_init("io::b", (size_t)N * N);
    float *a, *b, *c;
    cudaMalloc(&a, bytes); cudaMalloc(&b, bytes); cudaMalloc(&c, bytes);
    cudaMemcpy(a, ha, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int runs = 50, threads = 256, blocks = (N * N + threads - 1) / threads;

    // naive — apples-to-apples with coyote's emitted kernel
    for (int i = 0; i < 3; i++) gemm_naive<<<blocks, threads>>>(a, b, c, N);
    cudaEventRecord(t0);
    for (int i = 0; i < runs; i++) gemm_naive<<<blocks, threads>>>(a, b, c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float naive_ms;
    cudaEventElapsedTime(&naive_ms, t0, t1);
    naive_ms /= runs;

    // cuBLAS — the maturity gap, reported for context. Perf
    // reference only: cuBLAS is column-major, so this computes the
    // transposed-equivalent GEMM, not coyote's parity layout.
    cublasHandle_t h;
    cublasCreate(&h);
    float one = 1.f, zero = 0.f;
    for (int i = 0; i < 3; i++)
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, a, N, b, N, &zero, c, N);
    cudaEventRecord(t0);
    for (int i = 0; i < runs; i++)
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, a, N, b, N, &zero, c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float blas_ms;
    cudaEventElapsedTime(&blas_ms, t0, t1);
    blas_ms /= runs;

    float hc[8];
    cudaMemcpy(hc, c, sizeof hc, cudaMemcpyDeviceToHost);
    double gflop_n = 2.0 * N * N * N / (naive_ms * 1e6);
    double gflop_b = 2.0 * N * N * N / (blas_ms * 1e6);
    printf("gemm %dx%d naive: %.3f ms (%.1f GFLOP/s)\n", N, N, naive_ms, gflop_n);
    printf("gemm %dx%d cublas: %.3f ms (%.1f GFLOP/s)\n", N, N, blas_ms, gflop_b);
    printf("sample: %f %f %f %f\n", hc[0], hc[1], hc[2], hc[3]);
    return 0;
}
