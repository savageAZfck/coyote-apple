// ffn.cu — matched workload vs bench/vs_cuda/ffn.cy
// Coyote: ffn_swiglu(x, w1, w2, w3) = (silu(x@w1) * (x@w3)) @ w2
// x[256]; w1,w3 declared f16[256,1024] → storage [1024,256] read
// as w[n*256+k]; w2 declared f16[1024,256] → storage [256,1024]
// read as w2[n*1024+k].
#include <cuda_runtime.h>
#include <stdio.h>
#include "init.h"

#define K1 256
#define N1 1024

__device__ float silu(float x) { return x / (1.f + expf(-x)); }

// one thread per hidden element: h = silu(x@w1) * (x@w3)
__global__ void ffn_up(const float *x, const float *w1, const float *w3,
                       float *h) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N1) return;
    float a = 0.f, b = 0.f;
    for (int k = 0; k < K1; k++) {
        a += x[k] * w1[n * K1 + k];
        b += x[k] * w3[n * K1 + k];
    }
    h[n] = silu(a) * b;
}

// one thread per output element: o = h @ w2, w2 stored [256,1024]
__global__ void ffn_down(const float *h, const float *w2, float *o) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= K1) return;
    float s = 0.f;
    for (int k = 0; k < N1; k++) s += h[k] * w2[n * N1 + k];
    o[n] = s;
}

int main(void) {
    float *hx = coyote_init("io::x", K1);
    float *hw1 = coyote_param("w1", K1 * N1);
    float *hw2 = coyote_param("w2", N1 * K1);
    float *hw3 = coyote_param("w3", K1 * N1);
    float *x, *w1, *w2, *w3, *h, *o;
    cudaMalloc(&x, K1 * 4);
    cudaMalloc(&w1, K1 * N1 * 4); cudaMalloc(&w3, K1 * N1 * 4);
    cudaMalloc(&w2, N1 * K1 * 4);
    cudaMalloc(&h, N1 * 4); cudaMalloc(&o, K1 * 4);
    cudaMemcpy(x, hx, K1 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w1, hw1, K1 * N1 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w2, hw2, N1 * K1 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w3, hw3, K1 * N1 * 4, cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int runs = 200;
    int t_up = (N1 + 255) / 256, t_dn = (K1 + 255) / 256;
    for (int i = 0; i < 5; i++) {
        ffn_up<<<t_up, 256>>>(x, w1, w3, h);
        ffn_down<<<t_dn, 256>>>(h, w2, o);
    }
    cudaEventRecord(t0);
    for (int i = 0; i < runs; i++) {
        ffn_up<<<t_up, 256>>>(x, w1, w3, h);
        ffn_down<<<t_dn, 256>>>(h, w2, o);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= runs;

    float hc[4];
    cudaMemcpy(hc, o, sizeof hc, cudaMemcpyDeviceToHost);
    printf("ffn %d->%d->%d swiglu: %.4f ms\n", K1, N1, K1, ms);
    printf("sample: %f %f %f %f\n", hc[0], hc[1], hc[2], hc[3]);
    return 0;
}
