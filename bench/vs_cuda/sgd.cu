// sgd.cu — matched workload vs bench/vs_cuda/sgd.cy
// Coyote (4 lines):
//   w1 -= 0.001 * grad(mse(gelu(x @ w1) @ w2, y), w1);
//   w2 -= 0.001 * grad(mse(gelu(x @ w1) @ w2, y), w2);
//   ret gelu(x @ w1) @ w2
// x[64]; w1 [64,256] decl → storage [256,64]; w2 [256,64] decl →
// storage [64,256]. Sequential statement semantics: w2's grad is
// evaluated AFTER w1's update — each backward recomputes the
// forward on current weights, exactly like coyote's update kernels.
#include <cuda_runtime.h>
#include <stdio.h>
#include "init.h"

#define K1 64
#define N1 256
#define LR 0.001f

__device__ float gelu_f(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
__device__ float gelu_d(float x) {
    float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    float t = tanhf(u);
    return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
}

// forward pass into shared scratch: z1,h,adj. Returns adj via out.
__global__ void fwd(const float *x, const float *w1, const float *w2,
                    const float *y, float *z1, float *h, float *adj) {
    int k = threadIdx.x;              // K1=64 threads: cover N1 via loop
    for (int n = k; n < N1; n += K1) {
        float s = 0.f;
        for (int j = 0; j < K1; j++) s += x[j] * w1[n * K1 + j];
        z1[n] = s;
        h[n] = gelu_f(s);
    }
    __syncthreads();
    if (k < K1) {
        float p = 0.f;
        for (int j = 0; j < N1; j++) p += h[j] * w2[k * N1 + j];
        adj[k] = 2.f * (p - y[k]) / K1;   // mse seed
    }
}

// grad wrt w1 (rhs leaf of x @ w1): dW1[n,j] = dpre1[n] * x[j]
//   where dh[n] = sum_i adj[i]*w2[i*N1+n], dpre1[n]=dh[n]*gelu'(z1[n])
// one thread per w1 element; w1 updated in place.
__global__ void upd_w1(const float *x, const float *w2, const float *z1,
                       const float *adj, float *w1) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N1) return;
    float dh = 0.f;
    for (int i = 0; i < K1; i++) dh += adj[i] * w2[i * N1 + n];
    float dp = dh * gelu_d(z1[n]);
    for (int j = 0; j < K1; j++) w1[n * K1 + j] -= LR * dp * x[j];
}

// grad wrt w2 (rhs leaf of h @ w2): dW2[n,k] = adj[n] * h[k]
// one thread per w2 element; w2 updated in place.
__global__ void upd_w2(const float *h, const float *adj, float *w2) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= K1 * N1) return;
    int n = gid / N1, k = gid % N1;
    w2[gid] -= LR * adj[n] * h[k];
}

int main(void) {
    float *hx = coyote_init("io::x", K1);
    float *hy = coyote_init("io::y", K1);
    float *hw1 = coyote_param("w1", N1 * K1);
    float *hw2 = coyote_param("w2", K1 * N1);
    float *x, *y, *w1, *w2, *z1, *h, *adj, *o;
    cudaMalloc(&x, K1 * 4); cudaMalloc(&y, K1 * 4);
    cudaMalloc(&w1, N1 * K1 * 4); cudaMalloc(&w2, K1 * N1 * 4);
    cudaMalloc(&z1, N1 * 4); cudaMalloc(&h, N1 * 4);
    cudaMalloc(&adj, K1 * 4); cudaMalloc(&o, K1 * 4);
    cudaMemcpy(x, hx, K1 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy, K1 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w1, hw1, N1 * K1 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w2, hw2, K1 * N1 * 4, cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int runs = 200;
    int w1blocks = (N1 + 255) / 256, w2blocks = (K1 * N1 + 255) / 256;

    // coyote ordering: upd w1 → upd w2 (recomputed fwd) → ret fwd
    for (int i = 0; i < 5; i++) {
        fwd<<<1, K1>>>(x, w1, w2, y, z1, h, adj);
        upd_w1<<<w1blocks, 256>>>(x, w2, z1, adj, w1);
        fwd<<<1, K1>>>(x, w1, w2, y, z1, h, adj);
        upd_w2<<<w2blocks, 256>>>(h, adj, w2);
    }
    cudaEventRecord(t0);
    for (int i = 0; i < runs; i++) {
        fwd<<<1, K1>>>(x, w1, w2, y, z1, h, adj);
        upd_w1<<<w1blocks, 256>>>(x, w2, z1, adj, w1);
        fwd<<<1, K1>>>(x, w1, w2, y, z1, h, adj);
        upd_w2<<<w2blocks, 256>>>(h, adj, w2);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= runs;

    // final forward for the sample — h holds gelu(z1) post-update
    fwd<<<1, K1>>>(x, w1, w2, y, z1, h, adj);
    float hc[4];
    cudaMemcpy(hc, h, sizeof hc, cudaMemcpyDeviceToHost);
    printf("sgd 2-layer %d->%d->%d: %.4f ms/step (5 kernels vs coyote's 3)\n",
           K1, N1, K1, ms);
    printf("hidden sample: %f %f %f %f\n", hc[0], hc[1], hc[2], hc[3]);
    return 0;
}
