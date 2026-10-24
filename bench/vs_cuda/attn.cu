// attn.cu — matched workload vs bench/vs_cuda/attn.cy
// Coyote: attn(q,k,v) = softmax(q @ k.T) @ v, L=64 D=64.
// q @ k.T on a non-param arg is a true transpose:
//   scores[l,j] = softmax_j( sum_d q[l*D+d] * k[j*D+d] )
//   out[l,d]    = sum_j scores[l*L+j] * v[j*D+d]
// (named rhs v reads storage flat as v[j*K+d] — row-major here.)
// One thread per output row, matching coyote's naive emit strategy.
#include <cuda_runtime.h>
#include <stdio.h>
#include "init.h"

#define L 64
#define D 64

__global__ void attn_naive(const float *q, const float *k,
                           const float *v, float *o) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= L) return;
    float scores[L];
    float mx = -1e30f;
    for (int j = 0; j < L; j++) {
        float s = 0.f;
        for (int d = 0; d < D; d++) s += q[l * D + d] * k[j * D + d];
        scores[j] = s;
        if (s > mx) mx = s;
    }
    float sum = 0.f;
    for (int j = 0; j < L; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
    for (int j = 0; j < L; j++) scores[j] /= sum;
    for (int d = 0; d < D; d++) {
        float s = 0.f;
        for (int j = 0; j < L; j++) s += scores[j] * v[j * D + d];
        o[l * D + d] = s;
    }
}

int main(void) {
    float *hq = coyote_init("io::q", L * D);
    float *hk = coyote_init("io::k", L * D);
    float *hv = coyote_init("io::v", L * D);
    float *q, *k, *v, *o;
    size_t bytes = L * D * sizeof(float);
    cudaMalloc(&q, bytes); cudaMalloc(&k, bytes);
    cudaMalloc(&v, bytes); cudaMalloc(&o, bytes);
    cudaMemcpy(q, hq, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(k, hk, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v, hv, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int runs = 200;
    int threads = 64, blocks = (L + threads - 1) / threads;
    for (int i = 0; i < 5; i++) attn_naive<<<blocks, threads>>>(q, k, v, o);
    cudaEventRecord(t0);
    for (int i = 0; i < runs; i++) attn_naive<<<blocks, threads>>>(q, k, v, o);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= runs;

    float hc[4];
    cudaMemcpy(hc, o, sizeof hc, cudaMemcpyDeviceToHost);
    printf("attn L=%d D=%d naive: %.4f ms\n", L, D, ms);
    printf("sample: %f %f %f %f\n", hc[0], hc[1], hc[2], hc[3]);
    return 0;
}
