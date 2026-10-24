// init.h — replicates coyote's init_weight_pub(name, numel) exactly:
// xorshift64 seeded by FNV-1a over the io name, so CUDA and Coyote
// see byte-identical input tensors.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float *coyote_init(const char *name, size_t numel) {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++) {
        s ^= *p;
        s *= 0x100000001B3ULL;
    }
    s ^= (uint64_t)numel;
    float *v = (float *)malloc(numel * sizeof(float));
    for (size_t i = 0; i < numel; i++) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v[i] = (float)((double)s / (double)UINT64_MAX - 0.5);
    }
    return v;
}

// param fallback in eval is also init_weight("param::<name>", numel)
static float *coyote_param(const char *name, size_t numel) {
    char buf[128];
    snprintf(buf, sizeof buf, "param::%s", name);
    return coyote_init(buf, numel);
}
