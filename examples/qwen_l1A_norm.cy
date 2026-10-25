// qwen28.cy — a full 28-layer Qwen3-0.6B decode, the organism way.
// 28 programs on the chip, each with its own instanced KV cache;
// the decode loop — gather, dispatch, argmax, witness — is all
// host code. every token generated is witnessed by the organism.
//
//   coyote run examples/qwen28.cy -o out/qwen28
//
// needs models/qwen3-0.6b/model.safetensors (HF checkpoint).
// generated from qwenorg.cy — per-layer params/fns, shared state spec.

param lnf: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.norm.weight"
param lmh: frozen f16[1024, 151936] @ "models/qwen3-0.6b/model.safetensors::model.embed_tokens.weight"
table emb: f16[151936, 1024] @ "models/qwen3-0.6b/model.safetensors::model.embed_tokens.weight"

state kv {
    k0lo: f16[64, 64]
    k0hi: f16[64, 64]
    v0lo: f16[64, 64]
    v0hi: f16[64, 64]
    k1lo: f16[64, 64]
    k1hi: f16[64, 64]
    v1lo: f16[64, 64]
    v1hi: f16[64, 64]
    k2lo: f16[64, 64]
    k2hi: f16[64, 64]
    v2lo: f16[64, 64]
    v2hi: f16[64, 64]
    k3lo: f16[64, 64]
    k3hi: f16[64, 64]
    v3lo: f16[64, 64]
    v3hi: f16[64, 64]
    k4lo: f16[64, 64]
    k4hi: f16[64, 64]
    v4lo: f16[64, 64]
    v4hi: f16[64, 64]
    k5lo: f16[64, 64]
    k5hi: f16[64, 64]
    v5lo: f16[64, 64]
    v5hi: f16[64, 64]
    k6lo: f16[64, 64]
    k6hi: f16[64, 64]
    v6lo: f16[64, 64]
    v6hi: f16[64, 64]
    k7lo: f16[64, 64]
    k7hi: f16[64, 64]
    v7lo: f16[64, 64]
    v7hi: f16[64, 64]
}

param l0_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.input_layernorm.weight"
param l0_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.post_attention_layernorm.weight"
param l0_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_norm.weight[0:64]"
param l0_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_norm.weight[64:128]"
param l0_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_norm.weight[0:64]"
param l0_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_norm.weight[64:128]"
param l0_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[0:64]"
param l0_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,0:64]"
param l0_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[64:128]"
param l0_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,64:128]"
param l0_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[128:192]"
param l0_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,128:192]"
param l0_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[192:256]"
param l0_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,192:256]"
param l0_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[256:320]"
param l0_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,256:320]"
param l0_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[320:384]"
param l0_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,320:384]"
param l0_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[384:448]"
param l0_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,384:448]"
param l0_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[448:512]"
param l0_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,448:512]"
param l0_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[512:576]"
param l0_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,512:576]"
param l0_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[576:640]"
param l0_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,576:640]"
param l0_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[640:704]"
param l0_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,640:704]"
param l0_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[704:768]"
param l0_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,704:768]"
param l0_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[768:832]"
param l0_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,768:832]"
param l0_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[832:896]"
param l0_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,832:896]"
param l0_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[896:960]"
param l0_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,896:960]"
param l0_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[960:1024]"
param l0_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,960:1024]"
param l0_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1024:1088]"
param l0_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1024:1088]"
param l0_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1088:1152]"
param l0_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1088:1152]"
param l0_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1152:1216]"
param l0_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1152:1216]"
param l0_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1216:1280]"
param l0_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1216:1280]"
param l0_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1280:1344]"
param l0_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1280:1344]"
param l0_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1344:1408]"
param l0_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1344:1408]"
param l0_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1408:1472]"
param l0_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1408:1472]"
param l0_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1472:1536]"
param l0_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1472:1536]"
param l0_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1536:1600]"
param l0_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1536:1600]"
param l0_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1600:1664]"
param l0_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1600:1664]"
param l0_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1664:1728]"
param l0_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1664:1728]"
param l0_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1728:1792]"
param l0_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1728:1792]"
param l0_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1792:1856]"
param l0_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1792:1856]"
param l0_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1856:1920]"
param l0_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1856:1920]"
param l0_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1920:1984]"
param l0_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1920:1984]"
param l0_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1984:2048]"
param l0_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1984:2048]"
param l0_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[0:64]"
param l0_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[0:64]"
param l0_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[64:128]"
param l0_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[64:128]"
param l0_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[128:192]"
param l0_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[128:192]"
param l0_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[192:256]"
param l0_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[192:256]"
param l0_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[256:320]"
param l0_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[256:320]"
param l0_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[320:384]"
param l0_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[320:384]"
param l0_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[384:448]"
param l0_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[384:448]"
param l0_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[448:512]"
param l0_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[448:512]"
param l0_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[512:576]"
param l0_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[512:576]"
param l0_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[576:640]"
param l0_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[576:640]"
param l0_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[640:704]"
param l0_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[640:704]"
param l0_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[704:768]"
param l0_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[704:768]"
param l0_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[768:832]"
param l0_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[768:832]"
param l0_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[832:896]"
param l0_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[832:896]"
param l0_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[896:960]"
param l0_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[896:960]"
param l0_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[960:1024]"
param l0_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[960:1024]"
param l0_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.gate_proj.weight"
param l0_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.up_proj.weight"
param l0_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.down_proj.weight"

param l1_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.input_layernorm.weight"
param l1_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.post_attention_layernorm.weight"
param l1_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_norm.weight[0:64]"
param l1_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_norm.weight[64:128]"
param l1_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_norm.weight[0:64]"
param l1_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_norm.weight[64:128]"
param l1_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[0:64]"
param l1_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,0:64]"
param l1_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[64:128]"
param l1_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,64:128]"
param l1_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[128:192]"
param l1_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,128:192]"
param l1_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[192:256]"
param l1_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,192:256]"
param l1_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[256:320]"
param l1_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,256:320]"
param l1_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[320:384]"
param l1_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,320:384]"
param l1_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[384:448]"
param l1_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,384:448]"
param l1_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[448:512]"
param l1_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,448:512]"
param l1_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[512:576]"
param l1_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,512:576]"
param l1_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[576:640]"
param l1_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,576:640]"
param l1_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[640:704]"
param l1_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,640:704]"
param l1_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[704:768]"
param l1_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,704:768]"
param l1_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[768:832]"
param l1_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,768:832]"
param l1_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[832:896]"
param l1_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,832:896]"
param l1_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[896:960]"
param l1_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,896:960]"
param l1_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[960:1024]"
param l1_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,960:1024]"
param l1_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1024:1088]"
param l1_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1024:1088]"
param l1_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1088:1152]"
param l1_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1088:1152]"
param l1_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1152:1216]"
param l1_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1152:1216]"
param l1_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1216:1280]"
param l1_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1216:1280]"
param l1_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1280:1344]"
param l1_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1280:1344]"
param l1_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1344:1408]"
param l1_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1344:1408]"
param l1_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1408:1472]"
param l1_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1408:1472]"
param l1_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1472:1536]"
param l1_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1472:1536]"
param l1_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1536:1600]"
param l1_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1536:1600]"
param l1_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1600:1664]"
param l1_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1600:1664]"
param l1_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1664:1728]"
param l1_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1664:1728]"
param l1_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1728:1792]"
param l1_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1728:1792]"
param l1_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1792:1856]"
param l1_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1792:1856]"
param l1_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1856:1920]"
param l1_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1856:1920]"
param l1_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1920:1984]"
param l1_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1920:1984]"
param l1_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1984:2048]"
param l1_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1984:2048]"
param l1_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[0:64]"
param l1_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[0:64]"
param l1_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[64:128]"
param l1_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[64:128]"
param l1_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[128:192]"
param l1_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[128:192]"
param l1_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[192:256]"
param l1_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[192:256]"
param l1_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[256:320]"
param l1_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[256:320]"
param l1_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[320:384]"
param l1_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[320:384]"
param l1_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[384:448]"
param l1_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[384:448]"
param l1_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[448:512]"
param l1_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[448:512]"
param l1_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[512:576]"
param l1_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[512:576]"
param l1_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[576:640]"
param l1_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[576:640]"
param l1_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[640:704]"
param l1_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[640:704]"
param l1_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[704:768]"
param l1_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[704:768]"
param l1_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[768:832]"
param l1_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[768:832]"
param l1_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[832:896]"
param l1_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[832:896]"
param l1_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[896:960]"
param l1_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[896:960]"
param l1_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[960:1024]"
param l1_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[960:1024]"
param l1_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.mlp.gate_proj.weight"
param l1_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.mlp.up_proj.weight"
param l1_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.mlp.down_proj.weight"




























fn l0(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l1_ln1);
    h

}



























fn head(h: f16[1024]) -> f16[151936] {
    rmsnorm(h, lnf) @ lmh
}

host fn main() -> i64 {
    let me = self_token();
    witness("born", me);
    print("qwen3-0.6b organism", me.id);
    var tok = 151643;                       // BOS
    var x = emb[tok];                       // f16[1024] — host gather
    var step = 0;
    while step < 8 {
        let p = onehot(step % 64, 64);
        let m = causal(step % 64, 64);
        let c = rope_cos(step, 64);
        let s = rope_sin(step, 64);
        x = l0(x, p, m, c, s);
        
        let lg = head(x);
        tok = argmax(lg);
        print("step", step, "tok", tok);
        witness("tok", tok);
        x = emb[tok];
        step += 1;
    }
    witness("done", step);
    return tok;
}
