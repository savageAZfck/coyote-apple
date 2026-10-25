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




























fn l0(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[3072] {
    let h = rmsnorm(x, l1_ln1);
    let q0l = rmsnorm(h @ l1_q0lo, l1_qnl);
    let q0h = rmsnorm(h @ l1_q0hi, l1_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l1_q1lo, l1_qnl);
    let q1h = rmsnorm(h @ l1_q1hi, l1_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l1_q2lo, l1_qnl);
    let q2h = rmsnorm(h @ l1_q2hi, l1_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l1_q3lo, l1_qnl);
    let q3h = rmsnorm(h @ l1_q3hi, l1_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l1_q4lo, l1_qnl);
    let q4h = rmsnorm(h @ l1_q4hi, l1_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l1_q5lo, l1_qnl);
    let q5h = rmsnorm(h @ l1_q5hi, l1_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l1_q6lo, l1_qnl);
    let q6h = rmsnorm(h @ l1_q6hi, l1_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l1_q7lo, l1_qnl);
    let q7h = rmsnorm(h @ l1_q7hi, l1_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l1_q8lo, l1_qnl);
    let q8h = rmsnorm(h @ l1_q8hi, l1_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l1_q9lo, l1_qnl);
    let q9h = rmsnorm(h @ l1_q9hi, l1_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l1_q10lo, l1_qnl);
    let q10h = rmsnorm(h @ l1_q10hi, l1_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l1_q11lo, l1_qnl);
    let q11h = rmsnorm(h @ l1_q11hi, l1_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l1_q12lo, l1_qnl);
    let q12h = rmsnorm(h @ l1_q12hi, l1_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l1_q13lo, l1_qnl);
    let q13h = rmsnorm(h @ l1_q13hi, l1_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l1_q14lo, l1_qnl);
    let q14h = rmsnorm(h @ l1_q14hi, l1_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l1_q15lo, l1_qnl);
    let q15h = rmsnorm(h @ l1_q15hi, l1_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l1_k0lo, l1_knl);
    let k0h = rmsnorm(h @ l1_k0hi, l1_knh);
    let v0l = h @ l1_v0lo;
    let v0h = h @ l1_v0hi;
    let rk0l = k0l * cos - k0h * sin;
    let rk0h = k0h * cos + k0l * sin;
    kv.k0lo[pos] = rk0l;
    kv.k0hi[pos] = rk0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l1_k1lo, l1_knl);
    let k1h = rmsnorm(h @ l1_k1hi, l1_knh);
    let v1l = h @ l1_v1lo;
    let v1h = h @ l1_v1hi;
    let rk1l = k1l * cos - k1h * sin;
    let rk1h = k1h * cos + k1l * sin;
    kv.k1lo[pos] = rk1l;
    kv.k1hi[pos] = rk1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l1_k2lo, l1_knl);
    let k2h = rmsnorm(h @ l1_k2hi, l1_knh);
    let v2l = h @ l1_v2lo;
    let v2h = h @ l1_v2hi;
    let rk2l = k2l * cos - k2h * sin;
    let rk2h = k2h * cos + k2l * sin;
    kv.k2lo[pos] = rk2l;
    kv.k2hi[pos] = rk2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l1_k3lo, l1_knl);
    let k3h = rmsnorm(h @ l1_k3hi, l1_knh);
    let v3l = h @ l1_v3lo;
    let v3h = h @ l1_v3hi;
    let rk3l = k3l * cos - k3h * sin;
    let rk3h = k3h * cos + k3l * sin;
    kv.k3lo[pos] = rk3l;
    kv.k3hi[pos] = rk3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l1_k4lo, l1_knl);
    let k4h = rmsnorm(h @ l1_k4hi, l1_knh);
    let v4l = h @ l1_v4lo;
    let v4h = h @ l1_v4hi;
    let rk4l = k4l * cos - k4h * sin;
    let rk4h = k4h * cos + k4l * sin;
    kv.k4lo[pos] = rk4l;
    kv.k4hi[pos] = rk4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l1_k5lo, l1_knl);
    let k5h = rmsnorm(h @ l1_k5hi, l1_knh);
    let v5l = h @ l1_v5lo;
    let v5h = h @ l1_v5hi;
    let rk5l = k5l * cos - k5h * sin;
    let rk5h = k5h * cos + k5l * sin;
    kv.k5lo[pos] = rk5l;
    kv.k5hi[pos] = rk5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l1_k6lo, l1_knl);
    let k6h = rmsnorm(h @ l1_k6hi, l1_knh);
    let v6l = h @ l1_v6lo;
    let v6h = h @ l1_v6hi;
    let rk6l = k6l * cos - k6h * sin;
    let rk6h = k6h * cos + k6l * sin;
    kv.k6lo[pos] = rk6l;
    kv.k6hi[pos] = rk6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l1_k7lo, l1_knl);
    let k7h = rmsnorm(h @ l1_k7hi, l1_knh);
    let v7l = h @ l1_v7lo;
    let v7h = h @ l1_v7hi;
    let rk7l = k7l * cos - k7h * sin;
    let rk7h = k7h * cos + k7l * sin;
    kv.k7lo[pos] = rk7l;
    kv.k7hi[pos] = rk7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.08838834764831845 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l1_o0lo + c0h @ l1_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.08838834764831845 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l1_o1lo + c1h @ l1_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.08838834764831845 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l1_o2lo + c2h @ l1_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.08838834764831845 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l1_o3lo + c3h @ l1_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.08838834764831845 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l1_o4lo + c4h @ l1_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.08838834764831845 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l1_o5lo + c5h @ l1_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.08838834764831845 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l1_o6lo + c6h @ l1_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.08838834764831845 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l1_o7lo + c7h @ l1_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.08838834764831845 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l1_o8lo + c8h @ l1_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.08838834764831845 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l1_o9lo + c9h @ l1_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.08838834764831845 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l1_o10lo + c10h @ l1_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.08838834764831845 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l1_o11lo + c11h @ l1_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.08838834764831845 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l1_o12lo + c12h @ l1_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.08838834764831845 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l1_o13lo + c13h @ l1_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.08838834764831845 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l1_o14lo + c14h @ l1_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.08838834764831845 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l1_o15lo + c15h @ l1_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l1_ln2);
    let u = n @ l1_wu;
    u
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
        let y = l0(x, p, m, c, s);
        
        let lg = y;
        tok = argmax(lg);
        print("step", step, "tok", tok);
        witness("tok", tok);
        x = emb[tok];
        step += 1;
    }
    witness("done", step);
    return tok;
}
