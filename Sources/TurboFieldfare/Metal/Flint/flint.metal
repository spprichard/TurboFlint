#include <metal_stdlib>
using namespace metal;

// ============================================================================
// flint — kernels specific to Recursive Flint (40B-A2.5B).
//
// Activation vectors are FP32 (Flint's reference runs in BF16, whose range FP16
// intermediates such as expert SiLU products can exceed). Only the attention
// query and the K/V cache are FP16.
// Weights are MLX affine int4 (group 64, BF16 scale + bias) except the int8
// embedding table and the BF16 router. Math follows recursive-oss/rec
// engine/src/rec_mlx/model.py; Scripts/reference/flint_reference.py is the
// FP32 oracle these kernels are checked against.
// ============================================================================

constant constexpr uint kFlintGroup = 64;

struct FlintQKParams {
    uint  num_q_heads;
    uint  num_kv_heads;
    uint  head_dim;
    uint  cache_slot;   // row in the K/V cache (ring slot for sliding layers)
    uint  position;     // absolute token position, for RoPE
    uint  use_rope;
    float theta;
    float eps;
};

// Byte offsets of the nine sub-tensors inside one routed-expert slot.
struct FlintExpertLayout {
    uint gate_w, gate_s, gate_b;
    uint up_w, up_s, up_b;
    uint down_w, down_s, down_b;
};

static inline float flint_silu(float x) { return x / (1.0f + exp(-x)); }

// Dot product of one MLX affine int4 row with x, computed by one SIMD group.
// Requires N % 256 == 0: each lane covers 8 contiguous elements of a 64-wide
// group per 256-element block, so s*sum(q*x) + b*sum(x) factors per group.
// Weight bytes are read as ushort pairs (rows are only guaranteed 2-aligned).
// Every lane returns the full sum.
static inline float flint_int4_dot(device const uint8_t* W_row,
                                   device const bfloat*  s_row,
                                   device const bfloat*  b_row,
                                   device const float*   x,
                                   uint N,
                                   uint lane) {
    float acc = 0.0f;
    const uint blocks = N / 256u;
    for (uint blk = 0; blk < blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const uint elem = byte_base * 2u;
        const float4 fa = *((device const float4*)(x + elem));
        const float4 fb = *((device const float4*)(x + elem + 4u));
        const float4 qa = float4(float(w4 & 0xFu), float((w4 >> 4) & 0xFu),
                                 float((w4 >> 8) & 0xFu), float((w4 >> 12) & 0xFu));
        const float4 qb = float4(float((w4 >> 16) & 0xFu), float((w4 >> 20) & 0xFu),
                                 float((w4 >> 24) & 0xFu), float((w4 >> 28) & 0xFu));
        const float qx = dot(qa, fa) + dot(qb, fb);
        const float xs = (fa.x + fa.y + fa.z + fa.w) + (fb.x + fb.y + fb.z + fb.w);
        acc += float(s_row[g]) * qx + float(b_row[g]) * xs;
    }
    return simd_sum(acc);
}

// y[m] = W[m, :] . x for an int4 [M, N] matrix. Eight rows per threadgroup.
kernel void flint_int4_gemv(device const uint8_t* W [[buffer(0)]],
                            device const bfloat*  S [[buffer(1)]],
                            device const bfloat*  B [[buffer(2)]],
                            device const float*   x [[buffer(3)]],
                            device float*         y [[buffer(4)]],
                            constant uint&        M [[buffer(5)]],
                            constant uint&        N [[buffer(6)]],
                            uint tg   [[threadgroup_position_in_grid]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8u + sg;
    if (row >= M) return;
    const uint groups = N / kFlintGroup;
    const float v = flint_int4_dot(W + ulong(row) * (N / 2u), S + ulong(row) * groups,
                                   B + ulong(row) * groups, x, N, lane);
    if (lane == 0) y[row] = v;
}

// out[m] = silu(Wg[m] . x) * (Wu[m] . x) for a dense or shared-expert MLP.
kernel void flint_int4_gate_up_silu(device const uint8_t* Wg [[buffer(0)]],
                                    device const bfloat*  Sg [[buffer(1)]],
                                    device const bfloat*  Bg [[buffer(2)]],
                                    device const uint8_t* Wu [[buffer(3)]],
                                    device const bfloat*  Su [[buffer(4)]],
                                    device const bfloat*  Bu [[buffer(5)]],
                                    device const float*   x  [[buffer(6)]],
                                    device float*         out [[buffer(7)]],
                                    constant uint&        M  [[buffer(8)]],
                                    constant uint&        N  [[buffer(9)]],
                                    uint tg   [[threadgroup_position_in_grid]],
                                    uint sg   [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8u + sg;
    if (row >= M) return;
    const uint groups = N / kFlintGroup;
    const ulong wo = ulong(row) * (N / 2u);
    const ulong so = ulong(row) * groups;
    const float g = flint_int4_dot(Wg + wo, Sg + so, Bg + so, x, N, lane);
    const float u = flint_int4_dot(Wu + wo, Su + so, Bu + so, x, N, lane);
    if (lane == 0) out[row] = flint_silu(g) * u;
}

// Int8 affine embedding row -> FP32 residual stream.
kernel void flint_embed_int8(device const uint8_t* table  [[buffer(0)]],
                             device const bfloat*  scales [[buffer(1)]],
                             device const bfloat*  biases [[buffer(2)]],
                             device float*         out    [[buffer(3)]],
                             constant uint&        token  [[buffer(4)]],
                             constant uint&        D      [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= D) return;
    const ulong row = ulong(token);
    const uint groups = D / kFlintGroup;
    const float q = float(table[row * D + gid]);
    const float s = float(scales[row * groups + gid / kFlintGroup]);
    const float b = float(biases[row * groups + gid / kFlintGroup]);
    out[gid] = q * s + b;
}

// RMSNorm of the FP32 residual into FP16, one threadgroup.
kernel void flint_rmsnorm(device const float*  x   [[buffer(0)]],
                          device const bfloat* w   [[buffer(1)]],
                          device float*        out [[buffer(2)]],
                          constant uint&       D   [[buffer(3)]],
                          constant float&      eps [[buffer(4)]],
                          uint lid   [[thread_position_in_threadgroup]],
                          uint lsize [[threads_per_threadgroup]],
                          uint lane  [[thread_index_in_simdgroup]],
                          uint sg    [[simdgroup_index_in_threadgroup]],
                          uint nsg   [[simdgroups_per_threadgroup]]) {
    threadgroup float partial[32];
    float ss = 0.0f;
    for (uint i = lid; i < D; i += lsize) { ss = fma(x[i], x[i], ss); }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = (lane < nsg) ? partial[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) partial[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(partial[0] / float(D) + eps);
    for (uint i = lid; i < D; i += lsize) { out[i] = x[i] * inv * float(w[i]); }
}

// Per-head QK RMSNorm + optional half-split RoPE. Threadgroup `head` in
// [0, NQ) handles a query head; [NQ, NQ+NKV) a key head, which also copies the
// matching value head. Keys/values go straight into the cache row.
// One thread per head dimension (head_dim threads, <= 256).
kernel void flint_qk_norm_rope(device const float*  q_in    [[buffer(0)]],
                               device const float*  k_in    [[buffer(1)]],
                               device const float*  v_in    [[buffer(2)]],
                               device const bfloat* q_norm  [[buffer(3)]],
                               device const bfloat* k_norm  [[buffer(4)]],
                               device half*         q_out   [[buffer(5)]],
                               device half*         k_cache [[buffer(6)]],
                               device half*         v_cache [[buffer(7)]],
                               constant FlintQKParams& p    [[buffer(8)]],
                               uint head [[threadgroup_position_in_grid]],
                               uint d    [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg   [[simdgroup_index_in_threadgroup]],
                               uint nsg  [[simdgroups_per_threadgroup]]) {
    threadgroup float partial[8];
    threadgroup float vals[256];
    const uint HD = p.head_dim;
    const bool is_q = head < p.num_q_heads;
    const uint kvh = head - p.num_q_heads;
    const float x = is_q ? float(q_in[head * HD + d]) : float(k_in[kvh * HD + d]);
    float ss = simd_sum(x * x);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < nsg; ++i) total += partial[i];
    const float w = is_q ? float(q_norm[d]) : float(k_norm[d]);
    vals[d] = x * rsqrt(total / float(HD) + p.eps) * w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y = vals[d];
    if (p.use_rope != 0u) {
        const uint halfd = HD / 2u;
        const uint i = d % halfd;
        const float inv_freq = exp2(-(2.0f * float(i) / float(HD)) * log2(p.theta));
        const float ang = float(p.position) * inv_freq;
        const float c = precise::cos(ang);
        const float s = precise::sin(ang);
        y = (d < halfd) ? (vals[d] * c - vals[d + halfd] * s)
                        : (vals[d - halfd] * s + vals[d] * c);
    }
    if (is_q) {
        q_out[head * HD + d] = half(y);
    } else {
        const ulong row = (ulong(p.cache_slot) * p.num_kv_heads + kvh) * HD + d;
        k_cache[row] = half(y);
        v_cache[row] = half(v_in[kvh * HD + d]);
    }
}

// attention_decode_combine with Flint's learned per-head sink: the sink is one
// extra logit in each head's softmax that contributes no value.
kernel void flint_attention_combine_sink(device const float*  m_in  [[buffer(0)]],
                                         device const float*  d_in  [[buffer(1)]],
                                         device const float*  o_in  [[buffer(2)]],
                                         device const bfloat* sinks [[buffer(3)]],
                                         device float*        out   [[buffer(4)]],
                                         constant uint&       HD    [[buffer(5)]],
                                         constant uint&       NC    [[buffer(6)]],
                                         uint q_head [[threadgroup_position_in_grid]],
                                         uint lid    [[thread_position_in_threadgroup]],
                                         uint lsize  [[threads_per_threadgroup]]) {
    device const float* m_row = m_in + q_head * NC;
    device const float* d_row = d_in + q_head * NC;
    device const float* o_base = o_in + q_head * NC * HD;
    const float sink = float(sinks[q_head]);
    float m_glob = sink;
    for (uint c = 0; c < NC; ++c) m_glob = max(m_glob, m_row[c]);
    float D = exp(sink - m_glob);
    for (uint c = 0; c < NC; ++c) D += d_row[c] * exp(m_row[c] - m_glob);
    const float inv_d = 1.0f / D;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) acc += o_base[c * HD + i] * exp(m_row[c] - m_glob);
        out[q_head * HD + i] = acc * inv_d;
    }
}

// Router: scores = sigmoid(W . x) over E experts (BF16 W), top-K by score
// (ties -> lower index), weights renormalised over the K picks.
// One threadgroup of up to 1024 threads; E <= 256.
kernel void flint_router_topk(device const float*  x       [[buffer(0)]],
                              device const bfloat* W       [[buffer(1)]],
                              device uint*         ids     [[buffer(2)]],
                              device float*        weights [[buffer(3)]],
                              constant uint&       E       [[buffer(4)]],
                              constant uint&       N       [[buffer(5)]],
                              constant uint&       K       [[buffer(6)]],
                              uint lid  [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint nsg  [[simdgroups_per_threadgroup]]) {
    threadgroup float scores[256];
    for (uint row = sg; row < E; row += nsg) {
        device const bfloat* w = W + ulong(row) * N;
        float acc = 0.0f;
        for (uint i = lane * 4u; i < N; i += 128u) {
            acc += float(w[i]) * float(x[i]) + float(w[i + 1]) * float(x[i + 1])
                 + float(w[i + 2]) * float(x[i + 2]) + float(w[i + 3]) * float(x[i + 3]);
        }
        acc = simd_sum(acc);
        if (lane == 0) scores[row] = 1.0f / (1.0f + exp(-acc));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint chosen[16];
        float picked[16];
        float sum = 0.0f;
        for (uint k = 0; k < K; ++k) {
            float best = -1.0f;
            uint bi = 0;
            for (uint e = 0; e < E; ++e) {
                bool taken = false;
                for (uint j = 0; j < k; ++j) taken = taken || (chosen[j] == e);
                if (!taken && scores[e] > best) { best = scores[e]; bi = e; }
            }
            chosen[k] = bi;
            picked[k] = best;
            sum += best;
        }
        for (uint k = 0; k < K; ++k) {
            ids[k] = chosen[k];
            weights[k] = picked[k] / (sum + 1e-20f);
        }
    }
}

// Routed experts, phase 1: h[e, r] = silu(gate_e[r] . x) * (up_e[r] . x) for
// each of the K selected experts, whose weights sit in `pool` slots.
kernel void flint_moe_gate_up(device const uint8_t* pool      [[buffer(0)]],
                              device const uint*    slot_base [[buffer(1)]],
                              device const float*   x         [[buffer(2)]],
                              device float*         h         [[buffer(3)]],
                              constant FlintExpertLayout& L   [[buffer(4)]],
                              constant uint&        I         [[buffer(5)]],
                              constant uint&        N         [[buffer(6)]],
                              constant uint&        K         [[buffer(7)]],
                              uint tg   [[threadgroup_position_in_grid]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8u + sg;
    if (row >= K * I) return;
    const uint e = row / I;
    const uint r = row % I;
    device const uint8_t* base = pool + slot_base[e];
    const uint groups = N / kFlintGroup;
    const uint wo = r * (N / 2u);
    const uint so = r * groups;
    const float g = flint_int4_dot(base + L.gate_w + wo, (device const bfloat*)(base + L.gate_s) + so,
                                   (device const bfloat*)(base + L.gate_b) + so, x, N, lane);
    const float u = flint_int4_dot(base + L.up_w + wo, (device const bfloat*)(base + L.up_s) + so,
                                   (device const bfloat*)(base + L.up_b) + so, x, N, lane);
    if (lane == 0) h[row] = flint_silu(g) * u;
}

// Routed experts, phase 2: out[m] = sum_e weight[e] * (down_e[m] . h[e]).
kernel void flint_moe_down(device const uint8_t* pool      [[buffer(0)]],
                           device const uint*    slot_base [[buffer(1)]],
                           device const float*   weights   [[buffer(2)]],
                           device const float*   h         [[buffer(3)]],
                           device float*         out       [[buffer(4)]],
                           constant FlintExpertLayout& L   [[buffer(5)]],
                           constant uint&        I         [[buffer(6)]],
                           constant uint&        M         [[buffer(7)]],
                           constant uint&        K         [[buffer(8)]],
                           uint tg   [[threadgroup_position_in_grid]],
                           uint sg   [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8u + sg;
    if (row >= M) return;
    const uint groups = I / kFlintGroup;
    const uint wo = row * (I / 2u);
    const uint so = row * groups;
    float acc = 0.0f;
    for (uint e = 0; e < K; ++e) {
        device const uint8_t* base = pool + slot_base[e];
        acc += weights[e] * flint_int4_dot(base + L.down_w + wo, (device const bfloat*)(base + L.down_s) + so,
                                           (device const bfloat*)(base + L.down_b) + so, h + e * I, I, lane);
    }
    if (lane == 0) out[row] = acc;
}

// h += a (+ b). FP32 residual stream, FP16 branch outputs.
kernel void flint_residual_add(device float*       h [[buffer(0)]],
                               device const float* a [[buffer(1)]],
                               constant uint&     D [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid < D) h[gid] += float(a[gid]);
}

kernel void flint_residual_add2(device float*       h [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                constant uint&     D [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < D) h[gid] += float(a[gid]) + float(b[gid]);
}

// Router, phase 1: scores[e] = sigmoid(W[e] . x), one SIMD group per expert row,
// eight rows per threadgroup so the 1 MB BF16 matrix spreads across GPU cores.
kernel void flint_router_scores(device const float*  x      [[buffer(0)]],
                                device const bfloat* W      [[buffer(1)]],
                                device float*        scores [[buffer(2)]],
                                constant uint&       E      [[buffer(3)]],
                                constant uint&       N      [[buffer(4)]],
                                uint tg   [[threadgroup_position_in_grid]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8u + sg;
    if (row >= E) return;
    device const bfloat4* w = (device const bfloat4*)(W + ulong(row) * N);
    device const float4* xv = (device const float4*)x;
    float acc = 0.0f;
    for (uint i = lane; i < N / 4u; i += 32u) acc += dot(float4(w[i]), float4(xv[i]));
    acc = simd_sum(acc);
    if (lane == 0) scores[row] = 1.0f / (1.0f + exp(-acc));
}

// Router, phase 2: top-K of E <= 256 scores by one SIMD group (ties -> lower
// index), weights renormalised over the picks.
kernel void flint_router_select(device const float* scores  [[buffer(0)]],
                                device uint*        ids     [[buffer(1)]],
                                device float*       weights [[buffer(2)]],
                                constant uint&      E       [[buffer(3)]],
                                constant uint&      K       [[buffer(4)]],
                                uint lane [[thread_index_in_simdgroup]]) {
    float v[8];
    for (uint j = 0; j < 8u; ++j) {
        const uint e = lane * 8u + j;
        v[j] = (e < E) ? scores[e] : -INFINITY;
    }
    float picked[16];
    uint chosen[16];
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint bj = 0;
        for (uint j = 0; j < 8u; ++j) if (v[j] > best) { best = v[j]; bj = j; }
        const float gbest = simd_max(best);
        const uint cand = (best == gbest) ? lane * 8u + bj : 0xFFFFFFFFu;
        const uint win = simd_min(cand);
        if (lane == win / 8u) v[win % 8u] = -INFINITY;
        chosen[k] = win;
        picked[k] = gbest;
        sum += gbest;
    }
    if (lane == 0) {
        for (uint k = 0; k < K; ++k) {
            ids[k] = chosen[k];
            weights[k] = picked[k] / (sum + 1e-20f);
        }
    }
}
