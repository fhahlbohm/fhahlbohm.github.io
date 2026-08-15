// Shared capture-pass color evaluation (spliced into capture_mlp.frag and
// capture_mlp.vert at their `// @inject mlp_eval` marker — some Android drivers
// return constants from fragment-stage integer texelFetch, so the same code must
// run in either stage; see main.js chooseCaptureStage).
//
// The residual MLP is evaluated as in SMERF: the (bias-free) weights are indexed
// as texels — one texel = 4 consecutive input weights of one output neuron — and
// each hidden layer accumulates `input_block * mat4(4 texels)` per 4-wide output
// block; the 3-row output layer reads 3 texels per input block. Texel order:
//   texel[layerBase + j * outDim + o] = W[o, 4j:4j+4]
// Two interchangeable backings, chosen by WEIGHTS_UNIFORM (see main.js):
// a vec4 uniform array (no per-splat texture fetches — the default, but capped
// by the device's MAX_*_UNIFORM_VECTORS, so only used when it fits), or a
// 64-wide RGBA16F texture (the always-available fallback).
//
// Injected defines: BAKED (1 = layer 0's view-independent share is
// pre-multiplied into the per-splat h0_static cache, 0 = the legacy layout where
// the per-splat textures hold the encoded features and layer 0 is the full
// trained matrix), N_L0_IN (layer-0 input width as stored in the file),
// N_MLP_TEX (1 or 2 per-splat fp16 textures), N_FEATURES / SH0_INPUT (legacy
// layout only), N_NEURONS, N_HIDDEN, SH_DEGREE_MASK, COLOR_ACT (0 relu / 1
// softplus(β10) / 3 additive: satexp + exp base + softplus residual),
// RESIDUAL_ACT (0 none / 1 tanh; ignored for COLOR_ACT 3), RENDER_BASE /
// RENDER_RESIDUAL (the reference renderer's base-color and residual debug
// switches), D1_OFF..D6_OFF (layer-0-input-array offsets of the
// view-direction SH degrees).
//
// Color model (matches the reference implementation; the exporter bakes the
// relu/softplus 0.5 shift into the stored base d, so base_activation reduces
// to exp(3d) for COLOR_ACT 3 and the identity otherwise):
//   h0       = W0 . [features, (SH_C0), sh_degrees(dir)]      (BAKED 0)
//            = h0_static + W0_dyn . sh_degrees(dir)           (BAKED 1)
//              with dir = normalize(mean - cameraPos)
//   residual = residual_activation(MLP layers 1.. applied to h0)
//   color    = color_activation(base + residual)
//   RENDER_RESIDUAL 0 -> color_activation(base)           (the degree-0 path:
//                        the renderer skips the MLP, residual = 0)
//   RENDER_BASE 0     -> |full color - base-only color|

uniform highp usampler2D splatData;
// per-splat fp16 payload, 8 values per RGBA32UI texel: h0_static (one per
// neuron) when BAKED, the encoded mlp features otherwise
uniform highp usampler2D mlpTex0;
#if N_MLP_TEX > 1
uniform highp usampler2D mlpTex1;
#endif
uniform vec3 cameraPos;
uniform vec3 cameraFwd; // normalized camera forward, for the behind-plane early-out
uniform vec2 uBase; // x = scale, y = offset: base pre-activation d = code8 * scale + offset

const float SH_C0 = 0.28209479177387814;

// weight texture layout constants (3-row output layer)
const int L0_TEXELS = (N_L0_IN / 4) * N_NEURONS;
const int HID_TEXELS = (N_NEURONS / 4) * N_NEURONS;
const int OUT_BASE = L0_TEXELS + (N_HIDDEN - 1) * HID_TEXELS;

// MLP working precision (MLP_FP16): mediump matches the reference
// implementation, which evaluates in fp16 (half weights + activations), and runs
// ~2x on mobile GPUs. Input assembly (features, SH degrees, normalize) stays
// highp regardless — only the weights and the layer matmul/accumulation use
// MLPP. (Desktop drivers treat mediump as fp32, so there it is a no-op.)
#ifdef MLP_FP16
#define MLPP mediump
#else
#define MLPP highp
#endif

// weight texel accessor: uniform array (immediate operands, no fetch) or the
// 64-wide RGBA16F texture (texel t at (t % 64, t / 64))
#ifdef WEIGHTS_UNIFORM
uniform MLPP vec4 uWeights[N_WEIGHT_TEXELS];
#define W(t) uWeights[t]
#else
uniform MLPP sampler2D mlpWeights;
#define W(t) texelFetch(mlpWeights, ivec2((t) & 63, (t) >> 6), 0)
#endif

// softplus with beta 10, linear above x > 2 (the numerically stable form)
vec3 softplus10(vec3 x) {
    vec3 soft = log(1.0 + exp(10.0 * min(x, vec3(2.0)))) * 0.1;
    return mix(soft, x, greaterThan(x, vec3(2.0)));
}

// color activation over base + residual; the exporter pre-bakes the 0.5 shift
// of the relu/softplus variants into the stored base, so it is not
// re-applied here (COLOR_ACT 3 = satexp is anchored at black and unshifted)
vec3 colorActivation(vec3 x) {
#if COLOR_ACT == 1
    return softplus10(x);
#elif COLOR_ACT == 3
    return 1.0 - exp(-x);
#else
    return max(x, vec3(0.0));
#endif
}

#if RENDER_RESIDUAL
// View-dependent residual for direction d. When BAKED, layer 0 starts from the
// per-splat h0_static and is fed only the view-direction inputs; otherwise it
// starts at zero and is fed the full [features, (SH_C0), degrees] vector.
vec3 mlpResidual(ivec2 uv, vec3 d) {
    // ---- layer-0 input vector ----
    float inp[N_L0_IN];
#if BAKED
    for (int i = 0; i < N_L0_IN; i++) inp[i] = 0.0; // only the padded tail matters
#else
    {
        uvec4 t0 = texelFetch(mlpTex0, uv, 0);
        for (int i = 0; i < 4; i++) {
            vec2 v = unpackHalf2x16(t0[i]);
            if (2 * i < N_FEATURES) inp[2 * i] = v.x;
            if (2 * i + 1 < N_FEATURES) inp[2 * i + 1] = v.y;
        }
#if N_MLP_TEX > 1
        uvec4 t1 = texelFetch(mlpTex1, uv, 0);
        for (int i = 0; i < 4; i++) {
            vec2 v = unpackHalf2x16(t1[i]);
            if (8 + 2 * i < N_FEATURES) inp[8 + 2 * i] = v.x;
            if (8 + 2 * i + 1 < N_FEATURES) inp[8 + 2 * i + 1] = v.y;
        }
#endif
    }
#if SH0_INPUT
    inp[N_FEATURES] = SH_C0;
#endif
#endif

    // view-direction SH degrees (SH basis polynomials, ascending)
    float x = d.x, y = d.y, z = d.z;
    float xy = x * y, xz = x * z, yz = y * z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
#if (SH_DEGREE_MASK & 2) != 0
    inp[D1_OFF + 0] = -0.48860251190291987 * y;
    inp[D1_OFF + 1] = 0.48860251190291987 * z;
    inp[D1_OFF + 2] = -0.48860251190291987 * x;
#endif
#if (SH_DEGREE_MASK & 4) != 0
    inp[D2_OFF + 0] = 1.0925484305920792 * xy;
    inp[D2_OFF + 1] = -1.0925484305920792 * yz;
    inp[D2_OFF + 2] = 0.94617469575755997 * z2 - 0.31539156525251999;
    inp[D2_OFF + 3] = -1.0925484305920792 * xz;
    inp[D2_OFF + 4] = 0.54627421529603959 * x2 - 0.54627421529603959 * y2;
#endif
#if (SH_DEGREE_MASK & 8) != 0
    inp[D3_OFF + 0] = 0.59004358992664352 * y * (-3.0 * x2 + y2);
    inp[D3_OFF + 1] = 2.8906114426405538 * xy * z;
    inp[D3_OFF + 2] = 0.45704579946446572 * y * (1.0 - 5.0 * z2);
    inp[D3_OFF + 3] = 0.3731763325901154 * z * (5.0 * z2 - 3.0);
    inp[D3_OFF + 4] = 0.45704579946446572 * x * (1.0 - 5.0 * z2);
    inp[D3_OFF + 5] = 1.4453057213202769 * z * (x2 - y2);
    inp[D3_OFF + 6] = 0.59004358992664352 * x * (-x2 + 3.0 * y2);
#endif
#if (SH_DEGREE_MASK & 112) != 0
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
#endif
#if (SH_DEGREE_MASK & 16) != 0
    inp[D4_OFF + 0] = 2.5033429417967046 * xy * (x2 - y2);
    inp[D4_OFF + 1] = 1.7701307697799304 * yz * (-3.0 * x2 + y2);
    inp[D4_OFF + 2] = 0.94617469575756008 * xy * (7.0 * z2 - 1.0);
    inp[D4_OFF + 3] = 0.66904654355728921 * yz * (3.0 - 7.0 * z2);
    inp[D4_OFF + 4] = -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293;
    inp[D4_OFF + 5] = 0.66904654355728921 * xz * (3.0 - 7.0 * z2);
    inp[D4_OFF + 6] = 0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0);
    inp[D4_OFF + 7] = 1.7701307697799304 * xz * (-x2 + 3.0 * y2);
    inp[D4_OFF + 8] = -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4;
#endif
#if (SH_DEGREE_MASK & 32) != 0
    inp[D5_OFF + 0] = 0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4);
    inp[D5_OFF + 1] = 8.3026492595241645 * xy * z * (x2 - y2);
    inp[D5_OFF + 2] = -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0);
    inp[D5_OFF + 3] = 4.7935367849733241 * xy * z * (3.0 * z2 - 1.0);
    inp[D5_OFF + 4] = 0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0);
    inp[D5_OFF + 5] = 0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0);
    inp[D5_OFF + 6] = 0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0);
    inp[D5_OFF + 7] = 2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0);
    inp[D5_OFF + 8] = -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0);
    inp[D5_OFF + 9] = 2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4);
    inp[D5_OFF + 10] = 0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4);
#endif
#if (SH_DEGREE_MASK & 64) != 0
    float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;
    inp[D6_OFF + 0] = 1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4);
    inp[D6_OFF + 1] = 2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4);
    inp[D6_OFF + 2] = 2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0);
    inp[D6_OFF + 3] = -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0);
    inp[D6_OFF + 4] = 0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0);
    inp[D6_OFF + 5] = 0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0);
    inp[D6_OFF + 6] = 6.6747662381009842 * z2 - 20.024298714302954 * z4 + 14.684485723822165 * z6 - 0.31784601133814211;
    inp[D6_OFF + 7] = 0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0);
    inp[D6_OFF + 8] = 0.46060262975746175 * (x2 - y2) * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0);
    inp[D6_OFF + 9] = -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0);
    inp[D6_OFF + 10] = 0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4);
    inp[D6_OFF + 11] = 2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4);
    inp[D6_OFF + 12] = 10.247761577878714 * x2 * y4 - 10.247761577878714 * x4 * y2 + 0.6831841051919143 * x6 - 0.6831841051919143 * y6;
#endif

    // ---- layer 0: baked starts from h0_static, legacy from zero ----
    vec4 h[N_NEURONS / 4];
    vec4 act[N_NEURONS / 4];
#if BAKED
    {
        uvec4 t0 = texelFetch(mlpTex0, uv, 0);
        h[0] = vec4(unpackHalf2x16(t0.x), unpackHalf2x16(t0.y));
#if N_NEURONS > 4
        h[1] = vec4(unpackHalf2x16(t0.z), unpackHalf2x16(t0.w));
#endif
#if N_MLP_TEX > 1
        uvec4 t1 = texelFetch(mlpTex1, uv, 0);
        h[2] = vec4(unpackHalf2x16(t1.x), unpackHalf2x16(t1.y));
#if N_NEURONS > 12
        h[3] = vec4(unpackHalf2x16(t1.z), unpackHalf2x16(t1.w));
#endif
#endif
    }
#else
    for (int i = 0; i < N_NEURONS / 4; i++) h[i] = vec4(0.0);
#endif
    for (int j = 0; j < N_L0_IN / 4; j++) {
        vec4 b = vec4(inp[4 * j], inp[4 * j + 1], inp[4 * j + 2], inp[4 * j + 3]);
        for (int i = 0; i < N_NEURONS; i += 4) {
            int t = j * N_NEURONS + i;
            h[i / 4] += b * mat4(W(t), W(t + 1), W(t + 2), W(t + 3));
        }
    }

    // ---- hidden layers 1 .. N_HIDDEN-1 ----
    for (int l = 1; l < N_HIDDEN; l++) {
        int base = L0_TEXELS + (l - 1) * HID_TEXELS;
        for (int i = 0; i < N_NEURONS / 4; i++) {
            act[i] = max(h[i], vec4(0.0)); // ReLU
            h[i] = vec4(0.0);
        }
        for (int j = 0; j < N_NEURONS / 4; j++) {
            for (int i = 0; i < N_NEURONS; i += 4) {
                int t = base + j * N_NEURONS + i;
                h[i / 4] += act[j] * mat4(W(t), W(t + 1), W(t + 2), W(t + 3));
            }
        }
    }

    // ---- output layer (3 dims: rgb residual; 3 texels per input block) ----
    vec3 mlpOut = vec3(0.0);
    for (int j = 0; j < N_NEURONS / 4; j++) {
        int t = OUT_BASE + j * 3;
        vec4 b = max(h[j], vec4(0.0));
        mlpOut += vec3(dot(b, W(t)), dot(b, W(t + 1)), dot(b, W(t + 2)));
    }

#if COLOR_ACT == 3
    return softplus10(mlpOut); // additive combo: softplus residual is implied
#elif RESIDUAL_ACT == 1
    return tanh(mlpOut);
#else
    return mlpOut;
#endif
}
#endif

// Full capture-pass output for the splat at texture coordinate `texel`:
// rgb = final view-dependent color, a = opacity.
vec4 captureColor(ivec2 texel) {
    uvec4 p = texelFetch(splatData, texel, 0);

    // pre-activation base (range-coded rgb8) + opacity (a8)
    vec3 d = vec3(uvec3(p.x & 0xFFu, (p.x >> 8u) & 0xFFu, (p.x >> 16u) & 0xFFu))
        * uBase.x + uBase.y;
    float opacity = float((p.x >> 24u) & 0xFFu) / 255.0;

#if COLOR_ACT == 3
    vec3 baseTerm = exp(3.0 * d); // exp base activation
#else
    vec3 baseTerm = d;            // identity base activation (0.5 shift pre-baked)
#endif
    vec3 baseColor = colorActivation(baseTerm);

#if RENDER_RESIDUAL
    vec3 pos = vec3(unpackHalf2x16(p.y), unpackHalf2x16(p.z).x);
    vec3 rel = pos - cameraPos;
    // Behind-plane early-out: the render pass culls center-behind splats anyway
    // (both the sorter's count and the vertex shader), so skip the MLP for them.
    // The base-only fallback only shows if the one-frame-late sort order still
    // references such a splat — where the stale ordering dominates any color error.
    vec3 full = dot(rel, cameraFwd) <= 0.0
        ? baseColor
        : colorActivation(baseTerm + mlpResidual(texel, normalize(rel)));
#else
    vec3 full = baseColor; // base color only
#endif

#if RENDER_BASE
    vec3 rgb = full;
#else
    vec3 rgb = abs(full - baseColor); // residual color only
#endif

    return vec4(clamp(rgb, 0.0, 1.0), opacity);
}
