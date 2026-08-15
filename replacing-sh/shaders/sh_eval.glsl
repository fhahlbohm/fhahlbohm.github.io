// Benchmark-only capture-pass evaluation: full degree-3 spherical harmonics in
// the packed bit-layout spark uses: bands 1-3, DC baked into the rgb8 color,
// levels quantized to 7/8/6 bits and bit-packed into RG/RGBA integer textures.
// Defines the same `captureColor(ivec2)` entry point as mlp_eval.glsl and is
// spliced into the identical capture wrappers, so the SH path runs in whichever
// stage the probe selected — giving an apples-to-apples "MLP eval vs SH eval"
// capture timing.
//
// The SH coefficient textures here hold DUMMY data (main.js fills them with a
// cheap hash): the evaluation cost is data-independent (no branching on values),
// so the runtime is representative while the resulting image is meaningless.

uniform highp usampler2D splatData;
uniform vec3 cameraPos;
uniform vec3 cameraFwd; // normalized camera forward, for the behind-plane early-out
uniform highp usampler2D sh1Tex;
uniform highp usampler2D sh2Tex;
uniform highp usampler2D sh3Tex;

// per-level max(|coef|) is 1.0 for the dummy data; folded into the decode scale
vec3 evaluatePackedSH1(uvec2 packedData, vec3 viewDir) {
    vec3 sh1_0 = vec3(ivec3(
        int(packedData.x << 25u) >> 25,
        int(packedData.x << 18u) >> 25,
        int(packedData.x << 11u) >> 25
    ));
    vec3 sh1_1 = vec3(ivec3(
        int(packedData.x << 4u) >> 25,
        int((packedData.x >> 3u) | (packedData.y << 29u)) >> 25,
        int(packedData.y << 22u) >> 25
    ));
    vec3 sh1_2 = vec3(ivec3(
        int(packedData.y << 15u) >> 25,
        int(packedData.y << 8u) >> 25,
        int(packedData.y << 1u) >> 25
    ));
    vec3 rgb = sh1_0 * (-0.4886025 * viewDir.y)
        + sh1_1 * (0.4886025 * viewDir.z)
        + sh1_2 * (-0.4886025 * viewDir.x);
    return rgb * (1.0 / 63.0);
}

vec3 evaluatePackedSH2(uvec4 packedData, vec3 viewDir) {
    vec3 sh2_0 = vec3(ivec3(
        int(packedData.x << 24u) >> 24,
        int(packedData.x << 16u) >> 24,
        int(packedData.x << 8u) >> 24
    ));
    vec3 sh2_1 = vec3(ivec3(
        int(packedData.x) >> 24,
        int(packedData.y << 24u) >> 24,
        int(packedData.y << 16u) >> 24
    ));
    vec3 sh2_2 = vec3(ivec3(
        int(packedData.y << 8u) >> 24,
        int(packedData.y) >> 24,
        int(packedData.z << 24u) >> 24
    ));
    vec3 sh2_3 = vec3(ivec3(
        int(packedData.z << 16u) >> 24,
        int(packedData.z << 8u) >> 24,
        int(packedData.z) >> 24
    ));
    vec3 sh2_4 = vec3(ivec3(
        int(packedData.w << 24u) >> 24,
        int(packedData.w << 16u) >> 24,
        int(packedData.w << 8u) >> 24
    ));
    vec3 rgb = sh2_0 * (1.0925484 * viewDir.x * viewDir.y)
        + sh2_1 * (-1.0925484 * viewDir.y * viewDir.z)
        + sh2_2 * (0.3153915 * (2.0 * viewDir.z * viewDir.z - viewDir.x * viewDir.x - viewDir.y * viewDir.y))
        + sh2_3 * (-1.0925484 * viewDir.x * viewDir.z)
        + sh2_4 * (0.5462742 * (viewDir.x * viewDir.x - viewDir.y * viewDir.y));
    return rgb * (1.0 / 127.0);
}

vec3 evaluatePackedSH3(uvec4 packedData, vec3 viewDir) {
    vec3 sh3_0 = vec3(ivec3(
        int(packedData.x << 26u) >> 26,
        int(packedData.x << 20u) >> 26,
        int(packedData.x << 14u) >> 26
    ));
    vec3 sh3_1 = vec3(ivec3(
        int(packedData.x << 8u) >> 26,
        int(packedData.x << 2u) >> 26,
        int((packedData.x >> 4u) | (packedData.y << 28u)) >> 26
    ));
    vec3 sh3_2 = vec3(ivec3(
        int(packedData.y << 22u) >> 26,
        int(packedData.y << 16u) >> 26,
        int(packedData.y << 10u) >> 26
    ));
    vec3 sh3_3 = vec3(ivec3(
        int(packedData.y << 4u) >> 26,
        int((packedData.y >> 2u) | (packedData.z << 30u)) >> 26,
        int(packedData.z << 24u) >> 26
    ));
    vec3 sh3_4 = vec3(ivec3(
        int(packedData.z << 18u) >> 26,
        int(packedData.z << 12u) >> 26,
        int(packedData.z << 6u) >> 26
    ));
    vec3 sh3_5 = vec3(ivec3(
        int(packedData.z) >> 26,
        int(packedData.w << 26u) >> 26,
        int(packedData.w << 20u) >> 26
    ));
    vec3 sh3_6 = vec3(ivec3(
        int(packedData.w << 14u) >> 26,
        int(packedData.w << 8u) >> 26,
        int(packedData.w << 2u) >> 26
    ));
    float xx = viewDir.x * viewDir.x;
    float yy = viewDir.y * viewDir.y;
    float zz = viewDir.z * viewDir.z;
    float xy = viewDir.x * viewDir.y;
    vec3 rgb = sh3_0 * (-0.5900436 * viewDir.y * (3.0 * xx - yy))
        + sh3_1 * (2.8906114 * xy * viewDir.z)
        + sh3_2 * (-0.4570458 * viewDir.y * (4.0 * zz - xx - yy))
        + sh3_3 * (0.3731763 * viewDir.z * (2.0 * zz - 3.0 * xx - 3.0 * yy))
        + sh3_4 * (-0.4570458 * viewDir.x * (4.0 * zz - xx - yy))
        + sh3_5 * (1.4453057 * viewDir.z * (xx - yy))
        + sh3_6 * (-0.5900436 * viewDir.x * (xx - 3.0 * yy));
    return rgb * (1.0 / 31.0);
}

vec4 captureColor(ivec2 texel) {
    uvec4 p = texelFetch(splatData, texel, 0);
    vec3 pos = vec3(unpackHalf2x16(p.y), unpackHalf2x16(p.z).x);
    vec4 rgba = vec4(uvec4(
        p.x & 0xFFu, (p.x >> 8u) & 0xFFu, (p.x >> 16u) & 0xFFu, (p.x >> 24u) & 0xFFu
    )) / 255.0;

    // Behind-plane early-out, identical to mlp_eval.glsl: the render pass culls
    // center-behind splats anyway, so both evaluators skip the same subset and
    // the capture timings stay comparable (without this the SH pass would shade
    // the whole scene while the MLP pass shades only the front half).
    vec3 rel = pos - cameraPos;
    if (dot(rel, cameraFwd) <= 0.0)
        return rgba; // DC only, the SH analogue of mlp_eval's base-only fallback

    vec3 viewDir = normalize(rel);
    vec3 sh = evaluatePackedSH1(texelFetch(sh1Tex, texel, 0).rg, viewDir)
        + evaluatePackedSH2(texelFetch(sh2Tex, texel, 0), viewDir)
        + evaluatePackedSH3(texelFetch(sh3Tex, texel, 0), viewDir);
    return vec4(clamp(rgba.rgb + sh, 0.0, 1.0), rgba.a);
}
