// Spherical Voronoi appearance model residual chunk (spliced into
// capture_common.glsl at its `// @inject residual_eval` marker; the wrapper
// provides the uniforms, the color activation, and the captureColor entry
// point).
//
// The residual is a soft spherical Voronoi partition of the direction sphere:
//     dist_s = ||site_s - dir||
//     w      = softmax_s(-temperature_s * dist_s)
//     R(dir) = Σ_s w_s * color_s
// Everything view-independent is baked at export time: the per-splat payload
// holds N_SITES sites of 7 fp16 values each — the UNIT site direction, the
// exp-activated temperature, and the activated site color — so the per-frame
// work is just the softmax. It is evaluated with a running maximum in a single
// pass over the site data, exactly like the reference renderer; the min(., 0)
// clamp keeps a fused multiply-add of -temperature * dist from exceeding the
// rounded maximum by half an ulp (which exp would turn into inf at extreme
// temperatures).
//
// Injected define: N_SITES.

#if RENDER_RESIDUAL
vec3 residualColor(ivec2 uv, vec3 d) {
    float p[N_PARAM_TEX * 8];
    loadParams(uv, p);
    float maxLogit = -1e30;
    float weightSum = 0.0;
    vec3 weightedColor = vec3(0.0);
    for (int s = 0; s < N_SITES; s++) {
        vec3 site = vec3(p[7 * s], p[7 * s + 1], p[7 * s + 2]);
        float logit = -p[7 * s + 3] * length(site - d);
        float newMax = max(maxLogit, logit);
        float rescale = exp(maxLogit - newMax);
        float w = exp(min(logit - newMax, 0.0));
        weightSum = weightSum * rescale + w;
        weightedColor = weightedColor * rescale
            + w * vec3(p[7 * s + 4], p[7 * s + 5], p[7 * s + 6]);
        maxLogit = newMax;
    }
    return weightedColor / weightSum;
}
#endif
