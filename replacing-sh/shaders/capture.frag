precision highp float;

// Per-camera color capture (fragment-stage variant): renders one texel per
// splat (same layout as splatData) into an RGBA8 target, so the per-frame
// splat pass just reads the finished color. Runs only when the camera moves.
// The evaluation lives in capture_common.glsl plus the active appearance
// model's residual chunk, shared with the vertex-stage variant (capture.vert)
// used on drivers whose fragment-stage integer texture fetches are broken.

// @inject color_eval

out vec4 outColor;

void main() {
    outColor = captureColor(ivec2(gl_FragCoord.xy));
}
