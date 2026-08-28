precision highp float;

// Per-camera color capture, vertex-stage variant: one 1-pixel POINT per splat,
// with all texture reads and the color evaluation in the VERTEX shader,
// writing into the same colorCache texel the fragment variant would. Used on
// drivers whose fragment-stage integer texelFetch returns constants (seen on
// Samsung Android; the vertex stage provably reads the same textures correctly
// there). Same amortization as the fragment variant: 1 eval per splat per
// camera move.

// @inject color_eval

uniform vec2 uCacheSize; // colorCache texture size in texels

flat out vec4 vColor;

void main() {
    int id = gl_VertexID;
    ivec2 texel = ivec2(id % 2048, id / 2048);
    vColor = captureColor(texel);
    // position the point exactly on its texel center
    vec2 ndc = (vec2(texel) + 0.5) / uCacheSize * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = 1.0;
}
