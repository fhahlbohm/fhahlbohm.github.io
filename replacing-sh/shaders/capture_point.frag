precision highp float;

// Trivial fragment stage of the vertex-variant capture: the point's vertex
// shader already computed the final color.

flat in vec4 vColor;
out vec4 outColor;

void main() { outColor = vColor; }
