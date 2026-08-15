// ============================================================================
// Fetch / file helpers
// ============================================================================

export async function fetchFile(path, type = "text") {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Unable to load ${path} (Error ${response.status})`);
    if (type === "none") return response;
    return response[type]();
}

// Fetch a URL into an ArrayBuffer, reporting fractional progress as it streams in.
export async function readUrlWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Unable to load ${url} (Error ${res.status})`);
    const total = Number(res.headers.get("content-length")) || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let offset = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        offset += value.length;
        if (total) onProgress(offset / total);
    }
    if (!total) onProgress(1);
    return new Blob(chunks).arrayBuffer();
}

// Render the model grid and resolve with the clicked manifest entry
// { name, url }. Re-armable after a failed load.
export function chooseModel(sections, models) {
    return new Promise((resolve) => {
        const el = (tag, className, text) => {
            const node = document.createElement(tag);
            node.className = className;
            if (text != null) node.textContent = text;
            return node;
        };

        const groups = new Map();
        for (const model of models) {
            const key = model.group ?? "Models";
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(model);
        }

        sections.replaceChildren();
        for (const [title, groupModels] of groups) {
            const grid = el("div", "gallery-grid");
            for (const model of groupModels) {
                const card = el("button", "card");
                card.type = "button";
                card.append(el("div", "card-title", model.name ?? model.url));
                card.addEventListener("click", () => resolve(model));
                grid.append(card);
            }
            const section = el("div", "gallery-section");
            section.append(el("div", "section-title", title), grid);
            sections.append(section);
        }
    });
}

// ============================================================================
// .ngsplat loading (neural appearance model)
//
// The file contains the exact texture payloads: splatData (RGBA32UI:
// pre-activation base rgb8 + opacity a8, pos fp16, quat 24b, log-scales 3x8b),
// one or two per-splat fp16 textures (8 values per RGBA32UI texel), and the
// residual-MLP weights as raw fp16 bits in capture-shader texel order (3-row
// output layer). It also carries the test-set viewpoints of the scene the
// model was trained on, used by the fixed-viewport benchmark mode. No
// repacking happens here; the only derived data is the float splat centers for
// the CPU depth sorter.
//
// Two layer-0 layouts are supported, distinguished by flags bit 2:
//   BAKED (bit set) — the per-splat textures hold h0_static, the
//     view-independent share of mlp layer 0 (one fp16 per neuron; the encoded
//     features and the constant SH_C0 input are pre-multiplied at export
//     time), and layer 0 of the weight stream holds only the view-direction
//     columns, zero-padded to a multiple of 4.
//   LEGACY (bit clear, the layout of older files) — the per-splat textures
//     hold the encoded features and layer 0 of the weight stream is the full
//     [neurons x (features + SH_C0 + degrees)] matrix, so the shader
//     assembles the whole input vector per splat.
// Both evaluate the identical color model; BAKED just moves ~31% of the
// per-splat MACs to export time at the same file size.
// ============================================================================

const MAGIC = "NGSPLAT\n";
export const SH_C0 = 0.28209479177387814;
// header colorActivation codes; code 2 (sigmoid) is defined by the
// format but is never written
export const COLOR_ACTIVATION_NAMES = ["relu", "softplus (β10)", "sigmoid", "satexp (additive)"];
export const RESIDUAL_ACTIVATION_NAMES = ["none", "tanh"];

// float16 bit pattern -> float32, via a lazily built 64K lookup table.
let halfTable = null;
function buildHalfTable() {
    const table = new Float32Array(65536);
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    for (let h = 0; h < 65536; h++) {
        const sign = (h & 0x8000) << 16;
        const exp = (h >> 10) & 0x1F;
        const mant = h & 0x3FF;
        if (exp === 0) {
            // subnormal: mant * 2^-24
            table[h] = (sign ? -1 : 1) * mant * 5.960464477539063e-8;
        } else if (exp === 31) {
            u32[0] = sign | 0x7F800000 | (mant << 13);
            table[h] = f32[0];
        } else {
            u32[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
            table[h] = f32[0];
        }
    }
    return table;
}
export function halfToFloat(bits) {
    if (!halfTable) halfTable = buildHalfTable();
    return halfTable[bits & 0xFFFF];
}

export function loadNgsplat(buffer) {
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
    if (magic !== MAGIC) throw new Error("Not a .ngsplat file (bad magic)");
    const dv = new DataView(buffer);
    let o = 8;
    const u32 = () => { const v = dv.getUint32(o, true); o += 4; return v; };
    const f32 = () => { const v = dv.getFloat32(o, true); o += 4; return v; };

    const numSplats = u32();
    const textureWidth = u32();
    const textureHeight = u32();
    const featureDim = u32(); // ENCODED feature dims (frequency encoding is pre-baked)
    const nFrequencies = u32(); // informational only
    const shDegreeMask = u32(); // bit l set = view-direction SH degree l (1..6) is an mlp input
    const colorActivation = u32(); // 0 relu | 1 softplus(β10) | 2 sigmoid (unused) | 3 additive/satexp
    const residualActivation = u32(); // 0 none | 1 tanh (0 written for colorActivation 3)
    const nNeurons = u32();
    const nHiddenLayers = u32();
    const flags = u32();
    const properAA = (flags & 1) !== 0;
    const sh0Input = (flags & 2) === 0; // bit 1 = constant SH_C0 mlp input DISABLED
    const baked = (flags & 4) !== 0;    // bit 2 = layer 0 is baked (see the header comment)
    const baseScale = f32(); // base pre-activation d = code8 * scale + offset
    const baseOffset = f32();
    const camCenter = [f32(), f32(), f32()];
    const camUp = [f32(), f32(), f32()];
    const camDistance = f32();

    // baked test-set viewpoints (benchmark mode): c2w rows (3x4, COLMAP
    // right/down/forward convention) + native intrinsics
    const nTestCameras = u32();
    const testCameras = [];
    for (let i = 0; i < nTestCameras; i++) {
        const c2w = new Float32Array(12);
        for (let j = 0; j < 12; j++) c2w[j] = f32();
        testCameras.push({
            c2w,
            fx: f32(), fy: f32(), cx: f32(), cy: f32(),
            width: f32(), height: f32(),
        });
    }
    const nWeightTexels = u32();

    if (textureWidth !== 2048) throw new Error(`texWidth must be 2048 (render_ewa.vert hardcodes it), got ${textureWidth}`);
    if (shDegreeMask & ~0b1111110) throw new Error("shDegreeMask has bits outside degrees 1-6");
    if (colorActivation === 2) throw new Error("colorActivation 2 (sigmoid) has no shader implementation");
    if (colorActivation > 3) throw new Error(`unknown colorActivation code ${colorActivation}`);
    if (residualActivation > 1) throw new Error(`unknown residualActivation code ${residualActivation}`);
    if (nNeurons % 4 !== 0 || nNeurons < 4) throw new Error(`nNeurons must be a multiple of 4 >= 4, got ${nNeurons}`);
    if (nHiddenLayers < 1) throw new Error(`nHiddenLayers must be >= 1, got ${nHiddenLayers}`);

    const degrees = [];
    for (let l = 1; l <= 6; l++) if (shDegreeMask & (1 << l)) degrees.push(l);
    const nDirectionValues = degrees.reduce((s, l) => s + 2 * l + 1, 0);
    // layer-0 input width as stored in the file, and the number of per-splat
    // fp16 values each layout keeps in its textures
    let l0In, nPerSplatValues;
    if (baked) {
        // only the view-direction SH values, zero-padded to /4 (features + SH_C0
        // are pre-multiplied into the per-splat h0_static cache: one per neuron)
        if (!nDirectionValues) throw new Error("baked layout needs at least one view-direction SH degree");
        if (nNeurons > 16)
            throw new Error(`nNeurons must be <= 16 in the baked layout, got ${nNeurons} `
                + `(the h0_static cache is capped at 2 textures = 16 fp16 values)`);
        l0In = Math.ceil(nDirectionValues / 4) * 4;
        nPerSplatValues = nNeurons;
    } else {
        // the full trained input: encoded features + the SH_C0 constant + degrees
        if (featureDim < 1 || featureDim > 16)
            throw new Error(`unsupported featureDim ${featureDim} (max 2 feature textures = 16 encoded dims)`);
        l0In = featureDim + (sh0Input ? 1 : 0) + nDirectionValues;
        if (l0In % 4 !== 0) throw new Error(`MLP input dim ${l0In} is not a multiple of 4`);
        nPerSplatValues = featureDim;
    }
    // 3-row output layer: (nNeurons / 4) input blocks x 3 output texels each
    const expectedTexels =
        (l0In / 4) * nNeurons + (nHiddenLayers - 1) * (nNeurons / 4) * nNeurons + (nNeurons / 4) * 3;
    if (nWeightTexels !== expectedTexels)
        throw new Error(`weight texel count ${nWeightTexels} != expected ${expectedTexels} `
            + `for the ${baked ? "baked" : "legacy"} layer-0 layout`);

    // slice() copies into fresh, aligned ArrayBuffers.
    const take = (Type, count) => {
        const arr = new Type(buffer.slice(o, o + count * Type.BYTES_PER_ELEMENT));
        o += count * Type.BYTES_PER_ELEMENT;
        return arr;
    };
    const texels = textureWidth * textureHeight;
    const weightData = take(Uint16Array, nWeightTexels * 4); // raw fp16 bits (RGBA16F upload)
    const splatData = take(Uint32Array, texels * 4);
    // per-splat fp16 payload: h0_static (baked) or the encoded features (legacy)
    const numMlpTex = Math.ceil(nPerSplatValues / 8);
    const mlpTexData = [];
    for (let k = 0; k < numMlpTex; k++) mlpTexData.push(take(Uint32Array, texels * 4));
    if (o !== buffer.byteLength)
        console.warn(`ngsplat: ${buffer.byteLength - o} trailing bytes ignored`);

    // Unpack the fp16 centers for the CPU depth sorter (the one derived buffer).
    const centers = new Float32Array(numSplats * 3);
    for (let i = 0; i < numSplats; i++) {
        const w1 = splatData[i * 4 + 1], w2 = splatData[i * 4 + 2];
        centers[i * 3] = halfToFloat(w1);
        centers[i * 3 + 1] = halfToFloat(w1 >>> 16);
        centers[i * 3 + 2] = halfToFloat(w2);
    }

    return {
        numSplats, textureWidth, textureHeight,
        featureDim, nFrequencies, shDegreeMask, degrees, colorActivation, residualActivation,
        nNeurons, nHiddenLayers, sh0Input, properAA, baked,
        nDirectionValues, l0In,
        baseScale, baseOffset,
        camCenter, camUp, camDistance,
        testCameras,
        weightData, nWeightTexels, splatData, mlpTexData, centers,
        fileBytes: buffer.byteLength,
    };
}

// full input width of the TRAINED mlp. In the baked layout the file's layer 0
// holds only the view-direction columns (model.l0In), the rest having been
// pre-multiplied into h0_static, so the two differ there.
export function mlpInputDim(model) {
    return model.featureDim + (model.sh0Input ? 1 : 0)
        + model.degrees.reduce((s, l) => s + 2 * l + 1, 0);
}

// #define block for the capture shader, derived from the model header. The
// degree offsets index into the layer-0 input array: in the baked layout it
// holds only the view-direction SH values (so they start at 0), in the legacy
// layout they follow the encoded features and the SH_C0 constant.
// renderBase / renderResidual mirror the reference renderer's base-color and
// residual debug switches.
export function captureDefines(model, { renderBase, renderResidual, weightSource, precision }) {
    let off = model.baked ? 0 : model.featureDim + (model.sh0Input ? 1 : 0);
    const degreeOff = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
    for (const l of model.degrees) {
        degreeOff[l] = off;
        off += 2 * l + 1;
    }
    return [
        `#define BAKED ${model.baked ? 1 : 0}`,
        `#define N_L0_IN ${model.l0In}`,
        `#define N_MLP_TEX ${model.mlpTexData.length}`,
        `#define N_FEATURES ${model.featureDim}`, // legacy layout only
        `#define SH0_INPUT ${model.sh0Input ? 1 : 0}`, // legacy layout only
        `#define N_NEURONS ${model.nNeurons}`,
        `#define N_HIDDEN ${model.nHiddenLayers}`,
        `#define N_WEIGHT_TEXELS ${model.nWeightTexels}`,
        `#define SH_DEGREE_MASK ${model.shDegreeMask}`,
        `#define COLOR_ACT ${model.colorActivation}`,
        `#define RESIDUAL_ACT ${model.residualActivation}`,
        `#define RENDER_BASE ${renderBase ? 1 : 0}`,
        `#define RENDER_RESIDUAL ${renderResidual ? 1 : 0}`,
        weightSource === "uniform" ? "#define WEIGHTS_UNIFORM" : "",
        precision === "fp16" ? "#define MLP_FP16" : "",
        `#define D1_OFF ${degreeOff[1]}`,
        `#define D2_OFF ${degreeOff[2]}`,
        `#define D3_OFF ${degreeOff[3]}`,
        `#define D4_OFF ${degreeOff[4]}`,
        `#define D5_OFF ${degreeOff[5]}`,
        `#define D6_OFF ${degreeOff[6]}`,
        "",
    ].join("\n");
}

// ============================================================================
// CPU reference implementation of the capture-pass color model — used by the
// viewer's startup probe as ground truth for choosing the capture stage.
// Must match shaders/mlp_eval.glsl exactly, which in turn mirrors the
// reference implementation:
//   base     = base_activation(base_colors)        (exp(3x) for satexp, else x;
//              the exporter pre-bakes the relu/softplus 0.5 shift into d)
//   residual = residual_activation(mlp(input))     (softplus β10 for satexp)
//   color    = color_activation(base + residual)
//   residual off -> color_activation(base)                      (degree-0 path)
//   base off     -> |full color - base-only color|
// ============================================================================

// Reconstruct the MLP weight matrices from the file's fp16 texel stream (layer 0
// is model.l0In wide: the view-direction columns when baked, the full trained
// input otherwise).
export function buildMlpLayers(model) {
    const N = model.nNeurons;
    const readLayer = (baseTexel, outDim, inDim) => {
        const w = Array.from({ length: outDim }, () => new Float64Array(inDim));
        for (let j = 0; j < inDim / 4; j++)
            for (let o = 0; o < outDim; o++) {
                const t = (baseTexel + j * outDim + o) * 4;
                for (let k = 0; k < 4; k++)
                    w[o][4 * j + k] = halfToFloat(model.weightData[t + k]);
            }
        return w;
    };
    const layers = [readLayer(0, N, model.l0In)];
    let base = (model.l0In / 4) * N;
    for (let l = 1; l < model.nHiddenLayers; l++) {
        layers.push(readLayer(base, N, N));
        base += (N / 4) * N;
    }
    layers.push(readLayer(base, 3, N));
    return layers;
}

// SH basis polynomials, degrees 1-6 ascending (matches mlp_eval.glsl)
function shDegreeValues(model, d) {
    const [x, y, z] = d;
    const xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z;
    const x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    const x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;
    const per = {
        1: [-0.48860251190291987 * y, 0.48860251190291987 * z, -0.48860251190291987 * x],
        2: [
            1.0925484305920792 * xy, -1.0925484305920792 * yz,
            0.94617469575755997 * z2 - 0.31539156525251999,
            -1.0925484305920792 * xz,
            0.54627421529603959 * x2 - 0.54627421529603959 * y2,
        ],
        3: [
            0.59004358992664352 * y * (-3.0 * x2 + y2),
            2.8906114426405538 * xy * z,
            0.45704579946446572 * y * (1.0 - 5.0 * z2),
            0.3731763325901154 * z * (5.0 * z2 - 3.0),
            0.45704579946446572 * x * (1.0 - 5.0 * z2),
            1.4453057213202769 * z * (x2 - y2),
            0.59004358992664352 * x * (-x2 + 3.0 * y2),
        ],
        4: [
            2.5033429417967046 * xy * (x2 - y2),
            1.7701307697799304 * yz * (-3.0 * x2 + y2),
            0.94617469575756008 * xy * (7.0 * z2 - 1.0),
            0.66904654355728921 * yz * (3.0 - 7.0 * z2),
            -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293,
            0.66904654355728921 * xz * (3.0 - 7.0 * z2),
            0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0),
            1.7701307697799304 * xz * (-x2 + 3.0 * y2),
            -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4,
        ],
        5: [
            0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4),
            8.3026492595241645 * xy * z * (x2 - y2),
            -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0),
            4.7935367849733241 * xy * z * (3.0 * z2 - 1.0),
            0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0),
            0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0),
            0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0),
            2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0),
            -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0),
            2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4),
            0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4),
        ],
        6: [
            1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4),
            2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4),
            2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0),
            -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0),
            0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0),
            0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0),
            6.6747662381009842 * z2 - 20.024298714302954 * z4 + 14.684485723822165 * z6 - 0.31784601133814211,
            0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0),
            0.46060262975746175 * (x2 - y2) * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0),
            -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0),
            0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4),
            2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4),
            10.247761577878714 * x2 * y4 - 10.247761577878714 * x4 * y2 + 0.6831841051919143 * x6 - 0.6831841051919143 * y6,
        ],
    };
    return model.degrees.flatMap((l) => per[l]);
}

// softplus with beta 10, linear above x > 2 (matches the shader)
const softplus10 = (x) => (x > 2 ? x : Math.log1p(Math.exp(10 * Math.min(x, 2))) * 0.1);

export function referenceCaptureColor(model, layers, i, camPos, camFwd, { renderBase, renderResidual }) {
    const w0 = model.splatData[i * 4];
    const opacity8 = (w0 >>> 24) & 0xFF;
    const satexp = model.colorActivation === 3;
    const act = (x) =>
        model.colorActivation === 1 ? softplus10(x)
        : satexp ? -Math.expm1(-x)
        : Math.max(x, 0);

    // pre-activation base d (range-coded rgb8; the relu/softplus 0.5 shift is baked
    // in by the exporter, the satexp base is stored unshifted)
    const d = [w0 & 0xFF, (w0 >>> 8) & 0xFF, (w0 >>> 16) & 0xFF]
        .map((c) => c * model.baseScale + model.baseOffset);
    const baseTerm = satexp ? d.map((v) => Math.exp(3.0 * v)) : d;

    // the capture shader skips the MLP for splats whose center is behind the
    // camera plane (they are culled by the render pass anyway)
    const c = [model.centers[i * 3], model.centers[i * 3 + 1], model.centers[i * 3 + 2]];
    let rel = [c[0] - camPos[0], c[1] - camPos[1], c[2] - camPos[2]];
    const inFront = rel[0] * camFwd[0] + rel[1] * camFwd[1] + rel[2] * camFwd[2] > 0;

    let residual = [0, 0, 0];
    if (renderResidual && inFront) {
        // per-splat fp16 payload: h0_static (baked) or the encoded features (legacy)
        const perSplat = [];
        for (let k = 0; k < model.mlpTexData.length; k++)
            for (let w = 0; w < 4; w++) {
                const word = model.mlpTexData[k][i * 4 + w];
                perSplat.push(halfToFloat(word), halfToFloat(word >>> 16));
            }
        const len = Math.hypot(...rel);
        rel = rel.map((v) => v / len);

        // layer 0: the baked layout starts from h0_static and feeds it only the
        // direction values, the legacy layout assembles the full input vector
        const inp = model.baked ? [] : [
            ...perSplat.slice(0, model.featureDim),
            ...(model.sh0Input ? [SH_C0] : []),
        ];
        inp.push(...shDegreeValues(model, rel));
        while (inp.length < model.l0In) inp.push(0);
        let h = layers[0].map((row, o) =>
            (model.baked ? perSplat[o] : 0) + row.reduce((s, w, j) => s + w * inp[j], 0));
        for (let l = 1; l < layers.length; l++) {
            h = h.map((v) => Math.max(v, 0));
            h = layers[l].map((row) => row.reduce((s, w, j) => s + w * h[j], 0));
        }
        residual = satexp ? h.map(softplus10)
            : model.residualActivation === 1 ? h.map(Math.tanh)
            : h;
    }
    const rgb = [0, 1, 2].map((k) => {
        const baseColor = act(baseTerm[k]);
        const full = renderResidual ? act(baseTerm[k] + residual[k]) : baseColor;
        const v = renderBase ? full : Math.abs(full - baseColor);
        return Math.round(Math.min(Math.max(v, 0), 1) * 255);
    });
    return [...rgb, opacity8];
}
