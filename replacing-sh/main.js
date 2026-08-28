import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import GUI from "lil-gui";
import {
    fetchFile, loadNgsplat, chooseModel, readUrlWithProgress,
    captureDefines, mlpInputDim, APPEARANCE_MODELS, APPEARANCE_NAMES,
    COLOR_ACTIVATION_NAMES, RESIDUAL_ACTIVATION_NAMES,
    buildMlpLayers, referenceCaptureColor, halfToFloat,
    applyAppearanceOverride,
} from "./utils.js";
import { createSplatSorter } from "./sorter.js";

// ngsplat-viewer — a Three.js viewer for Gaussian splatting exports
// (.ngsplat): standard 3DGS EWA splatting with CPU-sorted alpha blending,
// where the per-splat view-dependent color is
//   color_activation(base + residual(dir))
// with the residual coming from one of five appearance models: quantized
// spherical harmonics (SH), a soft spherical Voronoi partition of the
// direction sphere (SV), anisotropic spherical Gaussian lobes with or without
// a Gabor term (NASG / NASGabor), or a small shared residual MLP over
// per-splat features (Neural). The residual runs in a capture pass — a
// fragment shader over one texel per splat, re-run only when the camera
// moves — and the splat
// pass reads the cached color. The file also carries the test-set viewpoints
// of the training scene; the Benchmark button times each of them at a fixed
// viewport (default 10 renders per view at 1280x720 — resolution, render
// count, and whether sorting is timed are adjustable in the "Benchmark" GUI
// folder; native intrinsics are kept, so 720p matches the reference
// benchmark). See README.md.

const NEAR = 0.2, FAR = 1000.0;

// Benchmark configuration (GUI folder "Benchmark"), snapshotted at the start
// of each run. The viewport is fixed to the chosen preset while the native
// camera intrinsics are kept, so 720p matches the reference benchmark.
const BENCH_RESOLUTIONS = {
    "360p": [640, 360], "480p": [854, 480], "720p": [1280, 720],
    "1080p": [1920, 1080], "1440p": [2560, 1440], "2160p": [3840, 2160],
};
const benchSettings = {
    resolution: "720p",
    repeats: 10,         // timed renders per view
    timeSorting: false,  // include a forced re-sort (worker roundtrip + order
                         // upload) in every timed render
};

const state = {
    // "full" | "base" (residual off) | "residual" (base off) — drives the
    // renderBase/renderResidual flags below, which mirror the reference
    // renderer's base-color and residual debug switches.
    shading: "full",
    sigma: 3.329,          // cutoff in std-devs; minAlpha = exp(-sigma^2/2). 3.329 -> 1/255
    renderBase: true,      // off = |full - base-only| visualization
    renderResidual: true,  // off = base color only (residual eval skipped)
    scaleModifier: 1.0,    // global gaussian scale multiplier
    highRes: false,        // false: half native DPR on high-DPI; true: native capped at 2
    testView: -1,          // baked test-set viewpoint index (-1 = free camera)
    // Capture pass stage: fragment (default) or vertex (fallback for drivers
    // with broken fragment-stage integer fetches). Decided by the startup probe.
    mlpMode: "capture-fragment",
    // MLP weight backing: "uniform" (vec4 uniform array, no per-splat texture
    // fetches — the default whenever the weights fit the device uniform limit)
    // or "texture" (64-wide RGBA16F texture; the fallback, and always available).
    weightSource: "uniform",
    // MLP arithmetic precision: "fp16" (default — matches the reference
    // implementation, which evaluates the MLP in half precision, and ~2x on
    // mobile) or "fp32". Desktop drivers run mediump as fp32, so it only
    // differs on mobile.
    precision: "fp16",
};

// Bottom-centered overlay stack shared by the benchmark panel and the error
// field: children shrink to their longest line, stack vertically (flex `order`
// puts errors below the benchmark panel), and never exceed the viewport.
function overlayContainer() {
    let c = document.getElementById("overlays");
    if (!c) {
        c = document.createElement("div");
        c.id = "overlays";
        // Full-width click-transparent strip: children center themselves and
        // shrink to their content (an abspos parent with left:50% would cap the
        // shrink-to-fit width at half the viewport and force wrapping).
        c.style.cssText = "position:absolute;bottom:8px;left:0;right:0;" +
            "display:flex;flex-direction:column;align-items:center;gap:8px;" +
            "z-index:98;pointer-events:none;";
        document.body.append(c);
    }
    return c;
}
const OVERLAY_BASE_CSS = "border-radius:8px;padding:12px 14px;box-sizing:border-box;" +
    "font:12px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:break-word;" +
    "max-width:calc(100vw - 16px);overflow:auto;pointer-events:auto;";

// Visible error reporting (mobile browsers have no reachable console).
function showFatal(msg) {
    let el = document.getElementById("fatal");
    if (!el) {
        el = document.createElement("div");
        el.id = "fatal";
        el.style.cssText = OVERLAY_BASE_CSS +
            "background:#3a0d0d;color:#ffb4b4;border:1px solid #a33;order:2;max-height:45vh;";
        overlayContainer().append(el);
    }
    el.textContent += (el.textContent ? "\n" : "") + msg;
    console.error(msg);
}

async function main() {
    const canvas = document.getElementById("canvas");
    canvas.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        showFatal("WebGL context lost — the GPU workload is likely too heavy for this device.");
    });
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, depth: false });
    renderer.autoClear = false;
    renderer.debug.onShaderError = (gl, program, vs, fs) => {
        const log = (s) => (gl.getShaderInfoLog(s) || "").trim();
        showFatal(`Shader compile/link failed:\nprogram: ${(gl.getProgramInfoLog(program) || "").trim()}` +
            `\nvertex: ${log(vs)}\nfragment: ${log(fs)}`);
    };

    // --- gallery: pick a model (a manifest card, or ?model=URL) ---
    const gallery = document.getElementById("gallery");
    const sections = document.getElementById("gallery-sections");
    const loading = document.getElementById("loading");
    const progressFill = document.getElementById("progress-fill");
    const progressLabel = document.getElementById("progress-label");

    const manifest = await fetchFile("./models.json", "json").catch(() => null);
    const models = manifest?.models ?? [];
    const params = new URLSearchParams(location.search);
    const urlModel = params.get("model");
    // Headless verification: streamed reads let --virtual-time-budget expire
    // mid-download (virtual time races between chunks), so test mode loads the
    // model as one blocking fetch instead of the progress stream.
    const isTest = Boolean(params.get("test"));

    let model;
    while (true) {
        const source = urlModel ? { url: urlModel } : await chooseModel(sections, models);
        const label = source.name ?? source.url;
        loading.hidden = false;
        progressFill.style.width = "0%";
        const onProgress = (frac) => {
            const pct = Math.round(frac * 100);
            progressFill.style.width = pct + "%";
            progressLabel.innerText = `${label} — ${pct}%`;
        };
        try {
            const buffer = isTest
                ? await (await fetch(source.url)).arrayBuffer()
                : await readUrlWithProgress(source.url, onProgress);
            model = loadNgsplat(buffer);
            break;
        } catch (err) {
            console.error(err);
            loading.hidden = true;
            if (urlModel) throw err; // no gallery to fall back to
        }
    }
    loading.hidden = true;
    gallery.classList.add("hidden");

    // Gallery "Appearance model" selector (?appearance= for direct links): a
    // file not carrying the selected model gets dummy residual parameters
    // from dummy_params.json, so any model can be benchmarked on the same
    // gaussians; "auto" (the default) loads the file as-is.
    const appearanceParam = params.get("appearance");
    const wantedAppearance = APPEARANCE_MODELS.includes(appearanceParam)
        ? appearanceParam
        : document.getElementById("appearance-select")?.value ?? "auto";
    if (wantedAppearance !== "auto")
        model = applyAppearanceOverride(model, wantedAppearance,
            await fetchFile("./dummy_params.json", "json"));

    const { numSplats, textureWidth, textureHeight } = model;
    // Verification hooks: ?residual=0 / ?base=0 preset the capture toggles.
    const urlParams = params;
    if (urlParams.get("residual") === "0") state.renderResidual = false;
    if (urlParams.get("base") === "0") state.renderBase = false;
    state.shading = !state.renderResidual ? "base" : (!state.renderBase ? "residual" : "full");
    const isNeural = model.appearance === "neural";
    const modelName = APPEARANCE_NAMES[model.appearance]
        + (model.dummyParams ? " (dummy params)" : "");
    const bytesPerSplat = 16 + 4 * model.paramTexData.reduce((s, t) => s + t.components, 0);
    const sceneInfo = {
        appearance: modelName,
        splats: numSplats.toLocaleString(),
        fileSize: `${(model.fileBytes / 1024 / 1024).toFixed(1)} MB (${bytesPerSplat} B/splat)`,
        antialiasing: model.properAA ? "on" : "off",
        baseRange: `[${model.baseOffset.toFixed(2)}, `
            + `${(model.baseOffset + 255 * model.baseScale).toFixed(2)}]`,
        baseActivation: model.baseExp ? "exp" : "none",
        colorActivation: COLOR_ACTIVATION_NAMES[model.colorActivation],
        // for sv/nasg/nasgabor the residual activation is pre-applied at export
        residualActivation: RESIDUAL_ACTIVATION_NAMES[model.residualActivation]
            + (["sv", "nasg", "nasgabor"].includes(model.appearance) ? " (baked)" : ""),
        testViews: String(model.testCameras.length),
    };
    if (model.appearance === "sh") {
        sceneInfo.degrees = model.degrees.join(", ");
        sceneInfo.coefficients = `${3 * model.degrees.reduce((s, l) => s + 2 * l + 1, 0)} `
            + `(7/8/6-bit quantized)`;
    } else if (model.appearance === "sv") {
        sceneInfo.sites = String(model.nSites);
    } else if (model.appearance === "nasg" || model.appearance === "nasgabor") {
        sceneInfo.lobes = String(model.nLobes);
    } else {  // neural
        const rawFeatures = model.nFrequencies
            ? model.featureDim / (2 * model.nFrequencies) : model.featureDim;
        const nIn = mlpInputDim(model);
        Object.assign(sceneInfo, {
            features: model.nFrequencies
                ? `${rawFeatures} × freq(${model.nFrequencies}) → ${model.featureDim}`
                : `${model.featureDim}`,
            degrees: [...(model.sh0Input ? [0] : []), ...model.degrees].join(", "),
            // baked: the view-independent inputs (encoded features + the SH_C0
            // constant) are pre-multiplied into a per-splat h0_static cache, so the
            // shader evaluates only the view-direction columns (padded to a multiple
            // of 4); legacy files carry the encoded features and the full matrix
            layer0: model.baked
                ? `baked (${model.featureDim + (model.sh0Input ? 1 : 0)} of ${nIn} inputs pre-multiplied)`
                : "full input",
            inputs: String(nIn),
            hiddenLayers: String(model.nHiddenLayers),
            neurons: String(model.nNeurons),
            // effective (unpadded) weight count: the file stores the output layer
            // already trimmed to 3 rows and rejects padded input widths
            weights: (
                nIn * model.nNeurons
                + (model.nHiddenLayers - 1) * model.nNeurons * model.nNeurons
                + 3 * model.nNeurons
            ).toLocaleString(),
            mlpActivation: "relu", // structurally fixed by the reference implementation
        });
    }
    const perf = { frameTime: "—", sortTime: "—" };
    let fpsCtrl = null;
    let sortCtrl = null;
    let benchCtrl = null;
    const BENCH_BUTTON_LABEL = "Benchmark";

    // --- GPU textures (payloads uploaded exactly as stored in the file) ---
    const makeIntTex = (data, components = 4) => {
        const tex = new THREE.DataTexture(
            data, textureWidth, textureHeight,
            components === 2 ? THREE.RGIntegerFormat : THREE.RGBAIntegerFormat,
            THREE.UnsignedIntType
        );
        tex.minFilter = THREE.NearestFilter;
        tex.magFilter = THREE.NearestFilter;
        tex.needsUpdate = true;
        return tex;
    };
    const splatTexture = makeIntTex(model.splatData);
    // per-splat residual parameter textures (fp16 streams, or the packed sh
    // coefficient levels — the loader reports each texture's u32 components)
    const paramTextures = model.paramTexData.map((t) => makeIntTex(t.data, t.components));
    // MLP weights (Neural appearance model only): raw fp16 bits into a 64-wide
    // RGBA16F texture (texel t at (t % 64, t / 64) — mirrored by the W() macro
    // in eval_neural.glsl). A conventional aspect ratio; degenerate 1-wide tall
    // textures are a known weak spot of some mobile drivers.
    let weightsTexture = null;
    let weightsFloat = null;
    let uniformWeightsFit = false;
    let maxUniformVec = 0;
    if (isNeural) {
        const WEIGHTS_W = 64;
        const weightsH = Math.ceil(model.nWeightTexels / WEIGHTS_W);
        const weightsData = new Uint16Array(WEIGHTS_W * weightsH * 4);
        weightsData.set(model.weightData);
        weightsTexture = new THREE.DataTexture(
            weightsData, WEIGHTS_W, weightsH, THREE.RGBAFormat, THREE.HalfFloatType
        );
        weightsTexture.minFilter = THREE.NearestFilter;
        weightsTexture.magFilter = THREE.NearestFilter;
        weightsTexture.needsUpdate = true;
        // fp16 → fp32 weights for the uniform-array backing (flat vec4 stream)
        weightsFloat = new Float32Array(model.nWeightTexels * 4);
        for (let i = 0; i < weightsFloat.length; i++) weightsFloat[i] = halfToFloat(model.weightData[i]);
        // uniform arrays are capped by the device: only offer the option if the vec4
        // array (plus a few scalar uniforms) fits the smaller of the two stage limits
        const gl = renderer.getContext();
        maxUniformVec = Math.min(
            gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),
            gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS)
        );
        uniformWeightsFit = model.nWeightTexels + 8 <= maxUniformVec;
    }
    if (!uniformWeightsFit) state.weightSource = "texture"; // the uniform default needs the budget

    // --- camera + controls (world up from the file header, derived from the
    //     training cameras; the initial pose is baked test view 0 when available) ---
    const worldUp = new THREE.Vector3(...model.camUp).normalize();
    const camera = new THREE.PerspectiveCamera(60, 1, NEAR, FAR);
    camera.up.copy(worldUp);
    const controls = new OrbitControls(camera, canvas);

    // Places the interactive camera on a baked test-set viewpoint: position and
    // view direction from the c2w (its columns are right/down/forward), vertical
    // fov from the native intrinsics, and the orbit target on the view ray near
    // the scene center so orbiting pivots sensibly. The camera keeps the scene's
    // world up rather than the view's own up, so dragging orbits around the true
    // vertical; that also levels any roll in the capture, exactly as the first
    // controls.update() would (an unrolled view is reproduced unchanged, since
    // its up already lies in the plane spanned by worldUp and the view ray).
    const _basisX = new THREE.Vector3(), _basisY = new THREE.Vector3(), _basisZ = new THREE.Vector3();
    const _rot = new THREE.Matrix4();
    function applyTestViewPose(cam) {
        const m = cam.c2w;
        camera.position.set(m[3], m[7], m[11]);
        camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(cam.height / (2 * cam.fy)));
        camera.updateProjectionMatrix();
        const forward = new THREE.Vector3(m[2], m[6], m[10]).normalize();
        const center = new THREE.Vector3(...model.camCenter);
        const along = Math.max(center.sub(camera.position).dot(forward), 0.25 * model.camDistance);
        controls.target.copy(camera.position).addScaledVector(forward, along);
        camera.up.copy(worldUp);
        camera.lookAt(controls.target);
    }

    if (model.testCameras.length) {
        const startView = Math.min(Math.max(Number(urlParams.get("view")) || 0, 0), model.testCameras.length - 1);
        state.testView = startView;
        applyTestViewPose(model.testCameras[startView]);
    } else {
        // header hint fallback: orbit start on the +x axis of the scene center
        const center = new THREE.Vector3(...model.camCenter);
        camera.position.copy(center).add(new THREE.Vector3(model.camDistance, 0, 0));
        camera.lookAt(center);
        controls.target.copy(center);
    }

    // --- capture pass: per-splat color eval -> RGBA8 cache, one texel per splat ---
    const colorRT = new THREE.WebGLRenderTarget(textureWidth, textureHeight, {
        depthBuffer: false, type: THREE.UnsignedByteType,
        minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
    });
    const orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const fsGeo = new THREE.PlaneGeometry(2, 2);
    const fsVert = `precision highp float;
in vec3 position;
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`;

    // The shared capture scaffolding plus the file's appearance model residual
    // chunk are spliced into both capture variants: the fragment-stage
    // fullscreen quad (default) and the vertex-stage point pass (fallback for
    // drivers whose fragment-stage integer texelFetch is broken — observed on
    // Samsung Android; the vertex stage reads correctly).
    const captureFragWrap = await fetchFile("./shaders/capture.frag");
    const captureVertWrap = await fetchFile("./shaders/capture.vert");
    const capturePointFragSrc = await fetchFile("./shaders/capture_point.frag");
    const captureCommonGlsl = await fetchFile("./shaders/capture_common.glsl");
    const modelEvalGlsl = await fetchFile(`./shaders/eval_${model.appearance}.glsl`);
    const colorEvalGlsl = captureCommonGlsl.replace("// @inject residual_eval", modelEvalGlsl);
    const inject = (wrap, chunk) => wrap.replace("// @inject color_eval", chunk);
    const captureFragSrc = inject(captureFragWrap, colorEvalGlsl);
    const captureVertSrc = inject(captureVertWrap, colorEvalGlsl);

    const captureUniforms = {
        splatData: { value: splatTexture },
        cameraPos: { value: new THREE.Vector3() },
        cameraFwd: { value: new THREE.Vector3() },
        uBase: { value: new THREE.Vector2(model.baseScale, model.baseOffset) },
        uCacheSize: { value: new THREE.Vector2(textureWidth, textureHeight) },
    };
    for (let k = 0; k < paramTextures.length; k++)
        captureUniforms[`paramTex${k}`] = { value: paramTextures[k] };
    if (isNeural) {
        captureUniforms.mlpWeights = { value: weightsTexture }; // active only in texture mode
        captureUniforms.uWeights = { value: weightsFloat };     // active only in uniform mode
    } else if (model.appearance === "sh") {
        // per-level dequantization factor: level scale / signed quantization max
        captureUniforms.shScales = { value: new THREE.Vector3(
            model.shScales[0] / 63, model.shScales[1] / 127, model.shScales[2] / 31
        ) };
    }

    // fragment variant: fullscreen quad
    const captureScene = new THREE.Scene();
    const captureMesh = new THREE.Mesh(fsGeo, null);
    captureScene.add(captureMesh);
    // vertex variant: one 1-pixel point per splat (ids come from gl_VertexID)
    const pointsGeo = new THREE.BufferGeometry();
    pointsGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(numSplats), 1));
    const capturePoints = new THREE.Points(pointsGeo, null);
    capturePoints.frustumCulled = false;
    const capturePointScene = new THREE.Scene();
    capturePointScene.add(capturePoints);

    let captureMat = null;
    function buildCaptureMaterial() {
        const wantFragment = state.mlpMode !== "capture-vertex";
        const defines = captureDefines(model, state);
        const m = new THREE.RawShaderMaterial({
            glslVersion: THREE.GLSL3,
            uniforms: captureUniforms,
            vertexShader: wantFragment ? fsVert : defines + captureVertSrc,
            fragmentShader: wantFragment ? defines + captureFragSrc : capturePointFragSrc,
            depthTest: false, depthWrite: false, blending: THREE.NoBlending,
        });
        if (wantFragment) captureMesh.material = m;
        else capturePoints.material = m;
        if (captureMat) captureMat.dispose();
        captureMat = m;
    }

    // Choose the capture stage by comparing each variant's output against a CPU
    // reference (ground truth) for the first row of splats. Some drivers silently
    // corrupt fragment-stage integer fetches or the color evaluation (no GL error,
    // no compile failure — seen on Samsung Android), so only a functional readback
    // check catches it. The fragment path is cheaper (no per-point primitive/
    // binning work), so it wins whenever it is correct; comparing against ground
    // truth (rather than fragment-vs-vertex agreement) attributes blame to the
    // right variant when the two disagree.
    function chooseCaptureStage() {
        const TOL = 3; // RGBA8 steps: GPU fp32 vs CPU fp64 + output rounding
        // probe in fp32: it tests whether the fetch path is functional (the
        // Samsung corruption is full-scale), and fp16 drift (~1-3 steps on Apple
        // GPUs) would otherwise exceed TOL and wrongly reject a healthy stage
        const savedPrecision = state.precision;
        state.precision = "fp32";
        // probe with the texture weight backing regardless of the active default:
        // the probe exists to catch broken texture-fetch paths
        const savedWeights = state.weightSource;
        state.weightSource = "texture";
        const nProbe = Math.min(64, numSplats);
        captureUniforms.cameraPos.value.copy(camera.position);
        camera.getWorldDirection(captureUniforms.cameraFwd.value);
        const layers = isNeural ? buildMlpLayers(model) : null;
        const camPos = camera.position.toArray();
        const camFwd = captureUniforms.cameraFwd.value.toArray();
        const opts = { renderBase: state.renderBase, renderResidual: state.renderResidual };
        const expected = new Uint8Array(nProbe * 4);
        for (let i = 0; i < nProbe; i++)
            expected.set(referenceCaptureColor(model, layers, i, camPos, camFwd, opts), i * 4);

        const errorFor = (mode) => {
            state.mlpMode = mode;
            buildCaptureMaterial();
            renderer.setRenderTarget(colorRT);
            renderer.setClearColor(0x000000, 0);
            renderer.clear(true, false, false);
            renderer.render(mode === "capture-vertex" ? capturePointScene : captureScene, orthoCamera);
            const out = new Uint8Array(textureWidth * 4);
            renderer.readRenderTargetPixels(colorRT, 0, 0, textureWidth, 1, out);
            let maxErr = 0;
            for (let i = 0; i < nProbe * 4; i++)
                maxErr = Math.max(maxErr, Math.abs(out[i] - expected[i]));
            return maxErr;
        };
        const errFrag = errorFor("capture-fragment");
        let stage;
        if (errFrag <= TOL) {
            probeInfo = "auto";
            stage = "capture-fragment";
        } else {
            const errVert = errorFor("capture-vertex");
            probeInfo = "auto";
            console.warn(`ngsplat: fragment-stage capture mismatches the CPU reference (max err ${errFrag}, vertex ${errVert}) — ` +
                `using ${errVert <= errFrag ? "vertex" : "fragment"}-stage capture`);
            stage = errVert <= errFrag ? "capture-vertex" : "capture-fragment";
        }
        state.precision = savedPrecision;
        state.weightSource = savedWeights;
        return stage;
    }
    let probeInfo = "forced";
    const forcedMode = urlParams.get("capture");
    state.mlpMode = (forcedMode === "fragment" || forcedMode === "vertex")
        ? `capture-${forcedMode}`
        : chooseCaptureStage(); // always runs in the texture weight mode
    // apply the weight-source / precision choices after the stage probe
    if (urlParams.get("weights") === "uniform" && uniformWeightsFit) state.weightSource = "uniform";
    if (urlParams.get("weights") === "texture") state.weightSource = "texture";
    if (urlParams.get("precision") === "fp32" || urlParams.get("precision") === "fp16")
        state.precision = urlParams.get("precision");
    buildCaptureMaterial();

    // --- render pass: EWA splats, CPU-sorted back-to-front alpha blending ---
    const splatDecodeGlsl = await fetchFile("./shaders/splat_decode.glsl");
    const renderVert = (await fetchFile("./shaders/render_ewa.vert"))
        .replace("// @inject splat_decode", splatDecodeGlsl);
    const renderFrag = await fetchFile("./shaders/render_ewa.frag");

    const uniforms = {
        splatData: { value: splatTexture },
        colorCache: { value: colorRT.texture },
        viewMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        halfWH: { value: new THREE.Vector2() },
        uMinAlpha: { value: Math.exp(-state.sigma * state.sigma / 2) },
        uScaleModifier: { value: state.scaleModifier },
    };

    const geometry = new THREE.PlaneGeometry(2, 2);
    // CPU depth sorter: instance i draws splat splatIndex[i], refilled
    // back-to-front by the worker whenever the camera moves. `count` is the
    // number of leading in-front splats — the rest are behind the camera and
    // skipped entirely via the instanced draw count (as in PlayCanvas).
    const sortOrder = new Float32Array(numSplats);
    for (let i = 0; i < numSplats; i++) sortOrder[i] = i;
    const sortAttr = new THREE.InstancedBufferAttribute(sortOrder, 1);
    sortAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("splatIndex", sortAttr);
    let onSortApplied = null; // one-shot hook used by the benchmark to await a sort
    const sorter = createSplatSorter(model.centers, numSplats, (order, count, sortTime) => {
        sortAttr.array.set(order);
        sortAttr.needsUpdate = true;
        mesh.count = count;
        perf.sortTime = `${sortTime.toFixed(1)} ms (${count.toLocaleString()} drawn)`;
        if (sortCtrl) sortCtrl.updateDisplay();
        if (onSortApplied) { const fn = onSortApplied; onSortApplied = null; fn(); }
    });
    const _sortBackward = new THREE.Vector3();

    const material = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: (model.properAA ? "#define PROPER_AA\n" : "") + renderVert,
        fragmentShader: renderFrag,
        uniforms,
        depthTest: false, depthWrite: false,
        blending: THREE.NormalBlending,
        transparent: true,
    });
    const scene = new THREE.Scene();
    const mesh = new THREE.InstancedMesh(geometry, material, numSplats);
    mesh.frustumCulled = false;
    scene.add(mesh);

    let cameraChanged = true;
    // High Res toggle (from supersplat): auto mode treats DPR >= 2 as high-DPI
    // and renders at half native; High Res forces native, capped at 2.
    const effectiveDpr = () => {
        const dpr = window.devicePixelRatio || 1;
        return state.highRes ? Math.min(dpr, 2) : (dpr >= 2 ? dpr * 0.5 : dpr);
    };
    const resize = () => {
        const w = window.innerWidth, h = window.innerHeight;
        renderer.setPixelRatio(effectiveDpr());
        renderer.setSize(w, h);
        uniforms.halfWH.value.set(w / 2, h / 2); // CSS px (DPR-invariant projection math)
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        cameraChanged = true;
    };
    window.addEventListener("resize", resize);
    controls.addEventListener("change", () => { cameraChanged = true; });
    let testViewCtrl = null;
    controls.addEventListener("start", () => {
        // manual navigation leaves the pinned test view
        if (state.testView !== -1) {
            state.testView = -1;
            if (testViewCtrl) testViewCtrl.updateDisplay();
        }
    });

    setupGui();
    resize();

    // Per-camera uniform updates for the render and capture passes. The render
    // projection is taken from camera.projectionMatrix, so the benchmark can
    // substitute its own intrinsics-derived matrix without touching this.
    const updateCameraUniforms = () => {
        camera.updateMatrixWorld(true);
        uniforms.viewMatrix.value.copy(camera.matrixWorldInverse);
        uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
        captureUniforms.cameraPos.value.copy(camera.position);
        camera.getWorldDirection(captureUniforms.cameraFwd.value);
    };

    const runCapture = () => {
        renderer.setRenderTarget(colorRT);
        if (state.mlpMode === "capture-vertex") {
            renderer.setClearColor(0x000000, 0); // padded tail texels stay defined
            renderer.clear(true, false, false);
            renderer.render(capturePointScene, orthoCamera);
        } else {
            renderer.render(captureScene, orthoCamera); // quad covers every texel
        }
    };

    const clock = new THREE.Clock();
    let avgMs = 0;
    const testN = Number(urlParams.get("test")) || 0;
    const testNoRender = testN > 0 && urlParams.get("norender") === "1";
    let framesRendered = 0;

    const animationLoop = () => {
        const delta = clock.getDelta();
        controls.update();

        if (cameraChanged) {
            updateCameraUniforms();
            // camera backward axis (world-matrix Z column), as PlayCanvas passes
            _sortBackward.setFromMatrixColumn(camera.matrixWorld, 2).normalize();
            sorter.setCamera(camera.position, _sortBackward);
            runCapture(); // captured colors depend only on the camera position
            cameraChanged = false;
        }

        if (!testNoRender) {
            renderer.setRenderTarget(null);
            renderer.setClearColor(0x000000, 1);
            renderer.clear(true, false, false);
            renderer.render(scene, camera);
        }

        const ms = delta * 1000;
        avgMs = avgMs * 0.9 + ms * 0.1;
        perf.frameTime = avgMs.toFixed(2) + " ms";
        if (fpsCtrl) fpsCtrl.updateDisplay();

        // Refreshed every frame: headless Chrome only ticks a few rAF frames
        // under --virtual-time-budget, so the dump grabs whatever is latest.
        if (testN) { framesRendered++; writeTestResult(); }
    };
    renderer.setAnimationLoop(animationLoop);

    // Headless-verification mode (?test=N): dump the first N capture-pass texels
    // and full-render stats as JSON into a #result div (read via --dump-dom).
    function writeTestResult() {
        let div = document.getElementById("result");
        if (!div) {
            div = document.createElement("div");
            div.id = "result";
            document.body.append(div);
        }
        try {
            div.textContent = JSON.stringify(computeTestResult());
        } catch (err) {
            div.textContent = JSON.stringify({ error: String(err?.stack ?? err) });
        }
    }

    function computeTestResult() {
        const cacheBytes = window.__ngs.readColorCache(testN);
        // Capture-only mode (&norender=1): skip the full-scene test render —
        // software GL can't rasterize million-splat scenes in reasonable time.
        if (urlParams.get("norender") === "1") {
            return {
                numSplats,
                frames: framesRendered,
                cameraPos: camera.position.toArray(),
                cameraFwd: camera.getWorldDirection(new THREE.Vector3()).toArray(),
                colorCache: Array.from(cacheBytes.slice(0, testN * 4)),
                renderNonBlack: null,
                renderTotal: 0,
            };
        }
        const SIZE = 512;
        const rt = new THREE.WebGLRenderTarget(SIZE, SIZE, {
            depthBuffer: false, type: THREE.UnsignedByteType,
        });
        renderer.setRenderTarget(rt);
        renderer.setClearColor(0x000000, 1);
        renderer.clear(true, false, false);
        renderer.render(scene, camera);
        const px = new Uint8Array(SIZE * SIZE * 4);
        renderer.readRenderTargetPixels(rt, 0, 0, SIZE, SIZE, px);
        rt.dispose();
        let nonBlack = 0;
        for (let i = 0; i < px.length; i += 4)
            if (px[i] | px[i + 1] | px[i + 2]) nonBlack++;
        // PNG of the render for visual checks (readback rows are bottom-up: flip)
        const cnv = document.createElement("canvas");
        cnv.width = SIZE; cnv.height = SIZE;
        const ctx = cnv.getContext("2d");
        const img = ctx.createImageData(SIZE, SIZE);
        for (let y = 0; y < SIZE; y++)
            img.data.set(px.subarray((SIZE - 1 - y) * SIZE * 4, (SIZE - y) * SIZE * 4), y * SIZE * 4);
        for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;
        ctx.putImageData(img, 0, 0);
        return {
            numSplats,
            frames: framesRendered,
            cameraPos: camera.position.toArray(),
            cameraFwd: camera.getWorldDirection(new THREE.Vector3()).toArray(),
            colorCache: Array.from(cacheBytes.slice(0, testN * 4)),
            renderNonBlack: nonBlack,
            renderTotal: SIZE * SIZE,
            renderPng: cnv.toDataURL("image/png"),
        };
    }

    // ---- benchmark: fixed-viewport renders of the baked test-set viewpoints ----
    // Every test view is applied with its exact pose and native intrinsics
    // (the viewport is forced to the configured preset with the intrinsics
    // kept — at 720p exactly like the reference benchmark). First an untimed
    // warmup pass over the whole test set, then per view a full re-sort
    // (awaiting the worker), one untimed warmup render, and the configured
    // timed renders; every render includes the capture pass (the moving-camera
    // frame cost) and ends with a 1x1 readback so timings include GPU
    // completion (and are not vsync-quantized). By default the sort and its
    // attribute upload stay outside the timing, matching the viewer's
    // amortized async-sorter design — the numbers are the steady-state frame
    // cost at a novel camera position; the "Time sorting" setting instead
    // folds a forced re-sort into every timed render (end-to-end novel-view
    // cost). Files without baked cameras fall back to a seeded random-orbit
    // sequence. Everything runs as currently configured (shading, and for
    // Neural files precision and weight backing).
    let benchmarkRunning = false;

    // The benchmark button doubles as its progress bar: label text + a partial
    // background fill on the button element. Updates happen only between timed
    // sections (during the per-view event-loop yield), so the repaint never
    // lands inside the synchronous render+readback loop being measured.
    function updateBenchButton(label, frac) {
        if (!benchCtrl) return;
        benchCtrl.name(label);
        const btn = benchCtrl.domElement.querySelector("button");
        if (btn) {
            const pct = frac == null ? null : Math.round(frac * 100);
            btn.style.background = pct == null ? "" :
                `linear-gradient(to right, rgba(78, 161, 255, 0.45) ${pct}%, transparent ${pct}%)`;
        }
    }

    // off-axis projection from the native intrinsics at the benchmark viewport
    // (CV convention: pixel u = cx + fx * X/Z with z forward, v measured from top)
    function benchProjection(cam, W, H) {
        const n = NEAR, f = FAR;
        return new THREE.Matrix4().set(
            2 * cam.fx / W, 0, 1 - 2 * cam.cx / W, 0,
            0, 2 * cam.fy / H, 2 * cam.cy / H - 1, 0,
            0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
            0, 0, -1, 0
        );
    }

    function randomOrbitPoses(count) {
        const mulberry32 = (a) => () => {
            a |= 0; a = (a + 0x6D2B79F5) | 0;
            let t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
        const rand = mulberry32(0x5EED);
        const target = controls.target.clone();
        const up = camera.up.clone().normalize();
        const baseDist = camera.position.distanceTo(target);
        const e1 = new THREE.Vector3(1, 0, 0);
        if (Math.abs(up.dot(e1)) > 0.9) e1.set(0, 0, 1);
        e1.cross(up).normalize();
        const e2 = new THREE.Vector3().crossVectors(up, e1).normalize();
        return Array.from({ length: count }, () => {
            const azim = rand() * Math.PI * 2;
            const elev = (rand() * 0.9 - 0.2) * (Math.PI / 4); // -9deg .. +31deg
            const r = baseDist * (0.6 + 0.9 * rand());
            return {
                position: new THREE.Vector3()
                    .addScaledVector(e1, Math.cos(azim) * Math.cos(elev) * r)
                    .addScaledVector(e2, Math.sin(azim) * Math.cos(elev) * r)
                    .addScaledVector(up, Math.sin(elev) * r)
                    .add(target),
                target,
            };
        });
    }

    async function runBenchmark() {
        if (benchmarkRunning) return;
        benchmarkRunning = true;
        renderer.setAnimationLoop(null); // pause the app loop
        benchCtrl?.disable();
        updateBenchButton("preparing…", 0);

        // settings snapshot: GUI changes during the run have no effect
        const [benchW, benchH] = BENCH_RESOLUTIONS[benchSettings.resolution];
        const repeats = Math.max(1, Math.round(benchSettings.repeats));
        const timeSorting = benchSettings.timeSorting;

        const useTestCams = model.testCameras.length > 0;
        const views = useTestCams ? model.testCameras : randomOrbitPoses(16);
        const fallbackProjection = new THREE.Matrix4().copy(camera.projectionMatrix);
        if (!useTestCams) {
            // no baked cameras: benchmark viewport aspect with the current fov
            const saved = camera.aspect;
            camera.aspect = benchW / benchH;
            camera.updateProjectionMatrix();
            fallbackProjection.copy(camera.projectionMatrix);
            camera.aspect = saved;
        }
        const applyPose = (i) => {
            if (useTestCams) {
                const cam = views[i];
                const m = cam.c2w;
                _basisX.set(m[0], m[4], m[8]);
                _basisY.set(-m[1], -m[5], -m[9]);
                _basisZ.set(-m[2], -m[6], -m[10]);
                camera.position.set(m[3], m[7], m[11]);
                camera.quaternion.setFromRotationMatrix(_rot.makeBasis(_basisX, _basisY, _basisZ));
                camera.projectionMatrix.copy(benchProjection(cam, benchW, benchH));
            } else {
                camera.position.copy(views[i].position);
                camera.lookAt(views[i].target);
                camera.projectionMatrix.copy(fallbackProjection);
            }
            camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
        };

        // settings snapshot (used unchanged during the run, shown with the results
        // as an aligned key-value table with short lines)
        const kv = (k, v) => `${k.padEnd(9)} ${v}`;
        const captureStage = state.mlpMode === "capture-vertex" ? "vertex shader" : "fragment shader";
        const shadingName = { full: "full", base: "base only", residual: "residual only" }[state.shading];
        const settings = [
            kv("splats", numSplats.toLocaleString()),
            kv("views", useTestCams
                ? `${views.length} test-set viewpoints`
                : `${views.length} random orbits (no baked test cameras)`),
            kv("res", `${benchW}×${benchH} fixed (native intrinsics)`),
            kv("renders", `${repeats} timed + 1 warmup / view`),
            kv("sorting", timeSorting
                ? "timed (forced re-sort per render)" : "untimed (async, pre-applied)"),
            kv("stage", `${captureStage} (${probeInfo})`),
            ...(isNeural ? [
                kv("weights", state.weightSource === "uniform" ? "uniform array" : "texture"),
                kv("precision", state.precision),
            ] : []),
            kv("shading", shadingName),
            kv("σ cutoff", `${Math.round(state.sigma * 1000) / 1000}σ`),
            kv("scale", state.scaleModifier.toFixed(2)),
        ].join("\n");

        const savedPos = camera.position.clone();
        const savedQuat = camera.quaternion.clone();
        const savedHalfWH = uniforms.halfWH.value.clone();
        uniforms.halfWH.value.set(benchW / 2, benchH / 2);
        const rt = new THREE.WebGLRenderTarget(benchW, benchH, {
            depthBuffer: false, type: THREE.UnsignedByteType,
        });
        const px = new Uint8Array(4);

        // progress units for the button fill: one warmup pass over all views
        // followed by one measured pass over all views
        const V = views.length;
        const totalUnits = 2 * V;

        // apply view v's pose and await the worker's sort order for it (safety
        // timeout in case a message is lost). `jitter` nudges the camera by more
        // than the worker's 0.001 dedup epsilon so that a repeated sort of the
        // SAME view is not silently skipped (used when sorting is timed;
        // ~0.002 scene units is visually and cost-wise negligible).
        const sortView = async (v, jitter = 0) => {
            applyPose(v);
            if (jitter) {
                camera.position.x += jitter;
                camera.position.y += jitter;
                camera.position.z += jitter;
            }
            updateCameraUniforms();
            _sortBackward.setFromMatrixColumn(camera.matrixWorld, 2).normalize();
            await new Promise((resolve) => {
                const timer = setTimeout(resolve, 2000);
                onSortApplied = () => { clearTimeout(timer); resolve(); };
                sorter.setCamera(camera.position, _sortBackward);
            });
            onSortApplied = null;
        };

        const frame = () => {
            runCapture();
            renderer.setRenderTarget(rt);
            renderer.setClearColor(0x000000, 1);
            renderer.clear(true, false, false);
            renderer.render(scene, camera);
            renderer.readRenderTargetPixels(rt, 0, 0, 1, 1, px); // GPU sync
        };
        // yield the event loop between views (a rAF await would stall in
        // hidden/headless tabs, where rAF is throttled to zero)
        const yieldLoop = () => new Promise((resolve) => setTimeout(resolve, 0));

        // An UNTIMED warmup pass over the whole test set (sort + one render per
        // view — shader compilation, allocator and driver warm-up, so the first
        // measured view isn't special), then the measured pass: per view, await
        // the sort, run one untimed warmup render (absorbs the fresh
        // sort-attribute upload), and time `repeats` renders. With "Time
        // sorting" on, every timed sample is instead a forced re-sort (worker
        // roundtrip + order upload, jitter defeats the dedup) followed by one
        // full render — end-to-end novel-view cost. Orders are never cached
        // across views (4·splats bytes each would pile up on phones with many
        // test views); each view is sorted on demand right before its renders.
        const viewMeans = [];
        try {
            for (let v = 0; v < V; v++) {
                window.__ngsBenchProgress = { phase: "warmup", view: v, of: V };
                updateBenchButton(`${modelName} warmup ${v + 1}/${V}`, v / totalUnits);
                await sortView(v);
                frame();
                await yieldLoop();
            }
            for (let v = 0; v < V; v++) {
                window.__ngsBenchProgress = { phase: "measure", view: v, of: V };
                updateBenchButton(`${modelName} ${v + 1}/${V}`, (V + v) / totalUnits);
                await sortView(v);
                frame(); // warmup render: absorbs the fresh sort-attribute upload
                if (timeSorting) {
                    let total = 0;
                    for (let r = 0; r < repeats; r++) {
                        const t0 = performance.now();
                        await sortView(v, (r % 2 ? 1 : -1) * 0.002);
                        frame();
                        total += performance.now() - t0;
                    }
                    viewMeans.push(total / repeats);
                } else {
                    const t0 = performance.now();
                    for (let r = 0; r < repeats; r++) frame();
                    viewMeans.push((performance.now() - t0) / repeats);
                }
                await yieldLoop();
            }
        } finally {
            rt.dispose();
            onSortApplied = null;
            updateBenchButton(BENCH_BUTTON_LABEL, null);
            benchCtrl?.enable();
            uniforms.halfWH.value.copy(savedHalfWH);
            camera.position.copy(savedPos);
            camera.quaternion.copy(savedQuat);
            camera.updateProjectionMatrix(); // drop the benchmark projection
            cameraChanged = true;
            renderer.setAnimationLoop(animationLoop);
            benchmarkRunning = false;
        }

        // stats: label row + the appearance model's row, column-aligned monospace
        const sorted = [...viewMeans].sort((a, b) => a - b);
        const avg = sorted.reduce((s, v) => s + v, 0) / sorted.length;
        const cells = [
            ["avg", `${avg.toFixed(2)} ms`],
            ["med", `${sorted[sorted.length >> 1].toFixed(2)} ms`],
            ["min", `${sorted[0].toFixed(2)} ms`],
            ["max", `${sorted[sorted.length - 1].toFixed(2)} ms`],
            ["fps", (1000 / avg).toFixed(1)],
        ];
        const colW = cells.map(([k, v]) => Math.max(k.length, v.length));
        const header = "".padEnd(modelName.length) + "   " +
            cells.map(([k], i) => k.padEnd(colW[i])).join("   ").trimEnd();
        const row = modelName + "   " +
            cells.map(([, v], i) => v.padEnd(colW[i])).join("   ").trimEnd();
        const statsText = [header, row].join("\n");

        // on-screen results panel (tap to dismiss): bottom-centered in the shared
        // overlay stack, shrink-to-fit its longest line, above any error field
        document.getElementById("bench-results")?.remove();
        const el = document.createElement("div");
        el.id = "bench-results";
        el.style.cssText = OVERLAY_BASE_CSS +
            "background:#101318;color:#cfd2da;border:1px solid rgba(255,255,255,0.2);" +
            "cursor:pointer;order:1;";
        el.textContent =
            `Benchmark — ${views.length} views × ${repeats} renders, re-sorted per view, ` +
            `warmup pass\n\n` +
            `${settings}\n\n${statsText}\n\n(tap to dismiss)`;
        el.addEventListener("click", () => el.remove());
        overlayContainer().append(el);
        // machine-readable copy for the headless harness
        window.__ngsBenchmark = {
            model: model.appearance, dummyParams: Boolean(model.dummyParams),
            views: views.length, repeats, timeSorting,
            width: benchW, height: benchH, viewMeans,
        };
    }

    function setupGui() {
        const gui = new GUI({ title: "ngsplat-viewer" });
        gui.domElement.style.position = "absolute";
        gui.domElement.style.top = "8px";
        gui.domElement.style.right = "8px";
        gui.domElement.style.left = "auto";

        // lil-gui has no separator widget: a styled div appended to the root's
        // children container renders as a horizontal rule in insertion order
        const addSeparator = () => {
            const sep = document.createElement("div");
            sep.style.cssText = "border-top:1px solid rgba(255,255,255,0.18);margin:6px 8px;";
            gui.$children.appendChild(sep);
        };

        gui.add({ gallery: () => location.assign(location.pathname) }, "gallery").name("← Gallery");

        addSeparator();
        fpsCtrl = gui.add(perf, "frameTime").name("Frame time").disable();
        sortCtrl = gui.add(perf, "sortTime").name("Sort time").disable();
        addSeparator();

        // base-color / residual debug switches of the reference renderer
        gui.add(state, "shading", {
            "Full": "full",
            "Base color only": "base",
            "Residual only": "residual",
        }).name("Shading")
            .onChange(() => {
                state.renderResidual = state.shading !== "base";
                state.renderBase = state.shading !== "residual";
                buildCaptureMaterial();
                cameraChanged = true;
            });

        if (model.testCameras.length) {
            testViewCtrl = gui.add(state, "testView", -1, model.testCameras.length - 1, 1)
                .name("Test view")
                .onChange(() => {
                    if (state.testView < 0) return;
                    applyTestViewPose(model.testCameras[state.testView]);
                    cameraChanged = true;
                });
        }

        // Each option shows the N-sigma radius and the alpha floor it implies.
        const cutoffModes = [
            { sigma: 1 }, { sigma: 2 }, { sigma: 3 },
            { sigma: 3.329, label: "3.329σ  (α ≥ 1/255)" },
        ];
        const sigmaOptions = {};
        for (const m of cutoffModes) {
            const alpha = Math.exp(-m.sigma * m.sigma / 2);
            const label = m.label ?? `${Math.round(m.sigma * 100) / 100}σ  (α ≥ ${alpha.toFixed(2)})`;
            sigmaOptions[label] = m.sigma;
        }
        gui.add(state, "sigma", sigmaOptions).name("Sigma cutoff")
            .onChange(() => {
                uniforms.uMinAlpha.value = Math.exp(-state.sigma * state.sigma / 2);
            });

        gui.add(state, "scaleModifier", 0.05, 2.0, 0.05).name("Scale modifier")
            .onChange(() => { uniforms.uScaleModifier.value = state.scaleModifier; });

        // High res only changes anything on high-DPI displays (effectiveDpr is
        // the identity below a devicePixelRatio of 2), so don't offer it elsewhere
        if ((window.devicePixelRatio || 1) >= 2)
            gui.add(state, "highRes").name("High res").onChange(resize);

        if (isNeural) {
            // MLP weight backing — uniform array (default) vs texture (the uniform
            // option only when it fits the device's uniform-vector budget; the
            // control is pinned to texture + noted otherwise)
            const weightCtrl = gui.add(state, "weightSource", {
                "Uniform array": "uniform",
                "Texture": "texture",
            }).name("MLP weights")
                .onChange(() => { buildCaptureMaterial(); cameraChanged = true; });
            if (!uniformWeightsFit) {
                state.weightSource = "texture";
                weightCtrl.setValue("texture").disable()
                    .name(`MLP weights (uniform > ${maxUniformVec} vec4 limit)`);
            }

            gui.add(state, "precision", { "fp16": "fp16", "fp32": "fp32" }).name("MLP precision")
                .onChange(() => { buildCaptureMaterial(); cameraChanged = true; });
        }

        const modelFolder = gui.addFolder("Model info");
        modelFolder.add(sceneInfo, "appearance").name("Appearance").disable();
        modelFolder.add(sceneInfo, "splats").name("Splats").disable();
        modelFolder.add(sceneInfo, "fileSize").name("File size").disable();
        modelFolder.add(sceneInfo, "antialiasing").name("Mip filter").disable();
        modelFolder.add(sceneInfo, "baseRange").name("Base range").disable();
        if (model.appearance === "sh") {
            modelFolder.add(sceneInfo, "degrees").name("SH degrees").disable();
            modelFolder.add(sceneInfo, "coefficients").name("Coefficients").disable();
        } else if (model.appearance === "sv") {
            modelFolder.add(sceneInfo, "sites").name("Voronoi sites").disable();
        } else if (model.appearance === "nasg" || model.appearance === "nasgabor") {
            modelFolder.add(sceneInfo, "lobes")
                .name(model.appearance === "nasg" ? "Lobes" : "Gabor lobes").disable();
        } else {  // neural
            modelFolder.add(sceneInfo, "features").name("Features").disable();
            modelFolder.add(sceneInfo, "degrees").name("SH degrees").disable();
        }
        modelFolder.add(sceneInfo, "testViews").name("Test views").disable();
        if (isNeural) {
            modelFolder.add(sceneInfo, "layer0").name("Layer 0").disable();
            const mlpFolder = modelFolder.addFolder("MLP");
            mlpFolder.add(sceneInfo, "weights").name("Weights").disable();
            mlpFolder.add(sceneInfo, "inputs").name("Inputs").disable();
            mlpFolder.add(sceneInfo, "hiddenLayers").name("Hidden layers").disable();
            mlpFolder.add(sceneInfo, "neurons").name("Neurons / layer").disable();
            mlpFolder.add(sceneInfo, "mlpActivation").name("Activation").disable();
        }
        modelFolder.add(sceneInfo, "baseActivation").name("Base activation").disable();
        modelFolder.add(sceneInfo, "colorActivation").name("Color activation").disable();
        modelFolder.add(sceneInfo, "residualActivation").name("Residual activation").disable();
        modelFolder.close();

        const benchFolder = gui.addFolder("Benchmark");
        benchFolder.add(benchSettings, "resolution", Object.keys(BENCH_RESOLUTIONS))
            .name("Resolution");
        benchFolder.add(benchSettings, "repeats", 1, 100, 1).name("Timed renders / view");
        // include a forced re-sort (worker roundtrip + order upload) in every
        // timed render — end-to-end novel-view cost instead of steady-state
        benchFolder.add(benchSettings, "timeSorting").name("Time sorting");
        benchCtrl = benchFolder.add({ benchmark: () => runBenchmark() }, "benchmark")
            .name(BENCH_BUTTON_LABEL);
        benchFolder.close();
    }

    // ?bench=1 auto-runs the benchmark once the first frame is up and mirrors
    // the results into a #bench-json div (for headless / scripted runs).
    if (urlParams.get("bench")) {
        requestAnimationFrame(async () => {
            await runBenchmark();
            const div = document.createElement("div");
            div.id = "bench-json";
            div.textContent = JSON.stringify(window.__ngsBenchmark);
            document.body.append(div);
        });
    }

    // Hooks for headless verification harnesses.
    window.__ngs = {
        model, camera, controls, renderer, benchSettings,
        recapture() { runCapture(); },
        runBenchmark,
        // Read n texels of the capture-pass color cache (RGBA8).
        readColorCache(n) {
            const h = Math.ceil(n / textureWidth);
            const out = new Uint8Array(textureWidth * h * 4);
            renderer.readRenderTargetPixels(colorRT, 0, 0, textureWidth, h, out);
            return out;
        },
    };
    window.__ngsReady = true;
}

main().catch((err) => {
    window.__ngsError = String(err?.stack ?? err);
    showFatal(String(err?.stack ?? err));
});
