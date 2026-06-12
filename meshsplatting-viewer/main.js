import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { fetchFile, loadMSplat, perpendicularTo, readWithProgress, readUrlWithProgress, chooseModel } from "./utils.js";

async function main() {
    const canvas = document.getElementById("canvas");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, depth: true });

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();

    // --- pick a model: a gallery card or a drag & dropped file ---
    const info = document.getElementById("info");
    const gallery = document.getElementById("gallery");
    const sections = document.getElementById("gallery-sections");
    const loading = document.getElementById("loading");
    const progressFill = document.getElementById("progress-fill");
    const progressLabel = document.getElementById("progress-label");

    const manifest = await fetchFile("./models.json", "json");
    const models = manifest?.models ?? [];

    let msplat;
    while (true) {
        const source = await chooseModel(gallery, sections, models);
        const label = source.file ? source.file.name : (source.name ?? source.url);
        // The progress bar is only visible while a file is actively loading.
        loading.hidden = false;
        progressFill.style.width = "0%";
        const onProgress = (frac) => {
            const pct = Math.round(frac * 100);
            progressFill.style.width = pct + "%";
            progressLabel.innerText = `${label} — ${pct}%`;
        };
        try {
            const buffer = source.file
                ? await readWithProgress(source.file, onProgress)
                : await readUrlWithProgress(source.url, onProgress);
            msplat = loadMSplat(buffer);
            break;
        } catch (err) {
            console.error(err);
            loading.hidden = true; // gallery stays up so another model can be picked
        }
    }
    loading.hidden = true;

    gallery.classList.add("hidden");
    document.getElementById("info-stats").innerHTML = [
        ["Vertices", msplat.numVerts.toLocaleString()],
        ["Triangles", msplat.numTris.toLocaleString()],
        ["SH Degree", String(msplat.shDegree)],
    ].map(([k, v]) => `<span class="stat-label">${k}</span><span class="stat-value">${v}</span>`).join("");
    info.hidden = false; // reveal the info card now that a model is loaded
    document.getElementById("back").addEventListener("click", () => location.reload());

    // --- geometry ---
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(msplat.positions, 3));
    geometry.setAttribute("color", new THREE.Uint8BufferAttribute(msplat.colors, 3, true));
    geometry.setIndex(new THREE.Uint32BufferAttribute(msplat.indices, 1));

    // --- packed SH textures (indexed by vertex via gl_VertexID, 2048-wide tiling) ---
    const makeIntTex = (arr, components) => {
        const tex = new THREE.DataTexture(
            arr, msplat.texWidth, msplat.texHeight,
            components === 2 ? THREE.RGIntegerFormat : THREE.RGBAIntegerFormat,
            THREE.UnsignedIntType
        );
        tex.minFilter = THREE.NearestFilter;
        tex.magFilter = THREE.NearestFilter;
        tex.needsUpdate = true;
        return tex;
    };

    const { numSh } = msplat;
    const defines = `#define NUM_SH ${numSh}\n`;
    const uniforms = { cameraPos: { value: new THREE.Vector3() } };
    if (numSh >= 1) {
        uniforms.sh1Tex = { value: makeIntTex(msplat.sh1Data, 2) };
        uniforms.sh1Max = { value: msplat.sh1Max };
    }
    if (numSh >= 2) {
        uniforms.sh2Tex = { value: makeIntTex(msplat.sh2Data, 4) };
        uniforms.sh2Max = { value: msplat.sh2Max };
    }
    if (numSh >= 3) {
        uniforms.sh3Tex = { value: makeIntTex(msplat.sh3Data, 4) };
        uniforms.sh3Max = { value: msplat.sh3Max };
    }

    const vertexShaderSource = defines + await fetchFile("./shaders/render.vert");
    const fragmentShaderSource = defines + await fetchFile("./shaders/render.frag");

    const material = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: vertexShaderSource,
        fragmentShader: fragmentShaderSource,
        uniforms,
        depthTest: true,
        depthWrite: true,
        blending: THREE.NoBlending,
        side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false; // bounds aren't computed for the custom pipeline
    scene.add(mesh);

    // --- camera / orbit controls from the .msplat header ---
    camera.up.copy(msplat.camUp);
    const dir = perpendicularTo(msplat.camUp);
    camera.position.copy(msplat.camCenter).addScaledVector(dir, msplat.camDistance);
    camera.lookAt(msplat.camCenter);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(msplat.camCenter);
    controls.update();

    const resize = () => {
        const w = window.innerWidth, h = window.innerHeight;
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", resize);
    resize();

    let avgMs = 0;
    const fpsEl = document.getElementById("fps");
    const clock = new THREE.Clock();

    renderer.setAnimationLoop(() => {
        const delta = clock.getDelta();
        controls.update();
        uniforms.cameraPos.value.copy(camera.position);
        renderer.render(scene, camera);

        avgMs = avgMs * 0.9 + delta * 1000 * 0.1;
        fpsEl.innerText = `${(1000 / avgMs).toFixed(0)} fps · ${avgMs.toFixed(1)} ms`;
    });
}

main().catch((err) => {
    console.error(err);
});
