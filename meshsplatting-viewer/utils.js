import * as THREE from "three";

export async function fetchFile(path, type = "text") {
    try {
        const response = await fetch(path);
        if (!response.ok) {
            // noinspection ExceptionCaughtLocallyJS
            throw new Error("Unable to load file, Error " + response.status);
        }
        if (type === "none") return response;
        return response[type]();
    } catch (error) {
        console.error(error.message);
    }
}

// Unit direction perpendicular to `up`, used to place the initial camera. Picks the
// world axis least parallel to `up` so the cross product is well-conditioned (avoids
// the degenerate forward=-up case in the WebGPU viewer's draw()).
export function perpendicularTo(up) {
    const axes = [
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, 1),
    ];
    let best = axes[0], bestDot = Infinity;
    for (const a of axes) {
        const d = Math.abs(up.dot(a));
        if (d < bestDot) { bestDot = d; best = a; }
    }
    return new THREE.Vector3().crossVectors(up, best).normalize();
}

// Resolve with the first file the user drags onto the page, toggling the overlay's
// `dragging` highlight while a file hovers.
export function waitForDrop(dropZone) {
    return new Promise((resolve) => {
        const onDragOver = (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            dropZone.classList.add("dragging");
        };
        const onDragLeave = (e) => {
            if (e.relatedTarget === null) dropZone.classList.remove("dragging"); // left the window
        };
        const onDrop = (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragging");
            const file = e.dataTransfer.files[0];
            if (!file) return;
            window.removeEventListener("dragover", onDragOver);
            window.removeEventListener("dragleave", onDragLeave);
            window.removeEventListener("drop", onDrop);
            resolve(file);
        };
        window.addEventListener("dragover", onDragOver);
        window.addEventListener("dragleave", onDragLeave);
        window.addEventListener("drop", onDrop);
    });
}

// Read a File into an ArrayBuffer, reporting fractional progress as it streams in.
export async function readWithProgress(file, onProgress) {
    const data = new Uint8Array(file.size);
    const reader = file.stream().getReader();
    let offset = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        data.set(value, offset);
        offset += value.length;
        onProgress(offset / file.size);
    }
    return data.buffer;
}

// Fetch a URL into an ArrayBuffer, reporting fractional progress as it streams in. Unlike a File
// the total size isn't known up front, so chunks are collected and progress is driven by the
// Content-Length header when present (otherwise onProgress is only called at completion).
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

// Render the model grid (one section per `group`, preserving first-seen order) and resolve with
// the user's choice: a clicked manifest entry { name, url } or a dropped { file }. Idempotent —
// rebuilds `sections` each call — so it can be re-armed after a failed load. Cleans up its
// listeners once a choice is made.
export function chooseModel(gallery, sections, models) {
    return new Promise((resolve) => {
        const cleanups = [];
        const choose = (source) => {
            for (const fn of cleanups) fn();
            resolve(source);
        };
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
                card.addEventListener("click", () => choose({ name: model.name, url: model.url }));
                grid.append(card);
            }
            const section = el("div", "gallery-section");
            section.append(el("div", "section-title", title), grid);
            sections.append(section);
        }

        // Drag & drop highlights the whole gallery (mirrors waitForDrop above).
        const onDragOver = (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            gallery.classList.add("dragging");
        };
        const onDragLeave = (e) => {
            if (e.relatedTarget === null) gallery.classList.remove("dragging");
        };
        const onDrop = (e) => {
            e.preventDefault();
            gallery.classList.remove("dragging");
            const file = e.dataTransfer.files[0];
            if (file) choose({ file });
        };
        window.addEventListener("dragover", onDragOver);
        window.addEventListener("dragleave", onDragLeave);
        window.addEventListener("drop", onDrop);
        cleanups.push(() => {
            window.removeEventListener("dragover", onDragOver);
            window.removeEventListener("dragleave", onDragLeave);
            window.removeEventListener("drop", onDrop);
        });
    });
}

const TEXTURE_WIDTH = 2048;

// Parse a .msplat binary (format version 2 — see export_web.py). The SH coefficients
// are already compressed by the exporter: DC is baked into an 8-bit per-vertex color
// and levels 1-3 are quantized + bit-packed into integer textures, so loading is just
// slicing typed-array views out of the buffer.
//
//   "MSPLAT\n"(7B) | num_verts u32 | num_tris u32 | sh_degree u32
//   | sh1Max f32 | sh2Max f32 | sh3Max f32
//   | cam_center f32×3 | cam_up f32×3 | cam_distance f32
//   | positions f32×3×V | indices u32×3×T | colors u8×3×V
//   | sh1 u32×2×P | sh2 u32×4×P | sh3 u32×4×P   (P = padded texel count, levels present per degree)
export function loadMSplat(buffer) {
    const dv = new DataView(buffer);

    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 7));
    if (magic !== "MSPLAT\n") throw new Error("Not a .msplat file (bad magic)");

    let o = 7;
    const numVerts = dv.getUint32(o, true); o += 4;
    const numTris = dv.getUint32(o, true); o += 4;
    const shDegree = dv.getUint32(o, true); o += 4;
    const sh1Max = dv.getFloat32(o, true); o += 4;
    const sh2Max = dv.getFloat32(o, true); o += 4;
    const sh3Max = dv.getFloat32(o, true); o += 4;
    const camCenter = new THREE.Vector3(
        dv.getFloat32(o, true), dv.getFloat32(o + 4, true), dv.getFloat32(o + 8, true)); o += 12;
    const camUp = new THREE.Vector3(
        dv.getFloat32(o, true), dv.getFloat32(o + 4, true), dv.getFloat32(o + 8, true)); o += 12;
    const camDistance = dv.getFloat32(o, true); o += 4;

    const numSh = shDegree; // levels 1..shDegree carried in packed textures

    // SH textures are tiled TEXTURE_WIDTH wide and indexed by vertex; the exporter padded
    // the last row, so each level's data is exactly texWidth·texHeight texels.
    const texWidth = Math.min(TEXTURE_WIDTH, numVerts);
    const texHeight = Math.ceil(numVerts / TEXTURE_WIDTH);
    const texels = texWidth * texHeight;

    // slice() copies into a fresh, aligned ArrayBuffer (section offsets aren't 4-aligned).
    const take = (Type, count) => {
        const arr = new Type(buffer.slice(o, o + count * Type.BYTES_PER_ELEMENT));
        o += count * Type.BYTES_PER_ELEMENT;
        return arr;
    };

    const positions = take(Float32Array, numVerts * 3);
    const indices = take(Uint32Array, numTris * 3);
    const colors = take(Uint8Array, numVerts * 3);
    const sh1Data = numSh >= 1 ? take(Uint32Array, texels * 2) : null;
    const sh2Data = numSh >= 2 ? take(Uint32Array, texels * 4) : null;
    const sh3Data = numSh >= 3 ? take(Uint32Array, texels * 4) : null;

    return {
        positions, indices, colors,
        numVerts, numTris, shDegree, numSh,
        sh1Data, sh2Data, sh3Data,
        sh1Max, sh2Max, sh3Max,
        texWidth, texHeight,
        camCenter, camUp, camDistance,
    };
}
