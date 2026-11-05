import { decode, encode } from "fast-png";
import { parseArgs } from "jsr:@std/cli/parse-args";
import { initRenderer, renderImage, cleanup } from "./js/CaptureRenderer.js";

// Quick look at the reconstruction (hole filling vs. edge bleed) without generating a dataset or training.
// Renders the eval cameras once, compares them with the GT alpha and re-renders on every save in shaders/.
//   deno run -A preview.js data/pc/lego_normals.las [--evalDir data/lego_normals_eval] [--downscale 2]
//                                                    [--radiusRatio 1] [--label name] [--once]
const args = parseArgs(Deno.args, { string: ["evalDir", "label"], boolean: ["once"] });
const [lasFile] = args._;
if (!lasFile) {
    console.error("Usage: deno run -A preview.js <file.las> [--evalDir dir] [--downscale 2] [--radiusRatio 1] [--label name] [--once]");
    Deno.exit(2);
}
const stem = lasFile.split(/[\\/]/).pop().replace(/\.la[sz]$/i, "");
const evalDir = args.evalDir ?? `data/${stem}_eval`;
const downscale = args.downscale ?? 2;
const radiusRatio = args.radiusRatio ?? 1;
const label = args.label ?? "run";
const outRoot = `output/preview/${stem}`;

const config = JSON.parse(await Deno.readTextFile(`${evalDir}/eval.json`));
const width = Math.floor(config.canvasWidth / downscale);
const height = Math.floor(config.canvasHeight / downscale);

// GT is only loaded and downscaled once
const cameras = config.cameras.filter((cam) => {
    if (cam.projection === "PERSPECTIVE") return true;
    console.log(`Skipping ${cam.id}: only PERSPECTIVE cameras are supported`);
    return false;
});
const gts = await Promise.all(cameras.map(async (cam) =>
    downscaleRgba(decode(await Deno.readFile(`${evalDir}/${cam.gt}`)), downscale)));

const scene = await initRenderer({ width, height, lasFile });
// eval.json is in the LAS frame of lasOrigin, the renderer centres the cloud on its own origin
const origin = config.lasOrigin ?? [scene.centerX, scene.minY, scene.centerZ];
const offset = [origin[0] - scene.centerX, origin[1] - scene.minY, origin[2] - scene.centerZ];

await Deno.mkdir(outRoot, { recursive: true });
let runIndex = 0;
await renderAll();

if (!args.once) {
    console.log("\nWatching shaders/ - save a shader to re-render (Ctrl+C to stop). JS changes need a restart.");
    let timer = null;
    for await (const event of Deno.watchFs("shaders")) {
        if (event.kind !== "modify") continue;
        // one save fires several events, so wait until they stop
        clearTimeout(timer);
        timer = setTimeout(() => renderAll().catch((error) => console.error(error)), 300);
    }
}
cleanup();

async function renderAll() {
    const runName = `${label}_${String(runIndex++).padStart(2, "0")}`;
    const runDir = `${outRoot}/${runName}`;
    await Deno.mkdir(runDir, { recursive: true });
    // keep the shader next to its results, so a good combination can be found again
    await Deno.copyFile("shaders/composite.wgsl", `${runDir}/composite.wgsl`);

    const start = performance.now();
    const rows = [];
    for (let i = 0; i < cameras.length; i++) {
        const cam = cameras[i];
        const { rgba } = await renderImage({
            imageIndex: i,
            cameraPosition: cam.eye.map((v, k) => v + offset[k]),
            targetPosition: cam.target.map((v, k) => v + offset[k]),
            fovDeg: cam.fovDeg,
            mode: "POINTS",
            useReconstruction: true,
            radiusRatio,
        });
        const metrics = compareAlpha(rgba, gts[i]);
        rows.push({ view: cam.id, ...metrics });

        await Deno.writeFile(`${runDir}/${cam.id}.png`, encode({ width, height, data: rgba, channels: 4 }));
        await Deno.writeFile(`${runDir}/${cam.id}_compare.png`, compareImage(rgba, gts[i]));
    }

    const mean = (key) => rows.reduce((s, r) => s + r[key], 0) / rows.length;
    rows.push({ view: "MEAN", holes: mean("holes"), bleed: mean("bleed"), halo: mean("halo") });

    console.log(`\n${runName}  (${((performance.now() - start) / 1000).toFixed(1)}s)  ->  ${runDir}`);
    console.table(rows.map((r) => ({
        view: r.view,
        "holes %": (100 * r.holes).toFixed(2),
        "bleed %": (100 * r.bleed).toFixed(2),
        "halo %": (100 * r.halo).toFixed(2),
    })));

    const logPath = `${outRoot}/log.csv`;
    let exists = true;
    try { await Deno.stat(logPath); } catch { exists = false; }
    const lines = rows.map((r) => [runName, r.view, r.holes, r.bleed, r.halo].join(","));
    await Deno.writeTextFile(logPath, (exists ? "" : "run,view,holes,bleed,halo\n") + lines.join("\n") + "\n", { append: true });
}

// All three are relative to the GT object area, so views of different size are comparable.
//   holes - object pixels the render leaves (mostly) empty      -> gaps
//   bleed - alpha the render puts outside the object            -> fog / beady edges
//   halo  - pixels with partial alpha (neither empty nor solid) -> what 3DGS turns into low opacity fog
function compareAlpha(rgba, gt) {
    let object = 0, holes = 0, bleed = 0, halo = 0;
    for (let i = 3; i < rgba.length; i += 4) {
        const a = rgba[i] / 255;
        const isObject = gt.data[i] >= 128;
        if (isObject) {
            object++;
            if (a < 0.5) holes++;
        } else {
            bleed += a;
        }
        if (a > 0.005 && a < 0.99) halo++;
    }
    object = Math.max(object, 1);
    return { holes: holes / object, bleed: bleed / object, halo: halo / object };
}

// GT | render | error map (red = bleed, strength = alpha; blue = hole), all over grey
function compareImage(rgba, gt) {
    const grey = 96;
    const out = new Uint8Array(width * 3 * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            const a = rgba[i + 3] / 255;
            const gtA = gt.data[i + 3] / 255;
            const isObject = gt.data[i + 3] >= 128;
            for (let c = 0; c < 3; c++) {
                out[(y * width * 3 + x) * 4 + c] = gt.data[i + c] * gtA + grey * (1 - gtA);
                out[(y * width * 3 + x + width) * 4 + c] = rgba[i + c] * a + grey * (1 - a);
            }
            const luma = (rgba[i] + rgba[i + 1] + rgba[i + 2]) / 3 * a + grey * (1 - a);
            let error = [luma * 0.5, luma * 0.5, luma * 0.5];
            if (isObject && a < 0.5) error = [0, 90, 255];
            else if (!isObject && a > 0.005) error = [255 * a + luma * 0.5 * (1 - a), luma * 0.5 * (1 - a), luma * 0.5 * (1 - a)];
            out.set([...error, 255], (y * width * 3 + x + 2 * width) * 4);
            out[(y * width * 3 + x) * 4 + 3] = 255;
            out[(y * width * 3 + x + width) * 4 + 3] = 255;
        }
    }
    return encode({ width: width * 3, height, data: out, channels: 4 });
}

// Box filter by an integer factor (GT is 8-bit RGBA)
function downscaleRgba(image, factor) {
    const w = Math.floor(image.width / factor), h = Math.floor(image.height / factor);
    const channels = image.channels;
    const data = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const sum = [0, 0, 0, 0];
            for (let dy = 0; dy < factor; dy++) {
                for (let dx = 0; dx < factor; dx++) {
                    const j = ((y * factor + dy) * image.width + x * factor + dx) * channels;
                    for (let c = 0; c < 4; c++) sum[c] += c < channels ? image.data[j + c] : 255;
                }
            }
            for (let c = 0; c < 4; c++) data[(y * w + x) * 4 + c] = sum[c] / (factor * factor);
        }
    }
    return { width: w, height: h, data };
}
