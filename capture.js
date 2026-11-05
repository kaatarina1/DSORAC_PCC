import { encode } from "fast-png";
import { initRenderer, renderImage, cleanup } from "./js/CaptureRenderer.js";
import { parseArgs } from "jsr:@std/cli/parse-args";
import { 
    generateFibonacciHemisphereAroundCamera,
    imagesTxtEntry,
    writeCamerasTxt,
    writeImagesTxt,
    writePoints3DTxt
} from "./js/CameraPosition.js";

// deno task capture path/to/save --imgIndex i 
const args = parseArgs(Deno.args, { string: ["out"] });
const [outPath] = args._;
const hasTarget = args.targetX !== undefined && args.targetY !== undefined && args.targetZ !== undefined;
if (!outPath || !args.lasFile || !args.width || !args.height || (!hasTarget && !args.poses)) {
    console.error("Invalid input arguments: deno task capture <outputDir> --lasFile /path/tofile.las --width w --height h --tagetX 0 --targetY 0 --targetZ 0");
    console.error(`deno task capture ${outPath} --lasFile ${args.lasFile} --width ${args.width} --height ${args.height} --tagetX ${args.targetX} --targetY ${args.targetY} --targetZ ${args.targetZ}`);
    Deno.exit(2);
}
const target = [args.targetX, args.targetY, args.targetZ]
const fullSphera = args.fullSphera === 1;
const mode = args.mode ?? "POINTS";
await Deno.mkdir(`${outPath}/images`, { recursive: true });

const scene = await initRenderer({ width: args.width, height: args.height, lasFile: args.lasFile });

const sparseDir = `${outPath}/sparse/0`;
const imagesTxtPath = `${sparseDir}/images.txt`;
const camerasTxtPath = `${sparseDir}/cameras.txt`;
await Deno.mkdir(sparseDir, { recursive: true });

const type = args.type ?? "fibonacci";
const results = [];
const start = performance.now();
let failed = 0;

const samplingRate = args.samplingRate;


if (type === "fibonacci") {
    if (!args.noCams || !args.radius) {
        console.error("For generating images using fibonacci distribution arguments radius and noCams are mendatory");
        Deno.exit(2);
    }

    const existing = await readTxtIfExists(imagesTxtPath);
    const startIndex = existing ? Math.max(0, ...parseImageTxt(existing).map((e) => e.imageId)) : 0;
    if (existing) console.log(`Dataset exists - appending after image ${startIndex}`);

    const poses = generateFibonacciHemisphereAroundCamera(args.noCams, args.radius, target, fullSphera);

    let index = startIndex;
    for (const pose of poses) {
        const task = {
            imageIndex: index,
            cameraPosition: pose,
            targetPosition: target,
            mode: mode,
            useReconstruction: args.useReconstruction === 0 ? false : true,
            pointSize: args.pointSize ?? 0.2,
            filename: `${String(index).padStart(4, "0")}_${pose[0]}_${pose[1]}_${pose[2]}_${mode}.png`,
            radiusRatio: args.radiusRatio ?? 1
        };
        try {
            index++;
            const { imageIndex, rgba, metadata } = await renderImage(task);
            const png = encode({ width: metadata.width, height: metadata.height, data: rgba, channels: 4 });
            await Deno.writeFile(`${outPath}/images/${metadata.filename}`, png);
            results.push({...metadata, imageId: imageIndex + 1});

            const done = results.length + failed;
            const elapsed = (performance.now() - start) / 1000;
            console.log(`✓ ${done}/${poses.length} ${metadata.filename} | ${elapsed.toFixed(1)}s | ~${(elapsed / done * (poses.length - done)).toFixed(0)}s left`);
        } catch (error) {
            failed++;
            console.error(`✗ ${task.filename}:`, (error.cause ?? error).stack);
        }
    }

    if (results.length > 0) {
        if (existing) {
            await appendToDataset(existing, results);
        } else {
            await writeNewDataset(results, samplingRate);
        }
    }
} else {
    // --poses file.json: [{ "eye": [x, y, z], "target": [x, y, z] }, ...] rendered in one run
    let poses;
    if (args.poses) {
        poses = JSON.parse(await Deno.readTextFile(args.poses));
    } else {
        if (args.cameraX === undefined || args.cameraY === undefined || args.cameraZ === undefined) {
            console.error("For generating images using lookAt option argument camera (or --poses) is mendatory");
            Deno.exit(2);
        }
        poses = [{ eye: [args.cameraX, args.cameraY, args.cameraZ], target: target }];
    }

    const existing = await readTxtIfExists(imagesTxtPath);
    const startIndex = existing ? Math.max(0, ...parseImageTxt(existing).map((e) => e.imageId)) : 0;
    if (existing) console.log(`Dataset exists - appending after image ${startIndex}`);

    let index = startIndex;
    for (const { eye, target, radiusRatio } of poses) {
        const task = {
            imageIndex: index,
            cameraPosition: eye,
            targetPosition: target,
            mode: mode,
            useReconstruction: args.useReconstruction === 0 ? false : true,
            pointSize: args.pointSize ?? 0.2,
            filename: `${String(index).padStart(4, "0")}_${eye[0]}_${eye[1]}_${eye[2]}_${mode}.png`,
            radiusRatio: radiusRatio ?? args.radiusRatio ?? 1,
        };
        try {
            index++;
            const { imageIndex, rgba, metadata } = await renderImage(task);
            const png = encode({ width: metadata.width, height: metadata.height, data: rgba, channels: 4 });
            await Deno.writeFile(`${outPath}/images/${metadata.filename}`, png);
            results.push({...metadata, imageId: imageIndex + 1});

            const done = results.length + failed;
            const elapsed = (performance.now() - start) / 1000;
            console.log(`✓ ${done}/${poses.length} ${metadata.filename} | ${elapsed.toFixed(1)}s | ~${(elapsed / done * (poses.length - done)).toFixed(0)}s left`);
        } catch (error) {
            failed++;
            console.error(`✗ ${task.filename}:`, (error.cause ?? error).stack);
        }
    }

    if (results.length > 0) {
        if (existing) {
            await appendToDataset(existing, results);
        } else {
            await writeNewDataset(results, samplingRate);
        }
    }
}

cleanup();

console.log(`Done: ${results.length} images in ${((performance.now() - start) / 1000).toFixed(1)}s`);
if (failed > 0) Deno.exit(1);

async function readTxtIfExists(path) {
    try {
        return await Deno.readTextFile(path);
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) return null;
        throw error;
    }
}

function parseImageTxt(imageTxt) {
    return imageTxt.split("\n")
        .map((line) => line.trim().split(/\s+/))
        .filter((e) => e.length === 10 && !e[0].startsWith("#"))
        .map((e) => ({ imageId: Number(e[0]), filename: e[9] }));
}

async function writeNewDataset(entries, samplingRate) {
    await Deno.writeTextFile(camerasTxtPath, writeCamerasTxt(entries));
    await Deno.writeTextFile(imagesTxtPath, writeImagesTxt(entries));
    await Deno.writeFile(`${sparseDir}/points3D.txt`, writePoints3DTxt(scene.positions, scene.colorsRGB, samplingRate).stream());
}

async function appendToDataset(existing, entries) {
    await Deno.writeTextFile(imagesTxtPath, entries.map(imagesTxtEntry).join(""), {append: true});
}