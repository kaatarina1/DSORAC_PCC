import { DepthMap } from "./DepthMap.js";
import { Composer } from "./Composer.js";
import { SignedDistanceFiled } from "./SignedDistanceFiled.js";
import { Solver } from "./Solvers.js";
import { convertTexture } from "./Utils.js";
import { LasLoader } from "./LasLoader.js";
import * as mat4 from "./_Mat4.js";
import { computeCameraMatrix, computeOrthoTopDownMatrix, computeSphericalCameraMatrix } from "./CameraPosition.js";
import { AdaptiveGrid } from "./AdaptiveGrid.js";
import { loadShader } from "./LoadShader.js";

let device = null;
let renderingPipeline = null;
let preparationPipeline = null;
let localSortPipeline = null;
let globalSortPipeline = null;
const quadPipelines = {};
let pointclouds = [];
let canvas = null;
let useClassColorsBuffer = null;
let format = null;
let deviceLost = false;
let deviceLostPromise = null;

function toRGBA8(data, width, height, colorOrder = "rgb") {
    const pixels = new Uint8Array(width * height * 4);
    // Preveri, ali so podatki že v 0-255 obsegu ali v 0-1 obsegu
    const needsScaling = data instanceof Float32Array;
    
    if (needsScaling) {
        for (let i = 0; i < width * height; i++) {
            const r = Math.max(0, Math.min(255, Math.floor(data[i * 4] * 255)));
            const g = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 1] * 255)));
            const b = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 2] * 255)));
            const a = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 3] * 255)));

            // Podpora za RGB ali BGR vrstni red barv v odvisnosti od vhodnih podatkov
            if (colorOrder === "rgb") {
                pixels[i * 4] = r;
                pixels[i * 4 + 1] = g;
                pixels[i * 4 + 2] = b;
            } else {
                pixels[i * 4] = b;
                pixels[i * 4 + 1] = g;
                pixels[i * 4 + 2] = r;
            }
            pixels[i * 4 + 3] = a;
        }
    } else {
        for (let i = 0; i < width * height; i++) {
            const r = Math.max(0, Math.min(255, Math.floor(data[i * 4])));
            const g = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 1])));
            const b = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 2])));
            const a = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 3])));

            // Podpora za RGB ali BGR vrstni red barv v odvisnosti od vhodnih podatkov
            if (colorOrder === "rgb") {
                pixels[i * 4] = r;
                pixels[i * 4 + 1] = g;
                pixels[i * 4 + 2] = b;
            } else {
                pixels[i * 4] = b;
                pixels[i * 4 + 1] = g;
                pixels[i * 4 + 2] = r;
            }
            pixels[i * 4 + 3] = a;
        }
    }
    
    return pixels;
}

// Bounding box oblaka točk (izračunano enkrat med inicializacijo)
let bbMin = [Infinity, Infinity, Infinity];
let bbMax = [-Infinity, -Infinity, -Infinity];
let numberOfAllPoints = 0;

export async function initRenderer(config) {
    console.log('🔧 Worker initializing GPU...');
    // Na Windows Deno privzeto izbere Vulkan, kjer wgpu pri preverjanju budgeta upošteva
    // tudi majhne NVIDIA heap-e (32 MB) in zavrne vsako večjo alokacijo
    // ("not enough memory left" -> izguba naprave). DX12 tega problema nima,
    // zato ga nastavimo vedno, tudi če je v okolju ostal npr. "vulkan".
    if (Deno.build.os === "windows") {
        Deno.env.set("DENO_WEBGPU_BACKEND", "dx12");
        // Privzeti FXC prevajalnik potrebuje ~3 min za sorting shaderje, DXC pa nekaj sekund.
        // dxcompiler.dll se naloži preko PATH, zato dodamo tools/dxc.
        const dxcDir = new URL("../tools/dxc/", import.meta.url);
        let hasDxc = false;
        try { hasDxc = Deno.statSync(new URL("dxcompiler.dll", dxcDir)).isFile; } catch { /* ni DXC */ }
        if (hasDxc && !Deno.env.get("DENO_WEBGPU_DX12_COMPILER")) {
            const dxcPath = decodeURIComponent(dxcDir.pathname).replace(/^\/([A-Za-z]:)/, "$1");
            Deno.env.set("PATH", `${dxcPath};${Deno.env.get("PATH") ?? ""}`);
            Deno.env.set("DENO_WEBGPU_DX12_COMPILER", "dxc");
        }
        console.log(`🔧 WebGPU backend: dx12, shader compiler: ${Deno.env.get("DENO_WEBGPU_DX12_COMPILER") ?? "fxc"}`);
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });

    // Nekateri oblaki točk (npr. ljubljana_2/3) imajo toliko točk, da depthStorageBuffer
    // v DepthMap (nPoints * 4 bajtov, brez chunkanja) preseže privzeti
    // maxStorageBufferBindingSize (128 MB) in povzroči izgubo GPU naprave.
    // Zahtevamo najvišjo limito, ki jo adapter podpira.
    device = await adapter?.requestDevice({
        requiredFeatures: ["bgra8unorm-storage", "float32-filterable", "float32-blendable", "timestamp-query"]
            .filter((f) => adapter.features.has(f)),
        requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
        },
    });

    // Spremljamo izgubo GPU
    deviceLostPromise = device.lost.then((info) => {
        if (info.reason === "destroyed") return;
        deviceLost = true;
        console.error('🔥 GPU Device Lost:', info.message, info.reason);
    });

    format = "rgba8unorm";

    canvas = { width: config.width, height: config.height };

    console.log('🔧 Worker creating rendering pipeline...');
    const renderingCode = await loadShader("rendering.wgsl");
    const renderingModule = device.createShaderModule({ code: renderingCode });

    renderingPipeline = device.createRenderPipeline({
        vertex: { module: renderingModule },
        fragment: {
            module: renderingModule,
            entryPoint: "fragment",
            targets: [{ format: format }],
        },
        primitive: { topology: "point-list" },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: "less",
            format: "depth32float",
        },
        layout: "auto",
    });

    // ostali pipeline-i (DISKS, BILLBOARDS, GAUSSIANS) 
    // delajo z istim formatom in bind group layouti
    const shaders = {
        DISKS: "rendering_disks.wgsl",
        BILLBOARDS: "rendering_billboards.wgsl",
        GAUSSIANS: "rendering_gaussians.wgsl",
    };
    for (const [mode, shaderPath] of Object.entries(shaders)) {
        const code = await loadShader(shaderPath);
        const module = device.createShaderModule({ code });
        const enableBlend = mode === "GAUSSIANS";
        const blendState = {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        };
        quadPipelines[mode] = device.createRenderPipeline({
            vertex: { module },
            fragment: { module, targets: [{ format: format, ...(enableBlend ? { blend: blendState } : {}) }] },
            primitive: { topology: "triangle-list", cullMode: "none" },
            depthStencil: {
                depthWriteEnabled: !enableBlend,
                depthCompare: "less",
                format: "depth32float"
            },
            layout: "auto",
        });
    }

    console.log("🔧 Worker creating sorting pipelines...");
    try {
        // Ustvarimo compute pipeline-e za pripravo podatkov in sortiranje, če so shaderji na voljo
        const preparationCode = await loadShader("preparation.wgsl");
        const localSortCode = await loadShader("localSort.wgsl");
        const globalSortCode = await loadShader("globalSort.wgsl");

        preparationPipeline = device.createComputePipeline({
            compute: { module: device.createShaderModule({ code: preparationCode }), entryPoint: "main" },
            layout: "auto",
        });
        localSortPipeline = device.createComputePipeline({
            compute: { module: device.createShaderModule({ code: localSortCode }), entryPoint: "compute" },
            layout: "auto",
        });
        globalSortPipeline = device.createComputePipeline({
            compute: { module: device.createShaderModule({ code: globalSortCode }), entryPoint: "mergePass" },
            layout: "auto",
        });
        console.log("✅ Sorting pipelines created successfully");
    } catch (error) {
        console.warn("⚠️ Warning: Sorting pipelines not available:", error.message);
        console.log("   Sorting will be skipped if requested");
    }

    useClassColorsBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(useClassColorsBuffer, 0, new Uint32Array([0]));

    console.log(`🔧 Worker loading point cloud from ${config.lasFile}...`);
    const lasLoader = new LasLoader(config.lasFile);
    const lasData = await lasLoader.loadLasData();

    console.log(`📊 Worker loaded ${lasData.positions.length / 3} points`);

    // Izračunamo bounding box oblaka točk,
    // da ga lahko uporabimo za nastavitev near/far
    // ravnin kamere in za normalizacijo globin
    const positions = lasData.positions;
    bbMin = [Infinity, Infinity, Infinity];
    bbMax = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
        bbMin[0] = Math.min(bbMin[0], positions[i]);
        bbMin[1] = Math.min(bbMin[1], positions[i + 1]);
        bbMin[2] = Math.min(bbMin[2], positions[i + 2]);
        bbMax[0] = Math.max(bbMax[0], positions[i]);
        bbMax[1] = Math.max(bbMax[1], positions[i + 1]);
        bbMax[2] = Math.max(bbMax[2], positions[i + 2]);
    }
    console.log("📐 Point cloud bounding box:", bbMin, bbMax);

    await createPointClouds(positions, lasData.colors, lasData.normals);
    
    if (preparationPipeline && localSortPipeline && globalSortPipeline) {
        prewarmMergeBuffers(numberOfAllPoints);
        for (const pc of pointclouds) {
            buildSortBindGroups(pc);
        }
        console.log("✅ Sorting enabled");
    } else {
        console.log("ℹ️ Sorting pipelines not available");
    }

    console.log(`✅ Worker ready with ${pointclouds.length} point cloud batches`);

    return { ...lasData, bbMin, bbMax };
}

async function createPointClouds(positions, colors, normals) {
    pointclouds = [];
    const pointByteSize = 48;
    const maxPointsPerBuffer = 1024 * 1024;
    numberOfAllPoints = positions.length / 3;

    // Uporabimo adaptive grid za razdelitev točk v celice, 
    // kar omogoča renderiranje z gaussovkami in optimizacijo GPU pomnilnika.
    const grid = new AdaptiveGrid(positions, maxPointsPerBuffer);
    console.log(`📐 Worker adaptive grid: ${grid.cells.length} cells`);

    for (const cell of grid.cells) {
        const count = cell.indices.length;

        const pointData = new ArrayBuffer(maxPointsPerBuffer * pointByteSize);
        const pointDataView = new DataView(pointData);

        let sumX = 0, sumY = 0, sumZ = 0;
        cell.indices.forEach((j, slot) => {
            const posIndex = j * 3;
            const pointOffset = slot * pointByteSize;

            pointDataView.setFloat32(pointOffset,      positions[posIndex],     true);
            pointDataView.setFloat32(pointOffset +  4, positions[posIndex + 1], true);
            pointDataView.setFloat32(pointOffset +  8, positions[posIndex + 2], true);
            pointDataView.setUint32( pointOffset + 12, colors[j],               true);
            // normals (potrebujemo jih za DISKS / BILLBOARDS / GAUSSIANS shaders)
            if (normals) {
                pointDataView.setFloat32(pointOffset + 16, normals[posIndex],     true);
                pointDataView.setFloat32(pointOffset + 20, normals[posIndex + 1], true);
                pointDataView.setFloat32(pointOffset + 24, normals[posIndex + 2], true);
            }
            pointDataView.setFloat32(pointOffset + 28, 0.0, true); // depth (sort key)

            sumX += positions[posIndex];
            sumY += positions[posIndex + 1];
            sumZ += positions[posIndex + 2];
        });

        const centerX = sumX / count;
        const centerY = sumY / count;
        const centerZ = sumZ / count;

        const pointBuffer = device.createBuffer({
            size: maxPointsPerBuffer * pointByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(pointBuffer, 0, new Uint8Array(pointData));

        const renderingBindGroup = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: pointBuffer } },
                { binding: 1, resource: { buffer: useClassColorsBuffer } },
            ],
        });

        pointclouds.push({
            pointBuffer,
            renderingBindGroup,
            numberOfPoints: count,
            center: [centerX, centerY, centerZ],
        });
    }
}


function createDepthRangeBindGroup(minDepth, maxDepth, pipeline = null) {
    // če gre za QUADS/BILLBOARDS/GAUSSIANS, 
    // je pipeline različen od null, sicer pa uporabimo renderingPipeline, 
    const pl = pipeline || renderingPipeline;
    const buf = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([minDepth, maxDepth]));
    return {
        buffer: buf,
        bindGroup: device.createBindGroup({
            layout: pl.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    };
}

function createTargetPositionBindGroup(pos, pipeline, isSpherical = false, far = 0) {
    // RenderParams (shaders/rendering.wgsl) je targetPosition:vec4f + isSpherical:f32 + far:f32 + 2x pad = 32 bajtov
    const buf = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([pos[0], pos[1], pos[2], 0.0, isSpherical ? 1 : 0, far, 0, 0]));
    return {
        buffer: buf,
        bindGroup: device.createBindGroup({
            layout: pipeline.getBindGroupLayout(3),
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    };
}

function createScreenParamsBindGroup(targetPosition, cameraPosition, pointSize, pipeline, isOrtho, isSpherical, far) {
    // SceneParams buffer
    const sceneParamsBuffer = device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    const sceneData = new Float32Array(12);
    sceneData[0] = targetPosition[0]; sceneData[1] = targetPosition[1]; sceneData[2] = targetPosition[2]; sceneData[3] = 0;
    sceneData[4] = cameraPosition[0]; sceneData[5] = cameraPosition[1]; sceneData[6] = cameraPosition[2]; sceneData[7] = 0;
    sceneData[8] = pointSize; sceneData[9] = isOrtho ? 1 : 0; sceneData[10] = isSpherical ? 1 : 0; sceneData[11] = far ?? 0;
    device.queue.writeBuffer(sceneParamsBuffer, 0, sceneData);


    const sceneBG = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(3),
        entries: [{ binding: 0, resource: { buffer: sceneParamsBuffer } }],
    });

    return sceneBG;
}

function createMatricesBindGroup(projectionViewMatrix, viewMatrix, pipeline = null) {
    // če gre za QUADS/BILLBOARDS/GAUSSIANS, 
    // je pipeline različen od null, sicer pa uporabimo renderingPipeline, 
    const pl = pipeline || renderingPipeline;
    const mvpBuf = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(mvpBuf, 0, new Float32Array(projectionViewMatrix));

    const viewBuf = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(viewBuf, 0, new Float32Array(viewMatrix));

    return {
        mvpBuffer: mvpBuf,
        viewBuffer: viewBuf,
        bindGroup: device.createBindGroup({
            layout: pl.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: mvpBuf } },
                { binding: 1, resource: { buffer: viewBuf } },
            ],
        }),
    };
}

export async function renderImage(params, { onProgress } = {}) {
    const imageIndex = params.imageIndex;
    let depthTexture;
    try {
        const {
            cameraPosition,
            targetPosition,
            useReconstruction = true,
            colorOrder = "rgb",
            mode = "POINTS",
            pointSize = 0.01,
            fovDeg = 45,
            radiusRatio = 1,
        } = params;

        let pipeline;
        if (mode === "POINTS") {
            pipeline = renderingPipeline;
        } else {
            pipeline = quadPipelines[mode];
        }

        // Preveri stanje GPU naprave
        if (deviceLost || !device) {
            throw new Error('GPU device is lost or invalid');
        }

        // Zagotovimo da je GPU pripravljen
        await device.queue.onSubmittedWorkDone();

        // Izračunamo view in projection matriki
        let viewMatrix, projectionViewMatrix, near, far, cameraDirOrPosition = cameraPosition, isOrtho = false, isSpherical = false;

        if (params.projection === "ORTHOGRAPHIC") {
            isOrtho = true;
            ({ viewMatrix, projectionViewMatrix, near, far } = computeOrthoTopDownMatrix(params.eyeY, params.panX, params.panZ, params.zoom, canvas, bbMin, bbMax));
            cameraDirOrPosition = mat4.normalize([0, 1, 0]); // top-down
        } else if (params.projection === "SPHERICAL") {
            isSpherical = true;
            ({ viewMatrix, far } = computeSphericalCameraMatrix(cameraPosition, bbMin, bbMax, params.yawDeg));
            projectionViewMatrix = mat4.identity();
            near = 0.01;
        } else {
            ({ viewMatrix, projectionViewMatrix, near, far } = computeCameraMatrix(cameraPosition, targetPosition, canvas, bbMin, bbMax, fovDeg));
        }

        depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            format: "depth32float",
        });

        let outputData;

        // Reconstruction pipeline deluje edino za POINTS tip izrisovanja.
        const effectiveReconstruction = useReconstruction && mode === "POINTS";

        let density;
        if (effectiveReconstruction) {
            // await renderFullScene(projectionViewMatrix, viewMatrix, depthTexture, targetPosition);

            const depthMap = new DepthMap(canvas, device, viewMatrix, projectionViewMatrix, pointclouds, isSpherical);
            let depthBins;
            
            try {
                depthBins = await depthMap.groupDepthIntoBins({ near, far });
                if (!isSpherical) density = await depthMap.computeViewpointDensity();
                if (!depthBins || depthBins.length === 0) {
                    depthBins = [[0, 0.3], [0.3, 0.6], [0.6, 1.0]];
                }
                depthBins.reverse();
            } catch (error) {
                depthBins = [[0, 0.3], [0.3, 0.6], [0.6, 1.0]];
            } finally {
                depthMap.destroy();
            }

            onProgress?.({
                imageIndex,
                message: `Calculated ${depthBins.length} depth bins (near=${near.toFixed(4)}, far=${far.toFixed(4)})`
            });

            // Za vsako sliko naredimo novo instanco composer razreda
            const composer = new Composer(device, canvas.width, canvas.height, radiusRatio);
            
            // Zagotovimo, da Composer začne s čistimi podatki
            composer.reconstructions = [];
            composer.sdfs = [];
            composer.depthPoints = [];
            composer.depths = [];

            console.log(`🎨 Starting composition with ${depthBins.length} depth bins`);

            for (let i = 0; i < depthBins.length; i++) {
                const [minDepth, maxDepth] = depthBins[i];
                await renderPointsInDepthRange(
                    minDepth, maxDepth, near, far,
                    projectionViewMatrix, viewMatrix, depthTexture, composer,
                    targetPosition, isSpherical
                );
                
                console.log(`✓ Layer ${i + 1}/${depthBins.length} added. Composer now has ${composer.reconstructions.length} layers`);
                
                onProgress?.({
                    imageIndex,
                    layer: i,
                    totalLayers: depthBins.length
                });
            }
            
            console.log(`🎬 Compositing ${composer.reconstructions.length} layers...`);

            // Zagotovimo, da so vsi GPU ukazi zaključeni pred kompozicijo
            await device.queue.onSubmittedWorkDone();
            
            outputData = await composer.compositeDepths(composer);
        } else {
            onProgress?.({
                imageIndex,
                message: 'Capturing raw point cloud (reconstruction disabled)'
            });
            outputData = await capturePointCloudImage(projectionViewMatrix, viewMatrix, depthTexture, targetPosition, cameraDirOrPosition, pointSize, mode, isOrtho, isSpherical, far);
        }

        // Izračunamo gostoto točk
        if (density === undefined && mode === "POINTS" && !isSpherical) {
            const densityMap = new DepthMap(canvas, device, viewMatrix, projectionViewMatrix, pointclouds, isSpherical);
            try {
                density = await densityMap.computeViewpointDensity();
            } finally {
                densityMap.destroy();
            }
        }

        const rgba = toRGBA8(outputData, canvas.width, canvas.height, colorOrder);
        
        //Čakamo, da se vse GPU operacije zaključijo
        await device.queue.onSubmittedWorkDone();
        
        const { quat, t } = mat4.cameraToColmapPose(cameraPosition, viewMatrix);
        const { fx, fy, cx, cy, width, height } = mat4.getIntrinsics(
            canvas.width,
            canvas.height,
            fovDeg * Math.PI / 180
        );

        // Zadnja sinhronizacija GPU pred čiščenjem
        await device.queue.onSubmittedWorkDone();

        console.log(`✅ Image ${imageIndex} completed successfully`);
        return { 
            imageIndex,
            rgba,
            metadata: { quat, t, fx, fy, cx, cy, width, height, fovDeg, density, 
                filename: params.filename ?? `image_${imageIndex + 1}_${targetPosition.join("_")}.png`,
            },
        }
    } catch (error) {
        throw new Error(`Image ${imageIndex} failed: ${error.message}`, { cause: error });
    } finally {
        depthTexture.destroy();
    }
}

// renderFullScene vedno uporabi renderingPipeline (POINTS) za izračun globine.
async function renderFullScene(projectionViewMatrix, viewMatrix, depthTexture, targetPosition) {
    const { bindGroup: matricesBG, mvpBuffer: mvpBuf, viewBuffer: viewBuf } =
        createMatricesBindGroup(projectionViewMatrix, viewMatrix, renderingPipeline);

    const { bindGroup: depthRangeBG, buffer: drBuf } =
        createDepthRangeBindGroup(0, 1e6, renderingPipeline);

    const { bindGroup: targetBG, buffer: tBuf } =
        createTargetPositionBindGroup(targetPosition, renderingPipeline);

    const tempTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        format: format,
    });

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: tempTexture.createView(),
            loadOp: "clear",
            clearValue: [0, 0, 0, 0],
            storeOp: "store",
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: "clear",
            depthClearValue: 1,
            depthStoreOp: "store",
        },
    });

    renderPass.setPipeline(renderingPipeline);
    renderPass.setBindGroup(1, matricesBG);
    renderPass.setBindGroup(2, depthRangeBG);
    renderPass.setBindGroup(3, targetBG);

    for (const pointcloud of pointclouds) {
        renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
        renderPass.draw(pointcloud.numberOfPoints);
    }

    renderPass.end();
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    tempTexture.destroy();
    mvpBuf.destroy();
    viewBuf.destroy();
    drBuf.destroy();
    tBuf.destroy();
}

// renderPointsInDepthRange vedno uporablja renderingPipeline.
async function renderPointsInDepthRange(
    minDepth, maxDepth, near, far,
    projectionViewMatrix, viewMatrix, depthTexture, composer,
    targetPosition, isSpherical = false
) {
    const { bindGroup: matricesBG, mvpBuffer: mvpBuf, viewBuffer: viewBuf } =
        createMatricesBindGroup(projectionViewMatrix, viewMatrix, renderingPipeline);

    const { bindGroup: depthRangeBG, buffer: drBuf } =
        createDepthRangeBindGroup(minDepth, maxDepth, renderingPipeline);

    const { bindGroup: targetBG, buffer: tBuf } =
        createTargetPositionBindGroup(targetPosition, renderingPipeline, isSpherical, far);

    const captureTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_SRC |
               GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.RENDER_ATTACHMENT,
        format: format,
    });

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: captureTexture.createView(),
            loadOp: "clear",
            clearValue: [0, 0, 0, 0],
            storeOp: "store",
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: "clear",
            depthClearValue: 1,
            depthStoreOp: "store",
        },
    });

    renderPass.setPipeline(renderingPipeline);
    renderPass.setBindGroup(1, matricesBG);
    renderPass.setBindGroup(2, depthRangeBG);
    renderPass.setBindGroup(3, targetBG);

    for (const pointcloud of pointclouds) {
        renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
        renderPass.draw(pointcloud.numberOfPoints);
    }

    renderPass.end();
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    let reconstructionRead = device.createTexture({
        size: [canvas.width, canvas.height],
        format: "rgba32float",
        usage: GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.COPY_SRC,
    });

    await convertTexture(device, canvas.width, canvas.height, captureTexture, reconstructionRead);

    let reconstructionWrite = device.createTexture({
        size: [canvas.width, canvas.height],
        format: "rgba32float",
        usage: GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.COPY_SRC,
    });

    const sdf = new SignedDistanceFiled(device, captureTexture, canvas.width, canvas.height);
    const { sdf: sdfTexture, density: densityTexture } = await sdf.generateSDF();
    // const sdfTexture = await sdf.generateSDF();

    let solver = new Solver(canvas, device);
    await solver.sorRedBlack(captureTexture, reconstructionRead, reconstructionWrite);

    let pointsTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: "rgba32float",
        usage: GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.COPY_SRC,
    });

    await convertTexture(device, canvas.width, canvas.height, captureTexture, pointsTexture);

    // await composer.addLayers(sdfTexture, reconstructionRead, pointsTexture, (minDepth + maxDepth) / 2);
    await composer.addLayers(sdfTexture, densityTexture, reconstructionRead, pointsTexture, (minDepth + maxDepth) / 2);
    
    await device.queue.onSubmittedWorkDone();

    captureTexture.destroy();
    reconstructionRead.destroy();
    reconstructionWrite.destroy();
    sdfTexture.destroy();
    densityTexture.destroy();
    pointsTexture.destroy();
    mvpBuf.destroy();
    viewBuf.destroy();
    drBuf.destroy();
    tBuf.destroy();
    sdf.destroyTexture();
}

async function capturePointCloudImage(projectionViewMatrix, viewMatrix, depthTexture, targetPosition, cameraPosition, pointSize, mode, isOrtho, isSpherical, far) {
    // GAUSSIANS: najprej sortiramo točke v oblaku glede na njihovo globino
    // v odvisnosti od pogleda kamere, da zagotovimo pravilen back-to-front.
    if (mode === "GAUSSIANS") {
        // Za sferične zajeme ni prave projekcijske matrike (projectionViewMatrix je
        // identiteta), zato za sort key uporabimo dejansko view matriko in v shaderju
        // računamo radialno razdaljo od kamere namesto clip-space z/w.
        const sortMatrix = isSpherical ? viewMatrix : projectionViewMatrix;
        for (const pc of pointclouds) {
            sortPointCloud(pc, sortMatrix, isSpherical);
        }
        await device.queue.onSubmittedWorkDone();
    }

    // Sortiramo še same kose oblaka (batches) glede na globino njihovega centra,
    // da zagotovimo pravilno zaporedje risanja med različnimi batchi.
    const batchesWithDepth = pointclouds.map((pc) => ({
        pc,
        depth: getBatchDepth(pc, projectionViewMatrix, isSpherical, cameraPosition),
    }));
    batchesWithDepth.sort((a, b) => b.depth - a.depth);

    const captureTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.COPY_SRC |
               GPUTextureUsage.RENDER_ATTACHMENT,
        format: format,
    });

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: captureTexture.createView(),
            loadOp: "clear",
            clearValue: [0, 0, 0, 0],
            storeOp: "store",
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: "clear",
            depthClearValue: 1,
            depthStoreOp: "store",
        },
    });

    if (mode === "POINTS") {
        const drBuf = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(drBuf, 0, new Float32Array([0, 1e6]));
        const tBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(tBuf, 0, new Float32Array([targetPosition[0], targetPosition[1], targetPosition[2], 0, isSpherical, far, 0, 0]));

        const mvpBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(mvpBuf, 0, new Float32Array(projectionViewMatrix));
        const viewBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(viewBuf, 0, new Float32Array(viewMatrix));

        const matricesBG = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: mvpBuf } }, { binding: 1, resource: { buffer: viewBuf } }],
        });
        const depthRangeBG = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: drBuf } }],
        });
        const renderParamsBG = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(3),
            entries: [
                { binding: 0, resource: { buffer: tBuf } }
            ],
        });

        renderPass.setPipeline(renderingPipeline);
        renderPass.setBindGroup(1, matricesBG);
        renderPass.setBindGroup(2, depthRangeBG);
        renderPass.setBindGroup(3, renderParamsBG);
        for (const { pc } of batchesWithDepth) {
            renderPass.setBindGroup(0, device.createBindGroup({
                layout: renderingPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: pc.pointBuffer } },
                    { binding: 1, resource: { buffer: useClassColorsBuffer } },
                ],
            }));
            renderPass.draw(pc.numberOfPoints);
        }

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        mvpBuf.destroy(); viewBuf.destroy(); drBuf.destroy(); tBuf.destroy();

    } else {
        const pipeline = quadPipelines[mode];

        const mvpBuf = device.createBuffer({ 
            size: 64, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST 
        });
        device.queue.writeBuffer(mvpBuf, 0, new Float32Array(projectionViewMatrix));
        const viewBuf = device.createBuffer({ 
            size: 64, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST 
        });
        device.queue.writeBuffer(viewBuf, 0, new Float32Array(viewMatrix));

        const drBuf = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(drBuf, 0, new Float32Array([0, 1e6]));

        const sceneBG = createScreenParamsBindGroup(targetPosition, cameraPosition, pointSize, pipeline, isOrtho, isSpherical, far);

        const matricesBG = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: mvpBuf } }, { binding: 1, resource: { buffer: viewBuf } }],
        });
        const depthRangeBG = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: drBuf } }],
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(1, matricesBG);
        renderPass.setBindGroup(2, depthRangeBG);
        renderPass.setBindGroup(3, sceneBG);
        for (const { pc } of batchesWithDepth) {
            renderPass.setBindGroup(0, device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: pc.pointBuffer } },
                    { binding: 1, resource: { buffer: useClassColorsBuffer } },
                ],
            }));
            renderPass.draw(pc.numberOfPoints * 6);
        }

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        mvpBuf.destroy(); 
        viewBuf.destroy(); 
        drBuf.destroy();
    }

    // Read pixels back to CPU
    const bytesPerPixel = 4;
    const alignedBytesPerRow = Math.ceil((canvas.width * bytesPerPixel) / 256) * 256;
    const outputBuffer = device.createBuffer({
        size: alignedBytesPerRow * canvas.height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const readEncoder = device.createCommandEncoder();
    readEncoder.copyTextureToBuffer(
        { texture: captureTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
        { buffer: outputBuffer, bytesPerRow: alignedBytesPerRow, rowsPerImage: canvas.height },
        [canvas.width, canvas.height, 1]
    );
    device.queue.submit([readEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await outputBuffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(outputBuffer.getMappedRange());
    const tightData = new Uint8Array(canvas.width * canvas.height * bytesPerPixel);
    for (let y = 0; y < canvas.height; y++) {
        tightData.set(
            mapped.subarray(y * alignedBytesPerRow, y * alignedBytesPerRow + canvas.width * bytesPerPixel),
            y * canvas.width * bytesPerPixel
        );
    }

    outputBuffer.unmap();
    outputBuffer.destroy();
    captureTexture.destroy();

    return tightData;
}

// ============================================================================
// SORTING - Bitonic sort
// ============================================================================
// Ustvari buffer z parametri k in j za merge pass
// k - velikost merge bloka, j - korak znotraj merge bloka
const mergeParamCache = new Map();
function getMergeParamBuffer(k, j) {
    const key = (k * 100000 + j); // unikaten integer key
    if (!mergeParamCache.has(key)) {
        const buf = device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buf, 0, new Uint32Array([k, j]));
        mergeParamCache.set(key, buf);
    }
    return mergeParamCache.get(key);
}

// Pokliče getMergeParamBuffer za vse potrebne kombinacije k in j (glede na število točk)
function prewarmMergeBuffers(maxN) {
    for (let k = 512; k <= maxN * 2; k *= 2) {
        for (let j = k / 2; j >= 1; j = Math.floor(j / 2)) {
            getMergeParamBuffer(k, j);
        }
    }
}

function nextPow2(n) {
    if (n <= 0) return 1;
    return Math.pow(2, Math.floor(Math.log2(n)) + 1);
}

// Vstvari binding groupe za sortiranje - priprava, lokalno sortiranje in globalno sortiranje
function buildSortBindGroups(pointcloud) {
    if (!preparationPipeline || !localSortPipeline || !globalSortPipeline) {
        return;
    }

    const n = pointcloud.numberOfPoints;
    const N = Math.max(256, nextPow2(n));

    // Reusable view matrix buffer
    pointcloud.prepViewBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const numBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(numBuf, 0, new Uint32Array([N]));

    // Flag, ki pove preparation shaderju, ali gre za sferičen (panoramski) zajem -
    // v tem primeru prepViewBuffer vsebuje view matriko (ne projekcijsko), zato
    // je treba sort key računati kot radialno razdaljo od kamere.
    pointcloud.prepIsSphericalBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pointcloud.prepIsSphericalBuffer, 0, new Uint32Array([0]));

    // Bind group za pripravo (preparation pass)
    pointcloud.prepBG0 = device.createBindGroup({
        layout: preparationPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: pointcloud.pointBuffer } }],
    });
    pointcloud.prepBG1 = device.createBindGroup({
        layout: preparationPipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: pointcloud.prepViewBuffer } },
            { binding: 1, resource: { buffer: numBuf } },
            { binding: 2, resource: { buffer: pointcloud.prepIsSphericalBuffer } },
        ],
    });

    // Bind group za local sort
    pointcloud.localSortBG = device.createBindGroup({
        layout: localSortPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointcloud.pointBuffer } },
            { binding: 1, resource: { buffer: numBuf } },
        ],
    });

    // Bind groupi za global sort - potrebujemo jih več, glede na različne kombinacije k in j
    pointcloud.globalSortBGs = new Map();
    for (let k = 512; k <= n * 2; k *= 2) {
        for (let j = k / 2; j >= 1; j = Math.floor(j / 2)) {
            const key = k * 100000 + j;
            pointcloud.globalSortBGs.set(key, device.createBindGroup({
                layout: globalSortPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: pointcloud.pointBuffer } },
                    { binding: 1, resource: { buffer: numBuf } },
                    { binding: 2, resource: { buffer: getMergeParamBuffer(k, j) } },
                ],
            }));
        }
    }
}

function sortPointCloud(pointcloud, viewMatrix, isSpherical = false) {
    if (!preparationPipeline || !localSortPipeline || !globalSortPipeline) {
        console.warn("⚠️ Sorting pipelines not available, skipping sort");
        return;
    }

    const n = pointcloud.numberOfPoints;
    const N = Math.max(256, nextPow2(n));
    const numWorkgroups = Math.ceil(N / 256);

    console.log(`sorting: numberOfPoints=${pointcloud.numberOfPoints}, paddedSize=${n}, ratio=${n/pointcloud.numberOfPoints}`);


    device.queue.writeBuffer(pointcloud.prepViewBuffer, 0, new Float32Array(viewMatrix));
    device.queue.writeBuffer(pointcloud.prepIsSphericalBuffer, 0, new Uint32Array([isSpherical ? 1 : 0]));

    // Preparation pass
    // izračunamo globino vsake točke glede na trenutni pogled kamere in shranimo v buffer
    // To je sort key za kasnejše sortiranje
    let encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(preparationPipeline);
    pass.setBindGroup(0, pointcloud.prepBG0);
    pass.setBindGroup(1, pointcloud.prepBG1);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Local sort pass - sortiramo točke znotraj vsake skupine
    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    pass.setPipeline(localSortPipeline);
    pass.setBindGroup(0, pointcloud.localSortBG);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Global sort pass - sortiramo skupine med seboj (bitonic sort)
    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    pass.setPipeline(globalSortPipeline);
    for (let k = 512; k <= N; k *= 2) {
        for (let j = k / 2; j >= 1; j = Math.floor(j / 2)) {
            const key = k * 100000 + j;
            pass.setBindGroup(0, pointcloud.globalSortBGs.get(key));
            pass.dispatchWorkgroups(numWorkgroups);
        }
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
}

// Pomožna funkcija za izračun globine batcha glede na center celice in trenutni pogled kamere
function getBatchDepth(pointcloud, projectionViewMatrix, isSpherical = false, cameraPosition = null) {
    const [px, py, pz] = pointcloud.center;
    if (isSpherical) {
        // projectionViewMatrix je pri sferičnih zajemih identiteta (ni prave
        // projekcije), zato batche razvrstimo po radialni razdalji od kamere.
        const [cx, cy, cz] = cameraPosition;
        return Math.hypot(px - cx, py - cy, pz - cz);
    }
    const m = projectionViewMatrix;
    const clipZ = m[2]*px + m[6]*py + m[10]*pz + m[14];
    const clipW = m[3]*px + m[7]*py + m[11]*pz + m[15];
    return clipZ / clipW >= 0 ? clipZ / clipW : 10000;
}

export function cleanup() {
    if (device) {
        for (const pc of pointclouds) {
            pc.pointBuffer.destroy();
        }
        pointclouds = [];
    }
    useClassColorsBuffer?.destroy();
    device?.destroy();
    deviceLost = false;
    device = null;
}