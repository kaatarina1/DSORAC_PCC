import * as mat4 from "./js/_Mat4.js";
import { PointerController } from "./js/PointerController.js";
import { LasLoader } from "./js/LasLoader.js";
import { 
    generateFibonacciHemisphereAroundCamera, 
    writeCamerasTxt,
    writeImagesTxt,
    writePoints3DTxt,
    computeCameraMatrix,
    computeOrthoTopDownMatrix
} from "./js/CameraPosition.js";
import { RenderingControls } from "./js/RenderingControls.js";
import { AdaptiveGrid } from "./js/AdaptiveGrid.js";
import JSZip from "jszip";
import saveAs from "file-saver";

import WorkerScript from './imageGenerationWorker.js?worker';
import { SegmentationPipeline } from "./js/SegmentationPipeline.js";
import { queueBlobDownload } from "./js/Utils.js";

const CONFIG = {
    canvasWidth: 1024,
    canvasHeight: 512,
    workerCount: Math.min(7, navigator.hardwareConcurrency || 6),
    batchSize: 50,
    reconstructionColorOrder: "rgb",
    simpleRenderingColorOrder: "bgr",
    // hemisphereRadii: [1],
    // terrestrial
    // hemisphereRadii: [100, 50],
    // airborn
    // hemisphereRadii: [170],
    // object model
    // hemisphereRadii: [1],
    imagesPerCombination: 50, // 110
    // lasFile: "./data/pc/ljubljana_1_normals.las",
    lasFile: "./data/pc/ljubljana_3_normals.las",
    // lasFile: "./data/pc/ljubljana_3.las",
    // lasFile: "./data/pc/ljubljanica_normals.las",
    // lasFile: "./data/pc/zeleznica_normals.las",
    // lasFile: "./data/pc/bezigrad_normals.las",
    // lasFile: "./data/pc/fri_normals.las",
    // lasFile: "./data/pc/celje_normals.las",
    // lasFile: "./data/pc/barn_normals.las",
    // lasFile: "./data/pc/truck_normals.las",
    // lasFile: "./data/pc/ljubljana_3_draco_normals.las",
    // lasFile: "./data/pc/ljubljana_3_gpcc_normals.las",
    // lasFile: "./data/pc/bezigrad_draco_normals.las",
    // lasFile: "./data/pc/bezigrad_gpcc_normals.las",
    // lasFile: "./data/pc/lego_draco_normals.las",
    // lasFile: "./data/pc/lego_gpcc_normals.las",
    // lasFile: "./data/pc/ignatius_normals.las",
    // lasFile: "./data/pc/lego_normals.las",
    targetPositionsAndRadii: [
        // terrestrial
        {radii: 60, pos: [0, 15, 0]},
        // {radii: 50, pos: [-20, 5, -10]},
        // {radii: 50, pos: [20, 5, -10]},
        // {radii: 50, pos: [-20, 5, 10]},
        // {radii: 50, pos: [20, 5, 10]},
        // {radii: 30, pos: [5, 5, -5]},
        // {radii: 30, pos: [-8, 5, 2]},
        // {radii: 30, pos: [0, 5, -9]},
        // air borne
        // {radii: 186, pos: [0, 35.3, 0]},
        // {radii: 170, pos: [50, 5, 50]},
        // {radii: 170, pos: [-50, 5, 50]},
        // {radii: 170, pos: [50, 5, -50]},
        // {radii: 170, pos: [-50, 5, -50]},
        // object model - lego 
        // { radii: 49, pos: [0, 13, 0] },
        // object model - truck 
        // { radii: 10, pos: [0, 0.5, 0] },
        // object model - statue 
        //{ radii: 2.8, pos: [0, 1.3, 0] },
        // object model - barn 
        // { radii: 20, pos: [0, 0, 0] },
    ]
};

console.log(`Using ${CONFIG.workerCount} workers`);

class WorkerPool {
    constructor(workerCount) {
        this.workerCount = workerCount;
        this.workers = [];
        this.taskQueue = [];
        this.initialized = false;
    }

    async initialize(config) {
        console.log(`Initializing ${this.workerCount} workers...`);
        
        for (let i = 0; i < this.workerCount; i++) {
            console.log(`Creating worker ${i}...`);
            
            const worker = new WorkerScript();
            
            const initPromise = new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error(`Worker ${i} timeout`));
                }, 60000);

                worker.onmessage = (e) => {
                    if (e.data.type === 'INIT_COMPLETE') {
                        clearTimeout(timeout);
                        console.log(`✓ Worker ${i} ready`);
                        resolve();
                    } else if (e.data.type === 'ERROR') {
                        clearTimeout(timeout);
                        reject(new Error(e.data.error));
                    }
                };

                worker.onerror = (error) => {
                    clearTimeout(timeout);
                    console.error(`Worker ${i} creation error:`, error);
                    reject(error);
                };
            });

            worker.postMessage({
                type: 'INIT',
                data: {
                    width: config.width,
                    height: config.height,
                    mode: currentMode,
                    lasFile: config.lasFile,
                }
            });

            this.workers.push({ worker, id: i, busy: false });
            
            try {
                await initPromise;
            } catch (error) {
                console.error(`Failed to initialize worker ${i}:`, error);
                this.workers.forEach(({ worker }) => worker.terminate());
                this.workers = [];
                throw error;
            }
        }
        
        this.initialized = true;
        console.log(`✓ All ${this.workerCount} workers initialized`);
    }

    async processImages(tasks, progressCallback) {
        if (!this.initialized) {
            throw new Error('WorkerPool not initialized');
        }

        return new Promise((resolve, reject) => {
            const results = [];
            const processedImages = new Set();
            const failedImages = new Set();
            let completed = 0;
            let failed = 0;
            const totalTasks = tasks.length;
            let resolveTimer = null;

            this.workers.forEach(({ worker, id }) => {
                worker.onmessage = (e) => {
                    switch (e.data.type) {
                        case 'IMAGE_COMPLETE':
                            // Preveri, ali je ta slika že bila obdelana (v primeru podvajanja sporočil)
                            if (processedImages.has(e.data.imageIndex)) {
                                console.warn(`⚠️ Duplicate completion for image ${e.data.imageIndex}`);
                                break;
                            }
                            
                            processedImages.add(e.data.imageIndex);
                            completed++;
                            results.push({
                                imageIndex: e.data.imageIndex,
                                blob: e.data.blob,
                                metadata: e.data.metadata
                            });

                            if (progressCallback) {
                                progressCallback({
                                    completed,
                                    total: totalTasks,
                                    percentage: (completed / totalTasks) * 100,
                                    workerId: id
                                });
                            }

                            this.assignNextTask(id);
                            break;

                        case 'PROGRESS':
                            if (e.data.message) {
                                console.log(`Worker ${id}: ${e.data.message}`);
                            }
                            break;

                        case 'SAVE_PNG': {
                            // Prenos PNG-ja, ki ga je worker poslal (debug)
                            queueBlobDownload(e.data.blob, e.data.fileName);
                            break;
                        }

                        case 'ERROR':
                            const errorImageIndex = e.data.imageIndex;
                            if (!failedImages.has(errorImageIndex) && !processedImages.has(errorImageIndex)) {
                                failedImages.add(errorImageIndex);
                                failed++;
                                console.error(`Worker ${id} error on image ${errorImageIndex}:`, e.data.error);
                            }
                            // Ne kličemo assignNextTask tukaj, da preprečimo nadaljnje nalaganje nalog, če pride do napake
                            break;
                    }

                    // Preveri, ali so vse naloge zaključene (upoštevajoč tako uspešne kot neuspešne)
                    if (completed + failed >= totalTasks) {
                        if (resolveTimer) clearTimeout(resolveTimer);
                        resolveTimer = setTimeout(() => {
                            if (failed > 0) {
                                console.warn(`⚠️ Batch completed: ${completed} succeeded, ${failed} failed`);
                                console.warn(`Failed images:`, Array.from(failedImages));
                            }
                            resolve(results);
                        }, 500);
                    }
                };

                worker.onerror = (error) => {
                    console.error(`Worker ${id} fatal error:`, error);
                    failed++;
                    if (completed + failed >= totalTasks) {
                        if (resolveTimer) clearTimeout(resolveTimer);
                        resolveTimer = setTimeout(() => reject(error), 500);
                    }
                };
            });

            this.taskQueue = [...tasks];
            
            // Dodamo majhen zamik pri dodeljevanju nalog, da zmanjšamo začetno obremenitev GPU
            this.workers.forEach(({ id }, index) => {
                setTimeout(() => this.assignNextTask(id), index * 200);
            });
        });
    }

    assignNextTask(workerId) {
        const workerInfo = this.workers.find(w => w.id === workerId);
        
        if (this.taskQueue.length === 0) {
            workerInfo.busy = false;
            return;
        }

        const task = this.taskQueue.shift();
        workerInfo.busy = true;
        
        // Dodamo majhen zamik pri začetku novih nalog, da zmanjšamo obremenitev GPU
        setTimeout(() => {
            workerInfo.worker.postMessage({
                type: 'GENERATE_IMAGE',
                data: task
            });
        }, 100);
    }

    async shutdown() {
        console.log('Shutting down workers...');
        
        // Pošljemo sporočilo za zaustavitev vsem workerjem, da lahko počistijo GPU vire
        for (const { worker } of this.workers) {
            worker.postMessage({ type: 'SHUTDOWN' });
        }
        
        // Počakamo, da workerji obdelajo sporočilo za zaustavitev
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // Ustavimo vse workerje, da zagotovimo sprostitev vseh virov
        for (const { worker } of this.workers) {
            worker.terminate();
        }
        
        this.workers = [];
        this.initialized = false;
        console.log('✓ All workers terminated');
    }
}

// ============================================================================
// MAIN SETUP
// ============================================================================

const adapter = await navigator.gpu.requestAdapter();
const hasBGRA8unormStorage = adapter.features.has("bgra8unorm-storage");
const device = await adapter?.requestDevice({
    requiredFeatures: hasBGRA8unormStorage 
        ? ["bgra8unorm-storage", "float32-filterable", "float32-blendable", "timestamp-query"] 
        : ["float32-filterable", "float32-blendable", "timestamp-query"],
    requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
});

const canvas = document.querySelector("canvas");
canvas.width = CONFIG.canvasWidth;
canvas.height = CONFIG.canvasHeight;

const context = canvas.getContext("webgpu");
const format = hasBGRA8unormStorage
    ? navigator.gpu.getPreferredCanvasFormat()
    : "rgba8unorm";
context.configure({ device, format, alphaMode: "premultiplied" });

console.log("Loading point cloud...");
const lasLoader = new LasLoader(CONFIG.lasFile);
const lasData = await lasLoader.loadLasData();
const positions = lasData.positions;
const colors = lasData.colors;
const colorsRGB = lasData.colorsRGB;
const scaleFactor = lasData.scaleFactor;
const normals = lasData.normals;
const classifications = lasData.classifications;

function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
    const m = l - c / 2;
    let r = 0, g = 0, b = 0;
    const sec = Math.floor(h * 6) % 6;
    if (sec === 0) { r = c; g = x; }
    else if (sec === 1) { r = x; g = c; }
    else if (sec === 2) { g = c; b = x; }
    else if (sec === 3) { g = x; b = c; }
    else if (sec === 4) { r = x; b = c; }
    else { r = c; b = x; }
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

// RGBA zapakiran kot 0xAABBGGRR
const classPalette = new Uint32Array(256);

// fallback (če za razred ni definirana barva)
// je magenta (enostavno zaznamo odstopanje)
for (let i = 0; i < 256; i++) {
    classPalette[i] = 0xFFFF00FF;
}

// Barve razredov za vizualizacijo klasifikacije.
// Razredi 0–200 so generirani z zlatoreznim korakom po barvnem krogu:
// φ ≈ 0.618 zagotavlja, da sta zaporedni barvi vedno ~222° narazen,
// torej sosednji indeksi nikoli nimajo podobnih barv.
// Zraven se ciclično menja nasičenost/svetlost, kar dodatno razlikuje barve z enakim odtenkom.
const GOLDEN_RATIO = 0.618033988749895;
const SL_VARIANTS = [
    [0.88, 0.52],  // vivid mid
    [0.95, 0.36],  // deep / saturated dark
    [0.72, 0.66],  // bright light
];
for (let i = 0; i < 201; i++) {
    const h = (i * GOLDEN_RATIO) % 1.0;
    const [s, l] = SL_VARIANTS[i % SL_VARIANTS.length];
    const [r, g, b] = hslToRgb(h, s, l);
    classPalette[i] = (r | (g << 8) | (b << 16) | (0xFF << 24)) >>> 0;
}

const SEMANTIC_OVERRIDES = {
    // Unclassified / scene labels 
    0:   [160, 160, 160],  // unclassified — neutral gray
    1:   [235, 218, 198],  // leaving room — warm ivory (scene label)

    // INDOOR: living room scene 
    2:   [195, 165, 120],  // floor — warm tan
    6:   [225, 105,  20],  // table — vivid orange
    7:   [  0, 155, 172],  // chair — strong teal
    8:   [ 48,  95, 215],  // sofa — cobalt blue
    11:  [245, 215,  45],  // lamp — bright yellow
    13:  [ 90,  55,  18],  // cabinet — dark brown
    16:  [ 38, 175,  58],  // plant — green
    20:  [200,  20, 142],  // carpet — deep magenta
    73:  [160, 155, 145],  // pillar — warm gray
    107: [ 58, 102, 205],  // structure — blue
    108: [  0, 112, 122],  // solar panels — dark teal
    109: [ 48,  18,  80],  // shadows — dark purple
    159: [175, 232, 242],  // glass — light cyan
    164: [ 88,  98, 128],  // coffee table — slate-blue (contrast with wood table)
    165: [235,  72,  72],  // rug — coral red (contrast with magenta carpet)
    166: [ 48,  50,  55],  // tv stand — near black
    167: [105,  28, 172],  // armchair — indigo-purple (contrast with teal/blue sofa/chair)
    168: [ 80, 125, 132],  // furniture generic — muted slate-teal
    169: [255,  95, 178],  // painting — hot pink
    172: [152,  20,  20],  // fireplace — dark red
    173: [215, 152, 175],  // living room — soft rose (scene label)
    174: [232, 148,  52],  // doorway — amber
    175: [212, 192, 155],  // box — beige

    // STREET / URBAN: road / sidewalk / building / car / tree / grass / sky 
    40:  [228,  32,  32],  // car — vivid red
    41:  [148,  15,  15],  // truck — dark crimson
    47:  [ 25, 140,  35],  // tree — deep green
    48:  [148, 228,  28],  // grass — bright yellow-green
    49:  [190, 195, 208],  // sidewalk — light cool gray
    50:  [ 55,  60,  68],  // road — dark charcoal
    51:  [215, 132,  48],  // building — warm orange
    52:  [140,  82,  25],  // fence — brown
    59:  [165,  70,  18],  // roof — dark red-brown
    62:  [  0, 202, 235],  // window — cyan
    63:  [132,  18, 185],  // door — purple
    64:  [ 48,  58,  80],  // street — dark slate (slightly bluer than road)
    68:  [205, 158,  55],  // house — amber
    74:  [222, 205,  48],  // house exterior — golden yellow
    75:  [ 98, 115, 150],  // city — slate blue
    76:  [ 62, 172,  75],  // village — warm green
    77:  [205, 155,  65],  // town — amber
    78:  [ 88, 168, 255],  // sky — azure blue
    81:  [152,  95,  42],  // cabin — brown
    82:  [182, 132,  58],  // hut — orange-brown
    83:  [238,  95,  28],  // vehicle — orange-red
    84:  [255, 132,   0],  // construction site — safety orange
    87:  [182, 162, 110],  // driveway — tan
    95:  [ 95, 108, 142],  // parking lot — blue-gray
    101: [ 65, 108, 202],  // parking — blue
    111: [255,  45, 148],  // circles — bright pink (special marking)
    112: [128, 130, 135],  // concrete wall — medium gray
    127: [235,  90,  80],  // car door — coral-pink
    128: [ 32,  32,  38],  // car tire — near black
    135: [252, 170,   0],  // sign — vivid orange-yellow
    136: [158, 152, 142],  // stone pillar — warm gray
    137: [205, 205, 215],  // light poles — silver
    144: [255, 240, 138],  // streetlights — soft gold
    177: [128, 162, 202],  // train car — silver-blue
    180: [ 85, 110, 148],  // street scene — muted blue
    181: [202,  28, 162],  // tower — magenta
    182: [242, 108, 162],  // bell tower — pink
    184: [118, 138,  38],  // shed — olive
    185: [222,  72,  52],  // van — coral-red

    // NATURE: tree / grass / water / rocks / sky / flowers
    60:  [105, 138,  35],  // bush — olive-green (darker/yellower than grass)
    61:  [ 18, 100, 202],  // water — deep blue
    67:  [178, 192, 212],  // space station — silver
    69:  [ 75,  92, 112],  // battleship — dark steel
    70:  [ 15, 158,  38],  // lush — rich green
    71:  [115, 168,  38],  // grassy — yellow-olive
    72:  [195, 195, 202],  // bleacher — light gray
    80:  [ 15,  95,  18],  // forest — dark forest green
    86:  [155, 108,  55],  // dirt — earthy brown
    90:  [195, 162, 112],  // path — sandy tan
    91:  [142, 125,  92],  // rocks — warm gray
    92:  [200, 198,  45],  // field — golden-yellow
    93:  [212, 218, 224],  // fog — pale gray
    94:  [  0, 198, 205],  // pool — turquoise
    97:  [188, 218,  28],  // lawn — yellow-green (distinguishable from grass [148,228,28])
    104: [ 55, 155,  55],  // foliage — muted green
    105: [ 85, 135,  32],  // shrubs — olive
    110: [152, 232,   0],  // grassy area — vivid lime
    130: [255,  85, 155],  // flowers — hot pink
    133: [105, 118, 142],  // stone — blue-gray
    134: [188, 118, 245],  // flowerbed — lavender (contrast with hot-pink flowers)
    138: [ 78, 215,  45],  // leaves — bright green
    139: [162, 108,  40],  // wood — dark amber
    143: [ 75,  88, 112],  // lot — dark slate
    150: [110, 182,  92],  // grassy_field — sage green
    155: [115, 192,  98],  // garden — pastel green
    171: [ 12, 110,  25],  // pine — deep forest
    178: [255, 188, 195],  // cherry tree — light pink

    // SKY / ATMOSPHERE
    152: [225, 235, 248],  // clouds — pale blue-white
    162: [ 18,  20,  36],  // night-time — very dark
    163: [ 12,  12,  18],  // darkness — near black
    176: [  8,  12,  62],  // night sky — very dark blue
    179: [102, 128, 165],  // rainy — steel blue-gray

    //  GROUND / PAVEMENT (all co-occur on streets)
    57:  [112, 115, 125],  // pavement — medium gray
    58:  [255, 165,  45],  // crosswalk — vivid orange
    100: [178, 180, 188],  // concrete — light gray
    102: [ 42,  45,  50],  // asphalt — near black
    103: [ 95, 165, 242],  // parking space — light blue
    113: [195, 172, 130],  // pathway — sand
    118: [ 28,  30,  34],  // asphalt pavement — very dark (darker than asphalt)
    121: [152, 162, 185],  // concrete pavement — gray-blue
    122: [162, 152, 138],  // stone pavement — warm gray
    151: [162, 178, 158],  // concrete_path — gray-green
    170: [158, 175, 148],  // concrete path — gray-green slightly different hue

    // PAVEMENT MARKINGS (street scenes — vivid unique hues per type) 
    123: [148, 232,   0],  // pavement markings — vivid lime
    124: [  0, 232, 222],  // road markings — vivid cyan
    125: [255,  45, 180],  // sidewalk markings — vivid pink
    126: [ 45, 100, 255],  // parking lot markings — vivid blue
    145: [168,   0, 255],  // street markings — vivid purple
    153: [ 95, 130, 172],  // metal sign — steel blue
    154: [252, 200,   0],  // signage — vivid amber-yellow
    160: [255, 222,   0],  // markings — bright yellow
    161: [255, 128,   0],  // lines — bright orange

    // BRICK BUILDINGS (all appear together — unrelated hues for each) 
    114: [205, 125,  55],  // brick_house — orange
    115: [ 28,  48, 185],  // brick_roof — dark blue
    116: [  0, 198, 222],  // brick_window — cyan
    117: [ 95, 198,  25],  // brick building — lime green
    119: [248,  45, 148],  // brick wall surface — hot pink
    120: [212,  15, 192],  // brick wall — magenta
    140: [140,  35, 192],  // brick_pavement — purple
    141: [ 15, 130,  45],  // brick_fence — dark green
    142: [212, 148,  52],  // brick-paved road — warm amber-brown
    147: [215, 178,  38],  // brick_path — amber
    148: [ 95, 172, 242],  // brick_garage_window — light blue
    149: [200, 110,  45],  // brick house — terracotta
    156: [215,  35,  35],  // brick structure — vivid red
    157: [  0, 172, 162],  // brick facade — teal
};
for (const [cls, rgb] of Object.entries(SEMANTIC_OVERRIDES)) {
    const [r, g, b] = rgb;
    classPalette[Number(cls)] = (r | (g << 8) | (b << 16) | (0xFF << 24)) >>> 0;
}
const classColors = new Uint32Array(positions.length / 3);
for (let i = 0; i < classColors.length; i++) {
    classColors[i] = classPalette[classifications[i]];
}

// Log class -> color mapping (samo razredi, ki se dejansko pojavijo v podatkih)
const presentClasses = new Map();
for (const cls of classifications) {
    if (!presentClasses.has(cls)) {
        const packed = classPalette[cls];
        const r = (packed >>  0) & 0xFF;
        const g = (packed >>  8) & 0xFF;
        const b = (packed >> 16) & 0xFF;
        presentClasses.set(cls, `rgb(${r}, ${g}, ${b})`);
    }
}
console.log(`Class → color mapping (${presentClasses.size} classes):`);
for (const [cls, color] of [...presentClasses].sort((a, b) => a[0] - b[0])) {
    console.log(`  %c  %c class ${cls} → ${color}`, `background:${color};padding:0 8px`, 'background:none');
}

console.log(`Loaded ${positions.length / 3} points`);

function computeBBox(positions) {
    let bbMin = [Infinity, Infinity, Infinity];
    let bbMax = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
        bbMin[0] = Math.min(bbMin[0], positions[i]);
        bbMin[1] = Math.min(bbMin[1], positions[i + 1]);
        bbMin[2] = Math.min(bbMin[2], positions[i + 2]);
        bbMax[0] = Math.max(bbMax[0], positions[i]);
        bbMax[1] = Math.max(bbMax[1], positions[i + 1]);
        bbMax[2] = Math.max(bbMax[2], positions[i + 2]);
    }
    return { bbMin, bbMax };
}
// Izračunamo bounding box oblaka točk
let { bbMin, bbMax } = computeBBox(positions);
console.log("Point cloud bounding box:", bbMin, bbMax);

// Naložimo shaderje in ustvarimo render pipeline
const renderingCode = await fetch("./shaders/rendering.wgsl").then(r => r.text());
const renderingModule = device.createShaderModule({ code: renderingCode });

const renderingPipeline = device.createRenderPipeline({
    vertex: { module: renderingModule },
    fragment: {
        module: renderingModule,
        targets: [{ format }],
    },
    primitive: { topology: "point-list" },
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth32float",
    },
    layout: "auto",
});

// Prilagodimo rendering pipeline za quad/billboard/gaussian
async function loadQuadPipeline(shaderFile, enableBlend) {
    const code = await fetch(shaderFile).then(r => r.text());
    const module = device.createShaderModule({ code });
    const blendState = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    return device.createRenderPipeline({
        vertex: { module },
        fragment: { module, targets: [{ format, ...(enableBlend ? { blend: blendState } : {}) }] }, // blend dodamo samo če gre za gaussian
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: {
            depthWriteEnabled: !enableBlend,  // false za GAUSSIANS, saj pri gaussian za pravilen izris moremo sortirat 
            depthCompare: "less",
            format: "depth32float"
        },
        layout: "auto",
    });
}

const quadPipelines = {
    DISKS:      await loadQuadPipeline("./shaders/rendering_disks.wgsl",      false),
    BILLBOARDS: await loadQuadPipeline("./shaders/rendering_billboards.wgsl", false),
    GAUSSIANS:  await loadQuadPipeline("./shaders/rendering_gaussians.wgsl",  true),
}; // določimo kateri shader uporabljamo

// Render mode state
let currentMode = "POINTS";
let currentPointSize = 0.02; // 0.2

// SceneParams buffer
const sceneParamsBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

function writeSceneParams(tp, camPos, pointSize, isOrtho = false) {
    const data = new Float32Array(12);
    data[0] = tp[0];   data[1] = tp[1];   data[2] = tp[2];   data[3] = 0;   // targetPosition
    // cameraPos.w: 0 = perspektivna kamera, 1 = oortografska kamera (potrebno za dločanje orientacije diskov)
    data[4] = camPos[0]; data[5] = camPos[1]; data[6] = camPos[2]; data[7] = 0; // cameraPos
    data[8] = pointSize; data[9] = isOrtho ? 1 : 0; data[10] = 0; data[11] = 0; // pointSize + isOrtho + pad
    device.queue.writeBuffer(sceneParamsBuffer, 0, data);
}

const classUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(classUniformBuffer, 0, new Uint32Array([0, 0, 0, 0]));

// ============================================================================
// RENDERING CONTROLS UI
// ============================================================================
let useReconstruction = true; // State variable for reconstruction mode

// ortofoto 
let orthoMode  = false;
const sceneHeight = bbMax[1] - bbMin[1];
const sceneCX  = (bbMin[0] + bbMax[0]) / 2;
const sceneCZ  = (bbMin[2] + bbMax[2]) / 2;
let orthoEyeY  = bbMax[1] + sceneHeight * 0.8;
let orthoPanX  = sceneCX;
let orthoPanZ  = sceneCZ;
let orthoZoom  = Math.max(
    (bbMax[0] - bbMin[0]) / 2,
    (bbMax[2] - bbMin[2]) / 2
) * 1.15;

let segmentation = null;
const controls = new RenderingControls({
    modes: ["POINTS", "DISKS", "BILLBOARDS", "GAUSSIANS"],
    currentMode,
    currentPointSize,
    useReconstruction,
    onModeChange: (mode) => { currentMode = mode; },
    onSizeChange: (size) => { currentPointSize = size; },
    onReconstructionChange: (value) => { useReconstruction = value; },
    onOrthoChange: (value) => { orthoMode = value; },
    onColorModeChange: (mode) => {
        device.queue.writeBuffer(classUniformBuffer, 0, new Uint32Array([mode, 0, 0, 0]));
    },
    onMasksLoaded: (zipFile, votingMode) => segmentation.processZip(zipFile, votingMode),
    onDownloadLas: () => downloadLas(),
    onConfigLoaded: async (jsonFile) => captureImagesFromConfig(JSON.parse(await jsonFile.text())),
});
controls.mount(document.body);

// Sorting pipelines
const preparationCode  = await fetch("./shaders/preparation.wgsl").then(r => r.text());
const localSortCode    = await fetch("./shaders/localSort.wgsl").then(r => r.text());
const globalSortCode   = await fetch("./shaders/globalSort.wgsl").then(r => r.text());

const preparationPipeline = device.createComputePipeline({
    compute: { module: device.createShaderModule({ code: preparationCode }), entryPoint: "main" },
    layout: "auto",
});
const localSortPipeline = device.createComputePipeline({
    compute: { module: device.createShaderModule({ code: localSortCode }), entryPoint: "compute" },
    layout: "auto",
});
const globalSortPipeline = device.createComputePipeline({
    compute: { module: device.createShaderModule({ code: globalSortCode }), entryPoint: "mergePass" },
    layout: "auto",
});

// Matrix buffers: MVP (binding 0) + view matrix (binding 1) in group 1
const mvpBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const viewMatrixBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const matricesBindGroup = device.createBindGroup({
    layout: renderingPipeline.getBindGroupLayout(1),
    entries: [
        { binding: 0, resource: { buffer: mvpBuffer } },
        { binding: 1, resource: { buffer: viewMatrixBuffer } },
    ],
});

// Kontrole kamere
let cameraPosition = [0, 20, 0]; // 0, 170, 0
let cameraTarget = [0, 0, 0];
let yaw = 0;
let pitch = Math.PI / 2 - 0.1; 
let distance = 20; // 170

const pointerController = new PointerController(canvas);

pointerController.addEventListener("pointermove", (e) => {
    const dx = e.movementX;
    const dy = e.movementY;
    const sensitivity = 0.005;
    
    if (orthoMode) {
        const aspect = canvas.width / canvas.height;
        let halfW, halfH;
        if (canvas.width <= canvas.height) {
            halfW = orthoZoom;
            halfH = orthoZoom / aspect;
        } else {
            halfH = orthoZoom;
            halfW = orthoZoom * aspect;
        }
        const worldPerPixelX = (2 * halfW) / canvas.width;
        const worldPerPixelZ = (2 * halfH) / canvas.height;
        orthoPanX -= dx * worldPerPixelX;
        orthoPanZ -= dy * worldPerPixelZ;
    } else {
        yaw -= dx * sensitivity;
        pitch -= dy * sensitivity;
        pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2 - 0.1, pitch));
        updateCameraOrbit();
    }
});

let SORT = false;
function updateCameraOrbit() {
    cameraPosition[0] = cameraTarget[0] + distance * Math.cos(pitch) * Math.sin(yaw);
    cameraPosition[1] = cameraTarget[1] + distance * Math.sin(pitch);
    cameraPosition[2] = cameraTarget[2] + distance * Math.cos(pitch) * Math.cos(yaw);
    SORT = true;
}

updateCameraOrbit();

document.addEventListener("keydown", (event) => {
     if (orthoMode) {
        const panStep    = orthoZoom * 0.05;
        const heightStep = sceneHeight * 0.05;
        switch (event.key) {
            case "ArrowLeft":  orthoPanX -= panStep; break;
            case "ArrowRight": orthoPanX += panStep; break;
            case "ArrowUp":    orthoPanZ -= panStep; break;
            case "ArrowDown":  orthoPanZ += panStep; break;
            case "w": case "W":
                orthoEyeY += heightStep;
                console.log(`Ortho eye Y: ${orthoEyeY.toFixed(2)}`);
                break;
            case "s": case "S":
                orthoEyeY -= heightStep;
                orthoEyeY = Math.max(bbMax[1] + 0.5, orthoEyeY);
                console.log(`Ortho eye Y: ${orthoEyeY.toFixed(2)}`);
                break;
        }
        return;
    }
    const moveSpeed = 1.0;
    const forward = mat4.normalize([
        cameraTarget[0] - cameraPosition[0],
        0,
        cameraTarget[2] - cameraPosition[2],
    ]);
    const right = mat4.cross(forward, [0, 1, 0]);
    
    switch(event.key) {
        case "ArrowLeft":
            cameraPosition[0] -= right[0] * moveSpeed;
            cameraPosition[2] -= right[2] * moveSpeed;
            cameraTarget[0] -= right[0] * moveSpeed;
            cameraTarget[2] -= right[2] * moveSpeed;
            break;
        case "ArrowRight":
            cameraPosition[0] += right[0] * moveSpeed;
            cameraPosition[2] += right[2] * moveSpeed;
            cameraTarget[0] += right[0] * moveSpeed;
            cameraTarget[2] += right[2] * moveSpeed;
            break;
        case "ArrowUp":
            cameraPosition[0] += forward[0] * moveSpeed;
            cameraPosition[2] += forward[2] * moveSpeed;
            cameraTarget[0] += forward[0] * moveSpeed;
            cameraTarget[2] += forward[2] * moveSpeed;
            break;
        case "ArrowDown":
            cameraPosition[0] -= forward[0] * moveSpeed;
            cameraPosition[2] -= forward[2] * moveSpeed;
            cameraTarget[0] -= forward[0] * moveSpeed;
            cameraTarget[2] -= forward[2] * moveSpeed;
            break;
        case "w":
            cameraPosition[1] += moveSpeed;
            cameraTarget[1] += moveSpeed;
            console.log("Camera Y:", cameraPosition[1].toFixed(3));
            break;
        case "s":
            cameraPosition[1] -= moveSpeed;
            cameraTarget[1] -= moveSpeed;
            break;
    }
    SORT = true;
});

// ============================================================================
// PARALLEL GENERATION WITH BATCHING
// ============================================================================

let workerPool = null;
let zip = null;

document.addEventListener("keydown", async (event) => {
    if (event.key === "T" || event.key === "t") {
        await generateImagesParallel();
    }
});

async function generateImagesParallel() {
    let densitySum = 0; 
    const startTime = performance.now();
    const effectiveReconstruction = useReconstruction && currentMode === "POINTS";
    const useParallelBatching = effectiveReconstruction;
    // Vsak worker naloži svojo lastno kopijo celotnega oblaka točk na GPU.
    // Pri zelo velikih oblakih (npr. ljubljana_2/3) bi preveč vzporednih workerjev
    // skupaj zahtevalo preveliko količino VRAM samo za naložit točke točke, kar povzroči izgubo GPU naprave.
    // Omejimo število workerjev tako, da skupna poraba pomnilnika za točke ostane pod budgetom.
    const pointBufferBudgetBytes = 4 * 1024 ** 3; // 4 GB, konzervativna privzeta vrednost
    const bytesPerWorker = numberOfAllPoints * pointByteSize;
    const memorySafeWorkerCount = Math.max(1, Math.floor(pointBufferBudgetBytes / bytesPerWorker));
    const activeWorkerCount = useParallelBatching
        ? Math.min(CONFIG.workerCount, memorySafeWorkerCount)
        : 1;
    const activeBatchSize = useParallelBatching ? CONFIG.batchSize : Number.MAX_SAFE_INTEGER;
    const colorOrder = effectiveReconstruction || currentMode === "BILLBOARDS" || currentMode === "GAUSSIANS"
        ? CONFIG.reconstructionColorOrder
        : CONFIG.reconstructionColorOrder;

    console.log(
        `🚀 Starting generation (${useParallelBatching ? "parallel + batched" : "single worker, no batching"})...`
    );
    console.log(`Capture mode: ${effectiveReconstruction ? "reconstruction" : "point cloud only"} (mode=${currentMode})`);
    console.log(`Output color order: ${colorOrder.toUpperCase()}`);
    if (useParallelBatching && activeWorkerCount < CONFIG.workerCount) {
        console.log(
            `⚠️ Reduced worker count from ${CONFIG.workerCount} to ${activeWorkerCount} ` +
            `to fit point buffers (${numberOfAllPoints.toLocaleString()} points, ` +
            `~${(bytesPerWorker / 1024 ** 3).toFixed(2)} GB/worker) within the VRAM budget`
        );
    }

    try {
        // Najprej generiramo vse pozicije kamer in naloge, preden začnemo z delom workerjev
        const allTasks = [];
        let imageIndex = 0;
        // let count = 0;

        for (const { radii: radius, pos: target } of CONFIG.targetPositionsAndRadii) {
            const oversampledCount = Math.max(
                CONFIG.imagesPerCombination * 2,
                CONFIG.imagesPerCombination + 1
            );
            // if (count === 5) {
            //     break;
            // }
            // count++;

            const poses = generateFibonacciHemisphereAroundCamera(
                oversampledCount,
                radius,
                target
            );

            const selectedPoses = poses.slice(0, CONFIG.imagesPerCombination);

            console.log(
                `Generated ${selectedPoses.length}/${CONFIG.imagesPerCombination} camera poses around target ${target} at radius ${radius}`
            );

            for (const pos of selectedPoses) {
                allTasks.push({
                    cameraPosition: pos,
                    target: target,
                    imageIndex: imageIndex++,
                    targetPosition: target,
                    useReconstruction: effectiveReconstruction,
                    colorOrder,
                    radius,
                    mode: currentMode,
                    pointSize: currentPointSize,
                });
            }
        }

        console.log(`📋 Total images to generate: ${allTasks.length}`);
        
        // Initializiramo ZIP arhiv za shranjevanje rezultatov
        zip = new JSZip();

        const allResults = [];
        const numBatches = Math.ceil(allTasks.length / activeBatchSize);

        for (let batch = 0; batch < numBatches; batch++) {
            const start = batch * activeBatchSize;
            const end = Math.min(start + activeBatchSize, allTasks.length);
            const batchTasks = allTasks.slice(start, end);
            
            console.log(`\n📦 Processing batch ${batch + 1}/${numBatches} (images ${start + 1}-${end})...`);
            
            const batchResults = await runBatchedTasks(batchTasks, CONFIG.lasFile, allTasks.length, start, startTime, batch, numBatches, useParallelBatching, activeWorkerCount);
            allResults.push(...batchResults);
        }


        // Dodamo vse slike v ZIP arhiv in pripravimo podatke za izvoz
        console.log(`\n📦 Adding images to ZIP...`);
        let addedCount = 0;
        const receivedIndices = new Set();
        
        for (const result of allResults) {
            if (result && result.blob && result.metadata) {
                receivedIndices.add(result.imageIndex);
                zip.file(`images/${result.metadata.filename}`, result.blob);
                addedCount++;
            }
        }
        
        console.log(`✓ Added ${addedCount}/${allResults.length} images to ZIP`);
        
        // Preverimo ali obstajajo manjkajoče slike
        const expectedIndices = Array.from({length: allTasks.length}, (_, i) => i);
        const missingIndices = expectedIndices.filter(i => !receivedIndices.has(i));
        if (missingIndices.length > 0) {
            console.warn(`⚠️ Missing ${missingIndices.length} images:`, missingIndices.slice(0, 20));
        }
        
        const successfulResults = allResults.filter(r => r && r.metadata);        
        console.log(`✓ Successfully generated ${successfulResults.length}/${allTasks.length} images`);

        // Povprečna gostota projiciranih točk (točk na piksel)
        const densities = successfulResults
            .map(r => r.metadata.density)
            .filter(d => d !== undefined);
        if (densities.length > 0) {
            const avgDensity = densities.reduce((sum, d) => sum + d, 0) / densities.length;
            console.log(
                `📊 Average viewpoint density: ${avgDensity.toFixed(2)} points/pixel ` +
                `over ${densities.length} views @ ${CONFIG.canvasWidth}×${CONFIG.canvasHeight}`
            );
        }

        // Izvozimo rezultate (cameras.txt, images.txt, points3D.txt) v ZIP arhiv
        await exportResults(
            successfulResults
                .map(r => ({
                    imageId: r.imageIndex + 1,
                    filename: r.metadata.filename,
                    quat: r.metadata.quat,
                    t: r.metadata.t,
                    camera_id: 1,
                    name: r.metadata.filename,
                    fx: r.metadata.fx,
                    fy: r.metadata.fy,
                    cx: r.metadata.cx,
                    cy: r.metadata.cy,
                    width: r.metadata.width,
                    height: r.metadata.height,
                }))
        );

        const totalTime = (performance.now() - startTime) / 1000 / 60;

        console.log(`
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ✓ PARALLEL GENERATION COMPLETE
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    📊 Images: ${successfulResults.length}/${allTasks.length}
                    ⏱️  Time: ${totalTime.toFixed(1)} minutes
                    ⚡ Avg: ${successfulResults.length > 0 ? (totalTime * 60 / successfulResults.length).toFixed(1) : 'N/A'}s/image
                    🚀 Workers: ${activeWorkerCount}
                    📦 Batches: ${numBatches} (${useParallelBatching ? activeBatchSize : allTasks.length} images each)
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                `);

    } catch (error) {
        console.error("❌ Error:", error);
    } finally {
        if (workerPool) {
            await workerPool.shutdown();
        }
    }
}

async function runBatchedTasks(tasks, lasFile, allTasksCount, start, startTime, batch = 0, numBatches = 1, useParallelBatching = false, activeWorkerCount = 1) {
    // Ustvarimo nov WorkerPool za vsak batch, da zagotovimo popolno sprostitev GPU virov po vsakem batchu
    workerPool = new WorkerPool(activeWorkerCount);
    await workerPool.initialize({
        width: CONFIG.canvasWidth,
        height: CONFIG.canvasHeight,
        mode: currentMode,
        lasFile: lasFile,
    });

    // Obdelamo trenutni batch nalog in spremljamo napredek
    const batchResults = await workerPool.processImages(
        tasks,
        (progress) => {
            const totalCompleted = start + progress.completed;
            const elapsed = (performance.now() - startTime) / 1000 / 60;
            const remaining = (elapsed / totalCompleted) * (allTasksCount - totalCompleted);
            
            console.log(
                `✓ ${totalCompleted}/${allTasksCount} ` +
                `(${((totalCompleted / allTasksCount) * 100).toFixed(1)}%) | ` +
                `Batch: ${progress.completed}/${tasks.length} | ` +
                `Elapsed: ${elapsed.toFixed(1)}m | ` +
                `Remaining: ~${remaining.toFixed(1)}m`
            );
        }
    );
            
    console.log(`📊 Batch ${batch + 1} results: ${batchResults.length}/${tasks.length} images received`);

    if (useParallelBatching) {
        console.log(`⏳ Waiting for GPU and message queue to clear...`);
        await new Promise(resolve => setTimeout(resolve, 4000));
    }
            
    // Ustavimo workerje, da sprostimo GPU vire
    await workerPool.shutdown();
    console.log(`✓ Batch ${batch + 1} complete, workers shut down`);
            
    // Dodamo majhen zamik pred začetkom naslednjega batcha, da zagotovimo, da so vsi GPU viri sproščeni
    if (useParallelBatching && batch < numBatches - 1) {
        console.log(`⏳ Cooling down GPU before next batch...`);
        await new Promise(resolve => setTimeout(resolve, 2000));
    }

    return batchResults;
}

async function captureImagesFromConfig(configJson) {
    const startTime = performance.now();
    if (configJson.canvasWidth) CONFIG.canvasWidth = configJson.canvasWidth;
    if (configJson.canvasHeight) CONFIG.canvasHeight = configJson.canvasHeight;
    const zip = new JSZip();
    for (const model of configJson.models) {
        const modelName = model.lasFile.split('/').pop().replace(/\.las$/i, '');
        const loader = new LasLoader(model.lasFile);
        const modelData = await loader.loadLasData();

        const tasks = model.cameras.map((cam, i) => ({
            imageIndex: i,
            cameraPosition: cam.eye,
            target: cam.target ?? [cam.panX, cam.eyeY - 1, cam.panZ],
            projection: cam.projection ?? "PERSPECTIVE",
            eyeY: cam.eyeY, panX: cam.panX, panZ: cam.panZ, zoom: cam.zoom,
            yawDeg: cam.yawDeg,
            fovDeg: cam.fovDeg,
            mode: cam.technique ?? "POINTS",
            useReconstruction: cam.useReconstruction ?? true,
            pointSize: cam.pointSize ?? 0.2,
            colorOrder: CONFIG.reconstructionColorOrder,
            targetPosition: cam.target ?? [cam.panX, cam.eyeY - 1, cam.panZ],
        }));

        const results = await runBatchedTasks(tasks, model.lasFile, tasks, CONFIG.batchSize, startTime);

        for (const result of results) zip.file(`${modelName}/${result.metadata.filename}`, result.blob);
    }
    const blob = await zip.generateAsync({ type: "blob" });
    saveAs(blob, configJson.outputZipName ?? "batch_capture_from_config.zip");
}

async function exportResults(capturedData) {
    // Preverimo ali je ZIP objekt veljaven pred uporabo
    console.log("Exporting results to ZIP...");
    if (!zip) {
        console.error('❌ ZIP object is null - cannot export');
        return;
    }
    
    zip.file("cameras.txt", writeCamerasTxt(capturedData));
    zip.file("images.txt", writeImagesTxt(capturedData, scaleFactor));
    zip.file("points3D.txt", writePoints3DTxt(positions, colorsRGB, scaleFactor));

    const blob = await zip.generateAsync({ type: "blob" });
    saveAs(blob, "export_bundle.zip");
    console.log("✓ Export saved");
}

async function downloadLas() {
    if (!segmentation?.segClassByGlobalIndex) {
        console.warn('No segmentation result yet — run segmentation first.');
        return;
    }

    // remapping to the CLSS classification
    const SEG_TO_LAS = new Map([
    // Terrain (2)
    ...[
        48, 49, 50, 58, 71, 97, 110, 150, 92, 155,
        86, 91, 133, 90, 113, 84, 87,
        57, 118, 121, 122, 75, 76
    ].map(c => [c, 2]),

    // Low Vegetation (3)
    ...[
        60, 105, 104, 70, 130, 134, 16
    ].map(c => [c, 3]),

    // Middle Vegetation (4)
    [47, 4],

    // High Vegetation (5)
    ...[
        171, 80
    ].map(c => [c, 5]),

    // Building (6)
    ...[
        1, 63, 62, 59, 73,
        112, 119, 120, 136, 174,
        68, 74, 81, 82, 184,
        51, 107, 117, 149, 156,
        157, 187, 181, 182
    ].map(c => [c, 6]),

    // Water (9)
    ...[
        61, 94
    ].map(c => [c, 9]),

    // Moving Object / Noise (18)
    ...[
        40, 41, 83, 185
    ].map(c => [c, 18]),

    // Unclassified (1)
    [0, 1]
]);

    console.log('Fetching original LAS file...');
    const response = await fetch(CONFIG.lasFile);
    const arrayBuffer = await response.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);
    const view  = new DataView(arrayBuffer);

    const pointFormat  = bytes[104] & 0b1111;
    const recordLength = view.getUint16(105, true);
    const dataOffset   = view.getUint32(96,  true);
    // Formats 0-5: classification stored in user_data at byte 17
    // Formats 6-10: full uint8 classification at byte 16
    const classOff     = pointFormat <= 5 ? 17 : 16;

    const seg = segmentation.segClassByGlobalIndex;
    for (let j = 0; j < numberOfAllPoints; j++) {
        const segClass = seg[jToGlobalIndex[j]];
        bytes[dataOffset + j * recordLength + classOff] = SEG_TO_LAS.get(segClass) ?? 1;
    }

    const filename = CONFIG.lasFile.split('/').pop().replace(/\.las$/i, '_classified.las');
    const blob = new Blob([arrayBuffer], { type: 'application/octet-stream' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log(`Downloaded ${filename} (${numberOfAllPoints} points).`);
}

// Interactive preview render loop
const pointByteSize = 48;
const maxNumberOfPointsPerBuffer = 1024 * 1024;
const numberOfAllPoints = positions.length / 3;
const pointclouds = [];

// Ustvarimo adaptivno mrežo, ki razdeli točke v celice z največ maxNumberOfPointsPerBuffer točkami
// To nam omogoča (za razliko od naključnega deljenja), da nato te celice sortiramo
// glde na povprečno globino točk v celici, kar je potrebno za pravilen izris gausovk (back-to-front)
// Točke najprej sortiramo znotraj vsake celice, nato pa sortiramo še celice med seboj
// Celice niso razporejene v regularno mrežo, ampak so prilagojene gostoti točk
// saj točje v oblaku niso enakomerno proazdeljene - 
// v gostejših delih bo več manjših celic, v redkejših pa manj večjih celic
const grid = new AdaptiveGrid(positions, maxNumberOfPointsPerBuffer);
console.log(`Adaptive grid: ${grid.cells.length} cells`);

let globalPointOffset = 0;
for (const cell of grid.cells) {
    const count = cell.indices.length;
    const cellGlobalStart = globalPointOffset;

    const pointData = new ArrayBuffer(maxNumberOfPointsPerBuffer * pointByteSize);
    const pointDataView = new DataView(pointData);

    cell.indices.forEach((j, slot) => {
        const posIndex = j * 3;
        const pointOffset = slot * pointByteSize;
        pointDataView.setFloat32(pointOffset, positions[posIndex], true);
        pointDataView.setFloat32(pointOffset + 4, positions[posIndex + 1], true);
        pointDataView.setFloat32(pointOffset + 8, positions[posIndex + 2], true);
        pointDataView.setUint32( pointOffset + 12, colors[j], true);
        pointDataView.setFloat32(pointOffset + 16, normals[posIndex], true);
        pointDataView.setFloat32(pointOffset + 20, normals[posIndex + 1], true);
        pointDataView.setFloat32(pointOffset + 24, normals[posIndex + 2], true);
        pointDataView.setFloat32(pointOffset + 28, 0.0, true); // depth (sort key)
        pointDataView.setUint32( pointOffset + 32, classColors[j], true); // classColor (overwritten by segmentation)
        pointDataView.setUint32( pointOffset + 36, cellGlobalStart + slot, true); // _pad0 = stable global index
        pointDataView.setUint32( pointOffset + 40, classColors[j], true); // lasClassColor (permanent LAS ASPRS classes)
    });

    const pointBuffer = device.createBuffer({
        size: maxNumberOfPointsPerBuffer * pointByteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(pointBuffer, 0, pointData);

    const renderingBindGroup = device.createBindGroup({
        layout: renderingPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointBuffer } },
            { binding: 1, resource: { buffer: classUniformBuffer } },
        ],
    });

    // Določimo center vsake celice za kasnejše
    // sortiranje celic glede na globino (povprečje pozicij točk v celici)
    let cx = 0, cy = 0, cz = 0;
    for (const pi of cell.indices) {
        cx += positions[pi * 3];
        cy += positions[pi * 3 + 1];
        cz += positions[pi * 3 + 2];
    }
    cx /= count; cy /= count; cz /= count;

    pointclouds.push({
        pointBuffer,
        renderingBindGroup,
        numberOfPoints: count,
        center: [cx, cy, cz],
        globalStart: cellGlobalStart,
    });

    globalPointOffset += count;
}

// Maps original LAS point index j to GPU global index (needed for LAS export)
const jToGlobalIndex = new Uint32Array(numberOfAllPoints);
{
    let off = 0;
    for (const cell of grid.cells) {
        cell.indices.forEach((j, slot) => { jToGlobalIndex[j] = off + slot; });
        off += cell.indices.length;
    }
}

segmentation = new SegmentationPipeline({ device, numberOfAllPoints, pointclouds, classPalette, controls, bbMin, bbMax });

let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    format: "depth32float",
});

let lastSize = { width: null, height: null };

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
prewarmMergeBuffers(numberOfAllPoints);

// Vstvari binding groupe za sortiranje - priprava, lokalno sortiranje in globalno sortiranje
function buildSortBindGroups(pointcloud) {
    const n = pointcloud.numberOfPoints;

    // Reusable view matrix buffer
    pointcloud.prepViewBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const numBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(numBuf, 0, new Uint32Array([n]));

    // main_parallel.js (live pregled) ne podpira sferične projekcije, zato je
    // ta zastavica vedno 0 - obstaja samo zato, ker preparation.wgsl deli isti
    // pipeline layout z imageGenerationWorker.js.
    const isSphericalBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(isSphericalBuf, 0, new Uint32Array([0]));

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
            { binding: 2, resource: { buffer: isSphericalBuf } },
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

// Pripravimo bind groupe za vsak pointcloud batch
for (const pc of pointclouds) {
    buildSortBindGroups(pc);
}

function sortPointCloud(pointcloud, viewMatrix) {
    const n = pointcloud.numberOfPoints;
    const numWorkgroups = Math.ceil(n / 256);

    console.log(`sorting: numberOfPoints=${pointcloud.numberOfPoints}, paddedSize=${n}, ratio=${n/pointcloud.numberOfPoints}`);


    device.queue.writeBuffer(pointcloud.prepViewBuffer, 0, new Float32Array(viewMatrix));

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
    for (let k = 512; k <= n * 2; k *= 2) {
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
function getBatchDepth(pointcloud, projectionViewMatrix) {
    const [px, py, pz] = pointcloud.center;
    const m = projectionViewMatrix;
    const clipZ = m[2]*px + m[6]*py + m[10]*pz + m[14];
    const clipW = m[3]*px + m[7]*py + m[11]*pz + m[15];
    return clipZ / clipW;
}

// Debug: prikaži samo en batch, da lažje vidimo sortiranje in izris
let DEBUG_BATCH_INDEX = -1; // -1 = pokaži vse, 0..n = pokaži samo ta batch
document.addEventListener("keydown", (e) => {
    if (e.key === "n") DEBUG_BATCH_INDEX = (DEBUG_BATCH_INDEX + 1) % pointclouds.length;
    if (e.key === "b") DEBUG_BATCH_INDEX = -1; // back to all
    console.log(`Showing batch ${DEBUG_BATCH_INDEX} of ${pointclouds.length}`);
});

function frame() {
    const size = canvas.getBoundingClientRect();
    if (size.width !== lastSize.width || size.height !== lastSize.height) {
        canvas.width = lastSize.width = size.width;
        canvas.height = lastSize.height = size.height;
        depthTexture.destroy();
        depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            format: "depth32float",
        });
    }

    // Compute matrices with tight near/far from bounding box
    let viewMatrix, projectionViewMatrix;

    if (orthoMode) {
        const result = computeOrthoTopDownMatrix(
            orthoEyeY, orthoPanX, orthoPanZ, orthoZoom, canvas, bbMin, bbMax
        );
        viewMatrix           = result.viewMatrix;
        projectionViewMatrix = result.projectionViewMatrix;
    } else {
        const result = computeCameraMatrix(
            cameraPosition, cameraTarget, canvas, bbMin, bbMax
        );
        viewMatrix           = result.viewMatrix;
        projectionViewMatrix = result.projectionViewMatrix;
    }
    device.queue.writeBuffer(mvpBuffer, 0, new Float32Array(projectionViewMatrix));
    device.queue.writeBuffer(viewMatrixBuffer, 0, new Float32Array(viewMatrix));

    // Depth range: pass huge range so nothing is clipped in preview
    const depthRangeBuffer = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(depthRangeBuffer, 0, new Float32Array([-10, 1e10]));

    // Target position — vec4f padded for alignment
    const targetPosBuf = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const tp = CONFIG.targetPositionsAndRadii[0].pos;
    device.queue.writeBuffer(targetPosBuf, 0, new Float32Array([tp[0], tp[1], tp[2], 0.0, 0, 0, 0, 0]));

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
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

    if (currentMode === "POINTS") {
        // Original shader path — nespremenjeno
        const depthRangeBindGroup = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: depthRangeBuffer } }],
        });
        const targetPosBindGroup = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(3),
            entries: [{ binding: 0, resource: { buffer: targetPosBuf } }],
        });
        renderPass.setPipeline(renderingPipeline);
        renderPass.setBindGroup(1, matricesBindGroup);
        renderPass.setBindGroup(2, depthRangeBindGroup);
        renderPass.setBindGroup(3, targetPosBindGroup);
        for (const pointcloud of pointclouds) {
            renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
            renderPass.draw(pointcloud.numberOfPoints);
        }
    } else {
        // Če gre za GAUSSIANS, sortiramo pointcloude (znotraj vsake celice) glede na globino
        if (currentMode === "GAUSSIANS" && SORT) {
            for (const pc of pointclouds) {
                sortPointCloud(pc, projectionViewMatrix);
            }
            SORT = false;
        }
        writeSceneParams(tp, orthoMode ? [0, 1, 0] : cameraPosition, currentPointSize, orthoMode);
        const pipeline = quadPipelines[currentMode];
        const depthRangeBindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: depthRangeBuffer } }],
        });
        const sceneBindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(3),
            entries: [{ binding: 0, resource: { buffer: sceneParamsBuffer } }],
        });
        const matricesQuadBindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: mvpBuffer } },
                { binding: 1, resource: { buffer: viewMatrixBuffer } },
            ],
        });

        // Sortiramo batch-e glede na globino centra batch-a (povprečje pozicij točk v batch-u), 
        // da zagotovimo pravilen back-to-front izris za GAUSSIANS
        const batchesWithDepth = pointclouds.map((pc) => ({
            pc,
            depth: getBatchDepth(pc, projectionViewMatrix),
        }));
        batchesWithDepth.sort((a, b) => b.depth - a.depth);

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(1, matricesQuadBindGroup);
        renderPass.setBindGroup(2, depthRangeBindGroup);
        renderPass.setBindGroup(3, sceneBindGroup);

      // Debugging: če je DEBUG_BATCH_INDEX nastavljen na 0..n, prikažemo samo ta batch, sicer prikažemo vse
       const batchesToRender = DEBUG_BATCH_INDEX === -1
            ? batchesWithDepth
            : [batchesWithDepth[DEBUG_BATCH_INDEX]];

        for (const batch of batchesToRender) {
            const pc = batch.pc;
            renderPass.setBindGroup(0, device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: pc.pointBuffer } },
                    { binding: 1, resource: { buffer: classUniformBuffer } },
                ],
            }));
            renderPass.draw(pc.numberOfPoints * 6);
        }
    }

    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

console.log(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 READY - Batch Processing Mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Workers: ${CONFIG.workerCount}
Batch size: 20 images (auto-restart)
Press 'T' to start
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);

canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (orthoMode) {
        const factor = e.deltaY > 0 ? 1.1 : 0.9;
        orthoZoom = Math.max(0.5, orthoZoom * factor);
    } else {
        distance = Math.max(0.5, distance + e.deltaY * 0.01);
        updateCameraOrbit();
    }
}, { passive: false });