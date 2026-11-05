import { createLazPerf } from "laz-perf";
import { parseHeader } from "./Utils.js";

// LAS Point Data Record Format -> byte offset za RGB (null če ni barve)
// https://laspy.readthedocs.io/en/latest/intro.html
// X, Y in Z so vedno prva tri polja (12 bajtov), nato pa se razlikujejo glede na format.
const COLOR_OFFSETS = {
    0: null,  // Brez barve
    1: null,  // Brez barve
    2: { r: 20, g: 22, b: 24 },  // Barva se nahaja na offsetih 20, 22, 24 (16 bajtov do RGB)
    3: { r: 28, g: 30, b: 32 },  // Format 2 + GPS Time(8) = 28 (najprej ima GPS Time, potem RGB)
    4: { r: 20, g: 22, b: 24},  // Brez barve
    5: { r: 28, g: 30, b: 32 },  // Enako kot format 3, vendar za barvo ima še dodatna polja
    6: null,  // Brez barve
    7: { r: 30, g: 32, b: 34 },  // Format 6 + RGB
    8: { r: 30, g: 32, b: 34 },  // Format 7 + NIR (neka dodatna polja za barvo)
    9: null,  // Brez barve
    10: { r: 30, g: 32, b: 34 }, // Format 9 + RGB (RGB vrednosti pred dodatnimi polji, ki jih ima format 9)
};

// Class IDs are stored in user_data (byte 17) for LAS 1.2 formats (0-5),
// because the classification field there is only 5 bits (max 31).
// For LAS 1.4 formats (6-10), the classification field is a full uint8 at byte 16.
const CLASSIFICATION_OFFSETS = {
    0: 17, 1: 17, 2: 17, 3: 17, 4: 17, 5: 17,
    6: 16, 7: 16, 8: 16, 9: 16, 10: 16,
};

const NORMAL_OFFSETS = {
    0: null,  // Brez normal
    1: null,  // Brez normal
    2: { x: 26, y: 30, z: 34 },  // Format 2: RGB at 20-24, normals would be after
    3: { x: 34, y: 38, z: 42 },  // Format 3: After GPS time + RGB
    4: { x: 26, y: 30, z: 34 },  // Brez normal
    5: { x: 34, y: 38, z: 42 },  // Format 5: After GPS time + RGB
    6: null,  // Brez normal
    7: { x: 36, y: 40, z: 44 },  // Format 7: Standard 48 bytes + NIR, normals after
    8: { x: 38, y: 42, z: 46 },  // Format 8: Standard 50 bytes + 3 float32 normals (4+4+4=12 bytes)
    9: null,  // Brez normal
    10: { x: 36, y: 40, z: 44 }, // Format 10: Similar to 8 but different layout
};

export class LasLoader {
	constructor(filename) {
        this.lasName = filename;
	}

    async loadLasData() {
        const LazPerf = await createLazPerf({
            locateFile: (file) => `./node_modules/laz-perf/lib/laz-perf.wasm`,
        });

        let file;
        if (typeof Deno !== "undefined") {
            file = await Deno.readFile(this.lasName);
        } else {
            const response = await fetch(this.lasName);
            const arrayBuffer = await response.arrayBuffer();
            file = new Uint8Array(arrayBuffer);
        }
        
        const header = parseHeader(file);
        const {
            pointDataRecordFormat,
            pointDataRecordLength,
            pointCount,
            scale,
            offset,
        } = header;

        console.log("Header Info:", header);
        console.log(`Point format: ${pointDataRecordFormat}, record length: ${pointDataRecordLength} bytes`);

        const colorOffset = COLOR_OFFSETS[pointDataRecordFormat] ?? null;
        const normalOffset = NORMAL_OFFSETS[pointDataRecordFormat] ?? null;
        const hasColor = colorOffset !== null && 
                         pointDataRecordLength >= (colorOffset.b + 2);
        const hasNormals = normalOffset !== null && 
                           pointDataRecordLength >= (normalOffset.z + 4);

        console.log(`Point record length: ${pointDataRecordLength} bytes, has normals: ${hasNormals}`);

        if (!hasColor) {
            console.warn(
                `Point format ${pointDataRecordFormat} has no color data` +
                (colorOffset && pointDataRecordLength < colorOffset.b + 2
                    ? ` (record too short: ${pointDataRecordLength} bytes, need ${colorOffset.b + 3})`
                    : '') +
                `. Points will render white.`
            );
        }

        if (!hasNormals) {
            console.warn("Loaded las file does not have normals! Rendering using 2D isotropic gaussians or disks won't work.");
        }

        const laszip = new LazPerf.LASZip();
        const dataPtr = LazPerf._malloc(pointDataRecordLength);
        const filePtr = LazPerf._malloc(file.byteLength);

        LazPerf.HEAPU8.set(
            new Uint8Array(file.buffer, file.byteOffset, file.byteLength),
            filePtr
        );

        laszip.open(filePtr, file.byteLength);

        const classOffset = CLASSIFICATION_OFFSETS[pointDataRecordFormat] ?? 15;
        const positions = new Float32Array(laszip.getCount() * 3);
        const colorsRGB = new Float32Array(laszip.getCount() * 3);
        const colors = new Uint32Array(laszip.getCount());
        const normals = new Float32Array(laszip.getCount() * 3);
        const classifications = new Uint8Array(laszip.getCount());

        let minX = Infinity,
            minY = Infinity,
            minZ = Infinity;
        let maxX = -Infinity,
            maxY = -Infinity,
            maxZ = -Infinity;

        for (let i = 0; i < laszip.getCount(); ++i) {
            laszip.getPoint(dataPtr);

            const pointBuffer = new DataView(
                LazPerf.HEAPU8.buffer,
                dataPtr,
                pointDataRecordLength
            );

            const x = pointBuffer.getInt32(0, true) * scale[0] + offset[0];
            const z = -pointBuffer.getInt32(4, true) * scale[1] + offset[1];
            const y = pointBuffer.getInt32(8, true) * scale[2] + offset[2];

            let r = 1.0, g = 1.0, b = 1.0; // default white
            if (hasColor) {
                r = pointBuffer.getUint16(colorOffset.r, true) / 65535;
                g = pointBuffer.getUint16(colorOffset.g, true) / 65535;
                b = pointBuffer.getUint16(colorOffset.b, true) / 65535;
            }

            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z);
            maxZ = Math.max(maxZ, z);

            positions.set([x, y, z], i * 3);
            colorsRGB.set([r, g, b], i * 3);
            const packedColor =
                ((Math.round(r * 255) & 0xff) << 0) |
                ((Math.round(g * 255) & 0xff) << 8) |
                ((Math.round(b * 255) & 0xff) << 16) |
                (0xff << 24);
            colors.set([packedColor >>> 0], i);

            if (hasNormals) {
                const nx = pointBuffer.getFloat32(normalOffset.x, true);
                const nz = -pointBuffer.getFloat32(normalOffset.y, true);
                const ny = pointBuffer.getFloat32(normalOffset.z, true);
                normals.set([nx, ny, nz], i * 3);
            }
            
            classifications[i] = pointBuffer.getUint8(classOffset);
        }

        const centerX = (minX + maxX) / 2;
        const centerZ = (minZ + maxZ) / 2;
        const scaleFactor = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

        for (let i = 0; i < positions.length; i += 3) {
            positions[i] = (positions[i] - centerX);
            positions[i + 1] = (positions[i + 1] - minY);      // Use minY instead of centerY
            positions[i + 2] = (positions[i + 2] - centerZ);
        }
        LazPerf._free(filePtr);
        LazPerf._free(dataPtr);
        laszip.delete();

        return { positions, colors, colorsRGB, normals, classifications, scaleFactor, centerX, centerZ, minY };
    }
}