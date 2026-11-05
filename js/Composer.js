import { loadShader } from "./LoadShader.js";

export class Composer {
	constructor(device, width, height, radiusRatio) {
		this.device = device;
		this.width = width;
		this.height = height;

		this.radiusRatio = radiusRatio;
		
        this.depthPoints = [];
        this.reconstructions = [];
        this.sdfs = [];
        this.densities = [];
		this.depths = [];

        this.backgroundTexture = null;
	}

    async createCompositePipeline() {
		const compositeCode = await loadShader("composite.wgsl");

		const compositeModule = this.device.createShaderModule({
			code: compositeCode,
		});

		const compositePipeline = this.device.createComputePipeline({
			compute: {
				module: compositeModule,
				entryPoint: "main",
			},
			layout: "auto",
		});

		return compositePipeline;
	}

    // Function to load a background image
    loadBackgroundImage(imageUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                // Create a canvas to draw the image
                const tempCanvas = this.document.createElement('canvas');
                tempCanvas.width = this.width;
                tempCanvas.height = this.height;
                const ctx = tempCanvas.getContext('2d');
                
                // Draw the image, stretching it to fit the canvas dimensions
                ctx.drawImage(img, 0, 0, this.width, this.height);
                
                // Get image data
                const imageData = ctx.getImageData(0, 0, this.width, this.height);
                const rgbaData = imageData.data;
    
                // Convert RGBA to BGRA
                const bgraData = new Uint8Array(rgbaData.length);
                for (let i = 0; i < rgbaData.length; i += 4) {
                    bgraData[i] = rgbaData[i + 2];     // B <- R
                    bgraData[i + 1] = rgbaData[i + 1]; // G <- G
                    bgraData[i + 2] = rgbaData[i];     // R <- B
                    bgraData[i + 3] = rgbaData[i + 3]; // A <- A
                }
                
                // Create a WebGPU texture from the swizzled data
                this.backgroundTexture = this.device.createTexture({
                    size: [this.width, this.height],
                    format: this.format,
                    usage: 
                        GPUTextureUsage.TEXTURE_BINDING |
                        GPUTextureUsage.COPY_DST |
                        GPUTextureUsage.COPY_SRC |
                        GPUTextureUsage.STORAGE_BINDING
                });
    
                // Write the swizzled data to the texture
                this.device.queue.writeTexture(
                    { texture: this.backgroundTexture },
                    bgraData,
                    { bytesPerRow: this.width * 4 },
                    [this.width, this.height]
                );
    
                console.log("Background image loaded and swizzled to BGRA");
                resolve();
            };
            
            img.onerror = (event) => {
                console.error(`Error loading image from URL: ${imageUrl}`);
                console.error("Image Error Event:", event);
                console.error("Browser User Agent:", navigator.userAgent);
                reject(new Error(`Failed to load background image from URL: ${imageUrl}`));
            };
            
            img.src = imageUrl;
        });
    }
        
    // Add a function to initialize the background - call this during your app startup
    async initializeBackground(imageUrl) {
        try {
            await this.loadBackgroundImage(imageUrl);
            console.log("Background loaded successfully");
        } catch (error) {
            console.error("Failed to load background:", error);
            // Create a solid color background instead
            this.backgroundTexture = this.device.createTexture({
                size: [this.width, this.height],
                format: this.format,
                usage: 
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.STORAGE_BINDING
            });
            
            // Fill with a solid color (gray)
            const data = new Uint8Array(this.width * this.height * 4);
            for (let i = 0; i < data.length; i += 4) {
                data[i] = 128;     // R
                data[i + 1] = 128; // G
                data[i + 2] = 128; // B
                data[i + 3] = 255; // A
            }
            
            this.device.queue.writeTexture(
                { texture: this.backgroundTexture },
                data,
                { bytesPerRow: this.width * 4 },
                [this.width, this.height]
            );
        }
    }

    async addLayers(sdf, density, reconstruction, points, depth) {
		const reconstructionTexture = this.device.createTexture({
			size: [this.width, this.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.STORAGE_BINDING,
		});
		const sdfTexture = this.device.createTexture({
			size: [this.width, this.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.STORAGE_BINDING,
		});
        const densityTexture = this.device.createTexture({
			size: [this.width, this.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.STORAGE_BINDING,
		});
		const pointsTexture = this.device.createTexture({
			size: [this.width, this.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.STORAGE_BINDING,
		});

		const commandEncoder = this.device.createCommandEncoder();
		commandEncoder.copyTextureToTexture(
			{ texture: reconstruction },
			{ texture: reconstructionTexture },
			[this.width, this.height]
		);

		commandEncoder.copyTextureToTexture(
			{ texture: sdf },
			{ texture: sdfTexture },
			[this.width, this.height]
		);
        commandEncoder.copyTextureToTexture(
			{ texture: density },
			{ texture: densityTexture },
			[this.width, this.height]
		);
		commandEncoder.copyTextureToTexture(
			{ texture: points },
			{ texture: pointsTexture },
			[this.width, this.height]
		);
		this.device.queue.submit([commandEncoder.finish()]);

		this.depthPoints.push(pointsTexture);
		this.reconstructions.push(reconstructionTexture);
		this.sdfs.push(sdfTexture);
		this.densities.push(densityTexture);
		this.depths.push(depth);
	}

	async compositeDepths() {
		const reconstructionTextures = this.device.createTexture({
			size: {
				width: this.width,
				height: this.height,
				depthOrArrayLayers: this.reconstructions.length,
			},
			format: "rgba32float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
			dimension: "2d",
		});

		const sdfTextures = this.device.createTexture({
			size: {
				width: this.width,
				height: this.height,
				depthOrArrayLayers: this.sdfs.length,
			},
			format: "rgba32float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
			dimension: "2d",
		});

        const densityTextures = this.device.createTexture({
			size: {
				width: this.width,
				height: this.height,
				depthOrArrayLayers: this.densities.length,
			},
			format: "rgba32float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
			dimension: "2d",
		});

		const pointsTexture = this.device.createTexture({
			size: {
				width: this.width,
				height: this.height,
				depthOrArrayLayers: this.depthPoints.length,
			},
			format: "rgba32float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
			dimension: "2d",
		});

		const outputTexture = this.device.createTexture({
			size: [this.width, this.height],
			format: "rgba32float",
			usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
		});

		const commandEncoder = this.device.createCommandEncoder();

		for (let i = 0; i < this.reconstructions.length; i++) {
			commandEncoder.copyTextureToTexture(
				{ texture: this.reconstructions[i] },
				{
					texture: reconstructionTextures,
					origin: { x: 0, y: 0, z: i },
				},
				[this.width, this.height, 1]
			);
			commandEncoder.copyTextureToTexture(
				{ texture: this.sdfs[i] },
				{ texture: sdfTextures, origin: { x: 0, y: 0, z: i } },
				[this.width, this.height, 1]
			);
			commandEncoder.copyTextureToTexture(
				{ texture: this.densities[i] },
				{ texture: densityTextures, origin: { x: 0, y: 0, z: i } },
				[this.width, this.height, 1]
			);
			commandEncoder.copyTextureToTexture(
				{ texture: this.depthPoints[i] },
				{ texture: pointsTexture, origin: { x: 0, y: 0, z: i } },
				[this.width, this.height, 1]
			);
		}

		const depthsBuffer = this.device.createBuffer({
			size: this.depthPoints.length  * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
        this.device.queue.writeBuffer(depthsBuffer, 0, new Float32Array(this.depths))

		const radiusRatioBuffer = this.device.createBuffer({
			size: 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
        this.device.queue.writeBuffer(radiusRatioBuffer, 0, new Float32Array([this.radiusRatio]))

		const compositePipeline = await this.createCompositePipeline();

		const textureBindGroup = this.device.createBindGroup({
			layout: compositePipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: reconstructionTextures.createView({
						dimension: "2d-array",
					}),
				},
				{
					binding: 1,
					resource: sdfTextures.createView({
						dimension: "2d-array",
					}),
				},
				{
					binding: 2,
					resource: densityTextures.createView({
						dimension: "2d-array",
					}),
				},
				{
					binding: 3,
					resource: pointsTexture.createView({
						dimension: "2d-array",
					}),
				},
				{
					binding: 4,
					resource: outputTexture.createView(),
				},
			],
		});

		const uniformBindGroup = this.device.createBindGroup({
			layout: compositePipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: { buffer: depthsBuffer },
				},
				{
					binding: 1,
					resource: { buffer: radiusRatioBuffer },
				},
			],
		});

		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(compositePipeline);
		computePass.setBindGroup(0, textureBindGroup);
		computePass.setBindGroup(1, uniformBindGroup);
		computePass.dispatchWorkgroups(
			Math.ceil(this.width / 8),
			Math.ceil(this.height / 8)
		);
		computePass.end();

		const outputBuffer = this.device.createBuffer({
			size: this.width * this.height * 4 * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		commandEncoder.copyTextureToBuffer(
			{
				texture: outputTexture,
				mipLevel: 0,
				origin: { x: 0, y: 0, z: 0 },
			},
			{
				buffer: outputBuffer,
				bytesPerRow: this.width * 16,
				rowsPerImage: this.height,
			},
			[this.width, this.height, 1]
		);

		this.device.queue.submit([commandEncoder.finish()]);

		// Map the output buffer to read the data
		await outputBuffer.mapAsync(GPUMapMode.READ);
		const outputData = new Float32Array(outputBuffer.getMappedRange());

        const copiedData = new Float32Array(outputData);

		// saveTextureToPNG(copiedData, this.width, this.height, "output.png");

		// Clean up temporary resources
		outputBuffer.unmap();

		reconstructionTextures.destroy();
		sdfTextures.destroy();
		densityTextures.destroy();
		pointsTexture.destroy();
		outputTexture.destroy();
        depthsBuffer.destroy();
        outputBuffer.destroy()

        for (let i = 0; i < this.reconstructions.length; i++) {
            this.reconstructions[i].destroy();
            this.sdfs[i].destroy();
            this.densities[i].destroy();
            this.depthPoints[i].destroy();
        }

		this.reconstructions = [];
		this.sdfs = [];
        this.densities = [];
		this.depthPoints = [];
		this.depths = [];

        return copiedData;
	}
}