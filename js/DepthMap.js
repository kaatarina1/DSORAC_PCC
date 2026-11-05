import { loadShader } from "./LoadShader.js";

export class DepthMap {
	constructor(canvas, device, viewMatrix, projectionViewMatrix, pointclouds, isSpherical = false) {
		this.canvas = canvas;
		this.device = device;
		this.viewMatrix = viewMatrix;
		this.projectionViewMatrix = projectionViewMatrix;
		this.pointclouds = pointclouds;
		// Skupno število točk čez vse chunk-e (vsak chunk ima svoj GPU buffer)
		this.nPoints = pointclouds.reduce((sum, pc) => sum + pc.numberOfPoints, 0);

		this.depthStorageBuffer = this.device.createBuffer({
			size: this.nPoints * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.readbackBuffer = this.device.createBuffer({
			size: this.nPoints * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		// Matriki prideta kot navadni tabeli, zato ju zapišemo v uniform buffer-ja
		this.projMatrixBuffer = this.device.createBuffer({
			size: 64,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		this.device.queue.writeBuffer(this.projMatrixBuffer, 0, new Float32Array(projectionViewMatrix));

		this.viewMatrixBuffer = this.device.createBuffer({
			size: 64,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		this.device.queue.writeBuffer(this.viewMatrixBuffer, 0, new Float32Array(viewMatrix));

		this.isSphericalBuffer = this.device.createBuffer({
			size: 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		this.device.queue.writeBuffer(this.isSphericalBuffer, 0, new Float32Array([isSpherical ? 1 : 0]));
	}

	// Shader naložimo asinhrono ob prvi uporabi (v konstruktorju await ni dovoljen)
	async initPipeline() {
		if (this.computePipeline) return;

		const computeCode = await loadShader("visiblePointDepth.wgsl");
		this.computeShader = this.device.createShaderModule({ code: computeCode });

		this.computePipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.computeShader,
				entryPoint: "main",
			},
		});

		this.matrixBindGroup = this.device.createBindGroup({
			layout: this.computePipeline.getBindGroupLayout(1),
			entries: [
				{ binding: 0, resource: { buffer: this.projMatrixBuffer } },
				{ binding: 1, resource: { buffer: this.viewMatrixBuffer } },
				{ binding: 2, resource: { buffer: this.isSphericalBuffer } },
			],
		});

		// Za vsak chunk pripravimo bind group s [count, offset] uniform-om,
		// da vsi chunk-i pišejo v skupni depthStorageBuffer brez prekrivanja.
		this.chunkBindGroups = [];
		this.paramsBuffers = [];
		let offset = 0;
		for (const pc of this.pointclouds) {
			const paramsBuf = this.device.createBuffer({
				size: 8,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			});
			this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([pc.numberOfPoints, offset]));
			this.paramsBuffers.push(paramsBuf);

			this.chunkBindGroups.push({
				count: pc.numberOfPoints,
				bindGroup: this.device.createBindGroup({
					layout: this.computePipeline.getBindGroupLayout(0),
					entries: [
						{ binding: 0, resource: { buffer: pc.pointBuffer } },
						{ binding: 1, resource: { buffer: paramsBuf } },
						{ binding: 2, resource: { buffer: this.depthStorageBuffer } },
					],
				}),
			});
			offset += pc.numberOfPoints;
		}
	}

	// Izračuna linearno view-space globino za vsako točko; točke izven pogleda dobijo -1.
	async extractDepthValues() {
		await this.initPipeline();

		const commandEncoder = this.device.createCommandEncoder();

		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(this.computePipeline);
		computePass.setBindGroup(1, this.matrixBindGroup);

		const workgroupSize = 256;
		for (const chunk of this.chunkBindGroups) {
			computePass.setBindGroup(0, chunk.bindGroup);
			computePass.dispatchWorkgroups(Math.ceil(chunk.count / workgroupSize));
		}
		computePass.end();

		commandEncoder.copyBufferToBuffer(
			this.depthStorageBuffer,
			0,
			this.readbackBuffer,
			0,
			this.nPoints * 4
		);

		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		await this.readbackBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = this.readbackBuffer.getMappedRange();
		const depthValues = new Float32Array(arrayBuffer.slice(0));

		this.readbackBuffer.unmap();

		return depthValues;
	}

	async computeViewpointDensity() {
		if (this.pointsInView === undefined) {
			const depthValues = await this.extractDepthValues();
			let inView = 0;
			for (let i = 0; i < depthValues.length; i++) {
				if (depthValues[i] > 0) inView++;
			}
			this.pointsInView = inView;
		}

		const pixels = this.canvas.width * this.canvas.height;
		return this.pointsInView / pixels;
	}

	
	// Deljenje globine v bin-e, ki vsebujejo približno enako število točk, ne glede na razdaljo kamere. 
	// Bin-i so v linearnih enotah pogleda, da jih shader lahko neposredno primerja.
	async groupDepthIntoBins({
		numBins = 12, // Število ciljanih binov (globinskih rezin)
		near = 0.05, // Near ravnina (mora biti usklajena s tisto, ki jo shader uporablja za renderiranje)
		far = 5.0, // Far ravnina (mora biti usklajena s tisto, ki jo shader uporablja za renderiranje)
		minPointsFraction = 0.1, // Minimalni delež točk v binu, preden se združi
		minBinWidth = 0.002, // Minimalna širina bin-a v linearni razdalji
	} = {}) {
		const depthValues = await this.extractDepthValues();

		// Globine so že linearne (view-space), točke izven pogleda imajo -1.
		const linear = [];
		let inView = 0;
		let minDepth = Infinity;
		let maxDepth = -Infinity;
		for (let i = 0; i < depthValues.length; i++) {
			const d = depthValues[i];
			if (d > 0 && Number.isFinite(d) && d >= near && d <= far) {
				inView++;
				linear.push(d);
				if (d < minDepth) minDepth = d;
				if (d > maxDepth) maxDepth = d;
			}
		}

		// Shranimo, da je computeViewpointDensity() po binningu zastonj.
		this.pointsInView = inView;

		if (linear.length === 0) {
			console.warn("No valid depth samples — returning single fallback bin");
			return [[near, far]];
		}

		const actualNear = Math.max(near, minDepth * 0.95);
		const actualFar  = Math.min(far,  maxDepth * 1.05);
		const range = actualFar - actualNear;

		const totalPoints = linear.length;
		const minPointsPerBin = Math.max(1, Math.floor(totalPoints * minPointsFraction));

		//const range = far - near;
		const resolution = 100;
		const histogram = new Float32Array(resolution);

		for (const d of linear) {
			const idx = Math.floor(((d - actualNear) / range) * (resolution - 1));
			if (idx >= 0 && idx < resolution) histogram[idx]++;
		}

		// Smooth
		const smoothed = new Float32Array(resolution);
		const smoothRadius = 6;
		for (let i = 0; i < resolution; i++) {
			let sum = 0, count = 0;
			for (let j = Math.max(0, i - smoothRadius); j < Math.min(resolution, i + smoothRadius); j++) {
				sum += histogram[j]; count++;
			}
			smoothed[i] = sum / count;
		}

		// Find peaks
		const peaks = [];
		for (let i = 1; i < resolution - 1; i++) {
			if (smoothed[i] > smoothed[i-1] && smoothed[i] >= smoothed[i+1]) {
				peaks.push({ depth: actualNear + (i / resolution) * range, density: smoothed[i] });
			}
		}
		peaks.sort((a, b) => b.density - a.density);
		const topPeaks = peaks.slice(0, numBins).sort((a, b) => a.depth - b.depth);

		const MIN_BIN = 2;
		let peakDepths = topPeaks.map(p => p.depth);
		if (peakDepths.length < MIN_BIN) {
			const sorted = Float32Array.from(linear).sort();
			peakDepths = [];
			for (let k = 0; k < MIN_BIN; k++) {
				peakDepths.push(sorted[Math.floor(((k + 0.5) / MIN_BIN) * (sorted.length - 1))]);
			}
		}

		// Voronoi boundaries
		const merged = peakDepths.map((d, i) => ({
			start: i === 0 ? near : (peakDepths[i - 1] + d) / 2,
			end:   i === peakDepths.length - 1 ? far : (d + peakDepths[i + 1]) / 2,
		}));


		// Razširimo prvega/ zadnjega bina na celoten near/far razpon, da nobena točka ne pade izven pokritosti.
		const bins = merged.map((b, i) => {
			let s = b.start;
			let e = b.end;
			// Prvi bin se razteza vse do near, da zajame vse točke blizu kamere
			if (i === 0) s = Math.max(0, near * 0.5);
			// Zadnji bin se razteza vse do far, da zajame vse točke daleč od kamere
			if (i === merged.length - 1) e = far * 1.5;
			return [s, e];
		});

		// Naredimo bin-e povezane (continous) in dodamo prekrivanje.
		// Prekrivanje zagotavlja, da točke blizu meje pojavijo v obeh
		// sosednjih rezinah, kar daje rekonstrukcijskemu solverju dovolj podatkov.
		for (let i = 1; i < bins.length; i++) {
			const mid = (bins[i - 1][1] + bins[i][0]) / 2;
			const binWidth = bins[i][1] - bins[i][0];
			const prevBinWidth = bins[i - 1][1] - bins[i - 1][0];
			const overlap = Math.min(binWidth, prevBinWidth) * 0.10;
			bins[i - 1][1] = mid + overlap;
			bins[i][0] = mid - overlap;
		}

		console.log(
			`Depth binning: ${totalPoints} valid points → ${bins.length} bins (linear view-space)`,
			bins
		);

		return bins;
	}

	// Sprosti vse GPU buffer-je te instance (klic po getDepthBins/groupDepthIntoBins,
	// sicer vsaka slika pusti ~2 × nPoints × 4 bajtov GPU pomnilnika).
	destroy() {
		this.depthStorageBuffer.destroy();
		this.readbackBuffer.destroy();
		this.projMatrixBuffer.destroy();
		this.viewMatrixBuffer.destroy();
		if (this.paramsBuffers) {
			for (const buf of this.paramsBuffers) {
				buf.destroy();
			}
			this.paramsBuffers = [];
		}
		this.chunkBindGroups = [];
		this.matrixBindGroup = null;
	}
}