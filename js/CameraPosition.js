import * as mat4 from "./_Mat4.js";


export function generateFibonacciHemisphereAroundCamera(samples, radius, target, fullSphera) {
	const goldenAngle = Math.PI * (3.0 - Math.sqrt(5.0));
	const poses = [];

	for (let i = 0; i < samples; i++) {
		const y = 1.0 - (i / (samples - 1)) * 2.0; // y od 1 do -1
		if (!fullSphera && y < 0) continue; // uporabimo samo zgornjo poloblo
		const r = Math.sqrt(1.0 - y * y);

		const theta = goldenAngle * i;

		const x = r * Math.cos(theta);
		const z = r * Math.sin(theta);

		// Pozicija kamere je na sferi okoli targeta
		const pos = [
			target[0] + radius * x,
			target[1] + radius * y,
			target[2] + radius * z,
		];

		poses.push(pos);
	}
	return poses;
}


// Izračuna tight near/far ravnini iz bounding boxa point clouda in trenutne pozicije kamere.
// To zagotavlja optimalno globinsko natančnost in preprečuje clipping.
function computeTightNearFar(
	eye,
	target,
	bbMin = [-1, -1, -1],
	bbMax = [1, 1, 1],
	padding = 1.1
) {
	// Kamera gleda proti targetu, normalizirano
	const dir = mat4.normalize([
		target[0] - eye[0],
		target[1] - eye[1],
		target[2] - eye[2],
	]);

	// Projecira vseh 8 oglišč bounding boxa na pogled kamere in najde min/max razdaljo
	let minDist = Infinity;
	let maxDist = -Infinity;
	for (let xi = 0; xi <= 1; xi++) {
		for (let yi = 0; yi <= 1; yi++) {
			for (let zi = 0; zi <= 1; zi++) {
				const corner = [
					xi === 0 ? bbMin[0] : bbMax[0],
					yi === 0 ? bbMin[1] : bbMax[1],
					zi === 0 ? bbMin[2] : bbMax[2],
				];
				// Signed distance along the view ray from eye
				const d =
					(corner[0] - eye[0]) * dir[0] +
					(corner[1] - eye[1]) * dir[1] +
					(corner[2] - eye[2]) * dir[2];
				if (d < minDist) minDist = d;
				if (d > maxDist) maxDist = d;
			}
		}
	}

	// Dodamo malo paddinga, da ne bomo tik na robu clippinga 
	let near = minDist / padding;
	let far = maxDist * padding;

	near = Math.max(near, 0.001); 
	far = Math.max(far, near + 0.01); // far more biti vsaj malo večji od near

	return { near, far };
}

// Izračuna view in projection matriko.
// Vrne tudi izračunane near/far ravnini, ki jih lahko uporabi depth binning za optimalno deljenje globine.
export function computeCameraMatrix(eye, target, canvas, bbMin, bbMax, fovDeg = 45) {
	const dir = [
		target[0] - eye[0],
		target[1] - eye[1],
		target[2] - eye[2],
	];
	const dirNorm = Math.hypot(...dir);
	const dirUnit = dir.map((v) => v / dirNorm);

	let up = [0, 1, 0];
	if (Math.abs(dirUnit[1]) > 0.99) {
		up = [0, 0, 1];
	}

	// Tight near/far
	const { near, far } = computeTightNearFar(
		eye,
		target,
		bbMin,
		bbMax
	);

	const viewMatrix = mat4.lookAt(eye, target, up);
	const projectionMatrix = mat4.perspective(
		fovDeg * (Math.PI / 180),
		canvas.width / canvas.height,
		near,
		far
	);
	const projectionViewMatrix = mat4.multiply(
		projectionMatrix,
		viewMatrix
	);

	return { viewMatrix, projectionMatrix, projectionViewMatrix, near, far };
}

export function writeCamerasTxt(captured) {
	const cam = captured[0];
	console.log("Writing cameras.txt with camera:", cam);
	return `# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE ${cam.width} ${cam.height} ${cam.fx} ${cam.fx} ${cam.cx} ${cam.cy}
`;
}

export function imagesTxtEntry(c) {
	return `${c.imageId} ${c.quat[0]} ${c.quat[1]} ${c.quat[2]} ${c.quat[3]} ${c.t[0]} ${c.t[1]} ${c.t[2]} 1 ${c.filename}\n\n`
}

export function writeImagesTxt(captured, scaleFactor = 1.0) {
	return "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" +
		captured.map(imagesTxtEntry).join("");
}

// Vrne Blob namesto enega dolgega niza, ker pri milijonih
// točk .join() na enem samem nizu porabi ogromno pomnilnika.
export function writePoints3DTxt(positions, colors, samplingRate = 100, scaleFactor = 1.0) {
	const totalPoints = positions.length / 3;
	
	console.log(`Writing points3D.txt with scale factor: ${scaleFactor}, sampling rate: ${samplingRate} (from ${totalPoints} points)`);

	const lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"];
	let id = 0;
	for (let pointIndex = 0; pointIndex < totalPoints; pointIndex += samplingRate) {
		const i = pointIndex * 3;
		const r = Math.round(colors[i] * 255);
		const g = Math.round(colors[i + 1] * 255);
		const b = Math.round(colors[i + 2] * 255);
		lines.push(
			`${id++} ${positions[i]} ${positions[i + 1]} ${positions[i + 2]} ${r} ${g} ${b} 0\n`
		);
	}

	console.log(`Wrote ${id} sampled points to points3D.txt`);

	return new Blob(lines, { type: "text/plain" });
}

export function computeOrthoTopDownMatrix(eyeY, panX, panZ, zoom, canvas, bbMin, bbMax) {
	const eye    = [panX, eyeY, panZ];
	const target = [panX, eyeY - 1, panZ]; // straight down
	const up     = [0, 0, 1];              // +Z at screen top

	const viewMatrix = mat4.lookAt(eye, target, up);
	
	const near = 0.1;
	const far  = Math.max(1.0, eyeY - bbMin[1] + 1.0);

	const aspect = canvas.width / canvas.height;
	let halfH, halfW;
	if (canvas.width <= canvas.height) {
		halfW = zoom;
		halfH = zoom / aspect;
	} else {
		halfH = zoom;
		halfW = zoom * aspect;
	}

	const projectionMatrix = mat4.orthographic(
		-halfW,  halfW,
		-halfH,  halfH,
		near,    far
	);

	const projectionViewMatrix = mat4.multiply(projectionMatrix, viewMatrix);

	return { viewMatrix, projectionMatrix, projectionViewMatrix, near, far };
}

export function computeSphericalCameraMatrix(eye, bbMin, bbMax, yawDeg) {
	let viewMatrix = mat4.lookAt(eye, [eye[0], eye[1], eye[2] - 1], [0, 1, 0]);
	if (yawDeg && yawDeg !== 0) {
		viewMatrix = mat4.multiply(mat4.rotateY(yawDeg * Math.PI/180), viewMatrix);
	}

	let far = 0;
	for (let xi = 0; xi <= 1; xi++) {
		for (let yi = 0; yi <= 1; yi++) {
			for (let zi = 0; zi <= 1; zi++) {
				const corner = [
					xi ? bbMax[0] : bbMin[0],
					yi ? bbMax[1] : bbMin[1],
					zi ? bbMax[2] : bbMin[2],
				]
				far = Math.max(far, Math.hypot(corner[0] - eye[0], corner[1] - eye[1], corner[2] - eye[2]));
			}	
		}
	}
	far = Math.max(far * 1.05, 0.01);
	return { viewMatrix, far };
}
