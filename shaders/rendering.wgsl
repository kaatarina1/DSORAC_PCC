struct Point {
    position:      vec3f,
    color:         u32,
    normal:        vec3f,
    depth:         f32,
    classColor:    u32,
    _pad0:         u32,
    lasClassColor: u32,
    _pad2:         u32,
}

struct RenderParams {
    targetPosition: vec4f,
    isSpherical: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<uniform> useClassColors: u32;
@group(1) @binding(0) var<uniform> matrix: mat4x4f;
@group(1) @binding(1) var<uniform> viewMatrix: mat4x4f;
@group(2) @binding(0) var<uniform> depthRange: vec2f;
@group(3) @binding(0) var<uniform> params: RenderParams;

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) @interpolate(flat) color: vec4f,
    @location(1) @interpolate(flat) isTarget: f32,
    @location(2) @interpolate(flat) depth: vec4f,
}

const PI: f32 = 3.14159265359;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
    let point = &points[index];

    var output: VertexOutput;

    // Compute linear view-space depth for filtering
    let viewPos = viewMatrix * vec4f((*point).position, 1);
    let linearDepth = -viewPos.z; // negate: camera looks along -Z

    if (params.isSpherical > 0.5) {
        let dist = length(viewPos.xyz);
        if (dist < depthRange.x || dist > depthRange.y) {
            output.clipPosition = vec4f(0.0, 0.0, 2.0, 0.0);
        } else {
            let dir = viewPos.xyz / max(dist, 1e-6); // Normalize 
            let lon = atan2(dir.x, -dir.z);
            let lat = asin(clamp(dir.y, -1.0, 1.0));
            output.clipPosition = vec4f(lon / PI, lat / (PI * 0.5), clamp(dist / max(params.far, 1e-6), 0.0, 1.0), 1.0);
        }
    } else {
        // Discard points outside the selected linear depth range
        if (linearDepth < depthRange.x || linearDepth > depthRange.y) {
            output.clipPosition = vec4f(0.0, 0.0, 2.0, 0.0); // Move outside clip space
        } else {
            output.clipPosition = matrix * vec4f((*point).position, 1);
        } 
    }

    let isTarget = distance((*point).position, params.targetPosition.xyz) < 0.02;
    output.isTarget = select(0.0, 1.0, isTarget);
    var pickedColor: u32;
    if (useClassColors == 1u) {
        pickedColor = (*point).classColor;
    } else if (useClassColors == 2u) {
        pickedColor = (*point).lasClassColor;
    } else {
        pickedColor = (*point).color;
    }
    output.color = unpack4x8unorm(pickedColor);
    output.depth = vec4f(1 / linearDepth, 0, 0, 1);
    return output;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
    return input.color.rgba;
}

@fragment
fn fragmentWithDepth(input: VertexOutput) -> @location(0) vec4f {
    return input.depth.rgba;
}
