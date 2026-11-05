@group(0) @binding(0) var reconstructionTextures: texture_2d_array<f32>;
@group(0) @binding(1) var sdfTextures: texture_2d_array<f32>;
@group(0) @binding(2) var densityTextures: texture_2d_array<f32>;
@group(0) @binding(3) var pointsTextures: texture_2d_array<f32>;
@group(0) @binding(4) var outputTexture: texture_storage_2d<rgba32float, write>;

@group(1) @binding(0) var<storage> depths: array<f32>;
@group(1) @binding(1) var<storage> radiusRatio: f32;

fn clampCoord(p: vec2<i32>, size: vec2<u32>) -> vec2<i32> {
    return clamp(p, vec2<i32>(0, 0), vec2<i32>(i32(size.x) - 1, i32(size.y) - 1));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.xy;
    let size = textureDimensions(outputTexture);
    if (index.x >= size.x || index.y >= size.y) {
        return;
    }

    let numberOfLayers = arrayLength(&depths);

    var minDepth = depths[0];
    var maxDepth = depths[0];
    for (var d = 1u; d < numberOfLayers; d++) {
        minDepth = min(minDepth, depths[d]);
        maxDepth = max(maxDepth, depths[d]);
    }
    let depthRange = max(maxDepth - minDepth, 1e-5);
    let densityScale = 1.0 / (radiusRatio * radiusRatio); 

    var result = vec4f(0.0, 0.0, 0.0, 0.0);

    for (var i = 0u; i < numberOfLayers; i++) {
        let col = textureLoad(reconstructionTextures, index, i, 0);
        let sdf = textureLoad(sdfTextures, index, i, 0);
        let pts = textureLoad(pointsTextures, index, i, 0);

        let sdfDist = sdf.a; // / radiusRatio;
        let hasData = select(0.0, 1.0, pts.a > 0.01);

        let normalizedDepth = (depths[i] - minDepth) / depthRange;
        let closeness = 1.0 - normalizedDepth; // 0 = zadaj, 1 = spredaj

        // Upadanje exponenta odvisnega od razdalje (sdf) je spredaj večje kot zadaj
        let k = mix(3.0 , 9.0, closeness) * 1000;
        let sdfAlpha = exp(-(sdfDist * sdfDist * k));
        let maxDist = mix(0.3, 0.3, densityScale);

        var alpha: f32;
        if (sdfDist > maxDist && hasData < 0.5) {
            alpha = 0.0;
        } else if (hasData > 0.5) {
            alpha = 1.0;
        } else {
            // Smerna gostota: glavna zaščita pred bleedom na pravih robovih 
            // (naredi bolj prosojne - TODO: poskusi dobit popolnoma brez bleeda
            // brez da bi imeli luknje v redkih področjih)
            let sampleDist = i32(max(2.0, f32(size.x) * 0.1));
            let dC = min(textureLoad(densityTextures, index, i, 0).a * densityScale, 1.0);
            let dN = min(textureLoad(densityTextures, vec2<u32>(clampCoord(vec2<i32>(index) + vec2<i32>(0, -sampleDist), size)), i, 0).a * densityScale, 1.0);
            let dS = min(textureLoad(densityTextures, vec2<u32>(clampCoord(vec2<i32>(index) + vec2<i32>(0,  sampleDist), size)), i, 0).a * densityScale, 1.0);
            let dE = min(textureLoad(densityTextures, vec2<u32>(clampCoord(vec2<i32>(index) + vec2<i32>( sampleDist, 0), size)), i, 0).a * densityScale, 1.0);
            let dW = min(textureLoad(densityTextures, vec2<u32>(clampCoord(vec2<i32>(index) + vec2<i32>(-sampleDist, 0), size)), i, 0).a * densityScale, 1.0);

            let directionalMin = min(min(dN, dS), min(dE, dW));
            let directionalAvg = (dN + dS + dE + dW) * 0.25;
            let directional = mix(directionalMin * directionalMin, directionalAvg * directionalAvg, 0.3 / mix(0.3, 1.0, radiusRatio));
            let support = max(dC * dC, directional);

            let confidenceModulator = clamp(support * 3.0, 0.0, 1.0);
            let confidence = mix(0.3, 1.0, confidenceModulator);

            alpha = sdfAlpha * confidence;
        }

        if (alpha < 0.005) {
            continue;
        }

        let color = mix(result.rgb, col.rgb, alpha);
        let a = alpha + result.a * (1.0 - alpha);

        result = vec4f(
            color,
            a
        );
    }

    textureStore(outputTexture, index, result);
}