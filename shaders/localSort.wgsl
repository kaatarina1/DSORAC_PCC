struct Point {
    position:   vec3f,
    color:      u32,
    normal:     vec3f,
    depth:      f32,
    classColor: u32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
}

@group(0) @binding(0) var<storage, read_write> points : array<Point>;
@group(0) @binding(1) var<uniform> numberOfPoints : u32;


var<workgroup> localPoints : array<Point, 256>;

@compute
@workgroup_size(256)
fn compute(@builtin(global_invocation_id) globalId : vec3u,
@builtin(local_invocation_id) localId : vec3u)
{
    let globalIndex = globalId.x;
    let localIndex = localId.x;

    // Naložimo podatke vshared memory
    localPoints[localIndex] = points[globalIndex];
    workgroupBarrier();

    // Izvedemo bitonic sort znotraj workgroup-e
    for (var k = 2u; k <= 256u; k *= 2u)
    {
        for (var j = k / 2u; j > 0u; j /= 2u)
        {
            let ixj = localIndex ^ j;

            if (ixj > localIndex && ixj < 256u)
            {
                let ascending = (globalIndex & k) == 0;
                let currDepth = localPoints[localIndex].depth;
                let otherDepth = localPoints[ixj].depth;

                let swap = (ascending && currDepth < otherDepth) || (!ascending && currDepth > otherDepth);

                if (swap)
                {
                    let temp = localPoints[localIndex];
                    localPoints[localIndex] = localPoints[ixj];
                    localPoints[ixj] = temp;
                }
            }
            workgroupBarrier();
        }
    }

    // Zapišemo sortirane rezultate nazaj v globalno pomnilnik (samo za aktivne niti)
    if (globalIndex < numberOfPoints)
    {
        points[globalIndex] = localPoints[localIndex];
    }
}
