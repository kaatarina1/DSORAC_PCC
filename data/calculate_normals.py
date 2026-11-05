import open3d as o3d
import numpy as np
import laspy
import struct
import io
from pathlib import Path
from scipy.spatial import cKDTree

name = "ljubljana_3"
# Poti glede na lokacijo skripte (data/pc/), da deluje ne glede na trenutni direktorij
base = str(Path(__file__).resolve().parent / "pc") + "/"

# Potrebno, da zagotovimo pravilno izvaja tudi če ima LAS datoteka že obstoječe (a napačne/duplikatne) 
# VLR-je z ekstra dimenzijami, ki bi lahko motili naš izračun normal. 
# Ta funkcija prebere surove bajte LAS datoteke, identificira in odstrani vse VLR-je z ekstra dimenzijami, 
# ki imajo imena 'nx', 'ny' ali 'nz', in nato ponovno zgradi VLR-je brez teh problematičnih dimenzij. 
# Na koncu vrne popravljen bajtovni tok, ki ga lahko varno naložimo z laspy brez tveganja za napake ali podvojene dimenzije.
def fix_duplicate_extra_bytes(path):
    with open(path, "rb") as f:
        raw = bytearray(f.read())

    offset_to_points = struct.unpack_from("<I", raw, 96)[0]
    num_vlrs         = struct.unpack_from("<I", raw, 100)[0]
    header_size      = struct.unpack_from("<H", raw, 94)[0]

    VLR_HEADER             = 54
    EXTRA_BYTES_RECORD_SIZE = 192

    pos = header_size
    vlr_spans = []
    for _ in range(num_vlrs):
        if pos + VLR_HEADER > len(raw):
            break
        user_id   = raw[pos+2:pos+18].rstrip(b'\x00').decode('ascii', errors='replace')
        record_id = struct.unpack_from("<H", raw, pos+18)[0]
        rec_len   = struct.unpack_from("<H", raw, pos+20)[0]
        data_start = pos + VLR_HEADER
        data_end   = data_start + rec_len
        data       = bytes(raw[data_start:data_end])
        vlr_spans.append((pos, data_end, user_id, record_id, data))
        pos = data_end

    eb_vlr_indices = [i for i, v in enumerate(vlr_spans) if v[2] == 'LASF_Spec' and v[3] == 4]

    print(f"  Found {len(eb_vlr_indices)} Extra Bytes VLR(s) — stripping all and rebuilding without normals...")

    seen_names = set()
    clean_records = []
    NORMAL_NAMES = {'nx', 'ny', 'nz'}  # Preskočimo te, saj bomo izračunali nove

    for idx in eb_vlr_indices:
        data = vlr_spans[idx][4]
        for off in range(0, len(data), EXTRA_BYTES_RECORD_SIZE):
            rec = data[off:off + EXTRA_BYTES_RECORD_SIZE]
            if len(rec) < EXTRA_BYTES_RECORD_SIZE:
                continue
            name = rec[4:36].rstrip(b'\x00').decode('ascii', errors='replace')
            if name in NORMAL_NAMES:
                print(f"    Skipping existing '{name}' extra dim (will recompute)")
                continue
            if name not in seen_names:
                seen_names.add(name)
                clean_records.append(rec)

    spans_to_remove = sorted([vlr_spans[i] for i in eb_vlr_indices], key=lambda x: x[0], reverse=True)
    removed_bytes = sum(s[1] - s[0] for s in spans_to_remove)
    for span in spans_to_remove:
        del raw[span[0]:span[1]]

    new_offset = offset_to_points - removed_bytes

    added_vlr_size = 0
    if clean_records:
        clean_data    = b''.join(clean_records)
        user_id_bytes = b'LASF_Spec\x00' + b'\x00' * 7
        description   = b'\x00' * 32
        new_vlr = (
            b'\x00\x00'
            + user_id_bytes
            + struct.pack("<H", 4)
            + struct.pack("<H", len(clean_data))
            + description
            + clean_data
        )
        raw[new_offset:new_offset] = new_vlr
        added_vlr_size = len(new_vlr)

    final_offset = new_offset + added_vlr_size
    struct.pack_into("<I", raw, 96, final_offset)

    new_num_vlrs = num_vlrs - len(eb_vlr_indices) + (1 if clean_records else 0)
    struct.pack_into("<I", raw, 100, new_num_vlrs)

    print(f"  Rebuilt: {len(clean_records)} non-normal extra dims kept, {new_num_vlrs} VLRs total")    
    return io.BytesIO(bytes(raw))


# Naložimo LAS datoteko
path = base + name + ".las"

# Počistimo morebitne problematične VLR-je z ekstra dimenzijami, ki bi lahko motili izračun normal
print("Patching raw bytes to remove duplicate/stale extra dims...")
buf = fix_duplicate_extra_bytes(path)

try:
    las = laspy.read(buf)
    print(f"Loaded after patch: {len(las.x)} points")
except Exception as e:
    print(f"Patch failed: {e}")
    raise

points = np.vstack((las.x, las.y, las.z)).T
origin = points.min(axis=0)
colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
src_header = las.header
print(f"X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}] ")
print(f"Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}] ")
print(f"Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}] ")

pcd = o3d.geometry.PointCloud()
# Točke premaknemo v lokalni izhodišče (origin), ob shranjevanju ga prištejemo nazaj
pcd.points = o3d.utility.Vector3dVector(points - origin)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Počistimo outlierje, da ne bodo motili normal.
# Indekse ohranimo, da lahko pravilno izberemo tudi klasifikacije preostalih točk.
# pcd, kept_idx = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=3.0)
# print(f"After cleaning: {len(pcd.points)} points kept")

# Izračun normal na downsampled oblaku, da bo hitreje in stabilneje 
pcd_down = pcd.voxel_down_sample(voxel_size=0.05) 
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=64)
)

pts = np.asarray(pcd.points)
x_range = pts[:,0].max() - pts[:,0].min()
y_range = pts[:,1].max() - pts[:,1].min()
z_range = pts[:,2].max() - pts[:,2].min()

center = pts.mean(axis=0)
if z_range <= y_range and z_range <= x_range:
    # Z is up (standard LAS)
    up_axis = np.array([0.0, 0.0, 1.0])
    print("Detected Z-up")
elif y_range <= x_range and y_range <= z_range:
    up_axis = np.array([0.0, 1.0, 0.0])
    print("Detected Y-up")
else:
    up_axis = np.array([1.0, 0.0, 0.0])
    print("Detected X-up")

pcd_down.orient_normals_consistent_tangent_plane(k=10)
down_normals = np.asarray(pcd_down.normals)

# Na koncu normale razširimo na celoten oblak: vsaki točki dodelimo normalo najbližje downsampled točke
down_pts = np.asarray(pcd_down.points)
down_normals = np.asarray(pcd_down.normals)
_, nn_idx = cKDTree(down_pts).query(pts, k=1, workers=-1)
full_normals = down_normals[nn_idx]
pcd.normals = o3d.utility.Vector3dVector(full_normals)
normals = np.asarray(pcd.normals)
print(f"Mean normal: [{normals[:,0].mean():.3f}, {normals[:,1].mean():.3f}, {normals[:,2].mean():.3f}]")
print(f"Normals computed: {normals.shape}")

# Shrani rezultat obdelave v novo LAS datoteko
filtered_points = np.asarray(pcd.points) + origin
filtered_colors = (np.asarray(pcd.colors) * 65535).astype(np.uint16)

header = laspy.LasHeader(
    point_format=laspy.PointFormat(src_header.point_format.id),
    version=src_header.version,
)
header.add_extra_dim(laspy.ExtraBytesParams(name="nx", type=np.float32))
header.add_extra_dim(laspy.ExtraBytesParams(name="ny", type=np.float32))
header.add_extra_dim(laspy.ExtraBytesParams(name="nz", type=np.float32))

out = laspy.LasData(header)
out.x     = filtered_points[:, 0]
out.y     = filtered_points[:, 1]
out.z     = filtered_points[:, 2]
out.red   = filtered_colors[:, 0]
out.green = filtered_colors[:, 1]
out.blue  = filtered_colors[:, 2]
out.nx    = normals[:, 0].astype(np.float32)
out.ny    = normals[:, 1].astype(np.float32)
out.nz    = normals[:, 2].astype(np.float32)
out.classification = np.asarray(las.classification)

out.write(base + name + "_normals.las")
print("Saved → " + name + "_normals.las")

las = laspy.read(base + name + "_normals.las")

print(las.point_format)
print(las.point_format.dimension_names)

# Structured array dtype — the ground truth
print(las.points.array.dtype)

lengths = np.linalg.norm(normals, axis=1)
print(f"Normal x: [{normals[:,0].min():.3f}, {normals[:,0].max():.3f}]")
print(f"Normal y: [{normals[:,1].min():.3f}, {normals[:,1].max():.3f}]")
print(f"Normal z: [{normals[:,2].min():.3f}, {normals[:,2].max():.3f}]")
print(f"Lengths:   min={lengths.min():.4f}  max={lengths.max():.4f}  (should be ~1.0)")