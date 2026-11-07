import open3d as o3d
import numpy as np
import laspy
import struct
import io
from pathlib import Path

name = str(Path(__file__).resolve().parents[1] / "data" / "pc" / "ljubljana_2")

def read_data(path):
    with open(path, "rb") as f:
        raw = bytearray(f.read())
   
    return io.BytesIO(bytes(raw))


# Naložimo LAS datoteko
path = name + ".las"
buf = read_data(path)

try:
    las = laspy.read(buf)
    print(f"Loaded after patch: {len(las.x)} points")
except Exception as e:
    print(f"Patch failed: {e}")
    raise

points = np.vstack((las.x, las.y, las.z)).T
colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
src_header = las.header
print(f"X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}] ")
print(f"Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}] ")
print(f"Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}] ")