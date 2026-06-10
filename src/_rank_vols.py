"""Scratch: rank LA meshes by volume to pick size extremes for QC."""
import glob, os, pyvista as pv

rows = []
for f in glob.glob("derivatives/meshes/*_LA.stl"):
    v = pv.read(f).volume / 1000.0
    rows.append((v, os.path.basename(f).replace("_LA.stl", "")))
rows.sort()
print("=== all meshes by volume (mL) ===")
for v, s in rows:
    print(f"{v:8.1f}  {s}")
print("\n3 smallest:")
for v, s in rows[:3]:
    print(f"  {v:7.1f}  {s}")
print("3 largest:")
for v, s in rows[-3:]:
    print(f"  {v:7.1f}  {s}")
