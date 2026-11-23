"""
Siybox - layer-list encoding of a cube on a 2D face, standalone.

Straight port of field_learning_lab/layer_list_encoding.py (2026-08-18), with the
ARC-task loader removed so it runs on its own. Same encoding, same functions,
sized to the cube Corey described on 2026-10-14: 100 grids of 100x100 = one cube.

Corey, 2026-08-18 ~05:54:
  "each cell would have a 1-10 value... if there's more than one they go in
   order 1-10, so it could be 3 - 3 - 4 - 7: 3 cells in that cell, one at the
   3rd layer, one at the 4th, one at the 7th. you could represent a full 3d
   cube that way 2d."

Corey, 2026-10-14:
  "take that 100 by 100 by 100 and make a 100 by 100 grid and map that cube to
   that... the first cell is being used by 25 of the 100 dimensions... you would
   mark the first cell and give it a depth of 25... do that for every cell...
   now take that 2D grid and put it into a new 3D, doing the same thing."

The objects:
  grid  : N x N list of lists, 0 = empty, nonzero = lit (value kept)
  cube  : K grids stacked, cube[z][r][c], z = layer 0..K-1
  face  : N x N; each cell = [count, (layer, value), (layer, value), ...]
          layers reported 1-based like he said it (1..K). Lossless.
  depth face   : one integer per cell = count only ("used by 25 of the 100").
  bitmask face : one integer per cell, bit z set if layer z is lit.
          Occupancy only, values dropped. K=100 -> 100-bit ints.
  shadow: last-lit-layer-wins. LOSSY. Kept only for contrast.

Recursion (cube of cubes): K faces are themselves K grids -> stack -> encode
again. A level-L cell = [count, (layer, level-(L-1) cell), ...]; a sub-cell is
lit when its own count is nonzero. Round-trips exactly at every level.

Pure python. No numpy.
"""
import random
import sys
import time


# ---------- data ----------
def make_cube(grids):
    """cube[z][r][c]; z = layer 0..K-1. Copies the grids."""
    return [[row[:] for row in g] for g in grids]


def random_grid(n, fill, rng, values=(1, 2, 3, 4, 5, 6, 7, 8, 9)):
    """N x N grid with roughly `fill` fraction of cells lit (demo data only)."""
    return [[rng.choice(values) if rng.random() < fill else 0 for _ in range(n)]
            for _ in range(n)]


def random_cube(k, n, fill, seed=0):
    rng = random.Random(seed)
    return [random_grid(n, fill, rng) for _ in range(k)]


# ---------- Corey's encoding ----------
def encode(cube):
    """2D face; each cell = [count, (layer, value), ...] in layer order.
    Layers reported 1-based (1..K)."""
    k = len(cube)
    n = len(cube[0])
    face = [[None] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            entries = [(z + 1, cube[z][r][c]) for z in range(k) if cube[z][r][c] != 0]
            face[r][c] = [len(entries)] + entries
    return face


def decode(face, k):
    """Face -> cube of K layers."""
    n = len(face)
    cube = [[[0] * n for _ in range(n)] for _ in range(k)]
    for r in range(n):
        for c in range(n):
            count, *entries = face[r][c]
            assert count == len(entries)
            for z1, v in entries:
                cube[z1 - 1][r][c] = v
    return cube


def depth_face(cube):
    """Just the count per cell: how many of the K layers use that cell."""
    k = len(cube)
    n = len(cube[0])
    return [[sum(1 for z in range(k) if cube[z][r][c] != 0) for c in range(n)]
            for r in range(n)]


def bitmask_face(cube):
    """Occupancy only: one number per cell, bit z set if layer z occupied."""
    k = len(cube)
    n = len(cube[0])
    face = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            m = 0
            for z in range(k):
                if cube[z][r][c] != 0:
                    m |= 1 << z
            face[r][c] = m
    return face


def shadow(cube):
    """Lossy contrast: last occupied layer wins."""
    k = len(cube)
    n = len(cube[0])
    face = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            for z in range(k):
                if cube[z][r][c] != 0:
                    face[r][c] = cube[z][r][c]
    return face


def count_cells(cube):
    return sum(1 for layer in cube for row in layer for v in row if v != 0)


def numbers_in_face(face):
    # count + 2 numbers per entry
    return sum(1 + 2 * (len(cell) - 1) for row in face for cell in row)


# ---------- recursion: cube of cubes ----------
def face_lit(cell):
    """A face cell is lit when its count is nonzero."""
    return cell[0] != 0


def encode_faces(faces):
    """K faces stacked as layers -> one level-up face.
    cell = [count, (layer, sub-face-cell), ...]; sub-cell lit iff its count != 0."""
    k = len(faces)
    n = len(faces[0])
    face = [[None] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            entries = [(z + 1, faces[z][r][c]) for z in range(k) if face_lit(faces[z][r][c])]
            face[r][c] = [len(entries)] + entries
    return face


def decode_faces(face, k):
    """Level-up face -> K sub-faces. Missing sub-cells are empty: [0]."""
    n = len(face)
    faces = [[[[0] for _ in range(n)] for _ in range(n)] for _ in range(k)]
    for r in range(n):
        for c in range(n):
            count, *entries = face[r][c]
            assert count == len(entries)
            for z1, cell in entries:
                faces[z1 - 1][r][c] = cell
    return faces


def encode_levels(cubes, levels):
    """cubes: K^(levels-1) cubes. Fold up `levels` times -> one face.
    levels=1: a single cube -> face. levels=2: K cubes -> K faces -> one face. ..."""
    k = len(cubes[0])
    faces = [encode(cb) for cb in cubes]
    for _ in range(levels - 1):
        groups = [faces[i:i + k] for i in range(0, len(faces), k)]
        faces = [encode_faces(g) for g in groups]
    assert len(faces) == 1, f"need K^(levels-1) cubes, got {len(cubes)}"
    return faces[0]


def decode_levels(face, k, levels):
    """Inverse of encode_levels -> list of cubes."""
    faces = [face]
    for _ in range(levels - 1):
        faces = [sub for f in faces for sub in decode_faces(f, k)]
    return [decode(f, k) for f in faces]


# ---------- run ----------
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100     # face size
    k = int(sys.argv[2]) if len(sys.argv) > 2 else n       # layers per cube
    fill = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    n2 = int(sys.argv[4]) if len(sys.argv) > 4 else 10     # size for the level-2 demo

    print(f"cube: {k} layers of {n}x{n} = {k*n*n} cells, fill ~{fill:.0%}")
    t0 = time.time()
    cube = random_cube(k, n, fill, seed=1)
    occ = count_cells(cube)
    print(f"occupied: {occ}  (built in {time.time()-t0:.1f}s)\n")

    # 1. layer-list face, lossless
    t0 = time.time()
    face = encode(cube)
    back = decode(face, k)
    print(f"layer-list face: round-trip identical = {back == cube}  ({time.time()-t0:.1f}s)")
    shown = 0
    for r in range(n):
        for c in range(n):
            if face[r][c][0] >= 3 and shown < 2:
                print(f"  ({r},{c}): {face[r][c][:5]} ...")
                shown += 1

    # 2. depth only - "this cell is in 25 of the 100 grids"
    d = depth_face(cube)
    print(f"\ndepth face (count per cell), first row: {d[0][:12]} ...")
    print(f"  cell (0,0) is used by {d[0][0]} of the {k} grids")

    # 3. one number per cell (occupancy bitmask, K bits)
    bm = bitmask_face(cube)
    occ_ok = all(((bm[r][c] >> z) & 1) == (1 if cube[z][r][c] != 0 else 0)
                 for z in range(k) for r in range(n) for c in range(n))
    print(f"\nbitmask face ({k}-bit ints): reproduces occupancy exactly = {occ_ok}")
    print(f"  cell (0,0) = {bm[0][0]}")

    # 4. shadow for contrast
    sh = shadow(cube)
    kept = sum(1 for row in sh for v in row if v != 0)
    print(f"\nshadow (last layer wins): keeps {kept} of {occ} -> loses {occ - kept}")

    # 5. sizes
    print("\nsize (numbers stored):")
    print(f"  raw cube        : {k*n*n}")
    print(f"  layer-list face : {numbers_in_face(face)}   (lossless, values kept)")
    print(f"  depth face      : {n*n}   (counts only)")
    print(f"  bitmask face    : {n*n}   (lossless occupancy, values dropped)")
    print(f"  shadow          : {n*n}   (LOSSY)")

    # 6. recursion: K cubes -> K faces -> one face of faces, and back
    #    run at a smaller size: K^2 * N^2 cells in pure python
    print(f"\nlevel 2 (cube of cubes) at {n2}x{n2}, {n2} layers, {n2} cubes:")
    cubes = [random_cube(n2, n2, fill, seed=100 + i) for i in range(n2)]
    t0 = time.time()
    face2 = encode_levels(cubes, 2)
    cubes_back = decode_levels(face2, n2, 2)
    print(f"  {n2*n2*n2*n2} cells -> one {n2}x{n2} face -> back: identical = {cubes_back == cubes}  ({time.time()-t0:.1f}s)")
    print(f"  face2 cell (0,0): {face2[0][0][0]} of {n2} cubes use that cell")

    # 7. level 3 - same thing again, K^2 cubes
    n3 = max(2, min(n2, 5))
    cubes3 = [random_cube(n3, n3, fill, seed=1000 + i) for i in range(n3 * n3)]
    face3 = encode_levels(cubes3, 3)
    back3 = decode_levels(face3, n3, 3)
    print(f"level 3 at {n3}: {n3*n3} cubes -> one face -> back: identical = {back3 == cubes3}")


if __name__ == "__main__":
    main()
