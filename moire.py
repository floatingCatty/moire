import numpy as np
from math import gcd, acos, degrees
from ase import Atoms
from ase.build import make_supercell
from ase.geometry import wrap_positions
from itertools import product

def moire_graphene(a=2.46, d=3.35, m=1, r=1, vacuum=26.65, symbol='C', stacking="AA"):
    """
    Construct a commensurate twisted bilayer graphene Moiré super-cell.
        method derived from: Lopes dos Santos, J. M. B., N. M. R. Peres, and A. H. Castro Neto. "Continuum model of the twisted graphene bilayer." 
        Physical Review B—Condensed Matter and Materials Physics 86.15 (2012): 155449.
    
    Parameters
    ----------
    a : float
        Graphene lattice constant (Å).
    d : float
        Inter-layer spacing between the two graphene sheets (Å).
    m, r : int
        Coprime integers that label the (m,r) commensurate structure.
    vacuum : float
        Extra vacuum added along z (Å).
    symbol : str
        Chemical symbol for the carbon atom.
        
    Returns
    -------
    theta_rad : float
        Twist angle in radians.
    theta_deg : float
        Twist angle in degrees.
    atoms : ase.Atoms
        ASE structure holding the full twisted Moiré bilayer.
    """

    # ---- 1. primitive graphene (single layer) --------------------------------
    a1 = np.array([ a/2,  np.sqrt(3)*a/2, 0.])
    a2 = np.array([-a/2,  np.sqrt(3)*a/2, 0.])
    delta1 = (a1 + a2)/3.0                      # A→B shift
    positions = [np.zeros(3), delta1.copy()]
    positions[0][2] = vacuum/2
    positions[1][2] = vacuum/2

    prim = Atoms([symbol, symbol],
                 positions=positions,
                 cell=[a1, a2, [0, 0, d + vacuum]],
                 pbc=(True, True, False))

    # ---- 2. twist angle -------------------------------------------------------
    cos_theta = (3*m**2 + 3*m*r + 0.5*r**2) / (3*m**2 + 3*m*r + r**2)
    theta = acos(cos_theta)         # radians
    theta_deg = degrees(theta)

    # ---- 3. super-cell real-space vectors ------------------------------------
    if gcd(r, 3) == 1:
        S2 = np.array([[ m,      m + r],
                       [-(m + r), 2*m + r]])
    else:  # gcd(r,3)=3
        S2 = np.array([[ m + r//3,  r//3],
                       [ -r//3,   m + 2*r//3 ]])

    # integer 3×3 matrix for ASE (no replication along z)
    S3 = np.zeros((3, 3), dtype=int)
    S3[:2, :2] = S2
    S3[2, 2] = 1

    # ---- 4. build bottom layer -----------------------------------------------
    bottom = make_supercell(prim, S3)

    # ---- 5. build and transform top layer ------------------------------------
    if stacking == "AA":
        top = _build_rotated_layer(bottom, theta_deg, d, buffer=1)
    elif stacking == "AB":
        positions = [np.zeros(3), 2*delta1.copy()]
        positions[0][2] = vacuum/2
        positions[1][2] = vacuum/2
        top = Atoms([symbol, symbol],
                 positions=positions,
                 cell=[a1, a2, [0, 0, d + vacuum]],
                 pbc=(True, True, False))
        top = make_supercell(top, S3)
        top = _build_rotated_layer(top, theta_deg, d, buffer=1)
        
    else:
        raise ValueError

    # rotate around cell centre → use origin (0,0,0)
    # top.rotate(theta_deg, 'z', center=(0, 0, 0))
    # top.translate([0, 0, d])       # shift by inter-layer distance

    # ---- 6. assemble & finalise ----------------------------------------------
    cell3d = np.vstack((bottom.cell[:2], [0, 0, d + vacuum]))
    bilayer = bottom + top
    bilayer.set_cell(cell3d)
    bilayer.set_pbc((True, True, False))

    return theta, theta_deg, bilayer

def _build_rotated_layer(bottom, theta_deg, d, buffer=1, tol=1e-3):
    """
    Return a fully populated, wrapped, duplicate-free top layer.

    Parameters
    ----------
    bottom   : Atoms   bottom Moiré layer (already in the target cell)
    theta_deg: float   twist angle in degrees
    d        : float   interlayer spacing (Å)
    buffer   : int     how many extra images to add in ±x, ±y before rotation
    tol      : float   duplicate-merging tolerance in Å
    """
    # 1. make extended copy
    ext = bottom.repeat((2*buffer + 1,
                         2*buffer + 1,
                         1))

    # 2. shift so (0,0,0) is at the geometric centre of the huge slab
    #    (makes rotation nicer)
    xy_shift = (ext.cell[0] + ext.cell[1]) * buffer
    ext.translate(-xy_shift)

    # 3. rotate and move up
    ext.rotate(theta_deg, 'z', center=(0, 0, 0))
    ext.translate([0, 0, d])

    # 4. wrap to target cell
    pos = wrap_positions(ext.positions,
                         cell=bottom.cell,
                         pbc=bottom.pbc)

    # 5. keep only atoms whose wrapped fractional coords lie in [0,1)
    s = np.inner(pos, np.linalg.inv(bottom.cell))
    mask = (s[:, 0] >= 0) & (s[:, 0] < 1) & (s[:, 1] >= 0) & (s[:, 1] < 1)
    pos = pos[~mask]
    nums = ext.numbers[~mask]

    # 6. merge duplicates (within tol)
    uniq, index = np.unique(np.round(pos / tol).astype(int), axis=0,
                            return_index=True)
    pos = pos[index]
    nums = nums[index]

    top = Atoms(numbers=nums, positions=pos,
                cell=bottom.cell, pbc=bottom.pbc)

    return top


def shift_graphene(a=2.46, d=3.35, n=3, i=1, j=1, vacuum=26.65, symbol="C"):
    a1 = np.array([ a/2,  np.sqrt(3)*a/2, 0.])
    a2 = np.array([-a/2,  np.sqrt(3)*a/2, 0.])
    delta1 = (a1 + a2)/3.0                      # A→B shift
    positions = [np.zeros(3), delta1.copy()]
    positions[0][2] = vacuum/2
    positions[1][2] = vacuum/2

    prim1 = Atoms([symbol, symbol],
                 positions=positions,
                 cell=[a1, a2, [0, 0, d + vacuum]],
                 pbc=(True, True, False))
    
    # positions = [np.zeros(3), 2*delta1.copy()]
    # positions[0][2] = vacuum/2
    # positions[1][2] = vacuum/2

    shift = (a1 * i + a2 * j) / n
    prim2 = Atoms([symbol, symbol],
                positions=positions,
                cell=[a1, a2, [0, 0, d + vacuum]],
                pbc=(True, True, False))
    prim2.wrap()
    prim2.translate([0, 0, d])
    prim2.translate(shift)

    cell3d = np.vstack((prim1.cell[:2], [0, 0, d + vacuum]))
    bilayer = prim1 + prim2
    bilayer.set_cell(cell3d)
    bilayer.rotate(-60, "z", rotate_cell=True)
    bilayer.set_pbc((True, True, False))

    return bilayer


def shift_graphene_trilayer(a=2.46, d=3.35, n=3, i=1, j=1, k=1, l=1, vacuum=26.65, symbol="C"):
    a1 = np.array([ a/2,  np.sqrt(3)*a/2, 0.])
    a2 = np.array([-a/2,  np.sqrt(3)*a/2, 0.])
    delta1 = (a1 + a2)/3.0                      # A→B shift
    positions = [np.zeros(3), delta1.copy()]
    positions[0][2] = vacuum/2
    positions[1][2] = vacuum/2

    prim1 = Atoms([symbol, symbol],
                 positions=positions,
                 cell=[a1, a2, [0, 0, 2*d + vacuum]],
                 pbc=(True, True, False))
    
    # positions = [np.zeros(3), 2*delta1.copy()]
    # positions[0][2] = vacuum/2
    # positions[1][2] = vacuum/2

    shift = (a1 * i + a2 * j) / n
    prim2 = Atoms([symbol, symbol],
                positions=positions,
                cell=[a1, a2, [0, 0, 2*d + vacuum]],
                pbc=(True, True, False))
    prim2.wrap()
    prim2.translate([0, 0, d])
    prim2.translate(shift)

    shift = (a1 * k + a2 * l) / n
    prim3 = Atoms([symbol, symbol],
                positions=positions,
                cell=[a1, a2, [0, 0, 2*d + vacuum]],
                pbc=(True, True, False))
    prim3.wrap()
    prim3.translate([0, 0, 2*d])
    prim3.translate(shift)

    cell3d = np.vstack((prim1.cell[:2], [0, 0, 2*d + vacuum]))
    trilayer = prim1 + prim2 + prim3
    trilayer.set_cell(cell3d)
    trilayer.rotate(-60, "z", rotate_cell=True)
    trilayer.set_pbc((True, True, False))

    return trilayer

# ----- symmetry generators ---------------------------------------------------
def rot60(i, j):               #  R(i,j) = (-j, i+j)
    return -j, i + j

def rotk(i, j, k):             #  R^k
    for _ in range(k % 6):
        i, j = rot60(i, j)
    return i, j

def mirror0(i, j):             #  M0(i,j) = (-i, i+j)
    return -i, i + j

def mirror_k(i, j, k):         #  Mk = R^k M0 R^-k
    u, v = rotk(i, j, -k)
    u, v = mirror0(u, v)
    return rotk(u, v, k)


def apply(op, i, j):
    typ, k = op
    return rotk(i, j, k) if typ == 'R' else mirror_k(i, j, k)

# ----- irreducible–mesh generator -------------------------------------------
def irreducible_mesh(N):
    """
    Return a sorted list of representatives of the D6 orbits
    inside the square [-N,N] x [-N,N].
    """

    OPS = [('R', k) for k in range(6)] + [('M', k) for k in range(6)] # Sym group for AA stacking

    reps, visited = [], set()

    for i, j in product(range(0, N), repeat=2):
        if i == 0 and j == 0:
            continue
        if (i, j) in visited:
            continue                              # already classified
        # --- build this point's orbit
        orbit = {(i, j)}
        frontier = [(i, j)]
        while frontier:
            p = frontier.pop()
            for ix, op in enumerate(OPS):
                q = apply(op, *p)
                q = (q[0] % N, q[1] % N)
                if q not in orbit:
                    orbit.add(q)
                    frontier.append(q)
        visited |= orbit                          # mark entire orbit

        # choose a representative that lies inside the window
        candidates = [p for p in orbit
                      if -N <= p[0] <= N and -N <= p[1] <= N]
        reps.append(min(candidates))              # lexicographically minimal

    reps.sort()
    return reps

# trilayer shift irreducible mesh generator
def irreducible_mesh_trilayer(N):
    reps_tri = []
    reps = irreducible_mesh(N=N)
    for irep in reps:
        if irep[0] == irep[1]:
            for i, j in product(range(0, N), repeat=2):
                if i == 0 and j == 0:
                    continue
                if i >=j:
                    reps_tri.append(irep+(i,j))
        else:
            for i, j in product(range(0, N), repeat=2):
                if i == 0 and j == 0:
                    continue
                reps_tri.append(irep+(i,j))
        
    return reps_tri




    
if __name__ == "__main__":
    print(irreducible_mesh_trilayer(5).__len__())
    # for i in range(1,5):
    #     atoms = shift_graphene(n=8, i=i, j=-i)
    #     atoms.write(f"shift_n8i{i}j{-i}.vasp")

    # theta, theta_deg, bilayer = moire_graphene(m=0,r=1)
    # bilayer.write("POSCAR")
    N = 5
    for i,j,k,l in irreducible_mesh_trilayer(N):
        atoms = shift_graphene_trilayer(i=i,j=j,k=k,l=l,n=N)
        atoms = make_supercell(atoms, P=np.diag([3,3,1]))
        atoms.write(f"grid{N}{N}_tri/shift_n{N}i{i}j{j}k{k}l{l}.vasp")

    # for i in range(6):
    #     print(mirror_k(3,1,i))
    # (-2,-2), (-2, -1), (-2,0), (-2,1), (-2,2), (-1,-1), (-1,1), (-1,2), (0,1)