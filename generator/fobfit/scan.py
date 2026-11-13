"""Scan loading, units, cleaning, key frame detection and the caliper check.

Key frame (all later work happens in it, in mm):
  x  along the blade axis, + toward the blade tip
  z  out of the button face (the key snaps into the jacket from +z)
  y  = z cross x
  origin on the blade axis, at the middle of the key body's length
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

from . import geom

UNITS = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0, 'inch': 25.4}


# ------------------------------------------------------------------------------------------------ loading
def load(path):
    """Read STL / PLY / OBJ (or anything trimesh reads) as one triangle mesh, without merging or repair."""
    m = trimesh.load(str(path), force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError('%s holds no triangles: it is a point cloud. Open the mesh (STL) as the scan. FobFit uses the '
                         'point cloud only to mark surface that the scanner software invented.' % Path(path).name)
    return m


# ------------------------------------------------------------------------------------------------ raw point cloud
CLOUD_SUFFIXES = ('.obj', '.ply', '.xyz', '.txt', '.pts')
INVENTED_MM = 0.5          # scan surface with no raw point this close was invented by the scanner software


def _obj_points(path):
    """Vertices of an OBJ ('v x y z [r g b]' lines) and whether it has faces."""
    pts, faces = [], False
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                pts.append(line[2:])
            elif line.startswith('f '):
                faces = True
                break
    if faces or not pts:
        return None, faces
    a = np.array(' '.join(pts).split(), dtype=float)
    k = len(pts[0].split())
    return a.reshape(-1, k)[:, :3], False


def load_cloud(path):
    """Points (n, 3) of a raw point cloud in file units, or None when the file holds triangles (it is a mesh)."""
    p = Path(path)
    if p.suffix.lower() == '.obj':
        pts, _ = _obj_points(p)
        return pts
    if p.suffix.lower() in ('.xyz', '.txt', '.pts'):
        a = np.loadtxt(p, comments=('#', '//'), ndmin=2)
        return a[:, :3] if a.shape[1] >= 3 else None
    g = trimesh.load(str(p), process=False)
    if isinstance(g, trimesh.Trimesh) and len(g.faces):
        return None
    v = np.asarray(getattr(g, 'vertices', []), float)
    return v if len(v) else None


def find_cloud(scan_path):
    """A raw point cloud next to the scan whose name shares the start of the scan name (at least half of it),
    e.g. 'toyota yaris mesh.stl' -> 'toyota yaris point cloud.obj'. Returns the path or ''."""
    s = Path(scan_path)
    best, best_n = '', 0
    for q in s.parent.iterdir():
        if q == s or q.suffix.lower() not in CLOUD_SUFFIXES or not q.is_file():
            continue
        a, b = s.stem.lower(), q.stem.lower()
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        if n >= max(4, len(a) / 2) and n > best_n:
            best, best_n = str(q), n
    if best and best.lower().endswith('.obj'):
        _, faces = _obj_points(best)
        if faces:
            return ''
    return best


def cloud_cover(points_key, cloud_key, r=INVENTED_MM):
    """Distance from each scan surface point to the nearest raw point (capped at 2 r); invented = no raw point
    within r."""
    from scipy.spatial import cKDTree
    t = cKDTree(cloud_key)
    d, _ = t.query(points_key, distance_upper_bound=2 * r, workers=-1)
    d = np.minimum(d, 2 * r)
    return d, d > r


def guess_units(mesh):
    """Unit name whose scale puts the longest side of the scan between 25 and 250 mm (a car key with its blade).
    mm wins when it fits. Returns (name, size in mm, note)."""
    e = float(np.max(mesh.extents))
    for name in ('mm', 'm', 'cm', 'inch'):
        if 25.0 <= e * UNITS[name] <= 250.0:
            note = '' if name == 'mm' else 'The scan looks like it is in %s (longest side %.4g); converted to mm.' % (name, e)
            return name, e * UNITS[name], note
    return 'mm', e, 'Longest side %.4g mm is outside 25-250 mm. Check the units.' % e


@dataclass
class CleanInfo:
    faces_in: int = 0
    faces_out: int = 0
    parts_in: int = 0
    parts_kept: int = 0
    area_dropped_pct: float = 0.0
    open_edges: int = 0
    noise_mm: float = 0.0
    smooth_iterations: int = 0
    noise_after_mm: float = 0.0
    hole_centres: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def clean(mesh_mm, keep_frac=0.05):
    """Merge vertices, drop degenerate and duplicate faces, drop loose fragments (parts with less than keep_frac of
    the largest part's area), estimate the scan noise. Returns (mesh, CleanInfo)."""
    info = CleanInfo(faces_in=len(mesh_mm.faces))
    m = trimesh.Trimesh(mesh_mm.vertices.copy(), mesh_mm.faces.copy(), process=True, validate=False)
    m.update_faces(m.nondegenerate_faces(height=1e-9))
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    n, lab = geom.face_components(m)
    area = np.bincount(lab, weights=m.area_faces, minlength=n)
    info.parts_in = int(n)
    keep = np.flatnonzero(area >= keep_frac * area.max())
    if len(keep) < n:
        info.area_dropped_pct = round(100.0 * (1 - area[keep].sum() / area.sum()), 2)
        m.update_faces(np.isin(lab, keep))
        m.remove_unreferenced_vertices()
        info.notes.append('Dropped %d loose fragments (%.2f %% of the surface).' % (n - len(keep), info.area_dropped_pct))
    info.parts_kept = int(len(keep))
    info.faces_out = len(m.faces)
    edges = m.edges_sorted
    _, cnt = np.unique(edges, axis=0, return_counts=True)
    info.open_edges = int((cnt == 1).sum())
    if info.open_edges:
        info.notes.append('The scan has holes (%d open edges).' % info.open_edges)
    info.noise_mm = round(noise_estimate(m), 3)
    return m, info


def noise_estimate(m, n=5000, seed=0, radius=1.5):
    """Scan noise (mm): median distance of vertices from the plane fitted to their neighbours within `radius`,
    times 1.25 (median to rms for a normal distribution). Sharp edges are a small share and the median ignores
    them. Vertices, not surface samples: samples between vertices average the noise away."""
    from scipy.spatial import cKDTree
    v = np.asarray(m.vertices)
    if len(v) < 200:
        return 0.0
    tree = cKDTree(v)
    q = v[np.random.default_rng(seed).choice(len(v), min(n, len(v)), replace=False)]
    d, nb = tree.query(q, k=49, workers=-1)
    ok = d[:, 12] < radius                        # enough neighbours close by
    d, nb, q = d[ok], nb[ok][:, 1:], q[ok]
    if len(q) < 50:
        return 0.0
    pts = v[nb]
    w = (d[:, 1:] < radius)[:, :, None]
    cnt = w.sum(1)
    c = (pts * w).sum(1) / cnt
    X = (pts - c[:, None]) * w
    C = np.einsum('nki,nkj->nij', X, X)
    _, vec = np.linalg.eigh(C)
    res = np.abs(np.einsum('ni,ni->n', q - c, vec[:, :, 0]))
    # 30th percentile of |N(0, s)| is 0.385 s. Noise moves every vertex; sharp features (button gaps, edges) only a
    # share of them, so a low percentile reads the noise and mostly ignores the features
    return float(np.percentile(res, 30) / 0.385)


def boundary_loops(m):
    """Closed loops of boundary edges (edges with one face), as vertex index arrays."""
    e = m.edges_sorted
    u, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    be = u[cnt == 1]
    if len(be) == 0:
        return []
    nb = {}
    for a, b in be:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    seen = set()
    loops = []
    for start in list(nb):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = [x for x in nb[cur] if x != prev and (x not in seen or (x == start and len(loop) > 2))]
            if not nxt:
                break
            n = nxt[0]
            if n == start:
                break
            loop.append(n)
            seen.add(n)
            prev, cur = cur, n
        if len(loop) >= 3:
            loops.append(np.array(loop))
    return loops


def fill_holes(m, max_extent=25.0, ring=2.5, step=0.2, noise=0.0):
    """Fill scan holes with patches that continue the surrounding surface: a quadric height field fitted to the
    vertices within `ring` mm of the hole's rim, triangulated over the hole and a little past its rim. A flat or
    ball-rolled bridge would cut inside a curved key (up to the sagitta of the hole) and the cavity would bind.
    Returns (mesh with patches, number of holes filled, largest hole extent)."""
    from scipy.spatial import cKDTree, Delaunay
    from matplotlib.path import Path as MPath
    loops = boundary_loops(m)
    if not loops:
        return m, 0, 0.0
    v = np.asarray(m.vertices)
    tree = cKDTree(v)
    vfn = m.vertex_normals
    patches = []
    centres = []
    biggest = 0.0
    fill_holes.envelope = 0
    for lp in loops:
        P = v[lp]
        ext = float(np.ptp(P, axis=0).max())
        if ext > max_extent:
            continue
        idx = np.unique(np.concatenate(tree.query_ball_point(P, ring)))
        Q = v[idx]
        if len(Q) < 12:
            continue
        c = Q.mean(0)
        w, ax = np.linalg.eigh(np.cov((Q - c).T))
        n, U, V = ax[:, 0], ax[:, 2], ax[:, 1]
        if np.dot(vfn[idx].mean(0), n) < 0:
            n = -n
            V = -V
        qu, qv, qh = (Q - c) @ U, (Q - c) @ V, (Q - c) @ n
        A = np.stack([np.ones_like(qu), qu, qv, qu * qu, qu * qv, qv * qv], axis=1)
        coef, *_ = np.linalg.lstsq(A, qh, rcond=None)
        resid = float(np.sqrt(np.mean((A @ coef - qh) ** 2)))
        # tangent planes at the rim (for holes over sharp features, where a quadric cuts inside the true corner)
        planes = []
        for pi in P[:: max(1, len(P) // 60)]:
            nb = tree.query_ball_point(pi, 1.5)
            if len(nb) < 6:
                continue
            R = v[nb]
            rc = R.mean(0)
            _, rv = np.linalg.eigh(np.cov((R - rc).T))
            ni = rv[:, 0] if rv[:, 0] @ n > 0 else -rv[:, 0]
            if ni @ n > 0.35:
                planes.append((rc, ni))
        pu, pv = (P - c) @ U, (P - c) @ V
        poly = MPath(np.stack([pu, pv], axis=1))
        g0, g1 = np.array([pu.min(), pv.min()]) - 0.6, np.array([pu.max(), pv.max()]) + 0.6
        gu, gv = np.meshgrid(np.arange(g0[0], g1[0] + step, step), np.arange(g0[1], g1[1] + step, step))
        G = np.stack([gu.ravel(), gv.ravel()], axis=1)
        inside = poly.contains_points(G, radius=0.4) | poly.contains_points(G, radius=-0.4)
        G = G[inside]
        if len(G) < 3:
            continue
        tri = Delaunay(G)
        F = tri.simplices
        cen = G[F].mean(1)
        F = F[poly.contains_points(cen, radius=0.4) | poly.contains_points(cen, radius=-0.4)]
        h = np.stack([np.ones(len(G)), G[:, 0], G[:, 1], G[:, 0] ** 2, G[:, 0] * G[:, 1], G[:, 1] ** 2], axis=1) @ coef
        if resid > max(0.08, 3.0 * noise) and planes:
            # the quadric fits badly (a corner or an edge under the hole): use the upper envelope of the rim's tangent
            # planes where it is higher. On a convex or a concave surface those planes lie on or outside the true
            # surface, so the fill can only add gap, never cut into the key (capped at 1 mm above the quadric)
            env = np.full(len(G), -1e9)
            for rc, ni in planes:
                hi = -((c - rc) @ ni + G[:, 0] * (U @ ni) + G[:, 1] * (V @ ni)) / (n @ ni)
                env = np.maximum(env, hi)
            h = np.maximum(h, np.minimum(env, h + 1.0))
            fill_holes.envelope = getattr(fill_holes, 'envelope', 0) + 1
        X = c + np.outer(G[:, 0], U) + np.outer(G[:, 1], V) + np.outer(h, n)
        pt = trimesh.Trimesh(X, F, process=False)
        flip = np.einsum('ij,j->i', pt.face_normals, n) < 0
        F[flip] = F[flip][:, ::-1]
        patches.append(trimesh.Trimesh(X, F, process=False))
        centres.append(P.mean(0).tolist())
        biggest = max(biggest, ext)
    fill_holes.centres = centres
    if not patches:
        return m, 0, 0.0
    out = trimesh.util.concatenate([m] + patches)
    out.remove_unreferenced_vertices()
    return out, len(patches), biggest


def denoise(m, target=0.03, max_iter=60, step=5):
    """Taubin smoothing (no shrinkage) in steps of `step` iterations until the noise estimate is below target.
    The distance field of a noisy scan follows the outer noise peaks, so noise must go before the field is built.
    Returns (mesh, iterations, noise after)."""
    m = m.copy()
    m.remove_unreferenced_vertices()
    it = 0
    nz = noise_estimate(m)
    while nz > target and it < max_iter:
        prev = m.copy()
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=step)
        nz2 = noise_estimate(m)
        if nz2 > 0.8 * nz:             # no real progress: what is left is shape, not noise; undo this step
            m = prev
            break
        it += step
        nz = nz2
    return m, it, nz


# ------------------------------------------------------------------------------------------------ frame
@dataclass
class Frame:
    T: list                      # 4x4 rigid, scan in mm -> key frame (rotation, origin); see matrix()
    unit: str = 'mm'
    unit_scale: float = 1.0
    caliper_scale: float = 1.0
    body_x: tuple = (0.0, 0.0)   # body from the ring end to the blade shoulder, key frame x
    width: float = 0.0           # largest width of the body (y)
    thickness: float = 0.0       # largest thickness of the body (z)
    blade_found: bool = False
    blade: dict = field(default_factory=dict)     # length, width, thickness, tip_x
    ring: dict = field(default_factory=dict)      # found, x, y, r (key frame)
    face_confidence: float = 0.0  # ratio of button-face to back roughness (>1.3 is a clear call)
    notes: list = field(default_factory=list)

    def matrix(self):
        """scan mm -> key frame mm, with the caliper scale (about the key frame origin)"""
        S = np.eye(4)
        S[:3, :3] *= self.caliper_scale
        return S @ np.asarray(self.T, float)

    def file_matrix(self):
        """scan file units -> key frame mm"""
        U = np.eye(4)
        U[:3, :3] *= self.unit_scale
        return self.matrix() @ U

    def to_dict(self):
        d = asdict(self)
        return d


def _pca(p):
    c = p.mean(0)
    w, v = np.linalg.eigh(np.cov((p - c).T))
    return c, v[:, ::-1].T                     # rows: major .. minor


def _profile(s, u, v, step=1.0):
    """Per 1 mm bin along s: (bin centres, thickness along v, width along u, count, centre u, centre v)."""
    lo, hi = np.floor(s.min()), np.ceil(s.max())
    edges = np.arange(lo, hi + step, step)
    k = np.clip(((s - lo) / step).astype(int), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        sel = k == b
        if sel.sum() < 8:
            out.append((edges[b] + step / 2, 0.0, 0.0, int(sel.sum()), 0.0, 0.0))
            continue
        uu, vv = u[sel], v[sel]
        u0, u1 = np.percentile(uu, [0.5, 99.5])
        v0, v1 = np.percentile(vv, [0.5, 99.5])
        out.append((edges[b] + step / 2, v1 - v0, u1 - u0, int(sel.sum()), (u0 + u1) / 2, (v0 + v1) / 2))
    return np.array(out)


def detect_frame(mesh_mm, unit='mm', unit_scale=1.0, blade_end=None, axis_points=None, flip_face=False,
                 face=None, n=300000):
    """Find the key frame of a cleaned scan in mm.

    blade_end:   None (detect), or a 3-vector in scan mm pointing from the body toward the blade
    axis_points: optional (tip, root) points in scan mm picked on the blade; sets the axis exactly
    flip_face:   turn the detected button face over
    face:        optional 3-vector in scan mm pointing out of the button face (overrides the detection)
    """
    P, _ = geom.sample_surface(mesh_mm, n, seed=3)
    c0, ax = _pca(P)
    notes = []
    if axis_points is not None:
        tip, root = (np.asarray(p, float) for p in axis_points)
        a = tip - root
        a /= np.linalg.norm(a)
    else:
        a = ax[0]
        if blade_end is not None and np.dot(a, blade_end) < 0:
            a = -a

    def basis(a):
        q = P - c0
        q = q - np.outer(q @ a, a)
        _, b = _pca(q)
        uu = b[0] - a * (b[0] @ a)
        uu /= np.linalg.norm(uu)
        vv = np.cross(a, uu)
        return uu, vv

    u, v = basis(a)
    s = (P - c0) @ a
    prof = _profile(s, (P - c0) @ u, (P - c0) @ v)
    ok = prof[:, 3] >= 8
    tmax = np.percentile(prof[ok, 1], 95)
    wmax = np.percentile(prof[ok, 2], 95)
    blade_like = ok & (prof[:, 1] < 0.45 * tmax) & (prof[:, 2] < 0.55 * wmax)

    def run(flags):
        k = 0
        for f in flags:
            if not f:
                break
            k += 1
        return k
    idx_ok = np.flatnonzero(ok)
    hi_run = run(blade_like[idx_ok[::-1]])
    lo_run = run(blade_like[idx_ok])
    blade_found = max(hi_run, lo_run) >= 8
    if axis_points is None and blade_end is None:
        if lo_run > hi_run:
            a = -a
            u, v = basis(a)
            s = (P - c0) @ a
            prof = _profile(s, (P - c0) @ u, (P - c0) @ v)
            ok = prof[:, 3] >= 8
            blade_like = ok & (prof[:, 1] < 0.45 * tmax) & (prof[:, 2] < 0.55 * wmax)
            idx_ok = np.flatnonzero(ok)
            hi_run, lo_run = lo_run, hi_run
        if not blade_found:
            notes.append('No blade found (is it folded or missing from the scan?). Check the blade end and axis.')
    blade_run = hi_run if axis_points is None else run(blade_like[idx_ok[::-1]])
    blade_found = blade_run >= 8
    s_tip = prof[idx_ok[-1], 0] + 0.5
    s_sh = prof[idx_ok[-blade_run], 0] - 0.5 if blade_found else s_tip
    # shoulder: first bin from the blade side (inward) whose thickness passes half the body thickness
    if blade_found:
        k = idx_ok[-blade_run] - 1
        while k > 0 and prof[k, 3] >= 8 and prof[k, 1] < 0.5 * tmax:
            k -= 1
        s_sh = prof[k, 0] + 0.5
    s_min = prof[idx_ok[0], 0] - 0.5
    # axis refinement from the blade
    # axis refinement: a line through the centres of the blade's cross-sections (1 mm slices), three passes
    bsel = (s > s_sh + 3.0) & (s < s_tip - 3.0)
    p_axis = c0
    if axis_points is not None:
        p_axis = np.asarray(axis_points[1], float)
    elif blade_found and bsel.sum() > 300:
        for _ in range(3):
            s = (P - c0) @ a
            u, v = basis(a)
            bsel = (s > s_sh + 3.0) & (s < s_tip - 3.0)
            q, sq = P[bsel] - c0, s[bsel]
            pu, pv = q @ u, q @ v
            kb = np.floor(sq).astype(int)
            cs, cu, cv = [], [], []
            for b in np.unique(kb):
                m_ = kb == b
                if m_.sum() < 20:
                    continue
                cs.append(sq[m_].mean())
                cu.append(np.mean(np.percentile(pu[m_], [2, 98])))
                cv.append(np.mean(np.percentile(pv[m_], [2, 98])))
            if len(cs) < 5:
                break
            cs, cu, cv = map(np.asarray, (cs, cu, cv))
            bu, au = np.polyfit(cs, cu, 1)
            bv, av = np.polyfit(cs, cv, 1)
            res = np.hypot(cu - (au + bu * cs), cv - (av + bv * cs))       # drop slices hit by scan holes
            keep = res <= max(3 * np.median(res), 0.05)
            if keep.sum() >= 5:
                bu, au = np.polyfit(cs[keep], cu[keep], 1)
                bv, av = np.polyfit(cs[keep], cv[keep], 1)
            p_axis = c0 + u * au + v * av
            a = a + bu * u + bv * v
            a /= np.linalg.norm(a)
        s = (P - c0) @ a
        u, v = basis(a)
        bsel = (s > s_sh + 3.0) & (s < s_tip - 3.0)
    # thin direction of the body (not the blade): z candidate
    body = P[(s < s_sh) & (s > s_min)]
    qb = body - c0
    qb = qb - np.outer(qb @ a, a)
    _, bb = _pca(qb)
    z = bb[1] - a * (bb[1] @ a)              # bb[2] is along a (no variance after the projection)
    z /= np.linalg.norm(z)
    if face is not None:
        if np.dot(z, face) < 0:
            z = -z
        conf = float('nan')
    else:
        conf, flip = _face_vote(body, a, z, c0)
        if flip:
            z = -z
        if conf < 1.3:
            notes.append('Button face is not clear (roughness ratio %.2f). Check it.' % conf)
    if flip_face:
        z = -z
    y = np.cross(z, a)
    R = np.vstack([a, y, z])
    s_mid = (s_min + s_sh) / 2
    o = p_axis + a * (s_mid - (p_axis - c0) @ a)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ o
    fr = Frame(T=T.tolist(), unit=unit, unit_scale=unit_scale, blade_found=bool(blade_found), notes=notes,
               face_confidence=round(float(conf), 2) if conf == conf else 0.0)
    fr.body_x = (round(float(s_min - s_mid), 2), round(float(s_sh - s_mid), 2))
    if blade_found:
        bs = P[bsel] if bsel.sum() > 0 else P[:0]
        bk = (bs - o) @ R.T if len(bs) else np.zeros((0, 3))
        fr.blade = dict(length=round(float(s_tip - s_sh), 1), tip_x=round(float(s_tip - s_mid), 1),
                        width=round(float(np.ptp(np.percentile(bk[:, 1], [1, 99]))), 2) if len(bk) else 0.0,
                        thickness=round(float(np.ptp(np.percentile(bk[:, 2], [1, 99]))), 2) if len(bk) else 0.0)
    return fr


def _face_vote(body, a, z, c0, cell=0.5):
    """Which side of the body carries the buttons: the rougher height map (buttons, gaps, rims). Returns
    (ratio rougher/smoother, flip) where flip means the button face is at -z."""
    y = np.cross(z, a)
    q = body - c0
    X, Y, Z = q @ a, q @ y, q @ z
    i = ((X - X.min()) / cell).astype(int)
    j = ((Y - Y.min()) / cell).astype(int)
    nx, ny = i.max() + 1, j.max() + 1
    top = np.full((nx, ny), -np.inf)
    bot = np.full((nx, ny), np.inf)
    np.maximum.at(top, (i, j), Z)
    np.minimum.at(bot, (i, j), Z)
    occ = np.isfinite(top)
    occ = ndimage.binary_closing(occ, iterations=2)
    inner = ndimage.binary_erosion(occ, iterations=int(3.0 / cell))
    rough = []
    for h, sgn in ((top, 1), (bot, -1)):
        # band-pass of the height map: features of about 1-5 mm (buttons, their gaps and rims) stay; scan noise
        # (finer) and the dome or wedge of the body (coarser) go
        good = np.isfinite(h)
        fill = np.where(good, h, 0.0)
        wsum = ndimage.gaussian_filter(good.astype(float), 0.7 / cell)
        fine = ndimage.gaussian_filter(fill, 0.7 / cell) / np.maximum(wsum, 1e-6)
        coarse = ndimage.gaussian_filter(fill, 3.0 / cell) / np.maximum(ndimage.gaussian_filter(good.astype(float), 3.0 / cell), 1e-6)
        r = np.abs(fine - coarse)[inner & good]
        r = r[r < 2.0]                         # larger jumps are scan holes (the other face shows through)
        rough.append(float(np.mean(np.minimum(r, 1.2))) if len(r) else 0.0)
    rt, rb = rough
    if rb > rt:
        return (rb / max(rt, 1e-6)), True
    return (rt / max(rb, 1e-6)), False


def to_key(mesh_scan, frame):
    m = mesh_scan.copy()
    m.apply_transform(frame.matrix())
    return m


def ring_hole(mesh_key, frame, cell=0.25, n=600000):
    """A through hole (along z) near the ring end: the key-ring hole. Returns dict(found, x, y, r)."""
    P, _ = geom.sample_surface(mesh_key, n, seed=5)
    x0, x1 = frame.body_x
    L = x1 - x0
    P = P[P[:, 0] < x0 + 0.45 * L]
    if len(P) < 100:
        return dict(found=False)
    lo = P[:, :2].min(0) - 2 * cell
    ij = ((P[:, :2] - lo) / cell).astype(int)
    shape = ij.max(0) + 3
    occ = np.zeros(shape, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    occ = ndimage.binary_closing(occ, iterations=1)
    lab, nl = ndimage.label(~occ)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    best = None
    for k in range(1, nl + 1):
        if k in border:
            continue
        cells = np.argwhere(lab == k)
        area = len(cells) * cell * cell
        if 2.0 <= area <= 120.0:
            if best is None or area > best[0]:
                best = (area, cells)
    if best is None:
        return dict(found=False)
    area, cells = best
    c = lo + (cells.mean(0) + 0.5) * cell
    r = float(np.sqrt(area / np.pi)) + cell / 2
    return dict(found=True, x=round(float(c[0]), 2), y=round(float(c[1]), 2), r=round(r, 2))


# ------------------------------------------------------------------------------------------------ calipers
def caliper_check(scan_sizes, cal):
    """Compare the scan's body length, width and thickness with caliper readings (mm). Width and thickness set the
    suggested uniform scale (least squares through zero); length (where the
    body ends is a judgement on both sides) is reported only. Returns a dict for the report and the window."""
    names = ('length', 'width', 'thickness')
    sc = dict(zip(names, scan_sizes))
    rows = []
    for k in names:
        if cal.get(k):
            rows.append(dict(name=k, scan=round(sc[k], 2), caliper=float(cal[k]),
                             diff_pct=round(100.0 * (sc[k] / float(cal[k]) - 1.0), 2)))
    if not rows:
        return dict(rows=[], scale=1.0, note='')
    use = [r for r in rows if r['name'] != 'length']
    if not use:
        use = rows
    s = sum(r['caliper'] * r['scan'] for r in use) / sum(r['scan'] ** 2 for r in use)
    L = next((r for r in rows if r['name'] == 'length'), None)        # reported, never used for the scale
    resid = max(abs(r['caliper'] - r['scan'] * s) for r in use)
    note = ''
    plausible = abs(s - 1) <= 0.02    # scanner scale errors are tenths of a percent; more means a reading or the scan is off
    if not plausible:
        note = ('The scan and the calipers differ by %.1f %%. That is more than a scanner scale error: check the readings '
                'and the units. The scan will not be rescaled. ' % (abs(s - 1) * 100))
    elif abs(s - 1) >= 0.003:          # below 0.3 % the readings themselves (about 0.05 mm on plastic) do not tell
        note = 'The scan is %.2f %% %s than the calipers. Rescale by x%.4f?' % (abs(s - 1) * 100,
                                                                              'smaller' if s > 1 else 'larger', s)
    if resid > 0.25:
        note += ' The sizes disagree by up to %.2f mm after scaling: check the readings or the scan.' % resid
    if L and L not in use and abs(L['diff_pct']) > 1.5:
        note += ' Length differs by %.1f %%; it is not used for the scale (where the body ends is a judgement).' % L['diff_pct']
    return dict(rows=rows, scale=round(float(s), 5), residual_mm=round(float(resid), 3), used=[r['name'] for r in use],
                note=note.strip(), plausible=bool(plausible))
