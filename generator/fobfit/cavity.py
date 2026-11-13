"""The jacket body around the key: signed distance field of the key, cavity with lips, keep-out zones, outer shape,
fit coupon, and the lip calculation. Works in the key frame (scan.py), grid arrays are indexed [z, y, x].

Shapes, as level sets of fields that are negative inside:
  phi        key signed distance (mm)
  K_c        phi < c                         the key grown by the clearance c
  sweep      cumulative min of phi - c along +z: everything the key passes through when it snaps in from +z
  band       the lip zones: columns within u inside the key's widest contour, on the long sides
  cavity     K_c, or (sweep and not band)    so the key drops in freely except at the lips
  outer      below the top of K_{c+w}, above the flat bed (the back prints flat, the rim follows the key's edge)
  jacket     outer and not cavity and not keep-out (ring hole, buttons, fold travel)
"""
from dataclasses import dataclass, asdict, field
import time
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from . import geom

# printed PETG, assumed (conservative; molded PETG is about 2 GPa): see BUILD_RECORD.md, lip calculation
E_PETG = 1600.0       # MPa
MU = 0.3              # PETG on the key's plastic
STRAIN_LIMIT = 0.015  # repeated snapping, bending across the layers
RHO_PETG = 1.27       # g/cm3


@dataclass
class Params:
    pitch: float = 0.2          # grid step (mm); the preview uses 0.4
    clearance: float = 0.30     # gap between key and cavity
    wall: float = 1.8           # side wall and rim, normal to the key surface
    back: float = 1.6           # thinnest part of the back under the key (a domed key makes the rest thicker)
    undercut: float = 0.0       # how far the lips reach inside the key's widest contour; 0 = choose (lip_choice)
    rim: float = 1.6            # rim and lip thickness above the key (caps the outer body)
    lip_end_gap: float = 7.0    # no undercut within this distance of the body ends
    lip_ramp: float = 4.0       # undercut ramps to zero over this length
    lip_land: float = 0.6       # straight part of the lip tip before its 45 degree lead-in
    end_wall: float = 1.8       # wall in front of the body at the blade end
    smooth: float = -1.0        # smoothing of the scan (mm, Gaussian sigma); -1 = from the scan noise
    close: float = 2.5          # scan holes up to about twice this are bridged
    ring_clear: float = 2.5     # room for the key-ring wire around the ring hole
    button_r: float = 6.0       # default radius of a button keep-out
    hood: bool = True           # nose hook: a cap over the ring-end slope; the key comes out tail first (hood_sweep)
    hood_len: float = 2.5       # hinge of the removal path this far behind the ring-end face (cap length)
    hood_tilt: float = 0.30     # removal: tail up by this angle (rad) about the hinge ...
    hood_slide: float = 3.5     # ... then this far back along the tilted key, then straight up
    hood_wall: float = 3.0      # cap thickness normal to the slope (it carries the whole turning load on the fold side)

    def to_dict(self):
        return asdict(self)



def smax(a, b, k):
    """Smooth maximum (polynomial, radius k): rounds the crease where two fields meet, so the iso-surface of a
    crease running at a shallow angle to the grid is smooth instead of beaded. Never below max(a, b)."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * (k * 0.25)


def smin(a, b, k):
    return -smax(-a, -b, k)


# ------------------------------------------------------------------------------------------------ grid and field
def grid_for(mesh_key, body_x, P, extra=0.0):
    x0, x1 = body_x
    v = mesh_key.vertices
    sel = (v[:, 0] >= x0 - 1.0) & (v[:, 0] <= x1 + 3.0)
    vv = v[sel] if sel.sum() > 10 else v
    pad = P.clearance + max(P.wall, P.back, P.end_wall) + 2.0 + extra
    lo = np.array([x0 - pad, vv[:, 1].min() - pad, vv[:, 2].min() - pad])
    hi = np.array([x1 + P.clearance + P.end_wall + 2.0, vv[:, 1].max() + pad, vv[:, 2].max() + pad])
    dims = tuple(int(n) for n in np.ceil((hi - lo) / P.pitch) + 1)
    return lo, dims


def axes(lo, pitch, shape):
    nz, ny, nx = shape
    Z = (lo[2] + np.arange(nz) * pitch).astype(np.float32)[:, None, None]
    Y = (lo[1] + np.arange(ny) * pitch).astype(np.float32)[None, :, None]
    X = (lo[0] + np.arange(nx) * pitch).astype(np.float32)[None, None, :]
    return X, Y, Z


def key_sdf(mesh_key, lo, pitch, dims, close=2.5, sigma=0.25, band=3.5, progress=None, log=None):
    """Signed distance of the key on the grid (negative inside), accurate in a band around the surface.

    Inside/outside: morphological closing of the scan surface (bridges holes and narrow gaps up to ~2*close),
    flood-filled from the grid border; if the flood leaks through a large hole the closing radius grows.
    Distance: exact point-to-triangle distance to the triangle of the nearest surface sample; where the scan has a
    hole the bridged surface (mask boundary) is used. Then a Gaussian smoothing of sigma mm takes out scan noise."""
    t0 = time.time()
    log = log if log is not None else {}
    nx, ny, nz = dims
    lo = np.asarray(lo, float)
    hi = lo + (np.array(dims) - 1) * pitch
    cen = mesh_key.triangles_center
    sel = np.all(cen > lo - 3.0, axis=1) & np.all(cen < hi + 3.0, axis=1)
    sub = trimesh.Trimesh(mesh_key.vertices, mesh_key.faces[sel], process=False)
    ns = int(np.clip(sub.area / 0.07 ** 2, 3e5, 3e6))
    pts, fid = geom.sample_surface(sub, ns, seed=11)
    ijk = np.rint((pts - lo) / pitch).astype(np.int64)
    ok = np.all(ijk >= 0, axis=1) & (ijk[:, 0] < nx) & (ijk[:, 1] < ny) & (ijk[:, 2] < nz)
    occ = np.zeros((nz, ny, nx), bool)
    occ[ijk[ok, 2], ijk[ok, 1], ijk[ok, 0]] = True
    if progress:
        progress(0.15, 'distance field')
    d_app = (ndimage.distance_transform_edt(~occ) * pitch).astype(np.float32)
    # solid by closing; grow the radius if the flood leaks through a large hole
    hull_vol = None
    for r in (close, close * 1.6, close * 2.4):
        far = d_app > r
        lab, _ = ndimage.label(far)
        border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                           lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        border = border[border > 0]
        outside0 = np.isin(lab, border)
        del lab
        solid = (ndimage.distance_transform_edt(~outside0) * pitch) > r
        vol = solid.sum() * pitch ** 3
        if hull_vol is None:
            try:
                hull_vol = trimesh.convex.convex_hull(pts[ok][::20]).volume
            except Exception:
                hull_vol = vol
        log.setdefault('closing', []).append(dict(r=round(float(r), 2), solid_cm3=round(float(vol) / 1000, 2),
                                                  hull_cm3=round(float(hull_vol) / 1000, 2)))
        if vol > 0.45 * hull_vol:
            break
    log['close_used'] = float(r)
    if progress:
        progress(0.35, 'exact distances')
    # exact distances in a band: outside up to `band`, inside up to 1.5 mm (deeper values are only signs).
    # Voxels within 1 mm of the scan use all samples; farther ones a sparser set (KD cost grows with distance^2)
    selb = (d_app < band) & (~solid | (d_app < 1.5))
    idx = np.nonzero(selb)
    gp = np.stack([lo[0] + idx[2] * pitch, lo[1] + idx[1] * pitch, lo[2] + idx[0] * pitch], axis=1)
    near = d_app[idx] <= 1.0
    tri = sub.triangles
    fn = sub.face_normals
    d_t = np.empty(len(gp), np.float32)
    vote = np.zeros(len(gp), np.float32)
    for mask, step in ((near, 1), (~near, 8)):
        tree = cKDTree(pts[::step])
        fsub = fid[::step]
        ii = np.flatnonzero(mask)
        for a in range(0, len(ii), 1_000_000):            # chunks keep the memory modest
            sel_i = ii[a:a + 1_000_000]
            q = gp[sel_i]
            _, j = tree.query(q, workers=-1)
            t = tri[fsub[j]]
            cp = geom.closest_on_triangles(q, t[:, 0], t[:, 1], t[:, 2])
            d_t[sel_i] = np.linalg.norm(q - cp, axis=1)
            vote[sel_i] = np.einsum('ij,ij->i', q - cp, fn[fsub[j]])
    # where the scan has a hole the nearest sample is far: use the bridged surface (mask boundary, voxel accurate)
    d_in = ndimage.distance_transform_edt(solid)
    d_out = ndimage.distance_transform_edt(~solid)
    d_mb = (np.where(solid, d_in, d_out)[idx] * pitch - pitch / 2).astype(np.float32)
    del d_in, d_out
    hole = d_t > np.maximum(d_mb, 0) + max(0.45, 2.5 * pitch)       # the mask boundary is only voxel accurate
    d = np.where(hole, np.maximum(d_mb, 0) + 0.15, d_t)
    # sign: the closed mask, except within 1.5 voxels of the scan where the mask is up to half a voxel fat; there the
    # nearest triangle's normal decides, if the scan's normals agree with the mask where the mask is reliable
    inside = solid[idx]
    on_mask = np.abs(d_t - np.maximum(d_mb, 0)) < 1.5 * pitch     # mask boundary on the scan (not a bridged gap)
    ref = (d_t > 2.0 * pitch) & (d_t < 3.0 * pitch) & on_mask
    agree = float(np.mean((vote[ref] < 0) == inside[ref])) if ref.any() else 0.0
    log['normals_agree_pct'] = round(100 * agree, 2)
    if agree > 0.97:
        close_ = (d_t < 1.5 * pitch) & on_mask
        inside = np.where(close_, vote < 0, inside)
    phi = np.where(solid, -band, band).astype(np.float32)
    phi[idx] = np.where(inside, -d, d)
    log['bridged_voxels_pct'] = round(100.0 * float(np.mean(hole)), 2)
    if sigma and sigma > 0:
        phi = ndimage.gaussian_filter(phi, sigma / pitch, mode='nearest').astype(np.float32)
    log['sdf_s'] = round(time.time() - t0, 1)
    log['samples'] = int(ns)
    log['band_voxels'] = int(len(gp))
    return phi


def column_crossing(f, lo_z, pitch, lowest=True):
    """Per column (y, x): z of the lowest (or highest) point where f < 0, interpolated; nan where none."""
    neg = f < 0
    has = neg.any(axis=0)
    nz = f.shape[0]
    np.seterr(divide='ignore', invalid='ignore')
    if lowest:
        k = np.argmax(neg, axis=0)
        k0 = np.clip(k - 1, 0, nz - 1)
        f1 = np.take_along_axis(f, k[None], 0)[0]
        f0 = np.take_along_axis(f, k0[None], 0)[0]
        t = np.where(np.abs(f1 - f0) > 1e-9, f0 / (f0 - f1), 0.0)
        z = lo_z + (k0 + np.clip(t, 0, 1)) * pitch
    else:
        k = nz - 1 - np.argmax(neg[::-1], axis=0)
        k1 = np.clip(k + 1, 0, nz - 1)
        f0 = np.take_along_axis(f, k[None], 0)[0]
        f1 = np.take_along_axis(f, k1[None], 0)[0]
        t = np.where(np.abs(f1 - f0) > 1e-9, f0 / (f0 - f1), 0.0)
        z = lo_z + (k + np.clip(t, 0, 1)) * pitch
    return np.where(has, z, np.nan)


# ------------------------------------------------------------------------------------------------ zones
def zone_field(X, Y, Z, zones, P):
    """Keep-out zones as one field (negative inside). zones: dict(ring=dict(x,y,r) or None,
    buttons=[dict(p, n, r)], fold=dict(pivot, side, length, half_width, half_thick) or None)."""
    out = None

    def add(f):
        nonlocal out
        out = f if out is None else np.minimum(out, f)
    ring = zones.get('ring')
    if ring and ring.get('found', True) and ring.get('x') is not None:
        R = float(ring['r']) + P.ring_clear
        disc = np.sqrt((X - ring['x']) ** 2 + (Y - ring['y']) ** 2) - R
        strip = np.maximum(np.abs(Y - ring['y']) - R, X - ring['x'])
        add(np.broadcast_to(np.minimum(disc, strip), np.broadcast_shapes(X.shape, Y.shape, Z.shape)))
    for b in zones.get('buttons') or []:
        p = np.asarray(b['p'], np.float32)
        n = np.asarray(b['n'], np.float32)
        n = n / np.linalg.norm(n)
        r = float(b.get('r') or P.button_r)
        dx, dy, dz = X - p[0], Y - p[1], Z - p[2]
        t = dx * n[0] + dy * n[1] + dz * n[2]
        rad = np.sqrt(np.maximum((dx - t * n[0]) ** 2 + (dy - t * n[1]) ** 2 + (dz - t * n[2]) ** 2, 0))
        add(np.maximum(np.maximum(rad - r, -t - 1.5), t - 10.0))     # from 1.5 mm inside to 10 mm out
    fold = zones.get('fold')
    if fold:
        pv = np.asarray(fold['pivot'], np.float32)
        s = float(np.sign(fold['side']) or 1)
        hw = float(fold.get('half_width', 4.5)) + P.clearance
        ht = float(fold.get('half_thick', 1.4)) + P.clearance + 0.4
        # everything from just below the blade upward, inside the swept half disc: a slot through the wall only
        # would leave the wall above it hanging (a floating overhang when printed)
        f = np.maximum((pv[2] - ht) - Z, np.sqrt((X - pv[0]) ** 2 + (Y - pv[1]) ** 2) - (float(fold['length']) + 1.5))
        add(np.maximum(f, -(s * (Y - pv[1])) - hw))
        if fold.get('sweep'):
            add(fold_sweep_field(X, Y, Z, fold))
    return out


FOLD_TRAVEL_DEG = 180.0      # blade open (0) to folded into the body (180)
NO_ZONE = 999.0              # sweep bin with no keep-out


def fold_margin(P):
    return float(P.clearance) + 0.4


def fold_sweep_table(pts, fold, P, dr=0.5):
    """Swept keep-out of the turning blade, from the points of the part that turns (key frame). Radius bins about
    the pivot, dr wide; each bin takes every point whose margin reaches it and stores the lowest z less the margin
    (floor) and the angle span from the earliest start angle to the latest angle at the end of the travel, both
    grown by the margin (angles in the fold direction, radians). The keep-out over a bin is everything above its
    floor inside its span: the root sweeps out wider and lower than the blade, and nothing hangs over a cut."""
    pv = np.asarray(fold['pivot'], float)
    s = float(np.sign(fold['side']) or 1)
    m = fold_margin(P)
    travel = np.radians(float(fold.get('travel_deg', FOLD_TRAVEL_DEG)))
    dx, dy = pts[:, 0] - pv[0], pts[:, 1] - pv[1]
    keep = dx > 0                      # the turning part lies beyond the pivot along the blade
    if keep.sum() < 50:
        return None
    pts, dx, dy = pts[keep], dx[keep], dy[keep]
    r = np.hypot(dx, dy)
    ph = np.arctan2(s * dy, dx)
    r0 = max(0.0, float(np.floor((r.min() - m) / dr) * dr))
    nb = int(np.ceil((r.max() + m - r0) / dr))
    floor, a, b = [], [], []
    order = np.argsort(r)
    rs = r[order]
    for i in range(nb):
        r_in, r_out = r0 + i * dr, r0 + (i + 1) * dr
        j0, j1 = np.searchsorted(rs, r_in - m), np.searchsorted(rs, r_out + m, side='right')
        if j1 <= j0:
            floor.append(NO_ZONE), a.append(0.0), b.append(0.0)
            continue
        sel = order[j0:j1]
        dang = min(np.pi / 2, m / max(r_in, m))
        floor.append(round(float(pts[sel, 2].min() - m), 3))
        a.append(round(float(ph[sel].min() - dang), 5))
        b.append(round(float(ph[sel].max() + travel + dang), 5))
    return dict(r0=round(r0, 3), dr=dr, floor=floor, a=a, b=b, margin_mm=round(m, 3),
                travel_deg=round(float(np.degrees(travel)), 1), points=int(len(pts)))


def fold_sweep_field(X, Y, Z, fold):
    """The swept keep-out as a field (negative inside): bin floor in z, angle span as arc length, radial ends."""
    sw = fold['sweep']
    pv = np.asarray(fold['pivot'], np.float32)
    s = np.float32(np.sign(fold['side']) or 1)
    dx, dy = X - pv[0], Y - pv[1]
    r = np.sqrt(dx * dx + dy * dy)
    ph = np.arctan2(s * dy, dx)
    fl = np.asarray(sw['floor'], np.float32)
    A = np.asarray(sw['a'], np.float32)
    B = np.asarray(sw['b'], np.float32)
    nb, r0, dr = len(fl), np.float32(sw['r0']), np.float32(sw['dr'])
    i = np.floor((r - r0) / dr).astype(np.int64)
    inb = (i >= 0) & (i < nb)
    ic = np.clip(i, 0, nb - 1)
    a, span = A[ic], np.minimum(B[ic] - A[ic], np.float32(2 * np.pi))
    d = np.mod(ph - a, np.float32(2 * np.pi))
    ang = np.where(d <= span, -np.minimum(d, span - d), np.minimum(d - span, np.float32(2 * np.pi) - d)) * r
    rad = np.maximum(r0 - r, r - (r0 + nb * dr))
    f2 = np.where(inb & (fl[ic] < NO_ZONE), np.maximum(ang, rad), np.maximum(np.abs(rad), np.float32(0.5)))
    return np.maximum(fl[ic] - Z, f2).astype(np.float32)


# ------------------------------------------------------------------------------------------------ jacket
def undercut_profile(X, body_x, P, u=None):
    """Lip undercut along x: full in the middle of the body, zero within lip_end_gap of the ends, linear ramps."""
    u = P.undercut if u is None else u
    x0, x1 = body_x
    a, b = x0 + P.lip_end_gap, x1 - P.lip_end_gap
    if b - a < 2 * P.lip_ramp:
        m = (a + b) / 2
        a, b = m - P.lip_ramp, m + P.lip_ramp
    # inside [a, b] the full undercut; over the ramps down to 0.2 mm; beyond them -clearance, i.e. no material
    # inside the cavity's outline at all (a thinner lip end would be a sliver under a voxel wide)
    d = np.minimum(X - a, b - X)
    ramp = np.clip(1.0 + d / P.lip_ramp, 0.0, 1.0)
    uu = np.where(d >= -P.lip_ramp, 0.2 + (u - 0.2) * ramp, -P.clearance) if u > 0.2 else np.full_like(d, -P.clearance)
    return uu.astype(np.float32), (a, b)


def ring_face(phi, lo, pitch, body_x, frac=0.6):
    """x of the ring-end face of the main body: where the body's width first reaches frac of its largest width,
    coming from the ring end. A narrow ring lug beyond it is not wrapped by the jacket."""
    chi = phi.min(axis=0)
    xs = lo[0] + np.arange(phi.shape[2]) * pitch
    w = (chi < 0).sum(axis=0) * pitch
    sel = (xs >= body_x[0]) & (xs <= body_x[1])
    wmax = w[sel].max()
    k = np.flatnonzero(sel & (w >= frac * wmax))
    return float(xs[k[0]]) if len(k) else float(body_x[0])



def plan_distance(phi, pitch):
    """Signed distance in plan (x, y) to the key's footprint outline, negative inside. The column minimum of phi is
    a 3D distance: a short bump nearby makes a deep column look close to the edge. So: Euclidean distance transform
    of the footprint, with the column minimum only within 1.5 voxels of the outline (sub-voxel there)."""
    cm = phi.min(axis=0)
    inside = cm < 0
    d_in = ndimage.distance_transform_edt(inside) * pitch
    d_out = ndimage.distance_transform_edt(~inside) * pitch
    sd = np.where(inside, -(d_in - pitch / 2), d_out - pitch / 2)
    return np.where(np.abs(cm) < 1.5 * pitch, cm, sd).astype(np.float32)


def hood_hinge(phi, lo, pitch, P, x_face):
    """Hinge of the hooked removal path: hood_len behind the ring-end face, one clearance above the key top."""
    xs = lo[0] + np.arange(phi.shape[2]) * pitch
    Cx = float(x_face + P.hood_len)
    sel = np.abs(xs - Cx) <= 0.6 * pitch
    if not sel.any():                       # a slice of the field that does not reach the hinge (lip_choice)
        return Cx, float('nan')
    top = column_crossing(phi[:, :, sel], lo[2], pitch, lowest=False)
    if np.isnan(top).all():
        return Cx, float('nan')
    return Cx, float(np.nanmax(top)) + P.clearance


def hood_sweep(phi, lo, pitch, P, x_face, progress=None):
    """The key swept along the hooked removal path, as a field without clearance (min over the poses of phi).
    Path: the tail turns up by hood_tilt about the hinge C (the nose dips under the cap), the key slides hood_slide
    back along its tilted axis (the nose leaves the cap), then it lifts straight up. The motion is rigid, so the
    cavity for clearance c is this field minus c (the same hinge for every clearance of the fit coupon).
    Steps: rotation 0.5 voxel at the farthest point, slide 0.75 voxel."""
    Cx, Cz = hood_hinge(phi, lo, pitch, P, x_face)
    psi, S = float(P.hood_tilt), float(P.hood_slide)
    nx = phi.shape[2]
    rmax = float(np.hypot(max(abs(lo[0] - Cx), abs(lo[0] + (nx - 1) * pitch - Cx)), 22.0))
    n1 = max(1, int(np.ceil(psi / (0.5 * pitch / rmax))))
    n2 = max(1, int(np.ceil(S / (0.75 * pitch))))
    ph = phi.astype(np.float32)

    def pose(a, sv):
        cs, sn = np.cos(a), np.sin(a)
        ax = lo[0] - Cx - sv * cs
        az = lo[2] - Cz - sv * sn
        ox = (Cx - lo[0] + cs * ax + sn * az) / pitch
        oz = (Cz - lo[2] - sn * ax + cs * az) / pitch
        M3 = np.array([[cs, 0, -sn], [0, 1, 0], [sn, 0, cs]])
        return ndimage.affine_transform(ph, M3, offset=[oz, 0, ox], order=1, cval=50.0, mode='constant')

    sw = ph.copy()
    end = ph
    for k in range(1, n1 + 1):
        end = pose(psi * k / n1, 0.0)
        sw = np.minimum(sw, end)
        if progress and k % 20 == 0:
            progress(k / (n1 + n2))
    for k in range(1, n2 + 1):
        end = pose(psi, S * k / n2)
        sw = np.minimum(sw, end)
    sw = np.minimum(sw, np.minimum.accumulate(end, axis=0))
    info = dict(hinge=(round(Cx, 2), round(Cz, 2)), tilt_deg=round(float(np.degrees(psi)), 1), slide_mm=S,
                steps=(n1, n2), cap_x=(round(float(x_face - P.clearance - P.end_wall), 2), round(Cx, 2)),
                cap_wall_mm=P.hood_wall)
    return sw.astype(np.float32), info


def jacket_field(phi, lo, pitch, body_x, P, zones, u=None, xslab=None, x_face=None, z_bed=None, sweep0=None):
    """Jacket material field g (negative inside), the cavity field and facts for the report.
    xslab: (x_mid, half_length) keeps only a slab of the jacket (fit coupon).
    x_face: ring-end face of the main body (ring_face); the jacket ends one wall beyond it.
    sweep0: the key swept along its removal path without clearance (hood_sweep); computed here when P.hood and
    not given; without the hood the key comes out straight up."""
    c, w = P.clearance, P.wall
    nz, ny, nx = phi.shape
    X, Y, Z = axes(lo, pitch, phi.shape)
    x_face = body_x[0] if x_face is None else x_face
    lip_body = (x_face, body_x[1])
    chi = plan_distance(phi, pitch)[None]                         # (1, ny, nx): plan distance to the widest contour
    hinfo = None
    if P.hood:
        if sweep0 is None:
            sweep0, hinfo = hood_sweep(phi, lo, pitch, P, x_face)
        else:
            Cx, Cz = hood_hinge(phi, lo, pitch, P, x_face)
            hinfo = dict(hinge=(round(Cx, 2), round(Cz, 2)), tilt_deg=round(float(np.degrees(P.hood_tilt)), 1),
                         slide_mm=P.hood_slide, cap_x=(round(float(x_face - c - P.end_wall), 2), round(Cx, 2)),
                         cap_wall_mm=P.hood_wall)
        sweep = sweep0 - np.float32(c)
        # over the cap the shell is hood_wall thick normal to the slope; the vertical rim cap is lifted so it does
        # not thin the cap on the steep slope (blend over 1 mm behind the hinge)
        Cx = hinfo['hinge'][0]
        hf = np.clip((Cx + 1.5 - X) / 1.0, 0.0, 1.0).astype(np.float32)
        w_x = (w + hf * max(P.hood_wall - w, 0.0)).astype(np.float32)
        rim_x = (P.rim + hf * (c + 1.75 * P.hood_wall - P.rim)).astype(np.float32)
    else:
        sweep = np.minimum.accumulate(phi - c, axis=0)
        w_x, rim_x = np.float32(w), np.float32(P.rim)
    top_c = column_crossing(phi - c, lo[2], pitch, lowest=False)  # top of K_c per column
    top_a = np.where(np.isnan(top_c), -1e3, top_c).astype(np.float32)[None]   # actual top, for the lip's floor
    # Reference height of the key's top for the lips and the rim. Near the widest contour the side is close to
    # vertical, so the top of K_c jumps by millimetres between neighbouring columns (and with scan noise). Take it
    # only from columns at least 0.3 mm inside the contour, extend it outward from the nearest such column, smooth it.
    stable = ~np.isnan(top_c) & (chi[0] < -0.3)
    if stable.any() and (~stable).any():
        _, (ii, jj) = ndimage.distance_transform_edt(~stable, return_indices=True)
        top_ref = top_c[ii, jj]
    else:
        top_ref = np.where(np.isnan(top_c), 0.0, top_c)
    top_ref = ndimage.gaussian_filter(top_ref.astype(np.float32), 0.5 / pitch, mode='nearest').astype(np.float32)[None]
    top_c, top_ext = top_ref, top_ref
    uX, lipx = undercut_profile(X, lip_body, P, u)
    K = np.float32(0.3)                                           # crease rounding (smax)
    band = -uX + smax(np.float32(0.0), Z - top_c - P.lip_land, K) - chi   # negative inside the band (45 degree lead-in)
    # lip material only above the topmost key surface of its column: a channel or a groove between two shells in a
    # lip column (a flip key's blade channel) stays part of the sweep, so nothing blocks it
    band = np.maximum(band, top_a - Z)
    if P.hood:
        # beyond the lips toward the blade end the tilting key swings forward (+x) low down: there the swept path is
        # cut in full, so the corners of the blade end do not block it (no lip lives there)
        band = np.maximum(band, np.float32(10.0) * (X - np.float32(lipx[1] + P.lip_ramp + 0.5)))
    cav = np.minimum(phi - c, np.maximum(sweep, -band))          # hard: sweep equals phi - c under the key
    del sweep, band
    # the outside need not follow the scan: a smoother field (0.8 mm) shapes it, so scan noise leaves no streaks
    phs = ndimage.gaussian_filter(phi, 0.8 / pitch, mode='nearest')
    chis = phs.min(axis=0)[None]
    o_top = np.minimum.accumulate((phs - (c + w_x))[::-1], axis=0)[::-1]
    del phs
    if z_bed is None:
        z_bed = float(np.nanmin(column_crossing(phi - (c + P.back), lo[2], pitch, lowest=True)))
    outer = np.maximum(o_top, z_bed - Z)
    del o_top
    outer = np.maximum(outer, Z - (top_ext + rim_x))              # rim and lips: at most `rim` above the key
    ch = 0.8                                                      # 45 degree chamfers on the outer bottom and top edges
    outer = np.maximum(outer, ch - (Z - z_bed) - (c + w_x - chis))
    outer = np.maximum(outer, (Z - (top_ext + rim_x)) + (chis - (c + w_x)) + ch)
    x_trim = body_x[1] + c + P.end_wall
    x_trim_lo = x_face - c - P.end_wall
    outer = np.maximum(outer, np.maximum(X - x_trim, x_trim_lo - X))
    if xslab is not None:
        xm, hl = xslab
        outer = np.maximum(outer, np.abs(X - xm) - hl)
    g = smax(outer, -cav, K)
    del outer
    zf = zone_field(X, Y, Z, zones, P)
    if zf is not None:
        g = np.maximum(g, -zf)
    g = g.astype(np.float32)
    # one body: drop islands, fill enclosed voids
    solid = g < 0
    lab, n = ndimage.label(solid)
    info = dict(islands_dropped=0, voids_filled=0)
    if n > 1:
        cnt = np.bincount(lab.ravel())
        cnt[0] = 0
        keep = int(np.argmax(cnt))
        g = np.where(solid & (lab != keep), np.float32(0.05), g)
        info['islands_dropped'] = int(n - 1)
        info['island_volume_mm3'] = round(float((cnt.sum() - cnt[keep]) * pitch ** 3), 2)
    del lab
    air = g >= 0
    lab, n = ndimage.label(air)
    if n > 1:
        border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                           lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        enclosed = air & ~np.isin(lab, border)
        if enclosed.any():
            g = np.where(enclosed, np.float32(-0.05), g)
            info['voids_filled'] = int(n - len(border[border > 0]))
    del lab
    g = np.where(np.abs(g) < 1e-5, np.float32(1e-5), g)
    g[0], g[-1], g[:, 0], g[:, -1], g[:, :, 0], g[:, :, -1] = 1, 1, 1, 1, 1, 1
    info.update(z_bed=round(z_bed, 3), lip_x=(round(float(lipx[0]), 2), round(float(lipx[1]), 2)),
                x_trim=(round(float(x_trim_lo), 2), round(float(x_trim), 2)), x_face=round(float(x_face), 2))
    if hinfo is not None:
        info['hood'] = hinfo
    return g, cav.astype(np.float32), info


def mesh_field(g, lo, pitch, simplify=0.01):
    """Field -> watertight manifold through manifold3d (validates and simplifies). The grid border is closed."""
    g = np.where(np.abs(g) < 1e-5, np.float32(1e-5), g).astype(np.float32)
    g[0], g[-1], g[:, 0], g[:, -1], g[:, :, 0], g[:, :, -1] = 1, 1, 1, 1, 1, 1
    tm = geom.iso_mesh(g, lo, pitch)
    if len(tm.faces) == 0:
        raise RuntimeError('the jacket field is empty')
    man = geom.to_man(tm)
    if str(man.status()) != 'Error.NoError':
        raise RuntimeError('jacket mesh is not manifold: %s' % man.status())
    if simplify:
        man = man.simplify(simplify)
    return man


# ------------------------------------------------------------------------------------------------ lip calculation
def lip_calc(phi, g, lo, pitch, body_x, P, u_used):
    """Plane-strain estimate of the lips at the middle of the lip zone (BUILD_RECORD.md, lip calculation).

    The open-front section opens like a U: the back plate bends and the side walls bend. Per side, per mm of lip:
      compliance C = h^2/2 * integral(dy / (E I_back(y)))  +  integral((h - z)^2 / (E I_wall(z)) dz),   I = t^3/12
      deflection force P = u / C,  strain = 6 P h / (E t^2) at the thinnest back and 6 P (h - z) / (E t(z)^2) in the wall
    Snap-in force   W = 2 * P * L_lip * (mu + tan a) / (1 - mu tan a),   a = 45 degree lead-in
    Hold per side   V = P * L_lip * (tan b + mu) / (1 - mu tan b),        b = angle of the lip's holding face
    Cam-out torque  T = V * key width  (lips only; the side walls carry part of the torque too, so this is low)."""
    c = P.clearance
    X, Y, Z = axes(lo, pitch, phi.shape)
    x = (body_x[0] + body_x[1]) / 2
    i = int(np.clip(round((x - lo[0]) / pitch), 0, phi.shape[2] - 1))
    ph = phi[:, :, i]
    gg = g[:, :, i]
    mat = gg < 0
    zs = lo[2] + np.arange(ph.shape[0]) * pitch
    ys = lo[1] + np.arange(ph.shape[1]) * pitch
    chi = ph.min(axis=0)
    inside = np.flatnonzero(chi < 0)
    if len(inside) < 5:
        return dict(ok=False, note='no key section at the lip zone centre')
    y_left, y_right = ys[inside[0]], ys[inside[-1]]
    width = float(y_right - y_left)
    cav_rows = np.flatnonzero((ph < c).any(axis=0))
    out = dict(ok=True, x=round(float(x), 2), key_width=round(width, 2), y_left=round(float(y_left), 2),
               y_right=round(float(y_right), 2), sides={})
    # back thickness under the cavity, for each y across the key
    tb = []
    for j in range(inside[0], inside[-1] + 1):
        col = mat[:, j]
        kc = np.flatnonzero(ph[:, j] < c)
        if len(kc) == 0:
            continue
        below = col[:kc[0]]
        tb.append(float(below.sum() * pitch))
    tb = np.array(tb)
    tb = tb[tb > 0.2]
    if len(tb) == 0:
        return dict(ok=False, note='no back under the key')
    tb_min = float(tb.min())
    z_bed = float(zs[np.flatnonzero(mat.any(axis=1))[0]])
    z_mid_back = z_bed + tb_min / 2
    Ib_int = float(np.sum(pitch / (E_PETG * tb ** 3 / 12.0)))     # integral dy / (E I_back)
    total = dict(P=0.0, V=0.0)
    for name, sgn in (('left', -1), ('right', 1)):
        jc = inside[0] if sgn < 0 else inside[-1]
        # lip tip: innermost material above the key on this side
        rows = range(ph.shape[1])
        tip = None
        for j in (range(jc, inside[0] + (inside[-1] - inside[0]) // 2, -1) if sgn > 0 else
                  range(jc, inside[0] + (inside[-1] - inside[0]) // 2)):
            kc = np.flatnonzero(ph[:, j] < c)
            if len(kc) == 0:
                continue
            above = np.flatnonzero(mat[kc[-1] + 1:, j])
            if len(above):
                tip = (j, kc[-1] + 1 + above[0])
            else:
                break
        z_eq = float(zs[int(np.argmin(ph[:, jc]))])
        if tip is None:
            out['sides'][name] = dict(lip=False)
            continue
        jt, kt = tip
        y_tip = float(ys[jt])
        z_tip = float(zs[kt])
        reach = float(sgn * (ys[jc] - y_tip))                    # how far inside the widest contour
        h = z_tip - z_mid_back
        # wall thickness for each z from the back up to the tip, on this side
        tw = []
        for k in range(int((z_mid_back - lo[2]) / pitch), kt + 1):
            run = mat[k, jc:] if sgn > 0 else mat[k, :jc + 1][::-1]
            nzr = np.flatnonzero(run)
            if len(nzr) == 0:
                continue
            seg = run[nzr[0]:]
            end = np.flatnonzero(~seg)
            tw.append((zs[k], (end[0] if len(end) else len(seg)) * pitch))
        tw = np.array(tw) if tw else np.zeros((0, 2))
        zrel = tw[:, 0] - z_mid_back
        Iw = E_PETG * np.maximum(tw[:, 1], 0.4) ** 3 / 12.0
        C = h * h / 2.0 * Ib_int + float(np.sum((h - zrel) ** 2 / Iw * pitch))
        Pu = u_used / C                                          # N per mm of lip
        eps_b = 6 * Pu * h / (E_PETG * tb_min ** 2)
        eps_w = float(np.max(6 * Pu * np.maximum(h - zrel, 0) / (E_PETG * np.maximum(tw[:, 1], 0.4) ** 2))) if len(tw) else 0.0
        # holding face angle at the tip: slope of the K_c surface there
        k0 = max(kt - 1, 1)
        gz = (ph[k0 + 1, jt] - ph[k0 - 1, jt]) / (2 * pitch)
        gy = (ph[k0, min(jt + 1, ph.shape[1] - 1)] - ph[k0, max(jt - 1, 0)]) / (2 * pitch)
        nrm = np.hypot(gy, gz) or 1.0
        b = float(np.degrees(np.arcsin(np.clip(abs(gz) / nrm, 0, 1))))
        out['sides'][name] = dict(lip=True, reach_mm=round(reach, 2), tip_z=round(z_tip, 2), widest_z=round(z_eq, 2),
                                  lever_h_mm=round(h, 2), compliance_mm_per_N_mm=round(C, 3),
                                  force_N_per_mm=round(Pu, 3), strain_back_pct=round(100 * eps_b, 2),
                                  strain_wall_pct=round(100 * eps_w, 2), hold_angle_deg=round(b, 1))
        total['P'] += Pu
    out['back_min_mm'] = round(tb_min, 2)
    return out


def lip_forces(lc, lip_len, design_torque):
    """Snap-in force, hold and cam-out torque from lip_calc and the lip length (mm)."""
    if not lc.get('ok'):
        return {}
    a = np.radians(45.0)
    res = dict(lip_length_mm=round(lip_len, 1))
    W = 0.0
    T = []
    strains = []
    for s in lc['sides'].values():
        if not s.get('lip'):
            continue
        Pt = s['force_N_per_mm'] * lip_len
        W += Pt * (MU + np.tan(a)) / (1 - MU * np.tan(a))
        tb = np.tan(np.radians(min(s['hold_angle_deg'], 89.0)))
        den = 1 - MU * tb
        V = Pt * (tb + MU) / den if den > 0.05 else float('inf')
        T.append(V * lc['key_width'] / 1000.0)
        strains.append(max(s['strain_back_pct'], s['strain_wall_pct']))
    res['snap_in_N'] = round(W, 1)
    res['cam_out_Nm'] = round(min(T), 2) if T and np.isfinite(min(T)) else ('self-locking' if T else 0.0)
    res['strain_max_pct'] = round(max(strains), 2) if strains else 0.0
    res['design_torque_Nm'] = design_torque
    return res


def lip_choice(phi, lo, pitch, body_x, P, zones, torque, x_face=None, z_bed=None,
               candidates=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0), snap_max=120.0, margin=1.5, sweep0=None, fine=0.05):
    """Pick the undercut: the deepest lip the plastic safely takes (defect 41; the designer: "Use all strain possible").
    Walk up the candidates while the strain stays under STRAIN_LIMIT and the snap-in force under snap_max, stop at the
    first candidate that fails, then try `fine` steps between the last allowed candidate and that one. The pick is the
    deepest allowed undercut below the first failure; if even the first candidate fails, the first candidate. Each row
    also says whether the cam-out torque reaches margin x the design torque. Returns (u, table), the table by undercut."""
    table = []

    def row(u):
        g, _, info = jacket_field(phi, lo, pitch, body_x, P, zones, u=u, x_face=x_face, z_bed=z_bed, sweep0=sweep0)
        lc = lip_calc(phi, g, lo, pitch, (x_face if x_face is not None else body_x[0], body_x[1]), P, u)
        L = float(info['lip_x'][1] - info['lip_x'][0] + P.lip_ramp)
        f = lip_forces(lc, L, torque)
        cam = f.get('cam_out_Nm', 0.0)
        cam = 99.0 if cam == 'self-locking' else float(cam or 0.0)
        ok = f.get('strain_max_pct', 99) <= 100 * STRAIN_LIMIT and f.get('snap_in_N', 999) <= snap_max
        t = dict(u=u, cam_out_Nm=cam, snap_in_N=f.get('snap_in_N'), strain_pct=f.get('strain_max_pct'),
                 hold_deg=min([sd['hold_angle_deg'] for sd in lc.get('sides', {}).values() if sd.get('lip')] or [0]),
                 allowed=bool(ok), reaches_margin=bool(cam >= margin * torque))
        table.append(t)
        return t

    last, fail = None, None
    for u in candidates:
        t = row(u)
        if not t['allowed']:
            fail = u
            break
        last = t
    if last is None:
        return table[0]['u'], table
    if fail is not None and fine:
        for k in range(1, int(round((fail - last['u']) / fine))):
            t = row(round(last['u'] + fine, 3))
            if not t['allowed']:
                break
            last = t
    table.sort(key=lambda t: t['u'])
    return last['u'], table
