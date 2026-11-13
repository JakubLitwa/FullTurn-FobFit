"""Checks of a finished jacket against a key, on the exported mesh (not on the fields that built it).

Signed distance to the jacket: nearest of dense surface samples of the jacket, sign from the face normals of the
4 nearest samples (the jacket is a clean manifold, so its normals point outward). Negative = inside material.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import geom


class JacketSDF:
    """Signed distance to the jacket (negative inside material). Distance: nearest of dense surface samples.
    Sign: a voxel stencil of the mesh (0.2 mm); within 0.25 mm of the surface, the nearest face's normal."""

    def __init__(self, part, n=2_000_000, seed=21, pitch=0.2):
        pts, fid = geom.sample_surface(part, n, seed=seed)
        self.part = part
        self.pts = pts
        self.nrm = part.face_normals[fid]
        self.tree = cKDTree(pts)
        self.lo = part.bounds[0] - 1.0
        self.pitch = pitch
        self.dims = tuple(int(v) for v in np.ceil((part.bounds[1] + 1.0 - self.lo) / pitch) + 1)
        self.mask = geom.inside_mask(part, self.lo, pitch, self.dims)

    def inside(self, q):
        ijk = np.rint((q - self.lo) / self.pitch).astype(np.int64)
        ok = np.all(ijk >= 0, axis=1) & (ijk[:, 0] < self.dims[0]) & (ijk[:, 1] < self.dims[1]) & (ijk[:, 2] < self.dims[2])
        out = np.zeros(len(q), bool)
        out[ok] = self.mask[ijk[ok, 2], ijk[ok, 1], ijk[ok, 0]]
        return out

    def enclosed(self, q):
        """Exact inside test (VTK ray casting) for a few points."""
        import vtk
        from vtk.util import numpy_support as ns
        if not hasattr(self, '_surf'):
            self._surf = geom.tm_to_vtk(self.part)
        pts = vtk.vtkPoints()
        pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(q, dtype=np.float64), deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        sel = vtk.vtkSelectEnclosedPoints()
        sel.SetInputData(pd)
        sel.SetSurfaceData(self._surf)
        sel.SetTolerance(1e-6)
        sel.Update()
        arr = ns.vtk_to_numpy(sel.GetOutput().GetPointData().GetArray('SelectedPoints'))
        return arr.astype(bool)

    def __call__(self, q, chunk=500_000):
        q = np.asarray(q, float)
        out = np.empty(len(q))
        for a in range(0, len(q), chunk):
            qq = q[a:a + chunk]
            d, j = self.tree.query(qq, workers=-1)
            ins = self.inside(qq)
            near = np.flatnonzero(d < 0.3)            # the stencil is 0.2 mm: test these exactly
            if len(near):
                ins[near] = self.enclosed(qq[near])
            out[a:a + chunk] = np.where(ins, -d, d)
        return out


def mesh_validity(part):
    import manifold3d as m3
    man = geom.to_man(part)
    bodies = part.split(only_watertight=False)
    return dict(watertight=bool(part.is_watertight), winding_consistent=bool(part.is_winding_consistent),
                manifold_status=str(man.status()), bodies=len(bodies), volume_cm3=round(float(part.volume) / 1000, 3),
                mass_g_solid=round(float(part.volume) / 1000 * 1.27, 2), triangles=int(len(part.faces)),
                euler=int(part.euler_number), size_mm=[round(float(v), 1) for v in part.extents])


def clearance(sdf, key_pts, c, reach=0.8):
    """Gap from key surface points to the jacket, for the points the jacket covers (gap < c + reach)."""
    g = sdf(key_pts)
    cov = g < c + reach
    gg = g[cov]
    if len(gg) == 0:
        return dict(points=0), cov, g
    st = dict(points=int(cov.sum()), covered_pct=round(100.0 * cov.mean(), 1), target=c,
              mean=round(float(gg.mean()), 3), p05=round(float(np.percentile(gg, 5)), 3),
              p50=round(float(np.percentile(gg, 50)), 3), p95=round(float(np.percentile(gg, 95)), 3),
              min=round(float(gg.min()), 3), max=round(float(gg.max()), 3),
              within_0_1_pct=round(100.0 * float(np.mean(np.abs(gg - c) <= 0.1)), 1),
              interference_pts=int((gg < -0.02).sum()))
    return st, cov, g


def snap_in(sdf, key_pts, lip_x, ramp, step=0.2, z_top=None, tol=0.06):
    """Move the key out of the jacket along +z (the reverse of snapping in) and record, for each key point, the
    deepest it gets into jacket material. Interference should appear only at the lips (x inside the lip zone and
    its ramps) and be at most the designed undercut. A point stops being tracked once it is above the jacket."""
    seat = sdf(key_pts)
    z_top = float(z_top if z_top is not None else key_pts[:, 2].max())
    near = seat < 3.0                 # only points that start near the jacket can touch it on the way out
    idx = np.flatnonzero(near)
    q0 = key_pts[idx]
    w = np.zeros(len(q0))
    t = step
    active = np.arange(len(q0))
    while len(active):
        q = q0[active] + [0.0, 0.0, t]
        d = sdf(q)
        w[active] = np.maximum(w[active], -d)
        active = active[q[:, 2] < z_top + 0.5]
        t += step
    worst = np.zeros(len(key_pts))
    worst[idx] = w
    x = key_pts[:, 0]
    lipzone = (x >= lip_x[0] - ramp) & (x <= lip_x[1] + ramp)
    hit = worst > tol
    out = dict(max_mm=round(float(worst.max()), 3),
               max_in_lip_zone_mm=round(float(worst[lipzone].max()) if lipzone.any() else 0.0, 3),
               max_outside_lip_zone_mm=round(float(worst[~lipzone].max()) if (~lipzone).any() else 0.0, 3),
               points=int(len(key_pts)), points_hit=int(hit.sum()), points_hit_outside_lip_zone=int((hit & ~lipzone).sum()),
               seated_min_gap_mm=round(float(seat.min()), 3), step_mm=step, tolerance_mm=tol)
    if hit.any():
        k = int(np.argmax(worst))
        out['deepest_at'] = [round(float(v), 2) for v in key_pts[k]]
        o = np.flatnonzero(hit & ~lipzone)
        if len(o):
            out['outside_at'] = [[round(float(v), 2) for v in key_pts[i]] for i in o[np.argsort(-worst[o])][:5]]
    return out, worst


def hood_path(sdf, key_pts, hinge, psi, S, lip_x, ramp, step=0.25, z_top=None, tol=0.1, n=24000, seed=5):
    """Take the key out of the exported jacket along the hooked removal path (the reverse of putting it on): the
    tail turns up by psi about the hinge (x, z), the key slides S back along its tilted axis, then it lifts
    straight up. For each key surface point, the deepest it gets into jacket material on the way, and where (x).
    Interference should appear only at the lips (deepest contact in the lip zone or its ramps) and be about the
    designed undercut; the nose cap must not be hit. Steps: step mm at the farthest point."""
    seat = sdf(key_pts)
    idx = np.flatnonzero(seat < 3.0)
    if len(idx) > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    q0 = key_pts[idx]
    cx, cz = float(hinge[0]), float(hinge[1])
    dx, dz = q0[:, 0] - cx, q0[:, 2] - cz
    r = np.hypot(dx, dz)
    n1 = max(1, int(np.ceil(psi * float(r.max()) / step)))
    n2 = max(1, int(np.ceil(S / step)))
    z_top = float(z_top if z_top is not None else sdf.part.bounds[1][2])

    def pose(a, s, t, i):
        ca, sa = np.cos(a), np.sin(a)
        q = q0[i].copy()
        q[:, 0] = cx + ca * dx[i] - sa * dz[i] + s * np.cos(psi)
        q[:, 2] = cz + sa * dx[i] + ca * dz[i] + s * np.sin(psi) + t
        return q

    # the poses in order: turn, slide, lift; each point moves at most disp mm from one pose to the next, so a point
    # that is d mm clear of the jacket cannot touch it in the next floor(d / disp) poses and is not tested there
    end_z = pose(psi, S, 0.0, np.arange(len(q0)))[:, 2]
    n3 = max(1, int(np.ceil((z_top + 0.5 - float(end_z.min())) / step)))
    phases = [((lambda k: (psi * k / n1, 0.0, 0.0)), n1, psi / n1 * r),
              ((lambda k: (psi, S * k / n2, 0.0)), n2, np.full(len(q0), S / n2)),
              ((lambda k: (psi, S, step * k)), n3, np.full(len(q0), step))]
    w = np.zeros(len(q0))
    wx = q0[:, 0].copy()              # where along x each point is deepest in material (lip or not is judged there)
    tests = 0
    for par, nk, disp in phases:
        skip = np.zeros(len(q0), dtype=np.int64)
        for k in range(1, nk + 1):
            i = np.flatnonzero(skip <= 0)
            skip -= 1
            if not len(i):
                continue
            q = pose(*par(k), i)
            d = sdf(q)
            tests += len(i)
            deeper = -d > w[i]
            w[i[deeper]] = -d[deeper]
            wx[i[deeper]] = q[deeper, 0]
            skip[i] = np.floor(np.maximum(d, 0.0) / np.maximum(disp[i], 1e-6)).astype(np.int64)
    lipzone = (wx >= lip_x[0] - ramp) & (wx <= lip_x[1] + ramp)
    hit = w > tol
    out = dict(hinge=[round(cx, 2), round(cz, 2)], tilt_deg=round(float(np.degrees(psi)), 1), slide_mm=S,
               poses=n1 + n2 + n3, points=int(len(q0)), point_tests=int(tests), step_mm=step, tolerance_mm=tol,
               max_in_lip_zone_mm=round(float(w[lipzone].max()) if lipzone.any() else 0.0, 3),
               max_outside_lip_zone_mm=round(float(w[~lipzone].max()) if (~lipzone).any() else 0.0, 3),
               points_hit_outside_lip_zone=int((hit & ~lipzone).sum()))
    o = np.flatnonzero(hit & ~lipzone)
    if len(o):
        o = o[np.argsort(-w[o])][:5]
        out['outside_at'] = [[round(float(v), 2) for v in q0[i]] for i in o]
        out['outside_contact_x'] = [round(float(wx[i]), 2) for i in o]
    return out


def zones_free(sdf, zones, P, bbox, n=40000, seed=4, fold_pts=None):
    """Material found inside the keep-out zones: points sampled in each zone (inside the jacket's box), the depth of
    the deepest one inside jacket material (0 = free). With fold_pts (the part of the key that turns), the fold is
    also checked by turning those points through the whole fold travel."""
    rng = np.random.default_rng(seed)
    lo, hi = bbox
    res = {}

    def sample(fn):
        q = rng.uniform(lo, hi, (n * 8, 3))
        q = q[fn(q)]
        return q[:n]
    ring = zones.get('ring')
    if ring and ring.get('found', True) and ring.get('x') is not None:
        R = float(ring['r']) + P.ring_clear * 0.9
        q = sample(lambda q: np.hypot(q[:, 0] - ring['x'], q[:, 1] - ring['y']) < R)
        if len(q):
            d = sdf(q)
            res['ring'] = dict(points=len(q), deepest_mm=round(float(max(0.0, -d.min())), 3))
    for i, b in enumerate(zones.get('buttons') or []):
        p, nn = np.asarray(b['p'], float), np.asarray(b['n'], float)
        nn /= np.linalg.norm(nn)
        r = float(b.get('r') or P.button_r) * 0.95

        def inb(q):
            v = q - p
            t = v @ nn
            return (t > -1.0) & (t < 9.5) & (np.linalg.norm(v - np.outer(t, nn), axis=1) < r)
        q = sample(inb)
        if len(q):
            d = sdf(q)
            res['button_%d' % (i + 1)] = dict(points=len(q), deepest_mm=round(float(max(0.0, -d.min())), 3))
    fold = zones.get('fold')
    if fold:
        pv = np.asarray(fold['pivot'], float)
        s = np.sign(fold['side']) or 1

        def inf(q):
            return ((np.abs(q[:, 2] - pv[2]) < float(fold.get('half_thick', 1.4)) + P.clearance) &
                    (np.hypot(q[:, 0] - pv[0], q[:, 1] - pv[1]) < float(fold['length'])) &
                    (s * (q[:, 1] - pv[1]) > -float(fold.get('half_width', 4.5))))
        q = sample(inf)
        if len(q):
            d = sdf(q)
            res['fold'] = dict(points=len(q), deepest_mm=round(float(max(0.0, -d.min())), 3))
        if fold_pts is not None and len(fold_pts):
            sw = fold_sweep_check(sdf, fold_pts, fold, bbox)
            if sw:
                old = res.get('fold', {}).get('deepest_mm', 0.0)
                res['fold'] = dict(res.get('fold', {}), **sw, deepest_mm=round(max(old, sw['swept_deepest_mm']), 3))
    return res


def fold_sweep_check(sdf, pts, fold, bbox, step_deg=1.0, voxel=0.4):
    """Turn the part of the key that turns about the pivot from open to folded, step_deg at a time, and measure the
    jacket around it at every step: the smallest gap (negative = inside material), the angle and the place."""
    pv = np.asarray(fold['pivot'], float)
    s = float(np.sign(fold['side']) or 1)
    travel = float(fold.get('travel_deg', 180.0))
    _, idx = np.unique(np.floor(pts / voxel).astype(np.int64), axis=0, return_index=True)
    p = np.asarray(pts, float)[idx]
    rel = p - pv
    lo, hi = np.asarray(bbox[0], float) - 1.0, np.asarray(bbox[1], float) + 1.0
    best = (np.inf, None, None)
    for deg in np.arange(0.0, travel + 1e-6, step_deg):
        a = np.radians(deg) * s
        c, sn = np.cos(a), np.sin(a)
        q = np.c_[pv[0] + c * rel[:, 0] - sn * rel[:, 1], pv[1] + sn * rel[:, 0] + c * rel[:, 1], p[:, 2]]
        q = q[np.all((q >= lo) & (q <= hi), axis=1)]
        if not len(q):
            continue
        d = sdf(q)
        j = int(np.argmin(d))
        if d[j] < best[0]:
            best = (float(d[j]), float(deg), q[j])
    if best[1] is None:
        return None
    return dict(swept_points=int(len(p)), swept_deg=travel, swept_step_deg=step_deg,
                swept_min_gap_mm=round(best[0], 3), swept_worst_deg=best[1],
                swept_worst_at=[round(float(v), 2) for v in best[2]],
                swept_deepest_mm=round(max(0.0, -best[0]), 3))


def wing_geometry(part, axis_p, axis_d, mount, wing_info):
    """Wing direction and lever arm measured on the mesh. axis_p, axis_d: a point on the blade axis and its
    direction in the frame of `part` (the true axis in the tests, the detected one in a run).
    Returns the angle between the wing's span and the plane square to the blade axis, and the lever arm."""
    v = part.vertices
    d = np.asarray(axis_d, float) / np.linalg.norm(axis_d)
    style = wing_info['style']
    ys = max(mount.y_right, -mount.y_left)
    if style == 'loop':
        z_hi = mount.z_bed + 2 * (wing_info['ring_id_mm'] / 2 + wing_info['ring_wall_mm']) + 6
        far = np.abs(v[:, 1]) > ys + 4.0
    else:
        far = (np.abs(v[:, 1]) > ys + 6.0) & (v[:, 2] < mount.z_bed + wing_info['thick_mm'] + 0.2)   # the plate only
    w = v[far]
    if len(w) < 20:
        return dict(ok=False)
    out = dict(ok=True)
    if style != 'loop':
        # span direction from two sections across the lobe (the flat lobe has vertices only at its corners): near the
        # root and through the centre of the round tip; the midpoint of each section is on the lobe's centre line
        angs, levers = [], []
        zt = mount.z_bed + wing_info['thick_mm'] * 0.5
        for s in (1, -1):
            lob = w[np.sign(w[:, 1]) == s]
            if len(lob) < 20:
                continue
            y_tip = float(np.max(s * lob[:, 1]))
            mids = []
            for yy in (ys + 8.0, y_tip - wing_info['tip_r_mm']):
                sec = part.section(plane_origin=[0, s * yy, 0], plane_normal=[0, 1, 0])
                if sec is None:
                    break
                pts = np.vstack([e for e in sec.discrete])
                pts = pts[np.abs(pts[:, 2] - zt) < wing_info['thick_mm'] * 0.5]
                if len(pts) < 2:
                    break
                mids.append(np.array([0.5 * (pts[:, 0].min() + pts[:, 0].max()), s * yy, 0.0]))
            if len(mids) < 2:
                continue
            span = mids[1] - mids[0]
            angs.append(float(np.degrees(np.arcsin(min(1.0, abs(span @ d) / np.linalg.norm(span))))))
            tipc = np.array([mids[1][0], s * (y_tip - wing_info['tip_r_mm']), axis_p[2]])
            rel = tipc - axis_p
            levers.append(float(np.linalg.norm(rel - (rel @ d) * d)))
        if not angs:
            return dict(ok=False)
        out.update(square_error_deg=round(float(max(angs)), 3), lever_mm=[round(l, 2) for l in levers])
    else:
        zc = wing_info['hole_centre_z']
        yc = wing_info['hole_centre_y']
        rel = np.array([float(mount.x_c), yc, zc]) - axis_p
        out.update(lever_mm=[round(float(np.linalg.norm(rel - (rel @ d) * d)), 2)])
        # ring plane square to the axis: the hole's axis is the extrusion direction (x)
        ring = w[np.abs(w[:, 1] - yc) < wing_info['ring_id_mm'] / 2 + wing_info['ring_wall_mm'] + 0.5]
        c, ax = np.linalg.eigh(np.cov((ring - ring.mean(0)).T))
        normal = ax[:, 0]                         # thinnest direction of the ring = its hole axis
        out['square_error_deg'] = round(float(np.degrees(np.arccos(min(1.0, abs(normal @ d))))), 3)
    return out


def _raster(loops, y0, z0, res, shape):
    """Even-odd fill of closed (y, z) loops into a (nz, ny) bool image."""
    from PIL import Image, ImageDraw
    acc = np.zeros(shape, bool)
    for lp in loops:
        if len(lp) < 3:
            continue
        im = Image.new('1', (shape[1], shape[0]), 0)
        ImageDraw.Draw(im).polygon([((y - y0) / res, (z - z0) / res) for y, z in lp], fill=1, outline=1)
        acc ^= np.asarray(im, bool)
    return acc


def snap_in_2d(part, key, x0, x1, lip_x, ramp, step_x=1.0, res=0.04):
    """Snap-in check on sections of the exported jacket and the key, every step_x along x.
    For each section, the key's section is lifted along +z (the reverse of snapping in) and, for each side, the
    largest horizontal overlap with jacket material is found: how far that side must flex to let the key pass.
    Inside the lip zone it should be about the designed undercut; outside it, zero."""
    rows = []
    for xs in np.arange(x0 + 0.5, x1 - 0.49, step_x):
        try:
            js = part.section(plane_origin=[xs, 0, 0], plane_normal=[1, 0, 0])
            ks = key.section(plane_origin=[xs, 0, 0], plane_normal=[1, 0, 0])
        except Exception:
            continue
        if js is None or ks is None:
            continue
        jl = [e[:, 1:3] for e in js.discrete]
        kl = [e[:, 1:3] for e in ks.discrete]
        allp = np.vstack(jl + kl)
        y0, z0 = allp.min(0) - 1.0
        y1, z1 = allp.max(0) + 1.0
        shape = (int((z1 - z0) / res) + 2, int((y1 - y0) / res) + 2)
        J = _raster(jl, y0, z0, res, shape)
        K = _raster(kl, y0, z0, res, shape)
        kany = K.any(axis=1)
        kr = np.flatnonzero(kany)
        if len(kr) == 0:
            continue
        kmin = np.where(kany, np.argmax(K, axis=1), -1)
        kmax = np.where(kany, shape[1] - 1 - np.argmax(K[:, ::-1], axis=1), -1)
        widest = kr[np.argmax(kmax[kr] - kmin[kr])]
        jc = int((kmin[widest] + kmax[widest]) // 2)
        out_r = out_l = 0
        nz = shape[0]
        cols = np.arange(shape[1])
        for t in range(1, nz - kr[0]):
            Kt = np.zeros_like(K)
            Kt[t:] = K[:nz - t]                      # the key lifted by t rows
            O = Kt & J                               # key cells that would sit in jacket material
            if not O.any():
                if t > (kr[-1] - kr[0]) + 5 and not Kt.any():
                    break
                continue
            rows_o = np.flatnonzero(O.any(axis=1))
            for z in rows_o:
                oc = cols[O[z]]
                kc = cols[Kt[z]]
                r_o, l_o = oc[oc > jc], oc[oc <= jc]
                if len(r_o):                          # right side material must move out past the key's cells
                    out_r = max(out_r, int(kc.max() - r_o.min() + 1))
                if len(l_o):
                    out_l = max(out_l, int(l_o.max() - kc.min() + 1))
        rows.append((float(xs), out_r * res, out_l * res))
    a = np.array(rows) if rows else np.zeros((0, 3))
    if not len(a):
        return dict(sections=0)
    inz = (a[:, 0] >= lip_x[0] - ramp) & (a[:, 0] <= lip_x[1] + ramp)
    core = (a[:, 0] >= lip_x[0]) & (a[:, 0] <= lip_x[1])
    flex = np.maximum(a[:, 1], a[:, 2])
    out = dict(sections=int(len(a)), step_mm=step_x, raster_mm=res,
               flex_in_lip_zone_max_mm=round(float(flex[inz].max()), 3) if inz.any() else 0.0,
               flex_lip_core_min_mm=round(float(np.minimum(a[core, 1], a[core, 2]).min()), 3) if core.any() else 0.0,
               flex_outside_lip_zone_max_mm=round(float(flex[~inz].max()), 3) if (~inz).any() else 0.0,
               profile=[[round(r[0], 1), round(r[1], 3), round(r[2], 3)] for r in a])
    if (~inz).any() and flex[~inz].max() > 0:
        out['worst_outside_x'] = round(float(a[~inz][np.argmax(flex[~inz]), 0]), 1)
    return out
