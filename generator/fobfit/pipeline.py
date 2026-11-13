"""One FobFit run: scan -> key frame -> distance field -> jacket + wing -> checks -> files.
The window (fobfit_gui.py) and the command line (fobfit_cli.py) both drive this module.
"""
from dataclasses import dataclass, field, asdict
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import manifold3d as m3

from . import geom, scan, cavity, wings, checks

M = m3.Manifold


@dataclass
class Job:
    scan_path: str
    out_dir: str = ''
    style: str = 'wing'
    lever: float = 0.0                 # 0 = the style's default
    side: int = 1                      # one-sided styles: +1 = +y side, -1 = -y side
    unit: str = 'auto'
    calipers: dict = field(default_factory=dict)   # length, width, thickness (mm)
    apply_scale: bool = False          # rescale the scan by the caliper result (never done silently)
    scale: float = 0.0                 # explicit scale (0 = none, or the caliper scale when apply_scale)
    flip_face: bool = False
    swap_ends: bool = False
    axis_points: list = None           # [tip, root] in scan mm, picked on the blade
    buttons: list = field(default_factory=list)    # dict(p, n, r) in the key frame
    ring: dict = None                  # None = detect; dict(x, y, r) = set; dict(found=False) = none
    fold: dict = None                  # dict(pivot, side, length) in the key frame
    params: dict = field(default_factory=dict)     # cavity.Params fields
    coupon: bool = False               # fit coupon: internal (tests); the window and the command line do not make it
    slice: bool = False
    preview_pitch: float = 0.3
    cloud_path: str = ''               # raw point cloud: '' = look next to the scan, 'none' = do not use one

    def to_dict(self):
        d = asdict(self)
        d.pop('coupon', None)          # internal, not a user setting
        return d


def peak_memory_mb():
    """Peak memory of this process so far (MB): Windows peak working set, else max resident size."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round((getattr(mi, 'peak_wset', 0) or mi.rss) / 2 ** 20)
    except Exception:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        return None


N_DEV = 200000          # scan surface samples for the fit check and the deviation map


def tool_out_dir(scan_path, style):
    """Default output folder: <tool folder>/out/<scan name>_<style>, so the scan folder stays untouched."""
    base = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
    return base / 'out' / ('%s_%s' % (Path(scan_path).stem, style))


class Run:
    """State of one run; each stage can be called again after a correction."""

    def __init__(self, job, progress=None):
        self.job = job
        self.P = cavity.Params(**{k: v for k, v in (job.params or {}).items() if hasattr(cavity.Params, k)})
        self.progress = progress or (lambda f, msg: None)
        self.log = dict(times={})
        self.raw = None
        self.frame = None

    def _t(self, name, t0):
        self.log['times'][name] = round(time.time() - t0, 2)
        self.log.setdefault('peak_memory_mb', {})[name] = peak_memory_mb()

    def load_cloud(self):
        """The raw point cloud, if there is one: it marks scan surface the scanner software invented."""
        self.cloud_mm, self.cloud_note, self.cloud_file = None, '', ''
        cp = self.job.cloud_path
        if cp.lower() == 'none':
            return
        if not cp:
            cp = scan.find_cloud(self.job.scan_path)
            if not cp:
                return
        try:
            pts = scan.load_cloud(cp)
        except Exception as e:
            self.cloud_note = 'Point cloud %s could not be read (%s); invented surface is not marked.' % (Path(cp).name, e)
            return
        if pts is None or len(pts) < 1000:
            self.cloud_note = '%s is not a point cloud; invented surface is not marked.' % Path(cp).name
            return
        self.cloud_mm = pts * scan.UNITS[self.unit]
        self.cloud_file = str(cp)

    # -------------------------------------------------------------------------------------------- stage 1
    def load(self):
        t0 = time.time()
        self.progress(0.02, 'reading the scan')
        self.raw = scan.load(self.job.scan_path)
        if self.job.unit == 'auto':
            self.unit, _, note = scan.guess_units(self.raw)
        else:
            self.unit, note = self.job.unit, ''
        self.unit_note = note
        mm = self.raw.copy()
        mm.apply_scale(scan.UNITS[self.unit])
        self.progress(0.05, 'cleaning the scan')
        self.mesh_raw, self.clean_info = scan.clean(mm)
        self.mesh_mm = self.mesh_raw
        ci = self.clean_info
        if ci.noise_mm > 0.04:
            self.progress(0.06, 'smoothing scan noise')
            self.mesh_mm, it, nz = scan.denoise(self.mesh_raw)
            ci.smooth_iterations = it
            ci.noise_after_mm = round(nz, 3)
            ci.notes.append('Scan noise %.2f mm smoothed to %.2f mm (%d Taubin passes).' % (ci.noise_mm, nz, it))
        else:
            ci.noise_after_mm = ci.noise_mm
        if ci.open_edges:
            # after the smoothing, so the patch fit sees the shape and not the noise
            self.progress(0.08, 'filling scan holes')
            filled, nh, ext = scan.fill_holes(self.mesh_mm, noise=ci.noise_after_mm)
            ci.hole_centres = list(getattr(scan.fill_holes, 'centres', []))
            ne = int(getattr(scan.fill_holes, 'envelope', 0))
            ci.notes.append('Filled %d scan holes (largest %.1f mm) with patches that follow the surface%s.'
                            % (nh, ext, ('; %d over sharp features, filled on the outside (more gap there)' % ne) if ne else ''))
            self.mesh_mm = filled
        self._t('load', t0)
        self.progress(0.09, 'reading the raw point cloud')
        t0 = time.time()
        self.load_cloud()
        if self.cloud_file:
            self._t('cloud', t0)
        self.detect()
        return self

    def invented(self, sp, covered):
        """Mark scan points with no raw point-cloud point within INVENTED_MM: surface the scanner software made up.
        Returns (flags or None, summary dict or None). Shares are of the surface the jacket covers."""
        if getattr(self, 'cloud_mm', None) is None:
            return None, None
        M = np.asarray(self.frame.matrix())
        ck = self.cloud_mm @ M[:3, :3].T + M[:3, 3]
        lo, hi = sp.min(axis=0) - 2, sp.max(axis=0) + 2
        ck = ck[np.all((ck >= lo) & (ck <= hi), axis=1)]
        self.cloud_crop = ck                                   # for the deviation map
        if len(ck) < 1000:
            inv = np.ones(len(sp), bool)
        else:
            _, inv = scan.cloud_cover(sp, ck)
        share = float(inv[covered].mean()) if covered.any() else 1.0
        if share > 0.5:
            self.cloud_note = ('Point cloud %s does not match the scan (%.0f %% of the covered scan surface has no '
                               'point within %.1f mm); not used.' % (Path(self.cloud_file).name, 100 * share,
                                                                     scan.INVENTED_MM))
            self.cloud_crop = None
            return None, None
        x0, x1 = self.frame.body_x
        regions = {}
        lc = self.lip
        lx = lc.get('lip_x') or (x0, x0)
        for name, sgn in (('lip_left', -1), ('lip_right', 1)):
            s = lc.get('sides', {}).get(name[4:], {})
            if not s.get('lip') or 'y_left' not in lc:
                continue
            ye = lc['y_left'] if sgn < 0 else lc['y_right']
            yin = ye - sgn * (s['reach_mm'] + 1.5)                   # 1.5 mm further in than the lip tip
            ylo, yhi = (min(ye + sgn * 1.0, yin), max(ye + sgn * 1.0, yin))   # from 1 mm outside the widest edge
            regions[name] = ((sp[:, 0] >= lx[0]) & (sp[:, 0] <= lx[1]) & (sp[:, 1] >= ylo) & (sp[:, 1] <= yhi) &
                             (sp[:, 2] >= s['widest_z'] - 1.0) & (sp[:, 2] <= s['tip_z'] + 0.5))
        regions['blade_end_wall'] = sp[:, 0] > x1 - 4.0
        other = np.ones(len(sp), bool)
        for m in regions.values():
            other &= ~m
        regions['other'] = other
        out = dict(radius_mm=scan.INVENTED_MM, cloud=Path(self.cloud_file).name, cloud_points=int(len(self.cloud_mm)),
                   covered_pct=round(100 * share, 1), regions={})
        per_pt = float(self.key_raw.area) / N_DEV          # samples are area-weighted over the whole scan
        for name, m in regions.items():
            mc = m & covered
            if not mc.any():
                continue
            out['regions'][name] = dict(points=int(mc.sum()), invented_pct=round(100 * float(inv[mc].mean()), 1),
                                        invented_mm2=round(per_pt * float((inv & mc).sum()), 1))
        return inv, out

    def detect(self):
        t0 = time.time()
        j = self.job
        M_old = self.frame.matrix() if self.frame is not None else None
        hint = None
        if self.frame is not None and j.swap_ends:
            R = np.asarray(self.frame.T)[:3, :3]
            hint = -R[0]
        face = None
        if self.frame is not None and not j.swap_ends:
            face = np.asarray(self.frame.T)[2, :3]
        axis = [np.asarray(p, float) for p in j.axis_points] if j.axis_points else None
        self.frame = scan.detect_frame(self.mesh_mm, self.unit, scan.UNITS[self.unit], blade_end=hint,
                                       axis_points=axis, flip_face=False, face=face)
        if j.flip_face:
            T = np.asarray(self.frame.T)
            T[1:3] *= -1          # turn over about the blade axis: y -> -y, z -> -z
            self.frame.T = T.tolist()
            j.flip_face = False
        j.swap_ends = False
        if j.scale:
            self.frame.caliper_scale = float(j.scale)
        if M_old is not None:
            self._carry_marks(self.frame.matrix() @ np.linalg.inv(M_old))
        self.key = scan.to_key(self.mesh_mm, self.frame)
        self.key_raw = scan.to_key(self.mesh_raw, self.frame) if self.mesh_raw is not self.mesh_mm else self.key
        if j.ring is None:
            self.ring = scan.ring_hole(self.key, self.frame)
        else:
            self.ring = dict(j.ring)
            self.ring.setdefault('found', True)
        self.frame.ring = self.ring
        self.sizes = None
        self.coarse = None
        self._t('detect', t0)
        return self

    def _carry_marks(self, A):
        """Keep the marks on the same spot of the key when the key frame changes (flip, swap, picked axis).
        A maps old key-frame mm to new key-frame mm."""
        j = self.job
        R = A[:3, :3]

        def pt(p):
            return (A @ np.append(np.asarray(p, float), 1.0))[:3]

        for b in j.buttons:
            b['p'] = pt(b['p']).tolist()
            nv = R @ np.asarray(b['n'], float)
            b['n'] = (nv / np.linalg.norm(nv)).tolist()
        if j.fold:
            q = pt(j.fold['pivot'])
            s = (R @ np.array([0.0, float(j.fold['side']), 0.0]))[1]
            j.fold = dict(j.fold, pivot=[float(q[0]), float(q[1]), 0.0], side=1 if s >= 0 else -1)
        if j.ring is not None and 'x' in j.ring:
            q = pt([j.ring['x'], j.ring['y'], 0.0])
            j.ring = dict(j.ring, x=float(q[0]), y=float(q[1]))

    def fold_side(self, x, y):
        """Which way a flip-key blade folds, from a pivot at key-frame (x, y): toward the nearer long edge of
        the body. Folding the far way would sweep the blade through the body."""
        v = np.asarray(self.key.vertices)
        s = v[np.abs(v[:, 0] - x) < 2.0, 1]
        if not len(s):
            return 1 if y >= 0 else -1
        return 1 if float(s.max()) - y <= y - float(s.min()) else -1

    def flip(self):
        """Turn the button face over (y -> -y, z -> -z) and redo what depends on it. Marks stay on the key."""
        self.job.flip_face = True
        return self.detect()

    def swap(self):
        self.job.swap_ends = True
        return self.detect()

    def set_scale(self, s):
        self.job.scale = float(s)
        self.frame.caliper_scale = float(s)
        self.key = scan.to_key(self.mesh_mm, self.frame)
        if self.ring.get('found') and self.job.ring is None:
            self.ring = scan.ring_hole(self.key, self.frame)
            self.frame.ring = self.ring
        self.sizes = None
        self.coarse = None

    # -------------------------------------------------------------------------------------------- stage 2
    def smooth_sigma(self):
        if self.P.smooth >= 0:
            return self.P.smooth
        return float(np.clip(1.5 * self.clean_info.noise_after_mm, 0.2, 0.4))

    def field(self, pitch):
        t0 = time.time()
        P = cavity.Params(**dict(self.P.to_dict(), pitch=pitch))
        lo, dims = cavity.grid_for(self.key, self.frame.body_x, P)
        flog = {}
        band = P.clearance + max(P.wall, P.back) + 1.2 + 2 * self.smooth_sigma()
        phi = cavity.key_sdf(self.key, lo, pitch, dims, close=P.close, sigma=self.smooth_sigma(), band=band,
                             progress=lambda f, m: self.progress(0.1 + 0.4 * f, m), log=flog)
        self._t('field_%.2f' % pitch, t0)
        self.log['field_%.2f' % pitch] = flog
        return phi, lo, P

    def body_sizes(self, phi, lo, P):
        """Length (from the profile), largest width and thickness of the body from the smoothed field."""
        x0, x1 = self.frame.body_x
        X, Y, Z = cavity.axes(lo, P.pitch, phi.shape)
        xs = X[0, 0]
        sel = (xs >= x0) & (xs <= x1)
        ph = phi[:, :, sel]
        bot = cavity.column_crossing(ph, lo[2], P.pitch, lowest=True)
        top = cavity.column_crossing(ph, lo[2], P.pitch, lowest=False)
        chi = ph.min(axis=0)                                  # (ny, nxs)
        # leave out the columns near filled scan holes: the patches are invented surface, and a hole at the widest
        # point would set the size (and a wrong rescale) by itself
        Mk = self.frame.matrix()
        hc = [(Mk @ np.append(c, 1))[:2] for c in (self.clean_info.hole_centres or [])]
        if hc:
            ys_ = lo[1] + np.arange(phi.shape[1]) * P.pitch
            Yg, Xg = np.meshgrid(ys_, xs[sel], indexing='ij')
            excl = np.zeros(chi.shape, bool)
            for cx, cy in hc:
                excl |= np.hypot(Xg - cx, Yg - cy) < 5.0
            if (~excl).any():
                top = np.where(excl, np.nan, top)
                bot = np.where(excl, np.nan, bot)
                chi = np.where(excl, np.float32(9.0), chi)
        T = float(np.nanmax(top) - np.nanmin(bot))
        neg = chi < 0
        ys = lo[1] + np.arange(chi.shape[0]) * P.pitch
        rows = np.flatnonzero(neg.any(axis=1))
        j0, j1 = rows[0], rows[-1]
        # sub-voxel: interpolate the crossing at both ends, over the column with the widest extent
        def edge(j_in, j_out):
            a, b = chi[j_in].min(), chi[j_out].min()
            return ys[j_in] + (ys[j_out] - ys[j_in]) * (a / (a - b)) if a < 0 < b else ys[j_in]
        W = float(edge(j1, min(j1 + 1, len(ys) - 1)) - edge(j0, max(j0 - 1, 0)))
        return float(x1 - x0), W, T

    def preview(self):
        """Quick field at preview_pitch: body sizes for the caliper check and a coarse jacket for the window."""
        phi, lo, P = self.field(self.job.preview_pitch)
        self.coarse = (phi, lo, P)
        self.sizes = self.body_sizes(phi, lo, P)
        self.cal = scan.caliper_check(self.sizes, self.job.calipers or {})
        return phi, lo, P

    # -------------------------------------------------------------------------------------------- stage 3
    def zones(self, P):
        z = dict(ring=self.ring if self.ring.get('found') else None, buttons=list(self.job.buttons or []))
        if self.job.fold:
            f = dict(self.job.fold)
            f.setdefault('half_width', (self.frame.blade.get('width') or 9.0) / 2)
            f.setdefault('half_thick', (self.frame.blade.get('thickness') or 2.8) / 2)
            if not f.get('length'):
                tip = self.frame.blade.get('tip_x') or (self.frame.body_x[1] + 40)
                f['length'] = float(tip - f['pivot'][0])
            self.fold_pts = self.fold_points(f)
            f['sweep'] = cavity.fold_sweep_table(self.fold_pts, f, P) if len(self.fold_pts) else None
            z['fold'] = f
        return z

    def fold_points(self, f, voxel=0.2):
        """The part of the key that turns when the blade folds (key frame): all beyond the body end (x > seam_x,
        default body end - 0.5 mm). Scan mesh and raw point cloud together: the cloud shows the real root (lower than
        the smoothed mesh), the mesh covers what the scanner did not see (the corners beside the root are a hole in
        the raw data, so they may be real). Thinned to one point per voxel."""
        seam = float(f.get('seam_x', self.frame.body_x[1] - 0.5))
        kp, _ = geom.sample_surface(self.key, 400000, seed=11)
        pts = [kp[kp[:, 0] > seam]]
        if getattr(self, 'cloud_mm', None) is not None:
            M = np.asarray(self.frame.matrix())
            ck = self.cloud_mm @ M[:3, :3].T + M[:3, 3]
            lo, hi = np.asarray(self.key.bounds)
            ck = ck[np.all((ck >= lo - 1.0) & (ck <= hi + 1.0), axis=1) & (ck[:, 0] > seam)]
            pts.append(ck)
        p = np.vstack(pts)
        _, idx = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_index=True)
        return p[np.sort(idx)]

    @staticmethod
    def fold_sweep_solid(f, big):
        """The swept keep-out as a solid: per radius bin an annular sector extruded from its floor upward."""
        sw = f['sweep']
        pv = np.asarray(f['pivot'], float)
        s = float(np.sign(f['side']) or 1)
        parts = []
        for i, (fl, a, b) in enumerate(zip(sw['floor'], sw['a'], sw['b'])):
            if fl >= cavity.NO_ZONE:
                continue
            r_in = max(sw['r0'] + i * sw['dr'] - 0.01, 0.0)
            span = min(b - a, 2 * np.pi - 1e-3)
            n = max(2, int(np.ceil(np.degrees(span) / 2.0)))
            t = a + np.linspace(0.0, span, n + 1)
            ro = (sw['r0'] + (i + 1) * sw['dr'] + 0.01) / np.cos(span / n / 2)     # chords outside the arc
            outer = np.c_[ro * np.cos(t), s * ro * np.sin(t)]
            inner = np.c_[r_in * np.cos(t[::-1]), s * r_in * np.sin(t[::-1])] if r_in > 0.02 else np.zeros((1, 2))
            cs = m3.CrossSection([np.vstack([outer, inner]).tolist()], m3.FillRule.NonZero)
            parts.append(M.extrude(cs, big).translate([float(pv[0]), float(pv[1]), float(fl)]))
        return M.batch_boolean(parts, m3.OpType.Add) if parts else None

    def zone_solids(self, zones, P, lo, hi):
        """The keep-out zones as solids (to cut the wing too)."""
        out = []
        big = float(np.max(hi - lo)) + 200
        ring = zones.get('ring')
        if ring:
            R = float(ring['r']) + P.ring_clear
            cyl = M.cylinder(big, R, R, 64, True).translate([ring['x'], ring['y'], 0])
            strip = M.cube([big, 2 * R, big], True).translate([ring['x'] - big / 2, ring['y'], 0])
            out.append(cyl + strip)
        for b in zones.get('buttons') or []:
            p, n = np.asarray(b['p'], float), np.asarray(b['n'], float)
            n = n / np.linalg.norm(n)
            r = float(b.get('r') or P.button_r)
            cyl = M.cylinder(11.5, r, r, 64)                 # along +z from 0: 1.5 mm inside to 10 mm out
            # rotate +z onto n
            zax = np.array([0, 0, 1.0])
            v = np.cross(zax, n)
            sang = np.linalg.norm(v)
            if sang < 1e-9:
                R3 = np.eye(3) if n[2] > 0 else np.diag([1, -1, -1.0])
            else:
                k = v / sang
                ang = np.arctan2(sang, zax @ n)
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R3 = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            start = p - 1.5 * n
            out.append(cyl.transform(np.hstack([R3, start[:, None]]).tolist()))
        f = zones.get('fold')
        if f:
            pv = np.asarray(f['pivot'], float)
            ht = float(f['half_thick']) + P.clearance + 0.4
            hw = float(f['half_width']) + P.clearance
            disc = M.cylinder(big, float(f['length']) + 1.5, float(f['length']) + 1.5, 128).translate([0, 0, -ht])
            s = np.sign(f['side']) or 1
            half = M.cube([big, big, big], True).translate([0, s * (big / 2 - hw), 0])
            out.append((disc ^ half).translate(list(pv)))      # from just below the blade upward
            if f.get('sweep'):
                sw = self.fold_sweep_solid(f, big)      # the root swept through the whole fold travel
                if sw is not None:
                    out.append(sw)
        return out

    def make(self, pitch=None, progress_base=0.0):
        """Fine field, jacket, wing, union, checks against the scan. Fills self.result."""
        P = cavity.Params(**dict(self.P.to_dict(), pitch=pitch or self.P.pitch))
        st = wings.STYLES[self.job.style]
        phi, lo, P = self.field(P.pitch)
        t0 = time.time()                                   # the field has its own time entry
        self.sizes = self.body_sizes(phi, lo, P)          # on the fine field (within about 0.03 mm on clean scans)
        self.cal = scan.caliper_check(self.sizes, self.job.calipers or {})
        self.scale_applied = 1.0
        s = float(self.cal.get('scale') or 1.0)
        if self.job.apply_scale and 0.003 <= abs(s - 1.0) <= 0.02:     # only a suggested, plausible rescale
            phi, lo, P = self.apply_scale(s, phi, lo, P)
        self.phi, self.lo, self.Pm = phi, lo, P
        zones = self.zones(P)
        self.x_face = cavity.ring_face(phi, lo, P.pitch, self.frame.body_x, frac=0.45)
        self.sweep0 = None
        if P.hood:
            self.progress(0.5, 'removal path')
            self.sweep0, _ = cavity.hood_sweep(phi, lo, P.pitch, P, self.x_face)
        self.progress(0.55, 'lips')
        if P.undercut > 0:
            u = P.undercut
            self.lip_table = None
        else:
            # choose the undercut on a 4 mm slab of the fine field at the middle of the lips (quick)
            xm = (self.x_face + self.frame.body_x[1]) / 2
            i0 = int(max(0, round((xm - 2.0 - lo[0]) / P.pitch)))
            i1 = int(min(phi.shape[2], round((xm + 2.0 - lo[0]) / P.pitch) + 1))
            los = np.array([lo[0] + i0 * P.pitch, lo[1], lo[2]])
            z_bed = float(np.nanmin(cavity.column_crossing(phi - (P.clearance + P.back), lo[2], P.pitch, lowest=True)))
            sw0 = None if self.sweep0 is None else np.ascontiguousarray(self.sweep0[:, :, i0:i1])
            u, self.lip_table = cavity.lip_choice(np.ascontiguousarray(phi[:, :, i0:i1]), los, P.pitch, self.frame.body_x, P,
                                                  zones, st['torque'], x_face=self.x_face, z_bed=z_bed, sweep0=sw0)
            self.log['lip_choice'] = self.lip_table
        self.progress(0.6, 'jacket body')
        g, cav, jinfo = cavity.jacket_field(phi, lo, P.pitch, self.frame.body_x, P, zones, u=u, x_face=self.x_face,
                                            sweep0=self.sweep0)
        self.lip_body = (self.x_face, self.frame.body_x[1])
        lc = cavity.lip_calc(phi, g, lo, P.pitch, self.lip_body, P, u)
        self.u_used = u
        self.progress(0.65, 'meshing')
        jm = cavity.mesh_field(g, lo, P.pitch)
        cm = cavity.mesh_field(cav + np.float32(0.05), lo, P.pitch, simplify=0.02)   # 0.05 inside: no coincident faces
        # wing mount from the jacket body at the bed
        mount = self.mount_from_field(g, lo, P, st)
        self.mount = mount
        wm, winfo = wings.make(self.job.style, mount, self.job.lever or None, self.job.side)
        self.progress(0.72, 'joining the wing')
        hi = lo + (np.array(phi.shape[::-1]) - 1) * P.pitch
        part = (jm + wm) - cm
        for zs in self.zone_solids(zones, P, lo, hi):
            part = part - zs
        part = part.simplify(0.005)
        if str(part.status()) != 'Error.NoError':
            raise RuntimeError('jacket CSG failed: %s' % part.status())
        self.part_man = part
        self.part = geom.clean_tm(part)
        self.zonesd = zones
        lip_len = float(jinfo['lip_x'][1] - jinfo['lip_x'][0] + P.lip_ramp)
        self.lip = dict(lc, forces=cavity.lip_forces(lc, lip_len, st['torque']), undercut_mm=u,
                        lip_x=jinfo['lip_x'], ramp=P.lip_ramp, choice=self.lip_table)
        self.jinfo = jinfo
        self.winfo = winfo
        self.cav_mesh = geom.to_tm(cm)
        self._t('make', t0)
        self.progress(0.8, 'checks')
        self.check()
        return self

    def apply_scale(self, s, phi, lo, P):
        """Rescale the scan by s about the key frame origin. The field scales exactly (phi(x) -> s phi(x / s)), so
        the grid is reused with origin and pitch times s. Marks, ring and frame facts scale with it."""
        self.frame.caliper_scale *= s
        self.key = scan.to_key(self.mesh_mm, self.frame)
        self.key_raw = scan.to_key(self.mesh_raw, self.frame) if self.mesh_raw is not self.mesh_mm else self.key
        fr = self.frame
        fr.body_x = (fr.body_x[0] * s, fr.body_x[1] * s)
        for k in ('tip_x', 'length', 'width', 'thickness'):
            if k in fr.blade:
                fr.blade[k] = fr.blade[k] * s
        if self.ring.get('found'):
            self.ring = dict(self.ring, x=self.ring['x'] * s, y=self.ring['y'] * s, r=self.ring['r'] * s)
            fr.ring = self.ring
        for b in self.job.buttons:
            b['p'] = [v * s for v in b['p']]
        if self.job.fold:
            self.job.fold['pivot'] = [v * s for v in self.job.fold['pivot']]
        self.sizes = tuple(v * s for v in self.sizes)
        self.scale_applied = s
        self.coarse = None
        P = cavity.Params(**dict(P.to_dict(), pitch=P.pitch * s))
        return (phi * np.float32(s)).astype(np.float32), np.asarray(lo) * s, P

    def mount_from_field(self, g, lo, P, st):
        X, Y, Z = cavity.axes(lo, P.pitch, g.shape)
        z_bed = self.jinfo_z_bed(g, lo, P)
        k = int(np.clip(round((z_bed + 0.6 - lo[2]) / P.pitch), 0, g.shape[0] - 1))
        x0, x1 = self.x_face, self.frame.body_x[1]
        xc = (x0 + x1) / 2
        root = max(10.0, min(st.get('root_len', st.get('ring_len', 10.0)), (x1 - x0) - 10.0))
        r_cap = band = None
        if P.hood:
            # with the nose hook the turning torque goes from the wing to the cap: put the root near the cap (short
            # twist length). The key-ring keep-out moves the root 1 mm clear of it, but only on the side it reaches:
            # the other side starts at the cap (defect 40: on the Yaris the ring rule had moved both roots 7 mm off
            # the cap, and the turn that lifts the fold side, which has no lip, twisted the open jacket between them)
            ring = self.ring if self.ring.get('found') else None
            r_cap = x0 + P.hood_len + 1.0
            r0 = r_cap
            if ring and ring.get('x') is not None:
                R = float(ring['r']) + P.ring_clear
                r0 = max(r0, float(ring['x']) + R + 1.0)
                band = (float(ring['y']) - R, float(ring['y']) + R)
            xc = min(xc, r0 + root / 2)
        xs = X[0, 0]
        k1 = int(np.clip(round((z_bed + 2.5 - lo[2]) / P.pitch), k + 1, g.shape[0]))
        ys = lo[1] + np.arange(g.shape[1]) * P.pitch

        def faces(sel):
            layer = (g[k:k1][:, :, sel] < 0).any(axis=0)  # (ny, nsel): material in the lowest 2.5 mm
            ymax = np.array([ys[np.flatnonzero(layer[:, i])].max() for i in range(layer.shape[1]) if layer[:, i].any()])
            ymin = np.array([ys[np.flatnonzero(layer[:, i])].min() for i in range(layer.shape[1]) if layer[:, i].any()])
            return float(ymin.max()), float(ymax.min())

        sel = np.flatnonzero(np.abs(xs - xc) <= root / 2)
        y_left, y_right = faces(sel)
        lo_r = lo_l = None
        if r_cap is not None and r_cap < xc - root / 2 - 0.5:
            sel_ext = np.flatnonzero((xs >= r_cap) & (xs <= xc + root / 2))
            yl_ext, yr_ext = faces(sel_ext)
            # a lobe's root strip reaches 2.5 mm into the side; the side is free when the ring band stays 1 mm off it
            if band is None or band[1] < y_right - 3.5:
                lo_r, y_right = float(r_cap), yr_ext
            if band is None or band[0] > y_left + 3.5:
                lo_l, y_left = float(r_cap), yl_ext
        top = cavity.column_crossing(g[:, :, sel[len(sel) // 2]][:, :, None], lo[2], P.pitch, lowest=False)
        return wings.Mount(x_c=float(xc), root_len=float(root), y_left=y_left, y_right=y_right,
                           z_bed=float(z_bed), z_top=float(np.nanmax(top)), x_lo_right=lo_r, x_lo_left=lo_l)

    def jinfo_z_bed(self, g, lo, P):
        solid_layers = np.flatnonzero((g < 0).any(axis=(1, 2)))
        k = solid_layers[0]
        # the field's bed plane is exact; take it from the crossing in the middle column
        bot = cavity.column_crossing(g, lo[2], P.pitch, lowest=True)
        return float(np.nanmin(bot))

    # -------------------------------------------------------------------------------------------- checks
    def check(self):
        t0 = time.time()
        P = self.Pm
        res = dict(mesh=checks.mesh_validity(self.part))
        sdf = checks.JacketSDF(self.part)
        self.sdf = sdf
        # the design surface (smoothed key, iso-surface of phi) and the raw scan points
        key_iso = geom.iso_mesh(self.phi, self.lo, P.pitch)
        self.key_iso = key_iso
        kp, _ = geom.sample_surface(key_iso, 200000, seed=8)
        x0, x1 = self.frame.body_x
        kp = kp[kp[:, 0] <= x1 + 3]
        st, cov, gap = checks.clearance(sdf, kp, P.clearance)
        bad = gap < -0.05
        if bad.any():
            st['interference_where'] = [[round(float(v), 2) for v in kp[i]] for i in np.flatnonzero(bad)[:: max(1, int(bad.sum() // 6))][:6]]
        res['clearance_design'] = st
        sp, _ = geom.sample_surface(self.key_raw, N_DEV, seed=9)
        sp = sp[(sp[:, 0] >= x0 - 3) & (sp[:, 0] <= x1 + 3)]
        st2, cov2, gap2 = checks.clearance(sdf, sp, P.clearance)
        res['clearance_scan'] = st2
        self.dev = dict(points=sp, gap=gap2, covered=cov2)
        t1 = time.time()
        inv, inv_sum = self.invented(sp, cov2)
        if getattr(self, 'cloud_file', ''):
            self._t('cloud_check', t1)
            t0 += time.time() - t1                         # keep 'checks' free of the cloud check's time
        if inv is not None:
            self.dev['invented'] = inv
            meas = cov2 & ~inv
            if meas.any():          # the deviation on measured surface only: the claim holds only there
                gm = gap2[meas]
                inv_sum['measured_within_0_1_pct'] = round(100.0 * float(np.mean(np.abs(gm - P.clearance) <= 0.1)), 1)
                inv_sum['measured_p05_p95'] = [round(float(np.percentile(gm, 5)), 3),
                                               round(float(np.percentile(gm, 95)), 3)]
        res['invented'] = inv_sum
        res['cloud_note'] = getattr(self, 'cloud_note', '')
        hood = self.jinfo.get('hood') if P.hood else None
        xs0 = self.jinfo['x_trim'][0]
        if hood:
            # the nose hook: the key comes out tail first along the hooked path (3D check); the straight-up 2D
            # section check covers only the part behind the nose, which the hook does not change
            res['hood_path'] = checks.hood_path(sdf, kp, hood['hinge'], P.hood_tilt, P.hood_slide, self.lip['lip_x'],
                                                self.lip['ramp'])
            xs0 = max(xs0, float(hood['hinge'][0]) + P.hood_slide + 0.5)
        sn = checks.snap_in_2d(self.part, self.key, xs0, self.jinfo['x_trim'][1], self.lip['lip_x'], self.lip['ramp'])
        res['snap_in'] = sn
        hi = self.part.bounds
        res['zones'] = checks.zones_free(sdf, self.zonesd, P, (hi[0], hi[1]), fold_pts=getattr(self, 'fold_pts', None))
        res['wing'] = checks.wing_geometry(self.part, np.zeros(3), np.array([1.0, 0, 0]), self.mount, self.winfo)
        self.checks = res
        self._t('checks', t0)
        return res

    # -------------------------------------------------------------------------------------------- outputs
    def print_pose(self):
        """Translation from the key frame to the print pose: flat back on z = 0, centred on x, y."""
        b = self.part.bounds
        return np.array([-(b[0, 0] + b[1, 0]) / 2, -(b[0, 1] + b[1, 1]) / 2, -b[0, 2]])

    def coupon(self, clearances=None, half_len=3.0):
        """Fit coupon: three 6 mm slices of the jacket at the middle of the lip zone, at clearances c-0.1, c, c+0.1,
        side by side; ring k carries k ribs on its outer -y side."""
        t0 = time.time()
        P = self.Pm
        c = P.clearance
        clearances = clearances or [round(c - 0.1, 2), round(c, 2), round(c + 0.1, 2)]
        xm = (self.x_face + self.frame.body_x[1]) / 2
        zones = self.zonesd
        rings = []
        for k, ci in enumerate(clearances):
            Pk = cavity.Params(**dict(P.to_dict(), clearance=ci))
            g, _, info = cavity.jacket_field(self.phi, self.lo, P.pitch, self.frame.body_x, Pk, zones, u=self.u_used,
                                             xslab=(xm, half_len), x_face=self.x_face,
                                             sweep0=getattr(self, 'sweep0', None))
            man = cavity.mesh_field(g, self.lo, P.pitch)
            tm = geom.to_tm(man)
            b = tm.bounds
            for r in range(k + 1):        # k+1 ribs on the outer -y side, near the bed, centred on the slice
                rib = M.cylinder(3.0, 0.7, 0.7, 24).translate([xm + (r - k / 2.0) * 1.8, b[0, 1] + 0.5, b[0, 2]])
                man = man + rib
            man = man.translate([(k - 1) * (2 * half_len + 7.0), 0, 0])
            rings.append(geom.clean_tm(man))          # each ring cleaned like the jacket (no zero-volume slivers)
        tm = trimesh.util.concatenate(rings)
        b = tm.bounds
        tm.apply_translation([-(b[0, 0] + b[1, 0]) / 2, -(b[0, 1] + b[1, 1]) / 2, -b[0, 2]])
        self.coupon_mesh = tm
        self.coupon_info = dict(clearances=clearances, slice_mm=2 * half_len, x=round(xm, 2),
                                ribs='ring with 1 rib = %.2f mm, 2 ribs = %.2f mm, 3 ribs = %.2f mm' % tuple(clearances),
                                mass_g_solid=round(float(tm.volume) / 1000 * cavity.RHO_PETG, 2),
                                bodies=len(tm.split(only_watertight=False)), watertight=bool(tm.is_watertight))
        self._t('coupon', t0)
        return tm

    def write(self, out_dir=None, slicer=None):
        from . import report
        t0 = time.time()
        out = Path(out_dir or self.job.out_dir or tool_out_dir(self.job.scan_path, self.job.style))
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(self.job.scan_path).stem
        name = '%s_FobFit_%s' % (stem, self.job.style)
        t = self.print_pose()
        jp = self.part.copy()
        jp.apply_translation(t)
        files = {}
        files['jacket_stl'] = out / (name + '.stl')
        jp.export(files['jacket_stl'])
        try:
            files['jacket_3mf'] = out / (name + '.3mf')
            jp.export(files['jacket_3mf'])
        except Exception as e:
            self.log['3mf_error'] = str(e)
            files.pop('jacket_3mf')
        if self.job.coupon:
            if not hasattr(self, 'coupon_mesh'):
                self.coupon()
            files['coupon_stl'] = out / (stem + '_FobFit_fit_coupon.stl')
            self.coupon_mesh.export(files['coupon_stl'])
        kd = geom.decimate(self.key, 150000)
        kd = kd.copy()
        kd.apply_translation(t)
        files['scan_in_print_pose'] = out / (stem + '_scan_in_print_pose.stl')
        kd.export(files['scan_in_print_pose'])
        self.files = files
        self.out = out
        self._t('export', t0)
        if slicer:
            self.progress(0.9, 'slicing')
            t0 = time.time()
            self.sliced = slicer(self, out)
            self._t('slice', t0)
        self.progress(0.95, 'report and pictures')
        t0 = time.time()
        report.write_all(self, out, name, t_report=t0)
        settings = dict(job=self.job.to_dict(), frame=self.frame.to_dict(), params=self.Pm.to_dict())
        (out / (stem + '_FobFit_settings.json')).write_text(json.dumps(settings, indent=1, default=float))
        self.progress(1.0, 'done')
        return out
