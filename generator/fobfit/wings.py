"""Wing styles. They know nothing about the scan: a Mount describes where the jacket offers a root, so the same
code serves the scan jacket and a caliper-sized fallback jacket.

Key frame: x along the blade axis, the axis itself on y = z = 0, the bed (flat back) at z = z_bed.
Lever arm: distance from the blade axis to the push point (the centre of the round tip; the centre of the hole for
the Loop). Every wing lies in the bed plane and runs along +y or -y, so it is square to the blade axis by build.
"""
from dataclasses import dataclass, asdict
import numpy as np
import manifold3d as m3

M = m3.Manifold
CS = m3.CrossSection

STYLES = {
    'wing': dict(label='Wing', lever=45.0, thick=3.5, tip_r=7.5, root_len=26.0, sides=(1, -1), torque=0.5,
                 use='two short wings, for an ignition'),
    'paddle': dict(label='Paddle', lever=70.0, thick=4.5, tip_r=9.0, root_len=32.0, sides=(1,), torque=1.5,
                   use='one long wing, for a stiff door lock'),
    'loop': dict(label='Loop', lever=40.0, ring_id=24.0, ring_wall=5.0, ring_len=10.0, torque=0.5, sides=(1,),
                 use='a ring for a hooked finger or the thumb'),
}
E_PETG = 1600.0      # MPa, printed PETG (assumed, as in cavity.py)
SIGMA_PETG = 30.0    # MPa, in-layer tensile strength of printed PETG (assumed, conservative)


@dataclass
class Mount:
    x_c: float          # middle of the root along x
    root_len: float     # length of jacket side available for the root
    y_left: float       # outer face of the jacket side at the root (negative)
    y_right: float      # outer face of the jacket side at the root (positive)
    z_bed: float        # bed plane
    z_top: float        # top of the jacket side at the root
    x_lo_right: float = None    # root start on the +y side when it reaches past x_c - root_len / 2 (to the cap)
    x_lo_left: float = None     # the same on the -y side

    def to_dict(self):
        return asdict(self)

    def root_x(self, s, root_len):
        """Root of the lobe on side s along x: (start, end)."""
        lo = self.x_lo_right if s > 0 else self.x_lo_left
        hi = self.x_c + root_len / 2
        return (float(lo) if lo is not None else self.x_c - root_len / 2), hi


def _rounded_slab(cs, t, R, z0, steps=6):
    """Convex 2D outline -> plate of thickness t standing on z0 with its top edge rounded by R (hull of insets)."""
    R = min(R, t * 0.6)
    parts = [cs.extrude(max(t - R, 0.05)).translate([0, 0, z0])]
    for k in range(1, steps + 1):
        th = np.pi / 2 * k / steps
        inset = R * (1 - np.cos(th))
        z = z0 + t - R + R * np.sin(th)
        parts.append(cs.offset(-inset, m3.JoinType.Round).extrude(0.02).translate([0, 0, z - 0.02]))
    return M.batch_hull(parts)


def _gusset(x0, x1, y_side, s, z, g):
    """Triangular fillet in the corner between the wing top and the jacket side, along x."""
    tri = CS([[(0.0, 0.0), (g, 0.0), (0.0, g)]]) if s > 0 else CS([[(0.0, 0.0), (0.0, g), (-g, 0.0)]])
    # cross-section in (y, z); extrude along w, then map (y, z, w) -> (x, y, z)
    m = tri.translate([y_side - s * 1.0, 0.0]).extrude(x1 - x0)
    return m.transform([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, z]])


def wing_lobe(mount, s, lever, thick, tip_r, root_len):
    y_side = mount.y_right if s > 0 else -mount.y_left
    x0, x1 = mount.root_x(s, root_len)
    xc = (x0 + x1) / 2                  # a root that reaches to the cap moves its tip with it: the lobe stays symmetric
    root = CS.square([x1 - x0, 2.5]).translate([x0, y_side - 2.5])
    tip = CS.circle(tip_r, 64).translate([xc, lever])
    cs = CS.batch_hull([root, tip])
    if s < 0:
        cs = cs.mirror([0, 1])
    slab = _rounded_slab(cs, thick, 1.5, mount.z_bed)
    g = _gusset(x0 + 2, x1 - 2, s * y_side, s, mount.z_bed + thick, 3.0)
    return slab + g


def loop_ring(mount, s, lever, ring_id, wall, ring_len):
    Ri, Ro = ring_id / 2, ring_id / 2 + wall
    y_side = mount.y_right if s > 0 else -mount.y_left
    zc = mount.z_bed + 5.0 + Ri                       # hole centre 5 mm above the bed
    yc = float(np.sqrt(max(lever ** 2 - zc ** 2, (y_side + Ri + 4) ** 2)))
    yc = max(yc, y_side + Ri + 3.0)
    arch = CS.batch_hull([CS.circle(Ro, 96).translate([yc, zc]),
                          CS.square([2 * Ro, 0.5]).translate([yc - Ro, mount.z_bed])])
    hole = CS.batch_hull([CS.circle(Ri, 96).translate([yc, zc]),
                          CS.square([0.6, 0.6]).translate([yc - 0.3, zc + Ri * np.sqrt(2) - 0.6])])   # 45 degree teardrop
    web = CS.square([yc - (y_side - 2.0), 5.0]).translate([y_side - 2.0, mount.z_bed])
    cs = (arch + web) - hole
    m = cs.extrude(ring_len).transform([[0, 0, 1, mount.x_c - ring_len / 2], [1, 0, 0, 0], [0, 1, 0, 0]])
    if s < 0:
        m = m.mirror([0, 1, 0])
    lever_real = float(np.hypot(yc, zc))
    return m, dict(hole_centre_y=round(s * yc, 2), hole_centre_z=round(zc, 2), lever_mm=round(lever_real, 1))


def make(style, mount, lever=None, side=1):
    """Wing solid (manifold) for a style name and the facts for the report."""
    st = dict(STYLES[style])
    lever = float(lever or st['lever'])
    info = dict(style=style, label=st['label'], use=st['use'], lever_mm=round(lever, 1),
                design_torque_Nm=st['torque'])
    sides = st['sides'] if style == 'wing' else (side,)
    parts = []
    if style == 'loop':
        for s in sides:
            m, extra = loop_ring(mount, s, lever, st['ring_id'], st['ring_wall'], st['ring_len'])
            parts.append(m)
            info.update(extra)
            info.update(ring_id_mm=st['ring_id'], ring_wall_mm=st['ring_wall'], ring_len_mm=st['ring_len'])
            lever = extra['lever_mm']
            info['lever_mm'] = lever
    else:
        for s in sides:
            parts.append(wing_lobe(mount, s, lever, st['thick'], st['tip_r'], st['root_len']))
        info['root_x_mm'] = {('+y' if s > 0 else '-y'): [round(v, 1) for v in mount.root_x(s, st['root_len'])]
                             for s in sides}
        y_root = max(mount.y_right, -mount.y_left)
        F = st['torque'] / (lever / 1000.0)
        arm = lever - y_root
        b = st['root_len']
        t = st['thick']
        sigma = 6 * F * arm / (b * t * t)
        b_avg = (st['root_len'] + 2 * st['tip_r']) / 2
        defl = F * arm ** 3 / (3 * E_PETG * b_avg * t ** 3 / 12)
        F25 = 2.5 / (lever / 1000.0)                 # EN 1303: torque a key must survive
        info.update(thick_mm=t, tip_r_mm=st['tip_r'], root_len_mm=b, tip_edge_mm=round(lever + st['tip_r'], 1),
                    force_N=round(F, 1), root_stress_MPa=round(sigma, 1),
                    root_stress_at_2_5Nm_MPa=round(6 * F25 * arm / (b * t * t), 1),
                    tip_deflection_mm=round(defl, 2), strength_assumed_MPa=SIGMA_PETG)
    info['side'] = 'both' if len(sides) == 2 else ('+y' if sides[0] > 0 else '-y')
    if style == 'loop':
        F = st['torque'] / (lever / 1000.0)
        info['force_N'] = round(F, 1)
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out, info
