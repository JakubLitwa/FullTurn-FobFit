"""Report, renders, cross-section and the deviation map of a run (pipeline.Run)."""
import json
from pathlib import Path

import numpy as np

from . import geom, render, cavity, scan

RHO = cavity.RHO_PETG
PURPLE = (0.72, 0.2, 0.78)       # invented scan surface in the deviation map
REGION_NAMES = dict(lip_left='Lip at -y', lip_right='Lip at +y', blade_end_wall='Blade-end wall (last 4 mm)',
                    other='Rest of the covered surface')
STAGE_NAMES = dict(load='Read and clean the scan', cloud='Read the raw point cloud', detect='Find the key frame',
                   make='Jacket body, lips, wing, join', checks='Checks on the part', cloud_check='Invented-surface check',
                   export='Write the files', slice='Slice (Bambu Studio)',
                   report='Pictures and report')
LIP_INVENTED_PCT = 5.0           # warn above these shares of invented surface
WALL_INVENTED_PCT = 10.0


# ------------------------------------------------------------------------------------------------ caliper box
def box_sdf(p, c, b, r):
    q = np.abs(p - c) - (b - r)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - r


def caliper_box(points, clearance):
    """The best jacket three caliper sizes can describe: a rounded box of the key's length, width and thickness,
    with the single edge radius (1-6 mm) that leaves the smallest mean gap while still holding the key.
    Returns (gap per point, description)."""
    lo, hi = points.min(0), points.max(0)
    c = (lo + hi) / 2
    b = (hi - lo) / 2 + clearance
    best = None
    for r in np.arange(1.0, min(6.0, b.min()) + 0.01, 0.5):
        d = -box_sdf(points, c, b, r)
        inside = np.mean(d >= clearance - 0.05)
        if inside >= 0.995 and (best is None or d.mean() < best[1].mean()):
            best = (r, d)
    if best is None:
        r = 1.0
        best = (r, -box_sdf(points, c, b, r))
    return best[1], dict(size_mm=[round(float(v), 2) for v in 2 * (b - clearance)], edge_radius_mm=float(best[0]))


# ------------------------------------------------------------------------------------------------ pictures
def _key_display(run, n=120000):
    return geom.decimate(run.key_iso if hasattr(run, 'key_iso') else run.key, n)


def renders(run, path):
    """Four views of the jacket on the key and one of the jacket alone, in the key frame."""
    from PIL import Image
    key = _key_display(run)
    x0, x1 = run.frame.body_x
    kd = key.copy()
    kd = kd.slice_plane([x1 + 45, 0, 0], [-1, 0, 0]) if kd.bounds[1, 0] > x1 + 45 else kd
    views = [((1.0, -1.3, 1.0), True), ((-1.2, 0.9, 1.1), True), ((0.25, -0.35, -1.0), True), ((1.0, 0.0, 0.18), True),
             ((0.9, -1.1, 1.3), False), ((-0.3, 1.2, 0.7), False)]
    imgs = []
    for d, with_key in views:
        sc = render.Scene(900, 680)
        sc.add(render.actor(run.part))
        if with_key:
            sc.add(render.actor(kd, color=render.KEY))
        sc.view(d)
        imgs.append(sc.image())
    rows = [np.hstack(imgs[i:i + 2]) for i in range(0, 6, 2)]
    Image.fromarray(np.vstack(rows)).save(path)


def section(run, path):
    """Cross-section of key and jacket at the middle of the lips, whole and one lip enlarged."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    xm = (run.lip_body[0] + run.lip_body[1]) / 2
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
    loops = {}
    for mesh, lab in ((run.key_iso, 'key'), (run.part, 'jacket')):
        try:
            sec = mesh.section(plane_origin=[xm, 0, 0], plane_normal=[1, 0, 0])
            loops[lab] = [e[:, 1:3] for e in sec.discrete] if sec is not None else []
        except Exception:
            loops[lab] = []
    sides = [s for s in run.lip.get('sides', {}).values() if s.get('lip')]
    for k in range(2):
        for lab, col, al in (('jacket', '#f28c28', 0.9), ('key', '#3a3d44', 0.6)):
            for pts in loops[lab]:
                ax[k].fill(pts[:, 0], pts[:, 1], color=col, alpha=al, lw=0)
                ax[k].plot(pts[:, 0], pts[:, 1], color=col, lw=0.8)
        ax[k].set_aspect('equal')
        ax[k].grid(alpha=0.25)
        ax[k].set_xlabel('y (mm)')
        ax[k].set_ylabel('z (mm), button face up')
    ax[0].set_title('Section across the key at x = %.1f mm (middle of the lips)' % xm, fontsize=11)
    kp = np.vstack(loops['key']) if loops['key'] else np.zeros((1, 2))
    yr = kp[:, 0].max() if len(kp) else 15
    if sides:
        zt = max(s['tip_z'] for s in sides)
        ax[1].set_xlim(yr - 4.5, yr + 3.5)
        ax[1].set_ylim(zt - 5.5, zt + 2.5)
    ax[1].set_title('Right lip enlarged: undercut %.2f mm, gap %.2f mm' % (run.lip['undercut_mm'], run.Pm.clearance), fontsize=11)
    ax[0].legend(handles=[Patch(color='#3a3d44', alpha=0.6, label='key (from the scan)'), Patch(color='#f28c28', label='jacket')],
                 loc='lower center', fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def deviation_map(run, path):
    """Scan surface coloured by its gap to the jacket cavity, three views, and a histogram against a cavity made
    from three caliper sizes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    c = run.Pm.clearance
    key = geom.decimate(getattr(run, 'key_raw', run.key), 160000)      # the raw scan: what the scanner saw
    x0, x1 = run.frame.body_x
    key = key.slice_plane([x1 + 4, 0, 0], [-1, 0, 0])
    gap_v = run.sdf(key.vertices)
    covered = gap_v < c + 0.8
    dev = np.where(covered, gap_v - c, np.nan)
    lim = 0.3
    vals = np.where(np.isnan(dev), 9.0, np.clip(dev, -lim, lim))
    inv_on = getattr(run, 'cloud_crop', None) is not None and 'invented' in run.dev
    if inv_on:      # covered surface the scanner software invented: purple (the NaN colour)
        _, inv_v = scan.cloud_cover(key.vertices, run.cloud_crop)
        vals = np.where(covered & inv_v, np.nan, vals)
    lut = render.diverging_lut(-lim, lim)
    lut.SetNanColor(*PURPLE, 1)
    lut.SetAboveRangeColor(0.82, 0.82, 0.84, 1.0)
    lut.UseAboveRangeColorOn()
    lut.SetBelowRangeColor(0.6, 0.0, 0.0, 1.0)
    lut.UseBelowRangeColorOn()
    imgs = []
    for d in ((0.0, -0.25, -1.0), (0.15, -1.0, 0.25), (0.15, 1.0, 0.25)):
        sc = render.Scene(700, 520)
        sc.add(render.actor(key, scalars=vals, lut=lut, rng=(-lim, lim), smooth=False))
        sc.view(d, up=(1, 0, 0) if abs(d[2]) > 0.9 else (0, 0, 1))
        imgs.append(sc.image())
    # numbers: scan cavity vs caliper box over the same covered points; measured surface only when the cloud says
    sp, gap = run.dev['points'], run.dev['gap']
    cov = run.dev['covered']
    g_all = gap[cov]
    use = cov & ~run.dev['invented'] if inv_on else cov
    g_scan = gap[use]
    body = (sp[:, 0] >= run.lip_body[0] - 1) & (sp[:, 0] <= x1)
    g_box, box = caliper_box(sp[body], c)
    g_box = g_box[use[body]]
    what = 'measured covered scan' if inv_on else 'covered scan'
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.8])
    for k, (img, t) in enumerate(zip(imgs, ('back', 'side at -y', 'side at +y'))):     # cameras at -z, -y, +y
        a = fig.add_subplot(gs[0, k])
        a.imshow(img)
        a.set_axis_off()
        a.set_title('Scan coloured by gap to the jacket: %s' % t, fontsize=10)
    a = fig.add_subplot(gs[1, :2])
    bins = np.linspace(-0.2, 3.0, 161)
    a.hist(np.clip(g_box, -0.2, 3.0), bins=bins, color='#9aa0a8', alpha=0.85,
           label='cavity from three caliper sizes (best rounded box %s mm, r %.1f mm)' % (' x '.join('%.1f' % v for v in box['size_mm']), box['edge_radius_mm']))
    a.hist(np.clip(g_scan, -0.2, 3.0), bins=bins, color='#2a9d4b', alpha=0.85, label='cavity from the scan (this jacket)')
    a.axvline(c, color='k', lw=1, ls='--')
    a.text(c + 0.03, a.get_ylim()[1] * 0.92, 'target gap %.2f mm' % c, fontsize=9)
    a.set_xlabel('gap between key surface and jacket (mm); gaps over 3 mm are counted at 3')
    a.set_ylabel('scan points' + (' (measured surface only)' if inv_on else ''))
    a.legend(fontsize=9)
    within_s = 100 * np.mean(np.abs(g_scan - c) <= 0.1)
    within_b = 100 * np.mean(np.abs(g_box - c) <= 0.1)
    within_all = 100 * np.mean(np.abs(g_all - c) <= 0.1)
    a.set_title('Within 0.1 mm of the target gap: %.0f %% of the %s (scan cavity) vs %.0f %% (caliper box)'
                % (within_s, what, within_b), fontsize=11)
    a2 = fig.add_subplot(gs[1, 2])
    a2.set_axis_off()
    cb = np.linspace(-lim, lim, 7)
    for i, v in enumerate(cb):
        rgb = [0.0] * 3
        lut.GetColor(v, rgb)
        a2.add_patch(plt.Rectangle((0.05, 0.17 + i * 0.105), 0.12, 0.095, color=rgb))
        a2.text(0.2, 0.2 + i * 0.105, '%+.2f mm' % v, fontsize=9)
    a2.text(0.05, 0.92, 'gap minus target', fontsize=10, weight='bold')
    if inv_on:
        a2.add_patch(plt.Rectangle((0.55, 0.45), 0.12, 0.1, color=PURPLE))
        a2.text(0.55, 0.58, 'invented by the\nscanner software:\nno raw point\nwithin %.1f mm' % scan.INVENTED_MM, fontsize=9,
                va='bottom')
    a2.text(0.05, 0.0, 'grey: not covered by the jacket\nred: less gap, blue: more gap'
            + ('\npurple: covered surface with no raw scan point' if inv_on else '\n(no point cloud: invented surface not marked)'),
            fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return dict(scan_within_0_1_pct=round(float(within_s), 1), box_within_0_1_pct=round(float(within_b), 1),
                scan_median=round(float(np.median(g_scan)), 3), box_median=round(float(np.median(g_box)), 3),
                scan_p95=round(float(np.percentile(g_scan, 95)), 3), box_p95=round(float(np.percentile(g_box, 95)), 3),
                caliper_box=box, points=int(len(g_scan)), measured_only=bool(inv_on),
                scan_within_0_1_all_covered_pct=round(float(within_all), 1))


# ------------------------------------------------------------------------------------------------ warnings
def warnings(run):
    w = []
    ch = run.checks
    fr = run.frame
    if run.unit_note:
        w.append(run.unit_note)
    w += fr.notes
    w += run.clean_info.notes
    if not run.ring.get('found'):
        w.append('No key-ring hole was found. If the key has one, mark it, or the jacket may block the ring.')
    cal = getattr(run, 'cal', {}) or {}
    if cal.get('note'):
        w.append('Calipers: ' + cal['note'] + (' Applied.' if run.frame.caliper_scale != 1.0 else ' Not applied.'))
    if not cal.get('rows') and run.frame.caliper_scale == 1.0:
        w.append('Scale not checked: no caliper sizes were given, so the scan sizes are used as they are. Check the '
                 'size and the gap on the first jacket.')
    ch_inv = ch.get('invented')
    if ch.get('cloud_note'):
        w.append(ch['cloud_note'])
    elif ch_inv is None:
        w.append('No raw point cloud next to the scan: surface the scanner software invented is not marked, so the fit '
                 'figures may include made-up surface.')
    else:
        for k, r in ch_inv['regions'].items():
            lim = LIP_INVENTED_PCT if k.startswith('lip') else WALL_INVENTED_PCT if k == 'blade_end_wall' else None
            if lim is not None and r['invented_pct'] > lim:
                how = ('Check the snap on the first jacket.'
                       if k.startswith('lip') else
                       'On the first jacket, check that the key snaps fully '
                       'in and does not slide toward the blade.')
                w.append('%s: %.0f %% of it (%.0f mm2) has no raw scan point within %.1f mm, so the scanner software '
                         'invented it and the cavity follows a guess there. %s'
                         % (REGION_NAMES[k], r['invented_pct'], r['invented_mm2'], ch_inv['radius_mm'], how))
    if cal.get('rows') and run.clean_info.noise_mm > 0.05:
        w.append('This scan is noisy (%.2f mm): its sizes are sure to about 1 %% only, so a caliper rescale of under '
                 '1 %% may be wrong. Prefer a cleaner scan.' % run.clean_info.noise_mm)
    m = ch['mesh']
    if not (m['watertight'] and m['bodies'] == 1 and m['manifold_status'] == 'Error.NoError'):
        w.append('Mesh check failed: %s' % m)
    cl = ch['clearance_design']
    if cl.get('interference_pts', 0) > 0:
        w.append('%d key points touch or cut into the jacket (deepest %.2f mm).' % (cl['interference_pts'], -cl['min']))
    sn = ch['snap_in']
    if sn.get('flex_outside_lip_zone_max_mm', 0) > 0.1:
        w.append('Snapping in, the key meets material outside the lips: a wall must flex %.2f mm at x = %s mm.'
                 % (sn['flex_outside_lip_zone_max_mm'], sn.get('worst_outside_x')))
    if sn.get('flex_in_lip_zone_max_mm', 0) > run.lip['undercut_mm'] + 0.5:
        w.append('A lip must flex %.2f mm, more than its designed undercut %.2f mm.' % (sn['flex_in_lip_zone_max_mm'], run.lip['undercut_mm']))
    hp = ch.get('hood_path')
    if hp:
        if hp['max_outside_lip_zone_mm'] > 0.1:
            w.append('Taking the key out along the nose-hook path, it meets material outside the lips: %.2f mm deep, '
                     'at x = %s mm. The key may not go in or come out by hand.'
                     % (hp['max_outside_lip_zone_mm'], ', '.join('%.1f' % v for v in hp.get('outside_contact_x', [])[:3])))
        if hp['max_in_lip_zone_mm'] > run.lip['undercut_mm'] + 0.5:
            w.append('On the nose-hook path a lip must flex %.2f mm, more than its designed undercut %.2f mm.'
                     % (hp['max_in_lip_zone_mm'], run.lip['undercut_mm']))
    for k, v in ch['zones'].items():
        if v['deepest_mm'] > 0.05:
            w.append('Material in the %s zone (%.2f mm deep).' % (k.replace('_', ' '), v['deepest_mm']))
        elif v.get('swept_min_gap_mm', 1.0) < 0.2:
            w.append('Turning the blade, the gap to the jacket falls to %.2f mm at %.0f degrees: the fold may rub.'
                     % (v['swept_min_gap_mm'], v['swept_worst_deg']))
    f = run.lip.get('forces', {})
    cam = f.get('cam_out_Nm')
    if isinstance(cam, (int, float)) and cam < run.winfo['design_torque_Nm'] and not run.jinfo.get('hood'):
        w.append('Estimated cam-out torque %.2f N m is below the design torque %.1f N m of this style. The side walls '
                 'carry part of the torque too; test before use.' % (cam, run.winfo['design_torque_Nm']))
    if f.get('strain_max_pct', 0) > 100 * cavity.STRAIN_LIMIT:
        w.append('Lip strain estimate %.1f %% is over the %.1f %% limit.' % (f['strain_max_pct'], 100 * cavity.STRAIN_LIMIT))
    sl = getattr(run, 'sliced', None) or {}
    mass = sl.get('jacket', {}).get('mass_g') or m['mass_g_solid']
    if mass > 15.0:
        w.append('Jacket mass %.1f g is over the 15 g target%s.' % (mass, '' if sl.get('jacket') else ' (solid estimate; slice for the real mass)'))
    wg = ch.get('wing', {})
    if wg.get('ok') and wg.get('square_error_deg', 0) > 1.0:
        w.append('Wing is %.2f degrees off square to the blade axis.' % wg['square_error_deg'])
    if not fr.blade_found:
        w.append('Blade not found; the blade axis is a guess from the key shape. Check it in the window.')
    if run.job.fold:
        w.append('The fold path of the blade is kept free: the blade and its root were turned through the whole fold and the '
                 'jacket is cut wherever they pass, with %.1f mm margin. Inside the swing the fold-side wall stops below the '
                 'blade, so that side has no lip there%s. Check the fit; if the blade can stay open, make the jacket '
                 'without the pivot mark.' % (cavity.fold_margin(run.P),
                                              '; the nose hook holds that side' if run.jinfo.get('hood')
                                              else ' and holds the key less well'))
    if run.job.fold is None and getattr(run, 'fold_hint', False):
        w.append('This may be a flip key: mark the blade pivot so the fold stays free.')
    return w


# ------------------------------------------------------------------------------------------------ report
def write_all(run, out, name, t_report=None):
    import time
    t_report = t_report or time.time()
    out = Path(out)
    files = run.files
    pics = {}
    pics['render'] = out / (name + '_render.png')
    renders(run, pics['render'])
    pics['section'] = out / (name + '_section.png')
    section(run, pics['section'])
    pics['deviation'] = out / (name + '_deviation_map.png')
    dev = deviation_map(run, pics['deviation'])
    run.deviation = dev
    w = warnings(run)
    run.warn = w
    run._t('report', t_report)
    sl = getattr(run, 'sliced', None) or {}
    times = dict(run.log.get('times') or {})
    times['total'] = round(sum(times.values()), 2)
    data = dict(name=name, scan=str(run.job.scan_path), style=run.winfo, job=run.job.to_dict(), params=run.Pm.to_dict(),
                frame=run.frame.to_dict(), clean=run.clean_info.__dict__, sizes_scan=getattr(run, 'sizes', None),
                calipers=getattr(run, 'cal', None), lips=run.lip, jacket=run.jinfo, mount=run.mount.to_dict(),
                checks=run.checks, deviation=dev, sliced=sl,
                warnings=w, times=times, peak_memory_mb=run.log.get('peak_memory_mb'),
                field_log={k: v for k, v in run.log.items() if k.startswith('field')},
                files={k: str(v) for k, v in files.items()}, pictures={k: str(v) for k, v in pics.items()})
    (out / (name + '_report.json')).write_text(json.dumps(data, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    (out / (name + '_report.md')).write_text(markdown(run, data), encoding='utf-8')
    run.report = data
    return data


def _f(v, fmt='%.1f'):
    return fmt % v if isinstance(v, (int, float)) else str(v)


def markdown(run, d):
    ch = d['checks']
    m = ch['mesh']
    cl = ch['clearance_design']
    cs = ch['clearance_scan']
    sn = ch['snap_in']
    lip = d['lips']
    f = lip.get('forces', {})
    wi = d['style']
    sl = d['sliced'] or {}
    jac = sl.get('jacket', {})
    L = []
    L.append('# FobFit report: %s' % d['name'])
    L.append('')
    L.append('Scan: `%s`. Style: %s (%s). Lever arm %.0f mm.' % (Path(d['scan']).name, wi['label'], wi['use'], wi['lever_mm']))
    L.append('')
    if d['warnings']:
        L.append('## Warnings')
        L.append('')
        for x in d['warnings']:
            L.append('- ' + x)
        L.append('')
    else:
        L.append('No warnings.')
        L.append('')
    L.append('## The part')
    L.append('')
    L.append('| Item | Value |')
    L.append('|---|---|')
    L.append('| Size (print pose, x y z) | %s mm |' % ' x '.join('%.1f' % v for v in m['size_mm']))
    L.append('| Mass | %s |' % ('%.1f g sliced (PETG, Bambu Lab printer, 0.4 mm nozzle, %s)' % (jac['mass_g'], jac.get('profile', '')) if jac.get('mass_g') else '%.1f g if solid; slice for the real mass' % m['mass_g_solid']))
    if jac.get('time_min'):
        L.append('| Print time | %.0f min (Bambu Studio estimate) |' % jac['time_min'])
    L.append('| Mesh | %s, %d body, %s |' % ('watertight' if m['watertight'] else 'NOT watertight', m['bodies'], m['manifold_status'].replace('Error.NoError', 'manifold')))
    L.append('| Print pose | flat back on the bed, opening up; no supports |')
    L.append('')
    L.append('## Fit')
    L.append('')
    L.append('Gap between the key and the cavity, target %.2f mm, over the %s %% of the key surface the jacket covers:' % (cl['target'], _f(cl.get('covered_pct'), '%.0f')))
    L.append('')
    L.append('| Surface | mean | 5 % | median | 95 % | min | within 0.1 mm of target |')
    L.append('|---|---|---|---|---|---|---|')
    for nm, s in (('Smoothed scan (what the cavity follows)', cl), ('Raw scan, with its noise', cs)):
        L.append('| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.0f %% |' % (nm, s['mean'], s['p05'], s['p50'], s['p95'], s['min'], s['within_0_1_pct']))
    L.append('')
    dv = d['deviation']
    L.append('Deviation map: `%s`. A cavity made from three caliper sizes (the best rounded box, %s mm, edge radius %.1f mm) '
             'puts %.0f %% of the same surface within 0.1 mm of the target gap; this scan cavity puts %.0f %% there. '
             'Median gap: caliper box %.2f mm, scan cavity %.2f mm.' % (Path(d['pictures']['deviation']).name,
                                                                       ' x '.join('%.1f' % v for v in dv['caliper_box']['size_mm']),
                                                                       dv['caliper_box']['edge_radius_mm'], dv['box_within_0_1_pct'],
                                                                       dv['scan_within_0_1_pct'], dv['box_median'], dv['scan_median']))
    if dv.get('measured_only'):
        L.append('These figures count measured surface only; with the invented surface included, the scan cavity puts '
                 '%.0f %% within 0.1 mm.' % dv['scan_within_0_1_all_covered_pct'])
    L.append('')
    cal = d.get('calipers') or {}
    if cal.get('rows'):
        L.append('Scale: checked against caliper sizes (%s); scale factor used %.4f.'
                 % (cal.get('note', ''), d['frame'].get('caliper_scale', 1.0)))
    else:
        L.append('Scale: not checked against a caliper measurement. The scan sizes are used as they are; the first jacket '
                 'is the first physical check.')
    L.append('')
    inv = ch.get('invented')
    L.append('## Invented surface')
    L.append('')
    if inv:
        L.append('The raw point cloud `%s` (%d points) marks scan surface that the scanner software made up: no raw '
                 'point within %.1f mm. %.0f %% of the surface the jacket covers is invented; it is purple in the '
                 'deviation map. The claim that the cavity follows the scan holds only on measured surface.'
                 % (inv['cloud'], inv['cloud_points'], inv['radius_mm'], inv['covered_pct']))
        L.append('')
        L.append('| Fit surface | invented | area |')
        L.append('|---|---|---|')
        for k, r in inv['regions'].items():
            L.append('| %s | %.1f %% | %.0f mm2 |' % (REGION_NAMES.get(k, k), r['invented_pct'], r['invented_mm2']))
        if 'measured_within_0_1_pct' in inv:
            L.append('')
            L.append('On measured surface only: %.0f %% within 0.1 mm of the target gap (5 %% to 95 %%: %.2f to %.2f mm).'
                     % (inv['measured_within_0_1_pct'], inv['measured_p05_p95'][0], inv['measured_p05_p95'][1]))
    else:
        L.append(ch.get('cloud_note') or 'No raw point cloud was found next to the scan, so invented surface is not '
                 'marked. Put the scanner\'s point cloud (OBJ, PLY or XYZ) next to the scan, or choose it in the window.')
    L.append('')
    L.append('## Lips')
    L.append('')
    sides = [s for s in lip.get('sides', {}).values() if s.get('lip')]
    L.append('| Item | Value |')
    L.append('|---|---|')
    L.append('| Undercut (reach past the widest contour) | %.2f mm%s |' % (lip['undercut_mm'], ' (chosen by the tool)' if lip.get('choice') else ''))
    has = {k: bool(s.get('lip')) for k, s in lip.get('sides', {}).items()}
    one = [k for k, v in has.items() if v] if sum(has.values()) == 1 else []
    yname = {'left': '-y', 'right': '+y'}
    where = ('at %s only (no lip at %s)' % (yname[one[0]], yname['right' if one[0] == 'left' else 'left'])
             if one else 'per side')
    L.append('| Lip length | %s mm %s, from x = %.1f to %.1f |' % (_f(f.get('lip_length_mm')), where, lip['lip_x'][0], lip['lip_x'][1]))
    if sides:
        L.append('| Holding face angle | %.0f deg |' % min(s['hold_angle_deg'] for s in sides))
        L.append('| Strain while snapping in (estimate) | %.2f %% (limit %.1f %%) |' % (f.get('strain_max_pct', 0), 100 * cavity.STRAIN_LIMIT))
    L.append('| Force to snap the key in (estimate) | %s N |' % _f(f.get('snap_in_N'), '%.0f'))
    cam_note = ('; only for a turn that lifts the %s edge: with no lip at %s, a turn the other way is not '
                'calculated' % (yname[one[0]], yname['right' if one[0] == 'left' else 'left'])) if one else ''
    if d['jacket'].get('hood'):
        cam_note += '. The nose hook stops cam-out in both directions, so the lips alone are not the limit'
    L.append('| Torque at which the key could cam out (estimate, lips only) | %s N m (design %.1f N m)%s |' % (_f(f.get('cam_out_Nm'), '%.2f'), wi['design_torque_Nm'], cam_note))
    L.append('')
    L.append('The estimates use a plane-strain model of the open U section (see BUILD_RECORD.md, lip calculation). '
             'Check the gap and the snap on the first jacket.')
    L.append('')
    hd = d['jacket'].get('hood')
    if hd:
        L.append('## Nose hook')
        L.append('')
        L.append('A cap of the jacket covers the key-ring end of the key from x = %.1f to %.1f mm, %.1f mm thick. A turn '
                 'in either direction lifts one edge of the key, and the cap holds that edge down, so the key cannot cam '
                 'out over the lip. To come off, the cap must break. The tool does not calculate the torque that breaks '
                 'it; test the first jacket on the lock.' % (hd['cap_x'][0], hd['cap_x'][1], hd['cap_wall_mm']))
        rx = wi.get('root_x_mm') or {}
        if len(rx) == 2 and abs(rx['+y'][0] - rx['-y'][0]) > 0.5:
            near = '+y' if rx['+y'][0] < rx['-y'][0] else '-y'
            far = '-y' if near == '+y' else '+y'
            L.append('')
            L.append('The %s wing root starts at x = %.1f mm, next to the cap, so a turn that lifts the %s edge goes from '
                     'the wing straight into the cap. The %s root starts at x = %.1f mm, clear of the key ring.'
                     % (near, rx[near][0], near, far, rx[far][0]))
        L.append('')
        L.append('To take the key out: lift the blade end of the key by about %.0f degrees, so the key turns about its '
                 'ring end under the cap; slide the key about %.1f mm toward the blade, out from under the cap; then lift '
                 'it out. To put it in: hook the ring end under the cap first, then press the blade end down until the lip '
                 'snaps.' % (hd['tilt_deg'], hd['slide_mm']))
        L.append('')
    L.append('## Wing')
    L.append('')
    L.append('| Item | Value |')
    L.append('|---|---|')
    L.append('| Lever arm, blade axis to push point | %.1f mm |' % wi['lever_mm'])
    L.append('| Force at the push point for %.1f N m | %.1f N (limit 22.2 N, ADA 309.4) |' % (wi['design_torque_Nm'], wi['force_N']))
    if 'root_stress_MPa' in wi:
        L.append('| Root stress at the design torque / at 2.5 N m | %.1f / %.1f MPa (printed PETG about %.0f MPa, assumed) |'
                 % (wi['root_stress_MPa'], wi['root_stress_at_2_5Nm_MPa'], wi['strength_assumed_MPa']))
        L.append('| Tip deflection at the design force | %.1f mm |' % wi['tip_deflection_mm'])
    wg = ch.get('wing', {})
    if wg.get('ok'):
        L.append('| Measured on the mesh: square to the blade axis | %.2f deg off |' % wg['square_error_deg'])
        L.append('| Measured on the mesh: lever arm | %s mm |' % ', '.join('%.1f' % v for v in wg['lever_mm']))
    L.append('')
    L.append('## Checks on the exported part')
    L.append('')
    hp = ch.get('hood_path')
    if hp:
        L.append('- Nose-hook path: %d points of the key were moved out of the exported jacket along the path (turn %.0f '
                 'degrees about x = %.1f, z = %.1f mm, slide %.1f mm, lift) in %d steps. The lips must flex up to %.2f mm '
                 '(designed undercut %.2f mm); outside the lip zones nothing must flex more than %.2f mm.'
                 % (hp['points'], hp['tilt_deg'], hp['hinge'][0], hp['hinge'][1], hp['slide_mm'], hp['poses'],
                    hp['max_in_lip_zone_mm'], lip['undercut_mm'], hp['max_outside_lip_zone_mm']))
    L.append('- Snap-in%s: on %d sections of the exported jacket, 1 mm apart, the key section was lifted out along the snap '
             'direction. The lips must flex up to %.2f mm (designed undercut %.2f mm); outside the lip zones nothing must flex '
             'more than %.2f mm (the raster step is %.2f mm).'
             % (' behind the nose hook' if hp else '', sn['sections'], sn['flex_in_lip_zone_max_mm'], lip['undercut_mm'],
                sn['flex_outside_lip_zone_max_mm'], sn['raster_mm']))
    L.append('- Seated key: %d of the covered key points touch or cut into the jacket.' % cl.get('interference_pts', 0))
    for k, v in ch['zones'].items():
        sw = ('; the turning part of the key (%d scan and point-cloud points) was turned through the whole fold, %.0f degrees '
              'in %.0f degree steps: smallest gap to the jacket %.2f mm, at %.0f degrees'
              % (v['swept_points'], v['swept_deg'], v['swept_step_deg'], v['swept_min_gap_mm'], v['swept_worst_deg'])
              if 'swept_min_gap_mm' in v else '')
        L.append('- %s zone: %s%s.' % (k.replace('_', ' ').capitalize(), 'free' if v['deepest_mm'] <= 0.05 else 'material %.2f mm deep' % v['deepest_mm'], sw))
    L.append('')
    tm = d.get('times') or {}
    pk = d.get('peak_memory_mb') or {}
    if tm:
        L.append('## Run time and memory')
        L.append('')
        L.append('Time of each stage of this run, and the peak memory of the program up to the end of that stage.')
        L.append('')
        L.append('| Stage | time | peak memory so far |')
        L.append('|---|---|---|')
        for k, v in tm.items():
            if k == 'total':
                continue
            nm = ('Distance field, %s mm grid' % k[6:]) if k.startswith('field_') else STAGE_NAMES.get(k, k.replace('_', ' '))
            L.append('| %s | %.1f s | %s |' % (nm, v,
                                               ('%d MB' % pk[k]) if pk.get(k) is not None else '-'))
        L.append('| **Total** | **%.0f s** | **%s** |' % (tm.get('total', 0), ('%d MB' % max(v for v in pk.values() if v))
                                                          if any(pk.values()) else '-'))
        L.append('')
    L.append('## Files')
    L.append('')
    for k, v in d['files'].items():
        L.append('- `%s`' % Path(v).name)
    for k, v in d['pictures'].items():
        L.append('- `%s`' % Path(v).name)
    L.append('')
    return '\n'.join(L)
