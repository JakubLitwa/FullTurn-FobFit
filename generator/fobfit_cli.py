"""FobFit from the command line: a 3D scan of a car key in, a printable FobFit jacket out.

  python fobfit_cli.py key_scan.stl
  python fobfit_cli.py key_scan.ply --style paddle --calipers 62.0,36.1,16.2 --apply-scale --slice
  python fobfit_cli.py key_scan.obj --settings key_scan_fobfit/key_scan_FobFit_settings.json   (repeat a window run)

Points (--button, --ring, --fold) are in the key frame the tool reports: x along the blade (+ toward the tip),
z out of the button face, origin on the blade axis at the middle of the body. The window sets them by clicking.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fobfit import pipeline, wings, slicer  # noqa: E402


def floats(s, n=None):
    v = [float(x) for x in s.split(',')]
    if n and len(v) != n:
        raise argparse.ArgumentTypeError('expected %d numbers, got %s' % (n, s))
    return v


def parse(argv=None):
    ap = argparse.ArgumentParser(description='Make a FobFit jacket from a key scan.')
    ap.add_argument('scan', help='STL, PLY or OBJ scan of the key')
    ap.add_argument('--out', default='', help='output folder (default: out/<scan name>_<style> in the FobFit folder)')
    ap.add_argument('--cloud', default='', help='raw point cloud of the scan (OBJ/PLY/XYZ) to mark invented surface; '
                                                'default: found next to the scan; "none" = do not use one')
    ap.add_argument('--style', default='wing', choices=list(wings.STYLES), help='wing (ignition), paddle (door lock), loop (finger ring)')
    ap.add_argument('--lever', type=float, default=0.0, help='lever arm in mm (default: 45 wing, 70 paddle, 40 loop)')
    ap.add_argument('--side', type=int, default=1, choices=(1, -1), help='side of a one-sided wing: 1 = +y, -1 = -y')
    ap.add_argument('--units', default='auto', choices=['auto', 'mm', 'cm', 'm', 'inch'])
    ap.add_argument('--calipers', type=lambda s: floats(s, 3), help='body length, largest width, largest thickness (mm)')
    ap.add_argument('--apply-scale', action='store_true', help='rescale the scan to the caliper sizes')
    ap.add_argument('--scale', type=float, default=0.0, help='rescale the scan by this factor')
    ap.add_argument('--flip-face', action='store_true', help='the button face is on the other side')
    ap.add_argument('--swap-ends', action='store_true', help='the blade is at the other end')
    ap.add_argument('--button', action='append', type=lambda s: floats(s, 7), default=[],
                    help='keep a button free: x,y,z,nx,ny,nz,radius (key frame)')
    ap.add_argument('--ring', type=lambda s: floats(s, 3), help='key-ring hole x,y,radius (key frame)')
    ap.add_argument('--no-ring', action='store_true', help='the key has no ring hole')
    ap.add_argument('--fold', type=lambda s: floats(s, 4), help='flip key: blade pivot x,y,z and the fold side (+1 / -1 in y)')
    ap.add_argument('--clearance', type=float, help='gap between key and cavity (mm, default 0.30)')
    ap.add_argument('--undercut', type=float, help='lip reach past the widest contour (mm, default: chosen)')
    ap.add_argument('--wall', type=float, help='side wall (mm, default 1.8)')
    ap.add_argument('--back', type=float, help='thinnest back (mm, default 1.6)')
    ap.add_argument('--pitch', type=float, help='grid step (mm, default 0.2)')
    ap.add_argument('--slice', action='store_true', help='slice the jacket with Bambu Studio (PETG) for the real mass and print time')
    ap.add_argument('--settings', help='settings JSON written by an earlier run (the window writes one)')
    return ap.parse_args(argv)


def job_from(a):
    params = {}
    if a.settings:
        s = json.loads(Path(a.settings).read_text())
        j = s.get('job', {})
        job = pipeline.Job(**{k: v for k, v in j.items() if k in pipeline.Job.__dataclass_fields__})
        job.scan_path = a.scan
        params = dict(s.get('params', {}))
        params.pop('pitch', None)
    else:
        job = pipeline.Job(scan_path=a.scan)
    job.style = a.style if a.style != 'wing' or not a.settings else job.style
    if a.out:
        job.out_dir = a.out
    if a.cloud:
        job.cloud_path = a.cloud
    for k in ('lever', 'side'):
        if getattr(a, k) not in (0.0, 1, None) or not a.settings:
            setattr(job, k, getattr(a, k))
    if a.units != 'auto':
        job.unit = a.units
    if a.calipers:
        job.calipers = dict(zip(('length', 'width', 'thickness'), a.calipers))
    job.apply_scale = job.apply_scale or a.apply_scale
    if a.scale:
        job.scale = a.scale
    job.flip_face = a.flip_face
    job.swap_ends = a.swap_ends
    for b in a.button:
        job.buttons.append(dict(p=b[0:3], n=b[3:6], r=b[6]))
    if a.no_ring:
        job.ring = dict(found=False)
    elif a.ring:
        job.ring = dict(x=a.ring[0], y=a.ring[1], r=a.ring[2], found=True)
    if a.fold:
        job.fold = dict(pivot=a.fold[0:3], side=int(a.fold[3]))
    for k in ('clearance', 'undercut', 'wall', 'back', 'pitch'):
        if getattr(a, k) is not None:
            params[k] = getattr(a, k)
    job.params = params
    job.coupon = False
    job.slice = a.slice
    return job


def main(argv=None):
    a = parse(argv)
    job = job_from(a)
    t0 = time.time()

    def prog(f, msg):
        print('  %3.0f %%  %-22s %5.1f s' % (100 * f, msg, time.time() - t0), flush=True)
    run = pipeline.Run(job, progress=prog)
    run.load()
    fr = run.frame
    print('Scan: %s, %s, %d -> %d triangles, noise %.2f mm. Body %.1f mm long, blade %s.'
          % (Path(job.scan_path).name, run.unit, run.clean_info.faces_in, run.clean_info.faces_out, run.clean_info.noise_mm,
             fr.body_x[1] - fr.body_x[0], 'found' if fr.blade_found else 'NOT found'))
    print('Point cloud: %s' % (('%s, %d points' % (Path(run.cloud_file).name, len(run.cloud_mm))) if run.cloud_file
                               else (run.cloud_note or 'none')))
    if job.calipers:
        run.preview()
        print('Calipers:', '; '.join('%s scan %.2f / caliper %.2f (%+.2f %%)' % (r['name'], r['scan'], r['caliper'], r['diff_pct'])
                                     for r in run.cal['rows']))
        if run.cal.get('note'):
            print('  ' + run.cal['note'])
    run.make()
    if job.calipers and job.apply_scale:
        print('Rescaled by x%.5f (from the fine field).' % run.scale_applied)
    run.write(slicer=slicer.for_run if job.slice else None)
    ch = run.checks
    sl = getattr(run, 'sliced', None) or {}
    print()
    print('Jacket: %s, %s mm, %s.' % (run.winfo['label'], ' x '.join('%.1f' % v for v in ch['mesh']['size_mm']),
                                     ('%.1f g, %.0f min sliced' % (sl['jacket']['mass_g'], sl['jacket']['time_min']))
                                     if sl.get('jacket', {}).get('mass_g') else '%.1f g if solid' % ch['mesh']['mass_g_solid']))
    c = ch['clearance_design']
    print('Gap: target %.2f, mean %.3f, 5-95 %% %.3f-%.3f mm, %.0f %% within 0.1 mm.' % (c['target'], c['mean'], c['p05'], c['p95'], c['within_0_1_pct']))
    inv = ch.get('invented')
    if inv:
        print('Invented surface (no raw point within %.1f mm): %.0f %% of the covered scan; %s.'
              % (inv['radius_mm'], inv['covered_pct'], ', '.join('%s %.0f %%' % (k.replace('_', ' '), r['invented_pct'])
                                                                 for k, r in inv['regions'].items())))
    tm = run.report.get('times', {})
    pk = run.report.get('peak_memory_mb') or {}
    print('Run time %.0f s (%s); peak memory %s MB.' % (tm.get('total', 0), ', '.join('%s %.0f s' % (k, v) for k, v in tm.items()
                                                                                   if k != 'total' and v >= 1),
                                                       max([v for v in pk.values() if v] or [0])))
    f = run.lip['forces']
    print('Lips: undercut %.2f mm, snap-in about %s N, cam-out about %s N m (design %.1f).'
          % (run.lip['undercut_mm'], f.get('snap_in_N'), f.get('cam_out_Nm'), run.winfo['design_torque_Nm']))
    for w in run.warn:
        print('WARNING: ' + w)
    print('Files in %s' % run.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
