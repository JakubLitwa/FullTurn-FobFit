"""Slicing with the Bambu Studio command line for a Bambu Lab H2S (0.4 mm nozzle) in PETG.

The command line resolves neither `inherits` nor `include` in the profiles it is given. Loaded as they are, the
machine profile slices with a stub start G-code and no filament load, and the process slices its inherited keys at
compiled defaults. So every profile is flattened here first (the recipe of PaddleUp/scripts/slice_parts.py).
The CLI sometimes writes its result and never exits: our own process is stopped once its output is 20 s old.
Other Bambu Studio processes (the designer's) are never touched.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

CANDIDATES = [r'C:\Program Files\Bambu Studio\bambu-studio.exe', '/Applications/BambuStudio.app/Contents/MacOS/BambuStudio',
              '/usr/bin/bambu-studio']
MACHINE = 'Bambu Lab H2S 0.4 nozzle'
PROCESS = '0.20mm Standard @BBL H2S'
FILAMENT = 'Bambu PETG HF @BBL H2S'
# jacket: 3 walls (1.26 mm) plus gap fill make the 1.8 mm sides solid; 15 % gyroid inside the back and the wing.
# 4 walls cost 0.7 g on a 17 g jacket and change nothing structural (sliced 2026-11-11, BUILD_RECORD.md)
OVERRIDES = {'wall_loops': '3', 'sparse_infill_density': '15%', 'sparse_infill_pattern': 'gyroid', 'enable_support': '0'}


def find_exe():
    for c in CANDIDATES:
        if Path(c).exists():
            return c
    w = shutil.which('bambu-studio')
    return w


def profiles_dir(exe):
    p = Path(exe).parent / 'resources' / 'profiles' / 'BBL'
    if not p.exists():
        p = Path(exe).parents[1] / 'Resources' / 'profiles' / 'BBL'
    return p


def index(bbl):
    idx = {}
    for sub in ('machine', 'process', 'filament'):
        for f in (bbl / sub).glob('*.json'):
            try:
                d = json.loads(f.read_text(encoding='utf-8'))
            except Exception:
                continue
            if 'name' in d:
                idx[(sub, d['name'])] = d
    return idx


def resolve(idx, sub, name):
    d = dict(idx[(sub, name)])
    if d.get('inherits'):
        base = resolve(idx, sub, d['inherits'])
        base.update({k: v for k, v in d.items() if k != 'inherits'})
        d = base
    d.pop('inherits', None)
    return d


def machine(idx, name=MACHINE):
    d = dict(idx[('machine', name)])
    base = machine(idx, d['inherits']) if d.get('inherits') else {}
    for inc in d.get('include') or []:
        base.update({k: v for k, v in idx[('machine', inc)].items() if k not in ('name', 'instantiation', 'from', 'setting_id')})
    base.update({k: v for k, v in d.items() if k not in ('inherits', 'include')})
    return base


def process(idx, overrides, name=PROCESS):
    d = resolve(idx, 'process', name)
    d.update(overrides)
    d['curr_bed_type'] = 'Textured PEI Plate'
    return d


def filament(idx, name=FILAMENT):
    d = resolve(idx, 'filament', name)
    d['from'] = 'system'
    d.pop('compatible_printers', None)
    d.pop('compatible_printers_condition', None)
    return d


def _stop(p, od):
    try:
        import psutil
        pr = psutil.Process(p.pid)
        if 'bambu' not in pr.name().lower() or str(od) not in ' '.join(pr.cmdline()):
            return False
    except Exception:
        pass
    p.kill()
    p.wait()
    return True


def _popen(cmd, lf):
    """Start the slicer. In FobFit.exe (PyInstaller) the exe's own DLL folder is on the DLL search path, and a child
    program inherits it: Bambu Studio then loads the exe's older msvcp140.dll and stops with error 127. So the exe
    starts it with the normal search path, then puts its own back. The Python run starts it as before."""
    if not (getattr(sys, 'frozen', False) and sys.platform == 'win32'):
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    import ctypes
    import os
    k32 = ctypes.windll.kernel32
    buf = ctypes.create_unicode_buffer(32768)
    n = k32.GetDllDirectoryW(len(buf), buf)
    own = os.path.normcase(getattr(sys, '_MEIPASS', '') or '\0')
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join(d for d in env.get('PATH', '').split(os.pathsep)
                                  if d and not os.path.normcase(d).startswith(own))
    k32.SetDllDirectoryW(None)
    try:
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    finally:
        k32.SetDllDirectoryW(buf.value if n else None)


def run_cli(cmd, od, limit=900, settle=20):
    out = od / 'out.3mf'
    t0 = time.time()
    with open(od / 'cli.log', 'ab') as lf:
        p = _popen(cmd, lf)
        while True:
            try:
                p.wait(timeout=3)
                break
            except subprocess.TimeoutExpired:
                pass
            if out.exists() and (od / 'result.json').exists() and time.time() - out.stat().st_mtime > settle:
                _stop(p, od)
                break
            if time.time() - t0 > limit:
                _stop(p, od)
                break
    return round(time.time() - t0, 1)


def mark_modified(path, idx, overrides, name=PROCESS):
    """List the overridden keys in the 3MF's different_settings_to_system, or the Bambu Studio GUI maps the
    project back to the unmodified system preset and drops them."""
    base = resolve(idx, 'process', name)
    keys = sorted(k for k, v in overrides.items() if str(base.get(k)) != str(v))
    with zipfile.ZipFile(path) as z:
        items = [(i, z.read(i.filename)) for i in z.infolist()]
    out = []
    for i, data in items:
        if i.filename == 'Metadata/project_settings.config':
            cfg = json.loads(data)
            diff = list(cfg.get('different_settings_to_system') or ['', '', ''])
            diff[0] = ';'.join(keys)
            cfg['different_settings_to_system'] = diff
            data = json.dumps(cfg, indent=4).encode('utf-8')
        out.append((i, data))
    tmp = path.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for i, data in out:
            z.writestr(i, data)
    tmp.replace(path)
    return keys


def gcode_facts(path):
    """Counts that show a full H2S start sequence was used (stub start G-code has none of these)."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('.gcode')]
        if not names:
            return {}
        g = z.read(names[0]).decode('utf-8', 'ignore')
    return dict(lines=g.count('\n'), M620=g.count('M620'), G29=g.count('G29'), M1002=g.count('M1002'))


def slice_stl(stl, out_3mf, overrides=None, work=None, exe=None):
    """Slice one STL (already in its print pose) to a Bambu 3MF project with G-code. Returns mass, time, facts."""
    exe = exe or find_exe()
    if not exe:
        return dict(error='Bambu Studio not found')
    bbl = profiles_dir(exe)
    idx = index(bbl)
    ov = dict(OVERRIDES, **(overrides or {}))
    od = Path(work or tempfile.mkdtemp(prefix='fobfit_slice_'))
    od.mkdir(parents=True, exist_ok=True)
    for f in list(od.glob('*.3mf')) + [od / 'result.json']:
        f.unlink(missing_ok=True)
    mf = od / 'machine.json'
    pf = od / 'process.json'
    ff = od / 'filament.json'
    mf.write_text(json.dumps(machine(idx), indent=1), encoding='utf-8')
    pf.write_text(json.dumps(process(idx, ov), indent=1), encoding='utf-8')
    ff.write_text(json.dumps(filament(idx), indent=1), encoding='utf-8')
    cmd = [exe, '--slice', '0', '--load-settings', '%s;%s' % (mf, pf), '--load-filaments', str(ff),
           '--outputdir', str(od), '--export-3mf', 'out.3mf', str(stl)]
    secs = run_cli(cmd, od)
    try:
        r = json.loads((od / 'result.json').read_text())
        pl = r['sliced_plates'][0]
        mass = sum(f['total_used_g'] for f in pl['filaments'])
        res = dict(mass_g=round(mass, 1), time_min=round(pl['total_predication'] / 60, 1),
                   profile='%s, %s, %s walls, %s %s infill' % (PROCESS.split(' @')[0], FILAMENT.split(' @')[0],
                                                               ov['wall_loops'], ov['sparse_infill_density'], ov['sparse_infill_pattern']),
                   warning=pl.get('warning_message', ''), cli_s=secs)
    except Exception as e:
        return dict(error='slicing failed: %s (see %s)' % (e, od / 'cli.log'), cli_s=secs)
    shutil.copyfile(od / 'out.3mf', out_3mf)
    res['keys_marked'] = mark_modified(Path(out_3mf), idx, ov)
    res['gcode'] = gcode_facts(Path(out_3mf))
    res['file'] = str(out_3mf)
    return res


def for_run(run, out):
    """Slice the jacket and the coupon of a pipeline run; the sliced projects replace the plain 3MFs."""
    res = {}
    out = Path(out)
    for key, stl_key in (('jacket', 'jacket_stl'), ('coupon', 'coupon_stl')):
        stl = run.files.get(stl_key)
        if not stl:
            continue
        target = Path(stl).with_suffix('.3mf')
        import os, uuid
        work = Path(tempfile.gettempdir()) / 'fobfit_slice' / ('%s_%d_%s' % (Path(stl).stem, os.getpid(), uuid.uuid4().hex[:6]))
        res[key] = slice_stl(stl, target, work=work)    # own folder per run: parallel runs must not share it
        if 'error' not in res[key]:
            run.files[stl_key.replace('_stl', '_3mf')] = target
    return res
