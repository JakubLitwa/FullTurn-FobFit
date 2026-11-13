"""FobFit window: choose a 3D scan of a car key, check what the tool found, press Make FobFit.

  python fobfit_gui.py                 (or the FobFit shortcut, or double-click FobFit.bat: no console window)
  Errors and printed output are also written to fobfit_log.txt in this folder.
  python fobfit_gui.py --demo demo.json --shots folder      (scripted run that saves screenshots; for testing)

3D view: left-drag turns, right-drag or the wheel zooms, middle-drag (or shift + left-drag) moves.
"""
import argparse
import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ------------------------------------------------------------------------------------------------ error log
# The shortcut starts FobFit with pythonw, which has no console window. Then print() output and Python errors
# would vanish, so they go to fobfit_log.txt next to this file, and an error also shows a short box naming it.
# FobFit.exe (PyInstaller): next to the exe, not inside its _internal folder
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
LOG = APP_DIR / 'fobfit_log.txt'
QUIET = False                           # True in a scripted --demo run: errors go to the log only, no boxes


def _open_log():
    global LOG
    for p in (LOG, Path(os.environ.get('TEMP', '.')) / 'fobfit_log.txt'):     # the FobFit folder, else TEMP
        try:
            if p.exists() and p.stat().st_size > 1_000_000:                    # keep the newest 200 kB
                p.write_bytes(p.read_bytes()[-200_000:])
            f = open(p, 'a', encoding='utf-8', errors='replace', buffering=1)
            f.write('\n==== FobFit started %s with %s ====\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), sys.executable))
            LOG = p
            return f
        except OSError:
            continue
    return None


LOG_FILE = _open_log()
if LOG_FILE is not None:
    if sys.stdout is None:              # pythonw: no console
        sys.stdout = LOG_FILE
    if sys.stderr is None:
        sys.stderr = LOG_FILE


def log_error(text):
    """Write an error with its time to the log (and to the console when there is one)."""
    line = '%s  %s\n' % (time.strftime('%H:%M:%S'), text.rstrip())
    if LOG_FILE is not None:
        try:
            LOG_FILE.write(line)
        except (OSError, ValueError):
            pass
    if sys.stderr is not None and sys.stderr is not LOG_FILE:
        try:
            sys.stderr.write(line)
        except (OSError, ValueError):
            pass


def system_box(title, text):
    """A Windows message box that needs no Tk (for errors before or outside the window)."""
    if QUIET:
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)       # MB_ICONERROR
    except Exception:
        pass


def _uncaught(kind, tb):
    log_error('%s\n%s' % (kind, tb))
    system_box('FobFit', 'FobFit stopped because of an error.\n\nThe details are in:\n%s' % LOG)


sys.excepthook = lambda t, v, tb: _uncaught('Error', ''.join(traceback.format_exception(t, v, tb)))

try:                                    # crisp window and screenshots that match on scaled displays
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass
try:                                    # own taskbar group, not Python's (the icon: see set_icon)
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('FobFit.Generator')
except Exception:
    pass

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import numpy as np
    from PIL import Image, ImageTk
    import vtk

    from fobfit import pipeline, scan, wings, render, geom, slicer, cavity, report
except Exception:                       # a missing package: say so, even without a console
    log_error('FobFit could not start:\n' + traceback.format_exc())
    system_box('FobFit', 'FobFit could not start: a Python package is missing or broken.\n\n'
                         'The details are in:\n%s' % LOG)
    raise SystemExit(1)

if LOG_FILE is not None:                # VTK warnings: to the log, not to a pop-up "vtkOutputWindow"
    _vtk_log = vtk.vtkFileOutputWindow()
    _vtk_log.SetFileName(str(LOG))
    _vtk_log.SetAppend(1)
    _vtk_log.SetFlush(1)
    vtk.vtkOutputWindow.SetInstance(_vtk_log)

STYLE_TEXT = {'wing': 'Wing: two short wings (ignition)', 'paddle': 'Paddle: one long wing (stiff door lock)',
              'loop': 'Loop: ring for a hooked finger or thumb'}


# ------------------------------------------------------------------------------------------------ 3D view
class View3D:
    """Offscreen VTK scene shown in a Tk canvas, with mouse turning, zooming, moving and picking."""

    def __init__(self, parent, on_pick):
        self.canvas = tk.Canvas(parent, bg='white', highlightthickness=0, cursor='fleur')
        self.canvas.pack(fill='both', expand=True)
        self.scene = render.Scene(900, 700, bg=(0.97, 0.97, 0.98))
        self.on_pick = on_pick
        self.on_miss = None
        self.photo = None
        self.pick_actor = None
        self.press = None
        self.pending = False
        self.canvas.bind('<Configure>', self._resize)
        self.canvas.bind('<ButtonPress-1>', lambda e: self._down(e, 1))
        self.canvas.bind('<ButtonPress-2>', lambda e: self._down(e, 2))
        self.canvas.bind('<ButtonPress-3>', lambda e: self._down(e, 3))
        for b in (1, 2, 3):
            self.canvas.bind('<B%d-Motion>' % b, self._drag)
            self.canvas.bind('<ButtonRelease-%d>' % b, self._up)
        self.canvas.bind('<MouseWheel>', self._wheel)
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)

    def _resize(self, e):
        if e.width > 50 and e.height > 50:
            self.scene.rw.SetSize(e.width, e.height)
            self.redraw()

    def _down(self, e, b):
        self.press = (e.x, e.y, b, e.x, e.y, bool(e.state & 0x0001))

    def _drag(self, e):
        if not self.press:
            return
        x0, y0, b, lx, ly, shift = self.press
        dx, dy = e.x - lx, e.y - ly
        cam = self.scene.ren.GetActiveCamera()
        if b == 1 and not shift:
            cam.Azimuth(-dx * 0.4)
            cam.Elevation(dy * 0.4)
            cam.OrthogonalizeViewUp()
        elif b == 3:
            cam.Dolly(1.0 + dy * 0.005)
        else:
            fp, pos = np.array(cam.GetFocalPoint()), np.array(cam.GetPosition())
            d = np.linalg.norm(pos - fp)
            up = np.array(cam.GetViewUp())
            right = np.cross(fp - pos, up)
            right /= np.linalg.norm(right)
            s = d * 0.0015
            mv = -right * dx * s + up * dy * s
            cam.SetFocalPoint(*(fp + mv))
            cam.SetPosition(*(pos + mv))
        self.scene.ren.ResetCameraClippingRange()
        self.press = (x0, y0, b, e.x, e.y, shift)
        self.redraw()

    def _up(self, e):
        if self.press and self.press[2] == 1 and abs(e.x - self.press[0]) < 4 and abs(e.y - self.press[1]) < 4:
            self._click(e.x, e.y)
        self.press = None

    def _wheel(self, e):
        self.scene.ren.GetActiveCamera().Dolly(1.1 if e.delta > 0 else 1 / 1.1)
        self.scene.ren.ResetCameraClippingRange()
        self.redraw()

    def _click(self, x, y):
        h = self.scene.rw.GetSize()[1]
        self.scene.rw.Render()
        if self.picker.Pick(x, h - 1 - y, 0, self.scene.ren) and self.picker.GetActor() is not None \
                and (self.pick_actor is None or self.picker.GetActor() is self.pick_actor):
            self.on_pick(np.array(self.picker.GetPickPosition()), np.array(self.picker.GetPickNormal()))
        elif self.on_miss:
            self.on_miss()

    def redraw(self):
        if self.pending:
            return
        self.pending = True
        self.canvas.after_idle(self._draw)

    def _draw(self):
        self.pending = False
        img = self.scene.image()
        self.photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def snapshot(self):
        return self.scene.image()


# ------------------------------------------------------------------------------------------------ the app
class App:
    def __init__(self, root):
        self.root = root
        root.title('FobFit generator')
        root.geometry('1440x900')
        self.q = queue.Queue()
        self.run = None
        self.busy = False
        self.pending_load = None    # a scan to read again once the running step ends (a choice made while busy)
        self.mode = None            # picking mode: 'button', 'ring', 'pivot', 'axis1', 'axis2'
        self.axis_pts = []
        self.view_mode = 'key'
        self.key_disp = None
        self.exe = slicer.find_exe()
        self._build()
        self.root.after(100, self._poll)

    # -------------------------------------------------------------------------------------------- layout
    def _build(self):
        st = ttk.Style()
        try:
            st.theme_use('vista')
        except Exception:
            pass
        st.configure('Big.TButton', font=('Segoe UI', 12, 'bold'), padding=8)
        st.configure('H.TLabel', font=('Segoe UI', 10, 'bold'))
        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True)
        left = ttk.Frame(main, width=400, padding=(10, 8))
        left.pack(side='left', fill='y')
        left.pack_propagate(False)
        right = ttk.Frame(main)
        right.pack(side='left', fill='both', expand=True)
        self.view = View3D(right, self._picked)
        self.view.on_miss = self._missed
        bar = ttk.Frame(right)
        bar.pack(fill='x')
        self.vm = tk.StringVar(value='key')
        for v, t in (('key', 'Key'), ('both', 'Jacket on key'), ('jacket', 'Jacket'), ('dev', 'Deviation map')):
            ttk.Radiobutton(bar, text=t, value=v, variable=self.vm, command=self._show).pack(side='left', padx=6, pady=3)
        ttk.Button(bar, text='Reset view', command=lambda: self._show(reset=True)).pack(side='right', padx=6)
        self.hint = ttk.Label(bar, text='', foreground='#a0522d')
        self.hint.pack(side='right', padx=10)

        # 1 scan
        ttk.Label(left, text='1  Scan', style='H.TLabel').pack(anchor='w')
        r = ttk.Frame(left)
        r.pack(fill='x', pady=(2, 0))
        ttk.Button(r, text='Open scan...', command=self.open_scan).pack(side='left')
        ttk.Label(r, text='  units').pack(side='left')
        self.units = tk.StringVar(value='auto')
        cb = ttk.Combobox(r, textvariable=self.units, values=['auto', 'mm', 'cm', 'm', 'inch'], width=6, state='readonly')
        cb.pack(side='left')
        cb.bind('<<ComboboxSelected>>', lambda e: self._reload())
        self.scan_info = ttk.Label(left, text='No scan loaded.', wraplength=375, justify='left')
        self.scan_info.pack(anchor='w', pady=(2, 0))
        r = ttk.Frame(left)
        r.pack(fill='x', pady=(2, 6))
        self.cloud_choice = ''          # '' = the point cloud found next to the scan, 'none' = do not use one
        self.cloud_info = ttk.Label(r, text='Raw point cloud: -', wraplength=230, justify='left')
        self.cloud_info.pack(side='left')
        ttk.Button(r, text='None', width=5, command=lambda: self.set_cloud('none')).pack(side='right')
        ttk.Button(r, text='Choose...', width=8, command=self.choose_cloud).pack(side='right', padx=(0, 4))

        # 2 check
        ttk.Label(left, text='2  Check what the tool found', style='H.TLabel').pack(anchor='w')
        self.found = ttk.Label(left, text='', wraplength=375, justify='left')
        self.found.pack(anchor='w')
        g = ttk.Frame(left)
        g.pack(fill='x', pady=2)
        self.btns = []
        for i, (t, cmd) in enumerate((('Flip button face', self.flip), ('Swap blade / ring ends', self.swap),
                                      ('Pick blade axis (2 clicks)', lambda: self.set_mode('axis1')),
                                      ('Mark a button', lambda: self.set_mode('button')),
                                      ('Mark ring hole', lambda: self.set_mode('ring')),
                                      ('No ring hole', self.no_ring),
                                      ('Mark flip-key pivot', lambda: self.set_mode('pivot')),
                                      ('Fold the other way', self.fold_other_way),
                                      ('Clear marks', self.clear_marks))):
            b = ttk.Button(g, text=t, command=cmd)
            b.grid(row=i // 2, column=i % 2, sticky='ew', padx=1, pady=1)
            self.btns.append(b)
        g.columnconfigure(0, weight=1)
        g.columnconfigure(1, weight=1)
        r = ttk.Frame(left)
        r.pack(fill='x')
        ttk.Label(r, text='mark radius (mm)').pack(side='left')
        self.mark_r = tk.DoubleVar(value=6.0)
        ttk.Spinbox(r, from_=2, to=15, increment=0.5, textvariable=self.mark_r, width=5).pack(side='left', padx=4)
        self.marks_info = ttk.Label(left, text='', wraplength=375, justify='left')
        self.marks_info.pack(anchor='w', pady=(0, 6))

        # 3 calipers
        ttk.Label(left, text='3  Calipers (optional): length, width, thickness (mm)', style='H.TLabel').pack(anchor='w')
        r = ttk.Frame(left)
        r.pack(fill='x', pady=2)
        self.cal = [tk.StringVar() for _ in range(3)]
        for v in self.cal:
            ttk.Entry(r, textvariable=v, width=8).pack(side='left', padx=(0, 4))
        ttk.Button(r, text='Check', command=self.check_cal).pack(side='left')
        self.rescale = tk.BooleanVar(value=False)
        self.rescale_cb = ttk.Checkbutton(left, text='Rescale the scan to the calipers', variable=self.rescale)
        self.rescale_cb.pack(anchor='w')
        self.cal_info = ttk.Label(left, text='', wraplength=375, justify='left')
        self.cal_info.pack(anchor='w', pady=(0, 6))

        # 4 wing
        ttk.Label(left, text='4  Wing style', style='H.TLabel').pack(anchor='w')
        self.style = tk.StringVar(value='wing')
        for k in ('wing', 'paddle', 'loop'):
            ttk.Radiobutton(left, text=STYLE_TEXT[k], value=k, variable=self.style, command=self._style_changed).pack(anchor='w')
        r = ttk.Frame(left)
        r.pack(fill='x', pady=2)
        ttk.Label(r, text='lever arm (mm)').pack(side='left')
        self.lever = tk.DoubleVar(value=wings.STYLES['wing']['lever'])
        ttk.Spinbox(r, from_=30, to=90, increment=1, textvariable=self.lever, width=5).pack(side='left', padx=4)
        ttk.Label(r, text='  side').pack(side='left')
        self.side = tk.StringVar(value='+y')
        ttk.Combobox(r, textvariable=self.side, values=['+y', '-y'], width=4, state='readonly').pack(side='left')
        self.adv_open = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text='Advanced fit settings', variable=self.adv_open, command=self._adv).pack(anchor='w', pady=(4, 0))
        self.adv = ttk.Frame(left)
        self.par = {}
        for i, (k, t, v) in enumerate((('clearance', 'gap (mm)', 0.30), ('undercut', 'lip undercut (0 = choose)', 0.0),
                                       ('wall', 'side wall (mm)', 1.8), ('back', 'thinnest back (mm)', 1.6))):
            ttk.Label(self.adv, text=t).grid(row=i, column=0, sticky='w')
            var = tk.DoubleVar(value=v)
            ttk.Spinbox(self.adv, from_=0, to=5, increment=0.05, textvariable=var, width=6).grid(row=i, column=1, padx=4)
            self.par[k] = var

        # 5 make
        self.make_box = ttk.Frame(left)
        self.make_box.pack(fill='x', pady=(8, 0))
        self.slice_on = tk.BooleanVar(value=bool(self.exe))
        ttk.Checkbutton(self.make_box, text='Slice (Bambu Studio %s)' % ('found' if self.exe else 'not found'),
                        variable=self.slice_on, state='normal' if self.exe else 'disabled').pack(anchor='w')
        self.make_btn = ttk.Button(self.make_box, text='Make FobFit', style='Big.TButton', command=self.make)
        self.make_btn.pack(fill='x', pady=4)
        self.prog = ttk.Progressbar(self.make_box, maximum=100)
        self.prog.pack(fill='x')
        self.status = ttk.Label(self.make_box, text='', wraplength=375, justify='left')
        self.status.pack(anchor='w')
        r = ttk.Frame(self.make_box)
        r.pack(fill='x')
        self.open_btn = ttk.Button(r, text='Open output folder', command=self.open_folder, state='disabled')
        self.open_btn.pack(side='left')
        self.out_dir = tk.StringVar(value='')
        ttk.Button(r, text='Output folder...', command=self.choose_out).pack(side='left', padx=4)
        self.result = tk.Text(left, height=12, wrap='word', font=('Segoe UI', 9), relief='flat', background='#f6f6f6')
        self.result.pack(fill='both', expand=True, pady=(6, 0))
        self.path = None

    def _adv(self):
        if self.adv_open.get():
            self.adv.pack(fill='x', before=self.make_box)
        else:
            self.adv.pack_forget()

    def _style_changed(self):
        self.lever.set(wings.STYLES[self.style.get()]['lever'])

    # -------------------------------------------------------------------------------------------- worker
    def work(self, fn, done=None, label=''):
        if self.busy:
            return
        self.busy = True
        self.make_btn.state(['disabled'])
        self.status.config(text=label)

        def target():
            try:
                res = fn()
                self.q.put(('done', done, res))
            except Exception as e:
                self.q.put(('error', e, traceback.format_exc()))
        threading.Thread(target=target, daemon=True).start()

    def progress(self, f, msg):
        self.q.put(('progress', f, msg))

    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == 'progress':
                    self.prog['value'] = 100 * m[1]
                    self.status.config(text=m[2])
                elif m[0] == 'done':
                    self.busy = False
                    self.make_btn.state(['!disabled'])
                    if m[1]:
                        m[1](m[2])
                elif m[0] == 'error':
                    self.busy = False
                    self.make_btn.state(['!disabled'])
                    self.status.config(text='Error: %s' % m[1])
                    self.result.delete('1.0', 'end')
                    self.result.insert('end', m[2])
                    self.show_error('Error: %s' % m[1], m[2])
                elif m[0] == 'call':
                    m[1]()
        except queue.Empty:
            pass
        finally:                        # an error in a step must not stop the window from listening
            self.root.after(100, self._poll)
        if self.pending_load is not None and not self.busy:
            p, self.pending_load = self.pending_load, None
            self.load(p)

    def show_error(self, what, tb):
        """Write the error to the log and show a short box that names the log (one box at a time)."""
        log_error('%s\n%s' % (what, tb))
        if QUIET or getattr(self, '_box_open', False):
            return
        self._box_open = True
        try:
            messagebox.showerror('FobFit', '%s\n\nThe details are in:\n%s' % (what, LOG), parent=self.root)
        finally:
            self._box_open = False

    # -------------------------------------------------------------------------------------------- steps
    def open_scan(self):
        p = filedialog.askopenfilename(title='Choose a key scan',
                                       filetypes=[('3D scans', '*.stl *.ply *.obj'), ('All files', '*.*')])
        if p:
            self.cloud_choice = ''
            self.load(p)

    def choose_cloud(self):
        p = filedialog.askopenfilename(title='Choose the raw point cloud of this scan',
                                       filetypes=[('Point clouds', '*.obj *.ply *.xyz *.txt *.pts'), ('All files', '*.*')])
        if p:
            self.set_cloud(p)

    def set_cloud(self, p):
        """The raw point cloud marks scan surface the scanner software invented; a change reloads the scan."""
        self.cloud_choice = p
        if not self._reload():
            self.cloud_info.config(text='Raw point cloud: %s' % ('none' if p == 'none' else Path(p).name))

    def _reload(self):
        """Read the current scan again (a new unit or point cloud choice). False when no scan is open."""
        p = self.pending_load or self.path
        if p:
            self.load(p)
        return bool(p)

    def load(self, path):
        if self.busy:
            # a step is still running on the old run: read the scan again with the new choice when it ends
            self.pending_load = path
            self.cloud_info.config(text='Raw point cloud: %s - used once the running step ends.' % (
                {'': 'the one next to the scan', 'none': 'none'}.get(self.cloud_choice, Path(self.cloud_choice).name)))
            return
        self.path = path
        job = pipeline.Job(scan_path=path, unit=self.units.get(), cloud_path=getattr(self, 'cloud_choice', ''))
        run = pipeline.Run(job, progress=self.progress)
        self.run = run
        self.result.delete('1.0', 'end')
        self.open_btn.state(['disabled'])

        def fn():                       # works on its own run: a newer load replaces self.run, not this one
            run.load()
            self.q.put(('call', lambda: self._loaded(run)))
            run.preview()
            return None
        self.work(fn, done=lambda r: self._previewed(run), label='Reading the scan...')

    def _loaded(self, run):
        if run is not self.run:         # a newer scan or choice replaced this one
            return
        r = run
        ci = r.clean_info
        self.scan_info.config(text='%s\n%s; %d triangles; noise about %.2f mm. %s%s' % (
            Path(r.job.scan_path).name, 'units: %s' % r.unit, ci.faces_out, ci.noise_mm,
            ' '.join(ci.notes), (' ' + r.unit_note) if r.unit_note else ''))
        if r.cloud_file:
            ct = 'Raw point cloud: %s (%d points). It marks invented surface.' % (Path(r.cloud_file).name, len(r.cloud_mm))
        elif r.cloud_note:
            ct = r.cloud_note
        else:
            ct = 'Raw point cloud: none%s. Invented surface is not marked.' % (
                '' if r.job.cloud_path == 'none' else ' found next to the scan')
        self.cloud_info.config(text=ct)
        self.key_disp = geom.decimate(r.key, 150000)
        self.vm.set('key')
        self._describe()
        self._show(reset=True)

    def _previewed(self, run=None):
        if run is not None and run is not self.run:
            return
        r = self.run
        L, W, T = r.sizes
        self.status.config(text='Scan body: length %.1f, width %.2f, thickness %.2f mm. Check the arrows, then press Make FobFit.' % (L, W, T))
        self.prog['value'] = 0
        if any(v.get().strip() for v in self.cal):
            self.check_cal()

    def _describe(self):
        r = self.run
        fr = r.frame
        t = ['Blade axis (blue arrow) points to the blade tip%s.' % ('' if fr.blade_found else ' - NO BLADE FOUND, pick the axis'),
             'Button face (green arrow) points up%s.' % ('' if fr.face_confidence >= 1.3 else ' - UNSURE, check it'),
             'Key-ring hole (red circle): %s.' % (('found, %.1f mm across' % (2 * r.ring['r'])) if r.ring.get('found') else 'none'),
             'Body %.1f mm long, blade %s mm.' % (fr.body_x[1] - fr.body_x[0], fr.blade.get('length', '-'))]
        self.found.config(text='\n'.join(t))
        nb = len(r.job.buttons)
        fold = (', flip-key pivot (orange pin; the orange arrow shows the way the blade folds)'
                if r.job.fold else '')
        self.marks_info.config(text='Marks: %d button(s)%s. Buttons on the face stay open anyway; mark side buttons.' % (
            nb, fold))

    def set_mode(self, m):
        if not self.run or self.busy:
            return
        self.mode = m
        self.vm.set('key')
        self._show()
        self.hint.config(text=self.MODE_TEXT[m])

    MODE_TEXT = {'axis1': 'Click the blade tip', 'axis2': 'Click the blade where it leaves the body',
                 'button': 'Click the middle of a button', 'ring': 'Click inside the key-ring hole (on its wall)',
                 'pivot': 'Click the blade pivot (the hinge of the flip key)'}

    def _missed(self):
        if self.run and self.mode:
            self.hint.config(text='That click missed the key. ' + self.MODE_TEXT[self.mode] + ' on the key itself.')

    def _picked(self, p, n):
        if not self.run or not self.mode:
            return
        r = self.run
        m = self.mode
        if m == 'axis1':
            self.axis_pts = [p]
            self.set_mode('axis2')
            return
        if m == 'axis2':
            Minv = np.linalg.inv(r.frame.matrix())
            tip = (Minv @ np.append(self.axis_pts[0], 1))[:3]
            root = (Minv @ np.append(p, 1))[:3]
            r.job.axis_points = [tip.tolist(), root.tolist()]
            self.mode = None
            self.hint.config(text='')
            self.work(lambda: (r.detect(), r.preview()), done=lambda _: self._after_frame(), label='Using the picked axis...')
            return
        if m == 'button':
            r.job.buttons.append(dict(p=p.tolist(), n=(n / np.linalg.norm(n)).tolist(), r=float(self.mark_r.get())))
        elif m == 'ring':
            r.job.ring = dict(found=True, x=float(p[0]), y=float(p[1]), r=max(1.5, float(self.mark_r.get()) / 2.4))
            r.ring = dict(r.job.ring)
        elif m == 'pivot':
            r.job.fold = dict(pivot=[float(p[0]), float(p[1]), 0.0], side=r.fold_side(float(p[0]), float(p[1])))
        self.mode = None
        self.hint.config(text='')
        self._describe()
        self._show()

    def _after_frame(self):
        self.key_disp = geom.decimate(self.run.key, 150000)
        self._describe()
        self._show(reset=True)
        self._previewed()

    def flip(self):
        if self.run and not self.busy:
            self.work(lambda: (self.run.flip(), self.run.preview()), done=lambda _: self._after_frame(), label='Turning the face over...')

    def swap(self):
        if self.run and not self.busy:
            self.work(lambda: (self.run.swap(), self.run.preview()), done=lambda _: self._after_frame(), label='Swapping the ends...')

    def no_ring(self):
        if self.run:
            self.run.job.ring = dict(found=False)
            self.run.ring = dict(found=False)
            self._describe()
            self._show()

    def fold_other_way(self):
        if not self.run:
            return
        f = self.run.job.fold
        if not f:
            self.hint.config(text='Mark the flip-key pivot first.')
            return
        f['side'] = -1 if f['side'] > 0 else 1
        self._describe()
        self._show()

    def clear_marks(self):
        if self.run:
            self.run.job.buttons, self.run.job.fold = [], None
            self._describe()
            self._show()

    def calipers(self):
        out = {}
        for k, v in zip(('length', 'width', 'thickness'), self.cal):
            try:
                if v.get().strip():
                    out[k] = float(v.get().replace(',', '.'))
            except ValueError:
                pass
        return out

    def check_cal(self):
        if not self.run or getattr(self.run, 'sizes', None) is None:
            self.cal_info.config(text='Load a scan first.')
            return
        cal = self.calipers()
        self.run.job.calipers = cal
        res = scan.caliper_check(self.run.sizes, cal)
        self.run.cal = res
        rows = ['%s: scan %.2f, calipers %.2f (%+.2f %%)' % (x['name'], x['scan'], x['caliper'], x['diff_pct']) for x in res['rows']]
        extra = []
        if self.run.clean_info.noise_mm > 0.05:
            extra = ['This scan is noisy: its sizes are sure to about 1 % only.']
        self.cal_info.config(text='\n'.join(rows + ([res['note']] if res.get('note') else ['Scan and calipers agree.']) + extra))
        self.rescale_cb.config(text='Rescale the scan to the calipers (preview: x%.4f; set exactly at Make)' % res['scale'])

    def choose_out(self):
        d = filedialog.askdirectory(title='Output folder')
        if d:
            self.out_dir.set(d)

    def make(self):
        if not self.run or self.busy:
            if not self.run:
                messagebox.showinfo('FobFit', 'Open a scan first.')
            return
        r = self.run
        j = r.job
        j.style = self.style.get()
        j.lever = float(self.lever.get())
        j.side = 1 if self.side.get() == '+y' else -1
        j.out_dir = self.out_dir.get()
        j.calipers = self.calipers()
        params = {k: float(v.get()) for k, v in self.par.items()}
        j.params = params
        r.P = cavity.Params(**params)
        j.apply_scale = bool(self.rescale.get())
        use_slicer = self.slice_on.get() and self.exe

        def fn():
            r.make()
            if r.scale_applied != 1.0:
                self.q.put(('call', lambda: setattr(self, 'key_disp', geom.decimate(r.key, 150000))))
            self.q.put(('call', self._made))
            r.write(slicer=slicer.for_run if use_slicer else None)
            return None
        self.work(fn, done=lambda _: self._written(), label='Making the jacket...')

    def _made(self):
        self.vm.set('both')
        self._show()

    def _written(self):
        r = self.run
        ch = r.checks
        sl = getattr(r, 'sliced', None) or {}
        jac = sl.get('jacket', {})
        cl = ch['clearance_design']
        f = r.lip['forces']
        lines = ['%s jacket, %s mm, %s.' % (r.winfo['label'], ' x '.join('%.0f' % v for v in ch['mesh']['size_mm']),
                                           ('%.1f g, %.0f min (sliced)' % (jac['mass_g'], jac['time_min'])) if jac.get('mass_g')
                                           else '%.1f g if solid' % ch['mesh']['mass_g_solid']),
                 'Mesh: %s, %d body.' % ('watertight' if ch['mesh']['watertight'] else 'NOT WATERTIGHT', ch['mesh']['bodies']),
                 'Gap: target %.2f mm, mean %.2f, 90 %% between %.2f and %.2f mm.' % (cl['target'], cl['mean'], cl['p05'], cl['p95']),
                 'Lips: undercut %.2f mm; snap-in about %s N; holds to about %s N m (estimates)%s.' % (
                     r.lip['undercut_mm'], f.get('snap_in_N'), f.get('cam_out_Nm'),
                     (lambda h: '; lip at %s only, so the hold is for one turning direction'
                      % ('-y' if h.get('left') else '+y') if sum(h.values()) == 1 else '')(
                         {k: bool(s.get('lip')) for k, s in r.lip.get('sides', {}).items()})),
                 'Wing: lever arm %.0f mm, %.1f N at %.1f N m.' % (r.winfo['lever_mm'], r.winfo['force_N'], r.winfo['design_torque_Nm']),
                 'Deviation: %.0f %% of the %s within 0.1 mm of the gap (three-caliper box: %.0f %%).' % (
                     r.deviation['scan_within_0_1_pct'],
                     'measured covered scan' if r.deviation.get('measured_only') else 'covered scan',
                     r.deviation['box_within_0_1_pct'])]
        inv = ch.get('invented')
        if inv:
            lines.append('Invented surface (no raw point within %.1f mm, purple in the deviation view): %.0f %% of the '
                         'covered scan; %s.' % (inv['radius_mm'], inv['covered_pct'], ', '.join(
                             '%s %.0f %%' % (k.replace('_', ' '), v['invented_pct']) for k, v in inv['regions'].items())))
        if r.warn:
            lines.append('')
            lines += ['WARNING: ' + w for w in r.warn]
        lines += ['', 'Files: %s' % r.out]
        self.result.delete('1.0', 'end')
        self.result.insert('end', '\n'.join(lines))
        self.status.config(text='Done.')
        self.open_btn.state(['!disabled'])
        self._show()

    def open_folder(self):
        if self.run and getattr(self.run, 'out', None):
            os.startfile(str(self.run.out)) if os.name == 'nt' else subprocess.Popen(['xdg-open', str(self.run.out)])

    # -------------------------------------------------------------------------------------------- drawing
    def _show(self, reset=False):
        r = self.run
        sc = self.view.scene
        sc.clear()
        self.view.pick_actor = None
        if r is None or self.key_disp is None:
            self.view.redraw()
            return
        mode = self.vm.get()
        has_part = getattr(r, 'part', None) is not None and getattr(r, 'checks', None) is not None
        if mode in ('both', 'jacket', 'dev') and not has_part:
            mode = 'key'
            self.vm.set('key')
        if mode == 'key':
            a = sc.add(render.actor(self.key_disp, color=(0.55, 0.58, 0.63)))
            self.view.pick_actor = a
            self._overlays()
        elif mode == 'both':
            sc.add(render.actor(r.part))
            sc.add(render.actor(self.key_disp, color=render.KEY))
        elif mode == 'jacket':
            sc.add(render.actor(r.part))
        elif mode == 'dev':
            k = geom.decimate(getattr(r, 'key_raw', r.key), 120000)
            gap = r.sdf(k.vertices) - r.Pm.clearance
            cov = gap < 0.8
            vals = np.where(cov, np.clip(gap, -0.3, 0.3), 9.0)
            inv_on = getattr(r, 'cloud_crop', None) is not None and 'invented' in r.dev
            if inv_on:                       # covered surface the scanner software invented: purple
                _, inv = scan.cloud_cover(k.vertices, r.cloud_crop)
                vals = np.where(cov & inv, np.nan, vals)
            lut = render.diverging_lut(-0.3, 0.3)
            lut.SetNanColor(*report.PURPLE, 1)
            lut.SetAboveRangeColor(0.82, 0.82, 0.84, 1.0)
            lut.UseAboveRangeColorOn()
            sc.add(render.actor(k, scalars=vals, lut=lut, rng=(-0.3, 0.3), smooth=False))
            sc.overlay(render.text_actor('gap minus target: red -0.3 mm, green 0, blue +0.3 mm; grey not covered'
                                         + ('; purple invented by the scanner software' if inv_on else ''), 10, 10, 14))
        if reset or not getattr(self, '_viewed', False):
            sc.view((0.9, -1.3, 1.1) if mode != 'dev' else (0.2, -0.4, -1.0))
            self._viewed = True
        else:
            sc.ren.ResetCameraClippingRange()
        self.view.redraw()

    def _overlays(self):
        r = self.run
        sc = self.view.scene
        fr = r.frame
        x0, x1 = fr.body_x
        tip = fr.blade.get('tip_x', x1 + 30) if fr.blade_found else x1 + 30
        zt = float(r.key.bounds[1, 2])
        sc.overlay(render.arrow_actor([x0 - 8, 0, 0], [tip + 10, 0, 0], color=(0.15, 0.35, 0.95), shaft=0.5, tip=1.6))
        sc.overlay(render.arrow_actor([0, 0, 0], [0, 0, zt + 14], color=(0.1, 0.65, 0.2), shaft=0.6, tip=1.8))
        if r.ring.get('found'):
            sc.overlay(render.ring_actor([r.ring['x'], r.ring['y'], zt + 0.5], [0, 0, 1], r.ring['r'] + 0.5))
        for b in r.job.buttons:
            sc.overlay(render.ring_actor(b['p'], b['n'], b['r'], color=(0.95, 0.2, 0.1)))
            sc.overlay(render.sphere_actor(b['p'], 0.8, color=(0.95, 0.2, 0.1)))
        if r.job.fold:
            # the pivot sits on the mid-plane, inside the key: draw a pin through the key, a ball on top and,
            # above the top face, an arc from the open blade (+x) round to the folded blade (-x)
            orange = (0.95, 0.55, 0.0)
            px, py = float(r.job.fold['pivot'][0]), float(r.job.fold['pivot'][1])
            zb = float(r.key.bounds[0, 2])
            sc.overlay(render.line_actor([px, py, zb - 4], [px, py, zt + 9], color=orange, width=5.0))
            sc.overlay(render.sphere_actor([px, py, zt + 9], 1.6, color=orange))
            s = 1.0 if r.job.fold['side'] > 0 else -1.0
            sc.overlay(render.arc_actor([px, py, zt + 2.5], 12.0, 0.0, s * np.pi * 0.92, color=orange))
        sc.overlay(render.text_actor('blue: blade axis, toward the tip    green: button face    red: ring hole and marks'
                                     '    orange: flip-key pivot and fold way', 10, 10, 14))


# ------------------------------------------------------------------------------------------------ scripted demo
def demo(app, script, shots):
    """Drive the window from a JSON script and save screenshots (for testing). Steps:
    {"load": path}, {"flip": true}, {"swap": true}, {"calipers": [L, W, T]}, {"rescale": true}, {"style": "paddle"},
    {"button": [x, y, z, nx, ny, nz, r]}, {"pivot": [x, y, z]}, {"make": true}, {"view": "dev"}, {"shot": "name"},
    {"wait": seconds}"""
    from PIL import ImageGrab
    shots = Path(shots)
    shots.mkdir(parents=True, exist_ok=True)
    steps = json.loads(Path(script).read_text())
    log = []

    def wait_idle(timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            app.root.update()
            if not app.busy and app.q.empty():
                for _ in range(10):
                    app.root.update()
                    time.sleep(0.05)
                if not app.busy:
                    return True
            time.sleep(0.05)
        return False

    def shot(name):
        app.root.update()
        time.sleep(0.3)
        app.root.update()
        x, y = app.root.winfo_rootx(), app.root.winfo_rooty()
        w, h = app.root.winfo_width(), app.root.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(shots / (name + '.png'))
        Image.fromarray(app.view.snapshot()).save(shots / (name + '_view3d.png'))
        log.append(dict(shot=name, t=time.strftime('%H:%M:%S'), status=app.status.cget('text'),
                        hint=app.hint.cget('text'), fold=copy.deepcopy(app.run.job.fold) if app.run else None,
                        buttons=copy.deepcopy(app.run.job.buttons) if app.run else None))

    app.root.lift()
    app.root.attributes('-topmost', True)
    for st in steps:
        k, v = next(iter(st.items()))
        if k == 'load':
            app.load(v)
        elif k == 'flip':
            app.flip()
        elif k == 'swap':
            app.swap()
        elif k == 'calipers':
            for var, x in zip(app.cal, v):
                var.set(str(x))
            app.check_cal()
        elif k == 'rescale':
            app.rescale.set(bool(v))
        elif k == 'style':
            app.style.set(v)
            app._style_changed()
        elif k == 'slice':
            app.slice_on.set(bool(v))
        elif k == 'out':
            app.out_dir.set(v)
        elif k == 'button':
            app.run.job.buttons.append(dict(p=v[0:3], n=v[3:6], r=v[6]))
            app._describe()
            app._show()
        elif k == 'pivot':
            app.run.job.fold = dict(pivot=v[0:3], side=int(v[3]) if len(v) > 3 else 1)
            app._describe()
            app._show()
        elif k == 'foldflip':
            app.fold_other_way()
        elif k == 'viewdir':
            app.view.scene.view(v)
            app.view.redraw()
        elif k == 'click':
            # a real pick: project a key-frame point to the view and click there, in the given mode
            app.set_mode(v['mode'])
            app.root.update()
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(*v['p'])
            app.view.scene.rw.Render()
            dx, dy = coord.GetComputedDoubleDisplayValue(app.view.scene.ren)
            h = app.view.scene.rw.GetSize()[1]
            app.view._click(int(round(dx)), int(round(h - 1 - dy)))
            log.append(dict(click=v, mode_after=app.mode, buttons=len(app.run.job.buttons), fold=app.run.job.fold))
        elif k == 'make':
            app.make()
        elif k == 'view':
            app.vm.set(v)
            app._show()
        elif k == 'wait':
            time.sleep(float(v))
        elif k == 'shot':
            wait_idle()
            shot(v)
            continue
        wait_idle()
    (shots / 'demo_log.json').write_text(json.dumps(log, indent=1))
    app.root.attributes('-topmost', False)
    app.root.destroy()


def set_icon(root):
    """The key-in-a-jacket icon (assets/, drawn by assets/make_icon.py) on the window, its dialogs and the taskbar."""
    a = Path(__file__).resolve().parent / 'assets'
    try:
        if sys.platform == 'win32' and (a / 'fobfit.ico').exists():
            root.iconbitmap(default=str(a / 'fobfit.ico'))      # every size, drawn for that size
            # Tk makes the outer window anew when it first shows, so set the taskbar icon on each new one
            done = set()

            def on_map(e):
                if e.widget is root and root.wm_frame() not in done:
                    done.add(root.wm_frame())
                    taskbar_icon(root, a / 'fobfit.ico')
            root.bind('<Map>', on_map, add='+')
        elif (a / 'fobfit.png').exists():
            root._icon = tk.PhotoImage(file=str(a / 'fobfit.png'))
            root.iconphoto(True, root._icon)
    except tk.TclError:
        pass


def taskbar_icon(root, ico):
    """Windows 11 draws the taskbar button from the app id, not from the window icon, so put the icon,
    the name and the command to open FobFit again on the window itself (also used when it is pinned)."""
    try:
        import uuid
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [('d1', ctypes.c_uint32), ('d2', ctypes.c_uint16), ('d3', ctypes.c_uint16),
                        ('d4', ctypes.c_ubyte * 8)]

        def guid(s):
            u = uuid.UUID(s)
            return GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))

        class PROPERTYKEY(ctypes.Structure):
            _fields_ = [('fmtid', GUID), ('pid', ctypes.c_uint32)]

        class PROPVARIANT(ctypes.Structure):
            _fields_ = [('vt', ctypes.c_ushort), ('r1', ctypes.c_ushort), ('r2', ctypes.c_ushort),
                        ('r3', ctypes.c_ushort), ('p', ctypes.c_void_p), ('p2', ctypes.c_void_p)]

        exe = Path(sys.executable)
        exe = exe.with_name('pythonw.exe') if exe.with_name('pythonw.exe').exists() else exe
        cmd = f'"{sys.executable}"' if getattr(sys, 'frozen', False) else f'"{exe}" "{Path(__file__).resolve()}"'
        values = ((2, cmd),                                          # RelaunchCommand
                  (3, f'{ico},0'),                                   # RelaunchIconResource
                  (4, 'FobFit generator'),                           # RelaunchDisplayNameResource
                  (5, 'FobFit.Generator'))                           # ID, last: it makes the taskbar read the rest
        store = ctypes.c_void_p()
        hwnd = wintypes.HWND(int(root.wm_frame(), 16))
        iid = guid('{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}')          # IPropertyStore
        if ctypes.windll.shell32.SHGetPropertyStoreForWindow(hwnd, ctypes.byref(iid), ctypes.byref(store)) != 0:
            return
        vt = ctypes.cast(ctypes.cast(store, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
        set_value = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(PROPERTYKEY),
                                       ctypes.POINTER(PROPVARIANT))(vt[6])
        try:
            for pid, val in values:
                buf = ctypes.create_unicode_buffer(val)               # the store keeps its own copy
                pv = PROPVARIANT(31, 0, 0, 0, ctypes.cast(buf, ctypes.c_void_p), None)    # VT_LPWSTR
                set_value(store, ctypes.byref(PROPERTYKEY(guid('{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}'), pid)),
                          ctypes.byref(pv))
            ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(vt[7])(store)              # Commit
        finally:
            ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vt[2])(store)             # Release
    except Exception:
        pass                                    # only the taskbar picture; never stop the window for it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scan', nargs='?', help='scan to open at start')
    ap.add_argument('--demo', help='JSON script of steps (testing)')
    ap.add_argument('--shots', default='gui_shots', help='folder for the demo screenshots')
    a = ap.parse_args()
    global QUIET
    QUIET = bool(a.demo)
    root = tk.Tk()
    set_icon(root)
    app = App(root)

    def fmt(t, v, tb):
        return ''.join(traceback.format_exception(t, v, tb))
    # errors in window callbacks, and in threads other than the step worker (that one reports its own)
    root.report_callback_exception = lambda t, v, tb: app.show_error('Error: %s' % v, fmt(t, v, tb))
    def thread_error(h):
        if h.exc_type is not SystemExit:
            what, tb = 'Error: %s' % h.exc_value, fmt(h.exc_type, h.exc_value, h.exc_traceback)
            app.q.put(('call', lambda: app.show_error(what, tb)))
    threading.excepthook = thread_error
    if a.demo:
        root.after(500, lambda: demo(app, a.demo, a.shots))
    elif a.scan:
        root.after(300, lambda: app.load(a.scan))
    root.mainloop()


if __name__ == '__main__':
    main()
