"""Offscreen VTK rendering: product renders, the deviation map and the window's 3D view share this."""
import numpy as np
import vtk
from vtk.util import numpy_support as ns

from . import geom

JACKET = (0.96, 0.55, 0.16)      # PETG orange
KEY = (0.22, 0.24, 0.27)         # dark key plastic
BG = (1.0, 1.0, 1.0)


def actor(tm, color=JACKET, opacity=1.0, scalars=None, lut=None, rng=None, smooth=True, edges=False):
    pd = geom.tm_to_vtk(tm)
    if smooth:
        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(pd)
        nf.SetFeatureAngle(40.0)
        nf.SplittingOn()
        nf.ConsistencyOn()
        nf.Update()
        pd = nf.GetOutput() if scalars is None else pd
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(pd)
    if scalars is not None:
        arr = ns.numpy_to_vtk(np.ascontiguousarray(scalars, dtype=np.float64), deep=True)
        pd.GetPointData().SetScalars(arr)
        m.SetScalarModeToUsePointData()
        m.SetLookupTable(lut)
        m.SetScalarRange(*rng)
        m.ScalarVisibilityOn()
    else:
        m.ScalarVisibilityOff()
    a = vtk.vtkActor()
    a.SetMapper(m)
    p = a.GetProperty()
    p.SetColor(*color)
    p.SetOpacity(opacity)
    p.SetSpecular(0.25)
    p.SetSpecularPower(30)
    p.SetDiffuse(0.85)
    p.SetAmbient(0.12)
    if edges:
        p.EdgeVisibilityOn()
    return a


def diverging_lut(lo, hi):
    """Blue (more gap) - green (on target) - red (less gap / interference)."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(lo, hi)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            u = t / 0.5
            c = (0.85 * (1 - u) + 0.15 * u, 0.15 * (1 - u) + 0.75 * u, 0.15 * (1 - u) + 0.25 * u)   # red -> green
        else:
            u = (t - 0.5) / 0.5
            c = (0.15 * (1 - u) + 0.15 * u, 0.75 * (1 - u) + 0.35 * u, 0.25 * (1 - u) + 0.9 * u)   # green -> blue
        lut.SetTableValue(i, c[0], c[1], c[2], 1.0)
    lut.Build()
    return lut


class Scene:
    """An offscreen render window; image() returns an (h, w, 3) uint8 array."""

    def __init__(self, w=1200, h=900, bg=BG):
        self.rw = vtk.vtkRenderWindow()
        self.rw.SetOffScreenRendering(1)
        self.rw.SetSize(w, h)
        self.rw.SetMultiSamples(8)
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(*bg)
        self.ren.SetUseFXAA(True)
        self.rw.AddRenderer(self.ren)
        self.ren.RemoveAllLights()
        for pos, inten in (((1, -1, 2), 0.75), ((-1.5, 1, 1), 0.35), ((0, 0.5, -2), 0.25)):
            L = vtk.vtkLight()
            L.SetLightTypeToCameraLight()
            L.SetPosition(*pos)
            L.SetFocalPoint(0, 0, 0)
            L.SetIntensity(inten)
            self.ren.AddLight(L)
        self.actors = []

    def add(self, a):
        self.ren.AddActor(a)
        self.actors.append(a)
        return a

    def clear(self):
        for a in self.actors:
            self.ren.RemoveActor(a)
        self.actors = []
        for a in list(getattr(self, 'overlays', [])):
            self.ren.RemoveActor(a)
        self.overlays = []

    def overlay(self, a):
        self.overlays = getattr(self, 'overlays', [])
        if hasattr(a, 'PickableOff'):
            a.PickableOff()                  # marks and arrows must not catch clicks meant for the key
        self.ren.AddActor(a)
        self.overlays.append(a)
        return a

    def view(self, direction, up=(0, 0, 1), zoom=1.0, center=None, bounds=None):
        cam = self.ren.GetActiveCamera()
        self.ren.ResetCamera(*bounds) if bounds is not None else self.ren.ResetCamera()
        fp = np.array(cam.GetFocalPoint()) if center is None else np.asarray(center, float)
        dist = cam.GetDistance()
        d = np.asarray(direction, float)
        d /= np.linalg.norm(d)
        cam.SetFocalPoint(*fp)
        cam.SetPosition(*(fp + d * dist))
        cam.SetViewUp(*up)
        cam.SetViewAngle(30)
        self.ren.ResetCamera(*bounds) if bounds is not None else self.ren.ResetCamera()
        cam.Zoom(zoom)
        self.ren.ResetCameraClippingRange()

    def image(self):
        self.rw.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.rw)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        img = w2i.GetOutput()
        w, h, _ = img.GetDimensions()
        a = ns.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3]
        return np.ascontiguousarray(a)

    def save(self, path):
        from PIL import Image
        Image.fromarray(self.image()).save(path)


def line_actor(p0, p1, color=(0.1, 0.3, 0.9), width=4.0):
    ls = vtk.vtkLineSource()
    ls.SetPoint1(*p0)
    ls.SetPoint2(*p1)
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(ls.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetLineWidth(width)
    a.GetProperty().SetLighting(False)
    return a


def arrow_actor(p0, p1, color=(0.1, 0.3, 0.9), shaft=0.6, tip=1.8):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    d = (p1 - p0) / L
    src = vtk.vtkArrowSource()
    src.SetShaftRadius(shaft / L)
    src.SetTipRadius(tip / L)
    src.SetTipLength(min(0.3, 4.0 / L))
    src.SetShaftResolution(24)
    src.SetTipResolution(24)
    # vtkArrowSource points along +x with unit length
    x = d
    t = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    y = np.cross(t, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    M = vtk.vtkMatrix4x4()
    for i in range(3):
        M.SetElement(i, 0, x[i] * L)
        M.SetElement(i, 1, y[i] * L)
        M.SetElement(i, 2, z[i] * L)
        M.SetElement(i, 3, p0[i])
    tr = vtk.vtkTransform()
    tr.SetMatrix(M)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetInputConnection(src.GetOutputPort())
    f.SetTransform(tr)
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(f.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetAmbient(0.4)
    return a


def sphere_actor(p, r=1.0, color=(0.9, 0.1, 0.1), opacity=1.0):
    s = vtk.vtkSphereSource()
    s.SetCenter(*p)
    s.SetRadius(r)
    s.SetThetaResolution(24)
    s.SetPhiResolution(16)
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetOpacity(opacity)
    return a


def ring_actor(center, normal, r, color=(0.9, 0.1, 0.1), width=4.0):
    src = vtk.vtkRegularPolygonSource()
    src.SetCenter(*center)
    src.SetNormal(*normal)
    src.SetRadius(r)
    src.SetNumberOfSides(48)
    src.GeneratePolygonOff()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(src.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetLineWidth(width)
    a.GetProperty().SetLighting(False)
    return a


def arc_actor(center, r, a0, a1, color=(0.95, 0.55, 0.0), tube=0.45, head=3.5, n=64):
    """A flat arc about the vertical axis through center, from angle a0 to a1 (radians, +x toward +y),
    drawn as a tube with an arrowhead at the a1 end."""
    c = np.asarray(center, float)
    pts = vtk.vtkPoints()
    t = np.linspace(a0, a1, n)
    for a in t:
        pts.InsertNextPoint(c[0] + r * np.cos(a), c[1] + r * np.sin(a), c[2])
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(n)
    for i in range(n):
        line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(cells)
    tf = vtk.vtkTubeFilter()
    tf.SetInputData(pd)
    tf.SetRadius(tube)
    tf.SetNumberOfSides(12)
    app = vtk.vtkAppendPolyData()
    app.AddInputConnection(tf.GetOutputPort())
    if head:
        e = c + r * np.array([np.cos(a1), np.sin(a1), 0.0])
        d = np.sign(a1 - a0) * np.array([-np.sin(a1), np.cos(a1), 0.0])
        cs = vtk.vtkConeSource()
        cs.SetCenter(*(e + d * head / 2))
        cs.SetDirection(*d)
        cs.SetHeight(head)
        cs.SetRadius(head * 0.45)
        cs.SetResolution(24)
        app.AddInputConnection(cs.GetOutputPort())
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(app.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetAmbient(0.4)
    return a


def text_actor(txt, x, y, size=18, color=(0.1, 0.1, 0.1)):
    t = vtk.vtkTextActor()
    t.SetInput(txt)
    t.SetDisplayPosition(int(x), int(y))
    tp = t.GetTextProperty()
    tp.SetFontSize(size)
    tp.SetColor(*color)
    tp.SetFontFamilyToArial()
    return t
