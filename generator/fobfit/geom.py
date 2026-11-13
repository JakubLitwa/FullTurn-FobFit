"""Mesh helpers shared by the other modules. All lengths in mm."""
import numpy as np
import trimesh
import manifold3d as m3
import vtk
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from vtk.util import numpy_support as ns

M = m3.Manifold


def to_tm(man):
    me = man.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(me.vert_properties)[:, :3].astype(float),
                           faces=np.asarray(me.tri_verts).astype(np.int64), process=False)


def to_man(tm):
    return M(m3.Mesh(vert_properties=np.ascontiguousarray(tm.vertices, dtype=np.float32),
                     tri_verts=np.ascontiguousarray(tm.faces, dtype=np.uint32)))


def tm_to_vtk(tm):
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(tm.vertices, dtype=np.float64), deep=True))
    conn = np.hstack([np.full((len(tm.faces), 1), 3, dtype=np.int64), tm.faces.astype(np.int64)]).ravel()
    cells = vtk.vtkCellArray()
    cells.ImportLegacyFormat(ns.numpy_to_vtkIdTypeArray(conn, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    return pd


def vtk_to_tm(pd, process=True):
    v = ns.vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    f = ns.vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return trimesh.Trimesh(v, f, process=process)


def iso_mesh(field, origin, pitch, level=0.0):
    """Iso-surface of a (nz, ny, nx) field; the solid is where field < level. Returns a trimesh with outward normals."""
    nz, ny, nx = field.shape
    img = vtk.vtkImageData()
    img.SetOrigin(*[float(o) for o in origin])
    img.SetSpacing(pitch, pitch, pitch)
    img.SetDimensions(nx, ny, nz)
    img.GetPointData().SetScalars(ns.numpy_to_vtk(np.ascontiguousarray(field, dtype=np.float32).ravel(), deep=True))
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, float(level))
    fe.ComputeNormalsOff()
    fe.ComputeGradientsOff()
    fe.Update()
    pd = fe.GetOutput()
    if pd.GetNumberOfPoints() == 0:
        return trimesh.Trimesh()
    tm = vtk_to_tm(pd, process=True)
    if tm.volume < 0:
        tm.invert()
    return tm


def decimate(tm, target_faces):
    """Quadric decimation (VTK) for display copies; returns tm itself when it is small enough."""
    if len(tm.faces) <= target_faces:
        return tm
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputData(tm_to_vtk(tm))
    dec.SetTargetReduction(1.0 - target_faces / len(tm.faces))
    dec.VolumePreservationOn()
    dec.Update()
    return vtk_to_tm(dec.GetOutput(), process=True)


def face_components(tm):
    """Connected components of faces (shared edges). Returns (n, labels per face)."""
    adj = tm.face_adjacency
    n = len(tm.faces)
    if len(adj) == 0:
        return n, np.arange(n)
    g = sparse.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)


def closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (n, 3). Ericson, Real-Time Collision Detection 5.1.5."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    bp = p - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    cp = p - c
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den
    w = vc / den
    out = a + ab * v[:, None] + ac * w[:, None]                    # interior
    # edge regions
    with np.errstate(divide='ignore', invalid='ignore'):
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = np.where(m, d1 / np.where(d1 - d3 == 0, 1, d1 - d3), 0)
        out[m] = (a + ab * t[:, None])[m]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = np.where(m, d2 / np.where(d2 - d6 == 0, 1, d2 - d6), 0)
        out[m] = (a + ac * t[:, None])[m]
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = np.where(m, (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1, (d4 - d3) + (d5 - d6)), 0)
        out[m] = (b + (c - b) * t[:, None])[m]
    # vertex regions
    m = (d1 <= 0) & (d2 <= 0)
    out[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3)
    out[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6)
    out[m] = c[m]
    return out


def sample_surface(tm, n, seed=0):
    pts, fid = trimesh.sample.sample_surface(tm, int(n), seed=seed)
    return np.asarray(pts), np.asarray(fid)


def rot_x(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [1, 0, 0])


def rot_z(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1])


def clean_tm(man_or_tm):
    """Trimesh for export: the mesh after a binary STL round trip (float32 vertices, merged, degenerate faces
    dropped, as a slicer reads it), keeping the largest body; zero-volume slivers split off by that round trip
    are dropped. Falls back to the plain conversion if the result is not one watertight body."""
    import io
    raw = to_tm(man_or_tm) if not isinstance(man_or_tm, trimesh.Trimesh) else man_or_tm.copy()
    buf = io.BytesIO(raw.export(file_type='stl'))
    tm = trimesh.load(buf, file_type='stl', process=True)
    parts = [p for p in tm.split(only_watertight=False) if abs(p.volume) > 1e-3]
    if len(parts) == 1 and parts[0].is_watertight and parts[0].is_winding_consistent and parts[0].volume > 0:
        return parts[0]
    return raw


def inside_mask(tm, origin, pitch, dims):
    """Voxel centres inside a closed mesh (scanline stencil), as a (nz, ny, nx) bool array."""
    img = vtk.vtkImageData()
    img.SetOrigin(*[float(o) for o in origin])
    img.SetSpacing(pitch, pitch, pitch)
    img.SetDimensions(*[int(d) for d in dims])
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    ns.vtk_to_numpy(img.GetPointData().GetScalars())[:] = 1
    st = vtk.vtkPolyDataToImageStencil()
    st.SetInputData(tm_to_vtk(tm))
    st.SetOutputOrigin(*[float(o) for o in origin])
    st.SetOutputSpacing(pitch, pitch, pitch)
    st.SetOutputWholeExtent(img.GetExtent())
    st.Update()
    im = vtk.vtkImageStencil()
    im.SetInputData(img)
    im.SetStencilConnection(st.GetOutputPort())
    im.ReverseStencilOff()
    im.SetBackgroundValue(0)
    im.Update()
    a = ns.vtk_to_numpy(im.GetOutput().GetPointData().GetScalars())
    return a.reshape(int(dims[2]), int(dims[1]), int(dims[0])).astype(bool)
