"""D435i fit prototypes, preserving the repository STL geometry and hole centres.

Run with Python 3 + numpy, scipy, trimesh, manifold3d and matplotlib.
STL input is metres; all output coordinates and dimensions are millimetres.
This is geometry/fit verification, not mechanical strength qualification.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/catch_camera_mpl')
if Path('/tmp/catch_camera_cad_deps').exists():
    sys.path.insert(0, '/tmp/catch_camera_cad_deps')
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = ROOT / 'catch_it_copy/assets/meshes'
DIAMETER = 3.4
N = 128


def cylinder(point, axis, radius, length):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    m = trimesh.creation.cylinder(radius=radius, height=length, sections=N)
    T = trimesh.geometry.align_vectors([0, 0, 1], axis)
    T[:3, 3] = point
    m.apply_transform(T)
    return m


def cut(mesh, cutters):
    return trimesh.boolean.difference([mesh] + cutters, engine='manifold')


def load(name):
    m = trimesh.load_mesh(SOURCES / name)
    m.apply_scale(1000)
    assert m.is_watertight and m.is_volume
    return m


def cylinder_boundary_check(mesh, centre, axis, thickness):
    """Inspect actual output bore vertices, rather than just construction inputs."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    d = mesh.vertices - centre
    along = d @ axis
    radius = np.linalg.norm(d - along[:, None] * axis, axis=1)
    selected = (abs(along) <= thickness / 2 + .01) & (abs(radius - DIAMETER / 2) < .01)
    assert selected.sum() >= 2 * N, selected.sum()
    assert radius[selected].min() > 1.699 and radius[selected].max() < 1.701
    return {'bore_vertex_count': int(selected.sum()),
            'vertex_diameter_min_max_mm': [float(2 * radius[selected].min()),
                                           float(2 * radius[selected].max())]}


def save(mesh, name, print_rotation=None):
    mesh.export(HERE / (name + '_mm.stl'))
    record = {'bounds_mm': mesh.bounds.tolist(), 'volume_mm3': float(mesh.volume),
              'triangles': len(mesh.faces), 'watertight': bool(mesh.is_watertight),
              'winding_consistent': bool(mesh.is_winding_consistent),
              'connected_components': len(mesh.split()),
              'sha256': hashlib.sha256((HERE / (name + '_mm.stl')).read_bytes()).hexdigest()}
    assert record['watertight'] and record['winding_consistent']
    assert record['connected_components'] == 1 and mesh.is_volume
    if print_rotation is not None:
        p = mesh.copy()
        T = np.eye(4)
        T[:3, :3] = print_rotation
        p.apply_transform(T)
        p.apply_translation(-p.bounds[0])
        p.export(HERE / (name + '_print_mm.stl'))
        record['print_size_mm'] = p.extents.tolist()
    return record


def main():
    HERE.mkdir(exist_ok=True)
    top_original = load('l3_Link.STL')
    wrist_original = load('p1_Link.STL')
    # Original l3 camera-facing surface: normal (-sqrt(3)/2,0,1/2),
    # passing through the origin. Existing bores are 12-gons of diameter 3.
    normal = np.array([-np.sqrt(3) / 2, 0, .5])
    surface_centres = np.array([[-12.69337567, y, -21.98557159] for y in (17.5, 62.5)])
    top_centres = surface_centres - 2.5 * normal
    top_cuts = [cylinder(c, normal, DIAMETER / 2, 9) for c in top_centres]
    top = cut(top_original, top_cuts)
    # The existing p1 camera holes are (105,15) and (105,60) in local X,Z.
    wrist_centres = np.array([[105, 1.5, z] for z in (15, 60)])
    wrist = cut(wrist_original,
                [cylinder(c, [0, 1, 0], DIAMETER / 2, 7) for c in wrist_centres])
    # Optional new holes on the original 45x80, 5-mm-thick extrusion foot.
    # They meet one longitudinal T-slot, not two fixed threaded chassis holes.
    foot_centres = np.array([[22.5, y, -2.5] for y in (17.5, 62.5)])
    top_m8 = cut(top, [cylinder(c, [0, 0, 1], 4.3, 9) for c in foot_centres])
    # Cheap gauge before printing the large pieces. Not a working mounting part.
    coupon = trimesh.creation.box([60, 12, 3])
    coupon.apply_translation([30, 6, 1.5])
    coupon = cut(coupon, [cylinder([x, 6, 1.5], [0, 0, 1], 1.7, 7)
                          for x in (7.5, 52.5)])
    top_rotation = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    wrist_rotation = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    outputs = {}
    outputs['top_l3_d435i'] = save(top, 'top_l3_d435i', top_rotation)
    outputs['top_l3_d435i_M8_foot'] = save(top_m8, 'top_l3_d435i_M8_foot', top_rotation)
    outputs['wrist_p1_d435i_FIT_PROTOTYPE'] = save(wrist, 'wrist_p1_d435i_FIT_PROTOTYPE', wrist_rotation)
    outputs['d435i_hole_gauge'] = save(coupon, 'd435i_hole_gauge', np.eye(3))
    checks = {'top_camera_bores': [cylinder_boundary_check(top, c, normal, 5) for c in top_centres],
              'wrist_camera_bores': [cylinder_boundary_check(wrist, c, [0, 1, 0], 3) for c in wrist_centres]}
    # The difference must be entirely inside the deliberate camera bore cutters.
    for old, new, cutters, tag in [(top_original, top, top_cuts, 'top'),
            (wrist_original, wrist, [cylinder(c, [0,1,0], 1.7, 7) for c in wrist_centres], 'wrist')]:
        added = cut(new, [old])
        extra_removed = cut(old, [new] + cutters)
        # Manifold/STL use float32; permit <0.001 mm^3 roundoff slivers.
        assert abs(added.volume) < .001 and abs(extra_removed.volume) < .001
        assert np.allclose(new.bounds, old.bounds, atol=1e-4)
        checks[tag + '_only_camera_bores_changed'] = True
        checks[tag + '_boolean_residual_mm3'] = {
            'added': float(abs(added.volume)), 'removed_outside_bores': float(abs(extra_removed.volume))}
    report = {'units': 'mm', 'camera_hole_diameter_mm': DIAMETER,
              'camera_hole_pitch_mm': 45, 'top_camera_surface_centres_mm': surface_centres.tolist(),
              'top_hole_axis': normal.tolist(), 'wrist_bore_midpoints_mm': wrist_centres.tolist(),
              'top_optional_M8_midpoints_mm': foot_centres.tolist(),
              'source_sha256': {n: hashlib.sha256((SOURCES/n).read_bytes()).hexdigest()
                                  for n in ['l3_Link.STL', 'p1_Link.STL']},
              'outputs': outputs, 'checks': checks,
              'physical_fit_verified': False, 'load_strength_verified': False,
              'assembly_collision_verified': False}
    (HERE / 'geometry_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    preview(top_m8, wrist, surface_centres, wrist_centres)
    print(json.dumps({'outputs': outputs, 'checks': checks}, indent=2))


def preview(top, wrist, top_points, wrist_points):
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    for i, (m, title) in enumerate([(top, 'Top: original l3 + D435i bores + optional M8 foot'),
                                  (wrist, 'Wrist: original p1, camera holes ONLY enlarged')]):
        ax = fig.add_subplot(2, 2, 2*i+1, projection='3d')
        ax.add_collection3d(Poly3DCollection(m.triangles, facecolor='#89b9d6',
                                           edgecolor='#395469', linewidth=.05))
        centre = m.bounds.mean(0); r = max(m.extents) / 2 * 1.1
        ax.set(xlim=(centre[0]-r, centre[0]+r), ylim=(centre[1]-r, centre[1]+r),
               zlim=(centre[2]-r, centre[2]+r), xlabel='X mm', ylabel='Y mm', zlabel='Z mm', title=title)
        ax.set_box_aspect((1,1,1)); ax.view_init(25, -55 if i == 0 else -70)
        ax = fig.add_subplot(2, 2, 2*i+2)
        if i == 0:
            # Camera mounting plane coordinates: u along the camera length (Y),
            # v along the sloping plate from its top edge.
            triangles = m.triangles
            normal = np.array([-np.sqrt(3)/2, 0, .5])
            mask = m.face_normals @ normal > .9999
            from matplotlib.collections import PolyCollection
            face = triangles[mask]
            uv = np.stack([face[:,:,1], face @ np.array([-.5, 0, -np.sqrt(3)/2])], axis=-1)
            ax.add_collection(PolyCollection(uv, facecolor='#c4e1ef', edgecolor='none'))
            xy = np.c_[top_points[:,1], top_points @ np.array([-.5,0,-np.sqrt(3)/2])]
            ax.set(xlim=(-5,85), ylim=(-5,50), xlabel='Original local Y (mm)', ylabel='Slope distance (mm)',
                   title='Top camera face: 2 x diameter 3.4; pitch 45 mm')
        else:
            from matplotlib.collections import PolyCollection
            face = m.triangles[m.face_normals @ [0,-1,0] > .9999]
            ax.add_collection(PolyCollection(face[:,:,[0,2]], facecolor='#c4e1ef', edgecolor='none'))
            xy = wrist_points[:,[0,2]]
            ax.set(xlim=(-5,120), ylim=(-5,80), xlabel='Original local X (mm)', ylabel='Original local Z (mm)',
                   title='Wrist: red holes are camera; all other holes preserved')
        ax.scatter(xy[:,0], xy[:,1], marker='+', color='#c52e2e', s=70)
        for x,y in xy: ax.add_patch(plt.Circle((x,y), 1.7, fill=False, color='#c52e2e'))
        p,q = xy
        offset = np.array([0,-7]) if i == 0 else np.array([8,0])
        ax.annotate('', xy=p+offset, xytext=q+offset,
                    arrowprops={'arrowstyle':'<->','color':'#c52e2e'})
        ax.text(*((p+q)/2+offset+np.array([2,2])), '45 mm', color='#c52e2e')
        ax.set_aspect('equal'); ax.grid(alpha=.2)
    fig.suptitle('D435i mount fit prototypes | millimetres | not load-qualified', fontsize=15)
    fig.subplots_adjust(left=.05, right=.97, bottom=.07, top=.91, hspace=.35, wspace=.18)
    fig.savefig(HERE/'mount_preview.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
