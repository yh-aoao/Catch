"""Derive a four-hole xArm/p1 fit candidate from official LEAP Hand v1 CAD.
Units are mm. Run with numpy, scipy, trimesh, manifold3d, matplotlib.
"""
from pathlib import Path
import sys, os, json, hashlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/catch_camera_mpl')
sys.path.insert(0, '/tmp/catch_camera_cad_deps')
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def loops_on_outer_face(m):
    mask = (m.face_normals @ [0,1,0] > .9999) & (abs(m.triangles_center[:,1]-3) < .001)
    f=m.faces[mask]
    edges=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    graph={}
    for a,b in unique[counts==1]:
        graph.setdefault(a,[]).append(b); graph.setdefault(b,[]).append(a)
    result=[]
    while graph:
        stack=[next(iter(graph))]; ids=[]
        while stack:
            a=stack.pop()
            if a not in graph: continue
            ids.append(a); stack.extend(graph.pop(a))
        result.append(m.vertices[ids])
    return result


def fit_circle(vertices):
    v=vertices[:,[0,2]]
    # Centre coordinates robust to odd-sided polygon meshes.
    origin=v.mean(0); w=v-origin
    sol=np.linalg.lstsq(np.c_[2*w,np.ones(len(w))],(w*w).sum(1),rcond=None)[0]
    return origin+sol[:2], float(np.sqrt(sol[2]+sum(sol[:2]**2)))


def main():
    blank=trimesh.load_mesh(HERE/'official/palm_upper_no_holes.stl')
    official=trimesh.load_mesh(HERE/'official/palm_upper_xarm_franka.stl')
    bores=[]
    for loop in loops_on_outer_face(official):
        if len(loop)==49:
            centre,radius=fit_circle(loop)
            bores.append((centre,radius))
    assert len(bores)==8
    # Four equal square-pattern holes at +/-17.5 from their shared centre.
    xs=sorted(c[0] for c,r in bores)
    square=[(c,r) for c,r in bores if -67<c[0]<-65 or -32<c[0]<-30]
    assert len(square)==4
    centre=np.mean([c for c,r in square],axis=0)
    expected=np.array([centre+[x,z] for x in [-17.5,17.5] for z in [-17.5,17.5]])
    actual=np.array([c for c,r in square])
    mismatch=max(min(np.linalg.norm(actual-e,axis=1)) for e in expected)
    assert mismatch<.0001
    p1=trimesh.load_mesh(ROOT/'catch_it_copy/assets/meshes/p1_Link.STL')
    p1.apply_scale(1000)
    p1_circles=[fit_circle(v) for v in loops_on_outer_face(p1)]
    p1_targets=np.array([[20,20],[20,55],[55,20],[55,55]])
    # On the +Y face these four p1 holes have diameter-12.8 head recesses;
    # their opposite-face through bores are diameter 6.8.
    p1_error=max(min(np.linalg.norm(c-t) for c,r in p1_circles if abs(r-6.4)<.01)
                  for t in p1_targets)
    assert p1_error<.0001
    cutters=[]
    for x,z in expected:
        c=trimesh.creation.cylinder(radius=3.5,height=22,sections=128)
        T=trimesh.geometry.align_vectors([0,0,1],[0,1,0]);T[:3,3]=[x,-2,z]
        c.apply_transform(T);cutters.append(c)
    derived=trimesh.boolean.difference([blank]+cutters,engine='manifold')
    # No new cut-outs in the centre: retain the official blank's supporting web.
    assert np.allclose(derived.bounds,blank.bounds,atol=.0001)
    residual=trimesh.boolean.difference([blank,derived]+cutters,engine='manifold')
    assert abs(residual.volume)<.001
    added=trimesh.boolean.difference([derived,blank],engine='manifold')
    assert abs(added.volume)<.001
    report={'units':'mm','official_mount_centre_XZ_mm':centre.tolist(),
            'four_hole_XZ_mm':expected.tolist(),'hole_diameter_mm':7,
            'square_pitch_mm':35,'square_diagonal_mm':float(35*np.sqrt(2)),
            'official_circle_fit_radii_mm':[r for c,r in square],
            'official_square_pattern_error_mm':mismatch,
            'source_p1_square_pattern_error_mm':p1_error,
            'p1_square_XZ_mm':[[20,20],[20,55],[55,20],[55,55]],
            'p1_alignment_XZ_translation_mm':(np.array([37.5,37.5])-centre).tolist(),
            'physical_hand_version_verified':False,'load_verified':False,
            'internal_fastener_clearance_verified':False,'meshes':{}}
    for name,m in [('leap_v1_p1_4hole_candidate',derived),('leap_v1_official_xarm',official)]:
        assert m.is_volume and m.is_watertight and m.is_winding_consistent and len(m.split())==1
        m.export(HERE/(name+'_mm.stl'))
        p=m.copy();T=np.eye(4);T[:3,:3]=Rotation.from_euler('x',-90,degrees=True).as_matrix()
        p.apply_transform(T);p.apply_translation(-p.bounds[0]);p.export(HERE/(name+'_print_mm.stl'))
        readback=trimesh.load_mesh(HERE/(name+'_print_mm.stl'))
        assert readback.is_volume and len(readback.split())==1 and abs(readback.bounds[0,2])<1e-5
        report['meshes'][name]={'size_print_mm':p.extents.tolist(),'volume_mm3':float(m.volume),
                               'watertight':True,'single_component':True}
    # Verify the exported candidate, not just cutter positions.
    found=[]
    for loop in loops_on_outer_face(derived):
        c,r=fit_circle(loop)
        if abs(r-3.5)<.001:found.append(c)
    assert len(found)==4
    err=max(min(np.linalg.norm(np.array(found)-e,axis=1)) for e in expected)
    assert err<.0001
    report['exported_hole_centre_error_mm']=err
    report['source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in (HERE/'official').iterdir() if f.is_file()}
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    preview(derived,expected,centre)
    print(json.dumps(report,indent=2))


def preview(m,holes,centre):
    fig=plt.figure(figsize=(13,7))
    ax=fig.add_subplot(121,projection='3d')
    ax.add_collection3d(Poly3DCollection(m.triangles,facecolor='#86b4d2',edgecolor='#40576b',linewidth=.08))
    c=m.bounds.mean(0);r=max(m.extents)/2*1.05
    ax.set(xlim=(c[0]-r,c[0]+r),ylim=(c[1]-r,c[1]+r),zlim=(c[2]-r,c[2]+r),title='Official LEAP v1 blank + four p1-matched holes',xlabel='X mm',ylabel='Y mm',zlabel='Z mm')
    ax.set_box_aspect((1,1,1));ax.view_init(25,-60)
    ax=fig.add_subplot(122)
    face=m.triangles[(m.face_normals@[0,1,0]>.9999)&(abs(m.triangles_center[:,1]-3)<.001)]
    offset=m.bounds[0,[0,2]]
    uv=face[:,:,[0,2]]-offset
    ax.add_collection(PolyCollection(uv,facecolor='#bddcec',edgecolor='none'))
    h=holes-offset
    for x,z in h:
        ax.add_patch(plt.Circle((x,z),3.5,fill=False,color='#c03432'))
        ax.plot(x,z,'+',color='#c03432')
    x0,z0=h.min(0);x1,z1=h.max(0)
    ax.annotate('',xy=(x0,z0-8),xytext=(x1,z0-8),arrowprops={'arrowstyle':'<->','color':'#c03432'})
    ax.text((x0+x1)/2,z0-13,'35 mm',ha='center',color='#c03432')
    ax.annotate('',xy=(x1+8,z0),xytext=(x1+8,z1),arrowprops={'arrowstyle':'<->','color':'#c03432'})
    ax.text(x1+10,(z0+z1)/2,'35 mm',color='#c03432',rotation=90,va='center')
    ax.add_patch(plt.Circle(centre-offset,37.5,fill=False,ls='--',color='#727272',label='p1 round area (orientation TBD)'))
    ax.set(xlim=(-5,90),ylim=(-5,130),xlabel='Width from blank edge (mm)',ylabel='Length from blank edge (mm)',title='4 x diameter 7 mm | p1 square pattern')
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('FIT CANDIDATE: verify hand version, screw access and load before operation',fontsize=13)
    fig.tight_layout();fig.savefig(HERE/'backplate_preview.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
