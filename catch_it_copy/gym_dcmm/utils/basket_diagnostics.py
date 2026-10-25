"""Read-only Basket arm diagnostics; never changes dynamics or rewards."""
import json
import numpy as np


class ArmDiagnostics:
    def __init__(self, model):
        self.model = model
        self.last_print = {}
        self.body_ids = {i for i in range(model.nbody)
                         if model.body(i).name in {'link1', 'link2', 'link3', 'link4', 'link5', 'link6'}}
        self.pairs = []
        for a in sorted(self.body_ids):
            for b in sorted(self.body_ids):
                if b <= a or model.body_parentid[a] == b or model.body_parentid[b] == a:
                    continue
                ga = np.flatnonzero(model.geom_bodyid == a)
                gb = np.flatnonzero(model.geom_bodyid == b)
                for group in (ga, gb):
                    if not any(model.geom_contype[g] or model.geom_conaffinity[g] for g in group):
                        print('[basket-self-config] body has no enabled collision geom: ' + model.body(int(model.geom_bodyid[group[0]])).name, flush=True) if len(group) else None
                # Prefer collision proxies; fall back to visual meshes, explicitly labelled.
                ca = [int(g) for g in ga if model.geom_contype[g] or model.geom_conaffinity[g]]
                cb = [int(g) for g in gb if model.geom_contype[g] or model.geom_conaffinity[g]]
                self.pairs.extend((int(g), int(h), not ca or not cb)
                                  for g in (ca or list(ga)) for h in (cb or list(gb)))
        print('[basket-self-config] nonadjacent arm geom pairs=' + str(len(self.pairs)) +
              '; distance probes bypass contact filtering; visual/convex proxies are not proof of physical collision', flush=True)

    def sample(self, data, episode_time):
        import mujoco
        m = self.model
        rows = []
        for index, c in enumerate(data.contact):
            a, b = int(c.geom1), int(c.geom2)
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if ba not in self.body_ids or bb not in self.body_ids or ba == bb:
                continue
            if m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, data, index, force)
            rows.append(dict(kind='contact', geoms=[a,b], distance=float(c.dist), normal_force=float(force[0])))
        for a, b, visual in self.pairs:
            distance = float(mujoco.mj_geomDistance(m, data, a, b, .01, None))
            if distance < .005:
                rows.append(dict(kind='geometry_proximity', geoms=[a,b], distance=distance,
                                 visual_fallback=visual))
        for row in rows:
            key = (row['kind'], *sorted(row['geoms']))
            if episode_time - self.last_print.get(key, -1.) < .1:
                continue
            self.last_print[key] = episode_time
            row.update(time=round(float(episode_time), 4),
                       pair=[dict(geom=m.geom(g).name or 'geom#'+str(g),
                                  body=m.body(int(m.geom_bodyid[g])).name,
                                  contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g]))
                             for g in row['geoms']],
                       arm_qpos=data.qpos[15:21].tolist())
            print('[basket-self] ' + json.dumps(row), flush=True)
