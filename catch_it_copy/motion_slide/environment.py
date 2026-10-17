"""New Slide dynamics/reward/termination; reuse robot observation and PID only."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv, RollTrackingEnv, hand_collision_ids
from .geometry import SlideConfig, frame, release_pose, intercept_target


class SlideEnv(DcmmVecEnv):
    def __init__(self, slide_config=None, slide_log=False, **kwargs):
        values = {} if slide_config is None else json.loads(Path(slide_config).read_text(encoding='utf-8-sig'))
        self.slide = SlideConfig(**values).validate()
        self.slide_log = slide_log
        kwargs.pop('object_motion', None)
        if kwargs.get('object_eval', False):
            raise ValueError('Slide v1 uses its own fixed box object; object_eval must be false')
        # Subclass construction bypasses the historical roll routing in __new__.
        super().__init__(object_motion='roll', **kwargs)

    def _reset_object_roll(self):
        root = ET.fromstring(self.Dcmm.model_xml_string)
        contact = root.find('contact')
        if contact is not None:
            for exclusion in list(contact):
                if exclusion.tag == 'exclude' and {exclusion.get('body1'), exclusion.get('body2')} == {'base_link', 'table_body'}:
                    contact.remove(exclusion)  # Ramp/base collision must be observable in Slide.
        body = root.find(".//body[@name='object']")
        geom = body.find(".//geom[@name='object']")
        inertial = body.find('inertial')
        if inertial is not None:
            body.remove(inertial)  # MuJoCo derives correct box inertia from mass/size.
        geom.attrib.pop('mesh', None)
        geom.set('type', 'box')
        geom.set('size', ' '.join(map(str, self.slide.half_size)))
        geom.set('mass', str(self.slide.mass))
        geom.set('friction', '{} .001 .0001'.format(self.slide.friction))
        geom.set('solref', '.01 1')
        joint = body.find('joint')
        if joint is not None:
            joint.set('damping', '0')
            joint.set('armature', '0')
        self.random_mass = self.slide.mass
        return ET.tostring(root, encoding='unicode')

    def _reset_simulation(self):
        super()._reset_simulation()
        model, data = self.Dcmm.model, self.Dcmm.data
        _, center, _, quat = frame(self.slide)
        body = model.body('table_body').id
        model.body_pos[body] = center
        model.body_quat[body] = quat
        model.geom_pos[self.table_geom_id] = 0.
        model.geom_quat[self.table_geom_id] = [1, 0, 0, 0]
        model.geom_size[self.table_geom_id] = [self.slide.width/2, self.slide.length/2, self.slide.thickness/2]
        model.geom_friction[self.table_geom_id] = [self.slide.friction, .001, .0001]
        self.object_pos3d, self.object_q = release_pose(self.slide, np.random.uniform(-self.slide.release_x_range, self.slide.release_x_range))
        self.object_vel6d = np.zeros(6)
        self.Dcmm.set_throw_pos_vel(np.r_[self.object_pos3d, self.object_q], np.zeros(6))
        data.ctrl[-1] = 0.
        # Refresh runtime geometry constants after changing ramp extents.
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        _, info = super().reset()
        self.object_throw = True  # No delayed launch or velocity injection.
        self.departed = self.caught = False
        self.hold = self.gap = 0.
        self.previous_action = None
        self.Dcmm.target_arm_qpos[:] = self.Dcmm.data.qpos[15:21]
        self.Dcmm.target_hand_qpos[:] = self.Dcmm.data.qpos[21:37]
        for key, target in [('arm', self.Dcmm.target_arm_qpos), ('hand', self.Dcmm.target_hand_qpos)]:
            self.action_buffer[key].clear()
            self.action_buffer[key].append(target.copy())
        info['slide_version'] = 'gravity_box_v1'
        return self._get_obs(), info

    def _get_ctrl(self):
        # Historical RollTracking PID routine has no Roll-specific target guard.
        # Actual collisions remain enabled and terminate in this task's own step.
        return RollTrackingEnv._get_ctrl(self)

    def step(self, action):
        action = {k: np.asarray(action[k], dtype=float).copy() for k in ('base', 'arm', 'hand')}
        if any(not np.isfinite(v).all() for v in action.values()):
            raise ValueError('Nonfinite Slide action')
        self.steps += 1
        self.Dcmm.target_base_vel[:2] = np.clip(action['base'], -1.5, 1.5)
        result, _ = self.Dcmm.move_ee_pose(np.clip(action['arm'], -.025, .025), measured_state=True)
        self.arm_limit = bool(result[1]) and bool(np.isfinite(result[0]).all())
        self.Dcmm.target_arm_qpos[:] = result[0] if self.arm_limit else self.Dcmm.data.qpos[15:21]
        if self.task == 'Catching':
            self.Dcmm.action_hand2qpos(np.clip(action['hand'], -.15, .15))
            for i in range(21, 37):
                jid = int(np.flatnonzero(self.Dcmm.model.jnt_qposadr == i)[0])
                if self.Dcmm.model.jnt_limited[jid]:
                    self.Dcmm.target_hand_qpos[i-21] = np.clip(self.Dcmm.target_hand_qpos[i-21], *self.Dcmm.model.jnt_range[jid])
        # Track keeps the reset hand pose; Catch learns all joints.
        self.update_target_ctrl()
        model, data = self.Dcmm.model, self.Dcmm.data
        hand_ids = hand_collision_ids(model, self.hand_start_id)
        dt = model.opt.timestep
        success, reason, touched = False, None, False
        bound = float(np.linalg.norm(self.slide.half_size))
        for _ in range(self.steps_per_policy):
            data.ctrl[:-1] = self._get_ctrl()
            data.ctrl[-1] = 0.
            mujoco.mj_step(model, data)
            mujoco.mj_rnePostConstraint(model, data)
            self.contacts = self._get_contacts()
            p = data.qpos[37:40]
            self.departed |= bool(p[1]+bound < self.slide.edge_y)
            contact = bool(np.isin(self.contacts['object_contacts'], hand_ids).any())
            touched |= contact and self.departed
            palm = data.geom_xpos[self.hand_start_id]
            near = np.linalg.norm(p[:2]-palm[:2]) < .13 and -.02 <= p[2]-palm[2] <= .16
            if self.departed and contact and near:
                self.caught = True
                self.gap = 0.
            else:
                self.gap += dt
            if self.caught and near and self.gap <= self.slide.contact_gap:
                self.hold += dt
            else:
                self.hold = 0.
                self.caught = False
            if self.contacts['base_contacts'].size:
                reason = 'base_collision'
            elif self.table_geom_id in self.contacts['hand_contacts']:
                reason = 'hand_ramp_collision'
            elif self.floor_id in self.contacts['object_contacts'] or p[2] < .03:
                reason = 'object_on_floor'
            elif abs(p[0]) > 1.3 or p[1] < -1.5 or p[1] > 4.:
                reason = 'out_of_bounds'
            if reason:
                break
            success = touched if self.task == 'Tracking' else self.hold >= self.slide.hold_seconds
            if success:
                reason = 'intercept' if self.task == 'Tracking' else 'retained'
                break
        mujoco.mj_forward(model, data)
        p, v = data.qpos[37:40].copy(), data.qvel[36:39].copy()
        palm = data.geom_xpos[self.hand_start_id].copy()
        target = intercept_target(p, v, self.slide, not self.departed)
        error = palm-target
        normalized = np.r_[action['base']/1.5, action['arm']/.025, action['hand']/.15]
        terms = dict(xy=2*np.exp(-np.sum(error[:2]**2)/.04), height=.5*np.exp(-error[2]**2/.02),
                     inside_ramp=-3*max(0., palm[1]-(self.slide.edge_y-.10)),
                     contact=1. if touched else 0., retention=2.*min(1., self.hold/self.slide.hold_seconds),
                     control=-.01*np.mean(normalized**2), smoothness=0.,
                     success=40. if success else 0., failure=-10. if reason and not success else 0., time=-.02)
        if self.previous_action is not None:
            terms['smoothness'] = -.02*np.mean((normalized-self.previous_action)**2)
        self.previous_action = normalized.copy()
        truncated = bool(not reason and data.time-self.start_time >= self.env_time)
        self.stage = 'grasping' if self.caught else 'tracking'
        self.step_touch = bool(touched)
        self.terminated = bool(reason)
        self.terminated_reason = reason or ('timeout' if truncated else None)
        info = self._get_info()
        info.update(success=bool(success), terminated_reason=self.terminated_reason, slide_target=target,
                    slide_departed=self.departed, slide_hold_seconds=self.hold, slide_reward_terms=terms)
        info['imgs'] = self.render()
        self.info = info
        if self.slide_log and (self.terminated or truncated):
            print('[slide-end] task={} success={} reason={} hold={:.3f}'.format(self.task, success, self.terminated_reason, self.hold), flush=True)
        return self._get_obs(), float(sum(terms.values())), self.terminated, truncated, info
