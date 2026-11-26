"""Original bridge behavior with configurable physical hoop diameter only."""
import xml.etree.ElementTree as ET
import numpy as np
from basket_mobile_bridge.environment import AutonomousTeacherEnv as Bridge

class AutonomousTeacherEnv(Bridge):
    def __init__(self,options=None,**kwargs):
        self.hoop_diameter=float((options or {}).get('hoop_diameter',.6))
        if not np.isfinite(self.hoop_diameter) or self.hoop_diameter<=2*.033275785333333335:
            raise ValueError('hoop_diameter must be finite and greater than ball diameter')
        super().__init__(options,**kwargs)
        self.basket_cfg.basket_radius=self.hoop_diameter/2.
    def _reset_object_throw_basket(self):
        root=ET.fromstring(super()._reset_object_throw_basket())
        basket=root.find("./worldbody/body[@name='basket_target']")
        if basket is None:raise ValueError('Physical basket geometry missing')
        count=0
        for geom in basket.findall('geom'):
            if not geom.get('name','').startswith('basket_seg_'):continue
            if geom.get('type')!='sphere':raise ValueError('Unexpected rim geometry type')
            position=np.fromstring(geom.get('pos',''),sep=' ')
            if position.shape!=(3,) or np.linalg.norm(position[:2])<=0:raise ValueError('Invalid rim segment position')
            position[:2]*=(self.hoop_diameter/2.)/np.linalg.norm(position[:2])
            geom.set('pos',' '.join(format(float(x),'.12g') for x in position));count+=1
        if count!=24:raise ValueError('Expected 24 original physical rim segments')
        # Do not change the global DcmmCfg: only this environment's scoring config.
        self.basket_cfg.basket_radius=self.hoop_diameter/2.
        return ET.tostring(root,encoding='unicode')
