import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from .core import Replay,TASKS,Student
from .run import optimize,validate
from .prepare_config import pin

class AllObjectsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c=json.loads((Path(__file__).parent/'configs/catch_all_objects_v2.json').read_text())
    def replay(self):
        r=Replay(self.c)
        for task,obj in r.groups:
            n=2
            e=dict(depth=torch.zeros(n,2,160,160,dtype=torch.uint8),proprio=torch.zeros(n,36),previous=torch.zeros(n,20),
                context=torch.zeros(n,10),target=torch.zeros(n,20),object_target=torch.zeros(n,6),object_valid=torch.ones(n,6),
                world_position=torch.zeros(n,3),contact_after=torch.zeros(n),basket_target=torch.ones(n,3),
                basket_valid=torch.full((n,),float(task=='basket')),basket_phase=torch.tensor([0.,1.]))
            e['context'][:,0]=float(task=='basket');r.add_episode(task+'/'+obj,e)
        return r
    def test_all18_conditions_every_update_equal_task_weight(self):
        r=self.replay()
        for _ in range(3):
            b=r.sample(1,'cpu')
            self.assertEqual(len(b['weights']),18)
            self.assertEqual(torch.bincount(b['task_id']).tolist(),[5,7,5,1])
            self.assertEqual(sorted(b['group_id'].tolist()),list(range(18)))
            for i in range(4):self.assertAlmostEqual(float(b['weights'][b['task_id']==i].sum()),.25,places=6)
        b=r.sample(2,'cpu');self.assertEqual(len(b['weights']),36)
        self.assertTrue(torch.all(torch.bincount(b['group_id'])==2))
    def test_18_sequence_backward_has_finite_gradients(self):
        model=Student(self.c);batch=self.replay().sample(1,'cpu')
        metrics=optimize(model,torch.optim.Adam(model.parameters(),lr=1e-4),batch,self.c)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertIsNotNone(model.basket_position_head[-1].weight.grad)
    def test_pin_actual_files_override_bounce_and_fail_without_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=copy.deepcopy(self.c)
            for task in TASKS:
                name=Path(c['tasks'][task]['checkpoint']).name
                if task=='bounce':continue
                (root/name).write_bytes(task.encode())
            bounce=root/'external'/'teacher.pth';bounce.parent.mkdir()
            torch.save(dict(bounce_grasp_version='controlled_hold_v1'),bounce)
            prepared=pin(c,root,bounce)
            self.assertEqual(prepared['tasks']['bounce']['checkpoint'],str(bounce))
            for task in TASKS:
                item=prepared['tasks'][task]
                self.assertEqual(item['sha256'],hashlib.sha256(Path(item['checkpoint']).read_bytes()).hexdigest())
            validate(prepared)
            with self.assertRaises(FileNotFoundError):pin(c,root,root/'missing.pth')
            self.assertIsNone(c['tasks']['bounce']['sha256'])
    def test_unpinned_server_only_teacher_rejected(self):
        with self.assertRaisesRegex(ValueError,'not pinned'):validate(self.c)

if __name__=='__main__':unittest.main()
