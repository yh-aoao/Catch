"""Synthetic contract tests; no success-rate claims."""
import copy
import contextlib
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import run
from .core import Student,Replay,TASKS,action_mask,losses,command_context
from distillation.environments import stack_tree


def config():
    c=json.loads((Path(__file__).parent/'configs/catch.json').read_text())
    c.update(device='cpu',iterations=2,num_envs_per_task=3,rollout_steps=4,updates_per_iteration=1,
             calibration_steps=4,max_episode_steps=16,capacity_frames_per_task=96,bptt_steps=2,
             eval_episodes=3,eval_start=1,eval_every=1,save_every=1)
    c['vision']['height']=c['vision']['width']=64
    c['model'].update(cnn_channels=[4,8,8],embedding=8,gru_hidden=16,mlp_hidden=[16,8])
    c['lr_schedule']=dict(kind='cosine',start=1,end=3,minimum=3e-5)
    return c


class FakeTeacher:
    def __init__(self,task):self.task=task;self.hash='fake-'+task
    def predict(self,obs):
        return np.full((len(obs['hand']),20),.2,np.float32)


class FakePool:
    def __init__(self,task,cfg,count,seed,viewer=False,objects=None):
        self.task=task;self.count=count;self.names=objects or cfg['objects']
        self.objects=[self.names[i%len(self.names)] for i in range(count)]
        self.steps=np.zeros(count,dtype=int);self.episodes=np.zeros(count,dtype=int);self.closed=False
        self.obs=self.packet()
    def packet(self):
        context=np.zeros((self.count,10),np.float32)
        if self.task=='basket':
            context[:,0]=1;context[:,1:6]=.3;context[:,6]=(self.steps>=2);context[:,9]=1
        teacher_obs=dict(hand=np.zeros((self.count,12),np.float32))
        return dict(depth=np.zeros((self.count,2,64,64),np.float16),
                    proprio=np.repeat(self.steps[:,None],36,axis=1).astype(np.float32),context=context,
                    object_target=np.ones((self.count,6),np.float32),object_valid=np.ones((self.count,6),np.float32)*(self.task!='basket'),
                    world_position=np.repeat(self.steps[:,None],3,axis=1).astype(np.float32),
                    object_name=np.array(self.objects),teacher_obs=teacher_obs)
    def step(self,actions):
        self.steps+=1;done=self.steps>=4+np.arange(self.count)%2
        infos=[dict(success=True,roll_eval_success=True,safe_basket_success=True,module_hand_contact=bool(self.steps[i]>=3),terminated_reason='synthetic') for i in range(self.count)]
        self.steps[done]=0;self.episodes[done]+=1;self.obs=self.packet()
        return self.obs,np.ones(self.count),done,infos
    def close(self):self.closed=True


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(19)

    def test_modes_masks_no_goal_leakage_and_sequence(self):
        c=config();s=Student(c)
        depth=torch.rand(2,5,2,64,64);robot=torch.rand(2,5,36);prev=torch.rand(2,5,20);ctx=torch.zeros(2,5,10)
        ctx[1,:,0]=1;ctx[1,2:,6]=1
        x=(depth,robot,prev,ctx)
        a,_,_=s(*x);hidden=None;parts=[]
        for j in range(5):
            aa,hidden,_=s(*(v[:,j:j+1] for v in x),hidden);parts.append(aa)
        torch.testing.assert_close(a,torch.cat(parts,1),atol=1e-6,rtol=1e-5)
        self.assertEqual(a[1,:2,2:].abs().sum().item(),0)
        self.assertEqual(a[1,2:,:2].abs().sum().item(),0)
        changed=ctx.clone();changed[0,:,1:]=float('nan')
        torch.testing.assert_close(a[0],s(depth,robot,prev,changed)[0][0],rtol=0,atol=0)
        # Basket goals affect the Basket path.
        changed=ctx.clone();changed[1,:,1:6]=3
        self.assertGreater((a[1]-s(depth,robot,prev,changed)[0][1]).abs().sum().item(),0)

    def make_replay(self,c):
        r=Replay(c)
        for task in TASKS:
            for obj in c['tasks'][task]['objects']:
                rows=[]
                for j in range(9):
                    ctx=np.zeros(10,np.float32)
                    if task=='basket':ctx[0]=1;ctx[1:6]=.3;ctx[6]=float(j>=3)
                    rows.append(dict(depth=np.zeros((2,64,64),np.float16),proprio=np.ones(36,np.float32),
                        previous=np.zeros(20,np.float32),context=ctx,target=np.ones(20,np.float32)*.2,
                        object_target=np.ones(6,np.float32),object_valid=np.ones(6,np.float32)*(task!='basket'),
                        world_position=np.array([j,0,0],np.float32),contact_after=np.float32(j==4)))
                r.add(task,obj,rows)
        return r

    def test_stratification_contact_masks_and_restore(self):
        c=config();r=self.make_replay(c);batch=r.sample(2,'cpu')
        self.assertEqual(len(batch['weights']),16)
        self.assertAlmostEqual(batch['weights'].sum().item(),1)
        for i in range(4):self.assertAlmostEqual(batch['weights'][batch['task_id']==i].sum().item(),.25)
        for t in ('roll','bounce'):
            for o in c['tasks'][t]['objects']:
                e=r.data[t+'/'+o][0]
                self.assertEqual(e['future_valid'][1,1].item(),1)
                self.assertEqual(e['future_valid'][2,1].item(),0)
        self.assertEqual(r.data['basket/original'][0]['future_valid'].sum().item(),0)
        rr=Replay(c);rr.load_state_dict(r.state_dict());self.assertEqual(rr.sizes,r.sizes)

    def test_losses_heads_and_padding(self):
        c=config();r=self.make_replay(c);batch=r.sample(1,'cpu');s=Student(c)
        x=[batch[k] for k in ('depth','proprio','previous','context')]
        _,_,logits,pred=s(*x,auxiliary=True)
        terms=losses(s,batch,0,9,logits,pred,c,batch['valid'].sum(1))
        terms['loss'].backward()
        for module in (s.head,s.basket_head,s.forecast,s.goal_encoder,s.gru):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None),0)
        s.zero_grad();batch['valid'][:,7:]=0
        _,_,logits,pred=s(*x,auxiliary=True)
        a=losses(s,batch,0,9,logits,pred,c,batch['valid'].sum(1))['loss']
        changed=copy.deepcopy(batch);changed['target'][:,7:]=-1;changed['future_target'][:,7:]=float('nan')
        b=losses(s,changed,0,9,logits,pred,c,batch['valid'].sum(1))['loss']
        torch.testing.assert_close(a,b)
        a.backward()

    def test_train_evaluate_snapshot_resume(self):
        c=config();teachers={t:FakeTeacher(t) for t in TASKS};adapters={t:run.ActionAdapter(v) for t,v in c['tasks'].items()}
        run.validate(c)
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'Pool',FakePool),patch.object(run,'provenance',return_value={'test':'synthetic'}),contextlib.redirect_stdout(io.StringIO()):
            p=Path(tmp);run.train(c,teachers,adapters,p)
            saved=torch.load(p/'resume_last.pth',weights_only=True)
            self.assertEqual(saved['iteration'],2);self.assertTrue((p/'student_000001.pth').exists())
            self.assertTrue(all(v==3 for t,v in [(k,json.loads((p/'eval_000002.json').read_text())[k]['episodes']) for k in TASKS]))
            c['iterations']=3;run.train(c,teachers,adapters,p,p/'resume_last.pth')
            saved=torch.load(p/'resume_last.pth',weights_only=True)
            self.assertEqual(saved['iteration'],3);self.assertAlmostEqual(saved['optimizer']['param_groups'][0]['lr'],3e-5)
            self.assertEqual(set(saved['replay']),set(self.make_replay(c).data))

    def test_default_schedule(self):
        c=json.loads((Path(__file__).parent/'configs/catch.json').read_text());run.validate(c)
        self.assertEqual([i for i in range(1,10001) if run.should_evaluate(i,c)],[5000,6000,7000,8000,9000,10000])

    def test_chunked_loss_and_basket_auxiliary_exclusion(self):
        c=config();batch=self.make_replay(c).sample(1,'cpu');s=Student(c)
        inputs=[batch[k] for k in ('depth','proprio','previous','context')]
        _,_,logits,prediction=s(*inputs,auxiliary=True)
        full=s.extra['future']
        whole=losses(s,batch,0,9,logits,prediction,c,batch['valid'].sum(1))['loss']
        total=0
        for begin,end in ((0,2),(2,6),(6,9)):
            s.extra={'future':full[:,begin:end]}
            total=total+losses(s,batch,begin,end,logits[:,begin:end],prediction[:,begin:end],c,batch['valid'].sum(1))['loss']
        torch.testing.assert_close(total,whole)
        s.extra={'future':full}
        changed=copy.deepcopy(batch);select=changed['task_id']==3
        changed['future_target'][select]=float('nan')
        changed['object_target'][select]=float('nan')
        altered=losses(s,changed,0,9,logits,prediction,c,batch['valid'].sum(1))['loss']
        torch.testing.assert_close(altered,whole)

    def test_phase_head_gradient_isolation(self):
        s=Student(config());depth=torch.rand(1,3,2,64,64);proprio=torch.rand(1,3,36);previous=torch.zeros(1,3,20)
        ctx=torch.zeros(1,3,10);ctx[...,0]=1;ctx[...,1:6]=.5
        s(depth,proprio,previous,ctx)[0].sum().backward()
        self.assertEqual(sum(p.grad.abs().sum().item() for p in s.head.parameters()),0)
        self.assertEqual(sum(p.grad.abs().sum().item() for p in s.forecast.parameters()),0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in s.basket_head.parameters()),0)

    def test_contact_includes_separate_collision_sphere(self):
        from .environments import hand_contact
        model=SimpleNamespace(geom_bodyid=np.array([0,1,2,0]),body=lambda _:SimpleNamespace(id=0))
        data=SimpleNamespace(contact=[SimpleNamespace(geom1=3,geom2=2)])
        env=SimpleNamespace(Dcmm=SimpleNamespace(model=model,data=data),hand_start_id=2)
        with patch('gym_dcmm.utils.roll_rewards.hand_collision_ids',return_value=np.array([2])):
            self.assertTrue(hand_contact(env))
            data.contact=[SimpleNamespace(geom1=1,geom2=2)]
            self.assertFalse(hand_contact(env))


if __name__=='__main__':unittest.main()
