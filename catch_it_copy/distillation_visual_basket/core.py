"""Explicit catch/basket command; no motion ID, object ID, mass or radius in forward."""
from collections import deque
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from distillation_visual_aux.core import Student as BaseStudent, camera_specs, encode_depth, auxiliary_losses

SCHEMA='visual_basket_autonomous_v1'
TASKS=('throw','roll','bounce','basket')
CONTEXT_DIM=10
PROPRIO_DIM=36


def robot_input(obs, joint_position, joint_velocity):
    parts=[obs['base']['v_lin_2d'],obs['arm']['ee_pos3d'],obs['arm']['ee_quat'],
           obs['arm']['ee_v_lin_3d'],obs['hand'],joint_position,joint_velocity]
    out=np.concatenate(parts,axis=-1).astype(np.float32)
    if out.shape[-1]!=PROPRIO_DIM or not np.isfinite(out).all():raise ValueError('Invalid robot-only state')
    return out


def command_context(obs, task):
    shape=obs['base']['v_lin_2d'].shape[:-1]
    result=np.zeros((*shape,10),np.float32)
    result[...,0]=float(task=='basket')
    return result


def action_mask(context):
    if torch.is_tensor(context):return torch.ones((*context.shape[:-1],20),device=context.device,dtype=context.dtype)
    return np.ones((*context.shape[:-1],20),np.float32)


class Student(BaseStudent):
    def __init__(self,config):
        super().__init__('catch',config['vision'],config['model'],config['auxiliary'])
        model=config['model']
        self.proprio_dim=PROPRIO_DIM
        self.mean=torch.zeros(PROPRIO_DIM)
        self.std=torch.ones(PROPRIO_DIM)
        self.basket_position_head=nn.Sequential(nn.Linear(self.hidden_dim,128),nn.ELU(),nn.Linear(128,3))
        self.gru=nn.GRU(len(self.camera_names)*model['embedding']+PROPRIO_DIM+20+1,self.hidden_dim,batch_first=True)
        layers=[];width=self.hidden_dim
        for size in model['mlp_hidden']:
            layers.extend([nn.Linear(width,size),nn.ELU()]);width=size
        layers.append(nn.Linear(width,20));self.basket_head=nn.Sequential(*layers)
        self.forecast=nn.Sequential(nn.Linear(self.hidden_dim,128),nn.ELU(),nn.Linear(128,9))
        self.fusion=nn.Linear(9,self.hidden_dim)
        self.extra={}

    def forward(self,depth,proprio,previous,context,hidden=None,auxiliary=False):
        b,t=depth.shape[:2]
        if tuple(depth.shape[2:])!=self.image_shape or proprio.shape!=(b,t,36) or previous.shape!=(b,t,20) or context.shape!=(b,t,10):
            raise ValueError('Invalid deployment input shapes')
        mode=context[...,:1]
        if not bool(((mode==0)|(mode==1)).all()):raise ValueError('Explicit catch=0/basket=1 command required')
        frames=depth.reshape(b*t,*self.image_shape).float()
        visual=torch.cat([enc(frames[:,i:i+1]) for i,enc in enumerate(self.encoders)],-1).reshape(b,t,-1)
        state=((proprio.float()-self.mean)/self.std).clamp(-5,5)
        # Only the explicit task command is consumed. Extra context is reserved zero padding.
        if bool((context[...,1:]!=0).any()):raise ValueError('Privileged context is forbidden')
        h,hidden=self.gru(torch.cat([visual,state,previous.float(),mode],-1),hidden)
        predicted=self.forecast(h)
        catch_logits=self.head(h+self.fusion(predicted))
        basket_logits=self.basket_head(h)
        logits=torch.where(mode.bool(),basket_logits,catch_logits)
        self.extra={'future':predicted,'basket_position':self.basket_position_head(h)}
        # Catch context is sanitized for masks too; phase values are irrelevant in catch mode.
        mask=action_mask(context)
        result=(logits.tanh()*mask,hidden,logits)
        if auxiliary:return (*result,self.aux_head(h))
        return result


class Replay:
    keys=('depth','proprio','previous','context','target','object_target','object_valid','world_position','contact_after','basket_target','basket_valid','basket_phase')
    def __init__(self,config):
        self.batch_per_task=config.get('batch_episodes_per_task',2)
        self.sampling_mode=config.get('sampling_mode','per_task')
        if self.sampling_mode not in ('per_task','per_object'):raise ValueError('Unknown replay sampling mode')
        self.groups=[(t,o) for t in TASKS for o in config['tasks'][t]['objects']]
        self.data={t+'/'+o:deque() for t,o in self.groups};self.sizes={k:0 for k in self.data}
        self.capacities={t+'/'+o:config['capacity_frames_per_task']//len(config['tasks'][t]['objects']) for t,o in self.groups}

    def add(self,task,obj,rows):
        episode={k:torch.from_numpy(np.stack([r[k] for r in rows])) for k in self.keys}
        self.add_episode(task+'/'+obj,episode)

    def add_episode(self,key,e):
        n=len(e['depth'])
        if not 0<n<=self.capacities[key] or any(len(e[k])!=n for k in self.keys):raise ValueError('Invalid episode length')
        if e['previous'][0].count_nonzero():raise ValueError('Replay must start at reset')
        task=key.split('/')[0]
        expected=1. if task=='basket' else 0.
        if not bool((e['context'][:,0]==expected).all()):raise ValueError('Replay command/task mismatch')
        future=torch.zeros(n,3,3);valid=torch.zeros(n,3)
        if task!='basket':
            contacts=e['contact_after'].reshape(n).bool().long().cumsum(0)
            prefix=torch.cat([torch.zeros(1,dtype=torch.long),contacts])
            for k,offset in enumerate((1,3,5)):
                if n>offset:
                    future[:-offset,k]=e['world_position'][offset:]-e['world_position'][:-offset]
                    valid[:-offset,k]=(prefix[offset:n]==0).float()
        e.update(future_target=future,future_valid=valid)
        while self.sizes[key]+n>self.capacities[key]:self.sizes[key]-=len(self.data[key].popleft()['depth'])
        self.data[key].append(e);self.sizes[key]+=n

    def ready(self):return all(self.data.values())

    def sample(self,per_object,device):
        if not self.ready():raise RuntimeError('Replay needs every task/object condition')
        entries=[]
        if self.sampling_mode=='per_object':
            if type(per_object) is not int or per_object<1:raise ValueError('Need positive episodes per object')
            for i,(task,obj) in enumerate(self.groups):
                for _ in range(per_object):entries.append((i,random.choice(self.data[task+'/'+obj])))
        else:
            for task in TASKS:
                indices=[i for i,(t,o) in enumerate(self.groups) if t==task]
                for _ in range(self.batch_per_task):
                    i=random.choice(indices);t,o=self.groups[i]
                    entries.append((i,random.choice(self.data[t+'/'+o])))
        length=max(len(e['depth']) for _,e in entries);batch={}
        for key in self.keys+('future_target','future_valid'):
            example=entries[0][1][key]
            x=torch.zeros(len(entries),length,*example.shape[1:],dtype=example.dtype)
            for i,(_,e) in enumerate(entries):x[i,:len(e[key])]=e[key]
            batch[key]=x.to(device)
        valid=torch.zeros(len(entries),length,device=device)
        weights=[];task_ids=[]
        for i,(index,e) in enumerate(entries):
            valid[i,:len(e['depth'])]=1
            task=self.groups[index][0];n=sum(t==task for t,o in self.groups)
            weights.append(1/(4*n*per_object) if self.sampling_mode=='per_object' else 1/(4*self.batch_per_task));task_ids.append(TASKS.index(task))
        batch.update(valid=valid,weights=torch.tensor(weights,device=device),task_id=torch.tensor(task_ids,device=device),
                     group_id=torch.tensor([index for index,e in entries],device=device))
        return batch

    def state_dict(self):return {key:list(value) for key,value in self.data.items()}
    def load_state_dict(self,saved):
        if set(saved)!=set(self.data):raise ValueError('Resume condition set changed')
        for k,episodes in saved.items():
            for e in episodes:self.add_episode(k,e)


def losses(student,batch,begin,end,logits,prediction,config,lengths):
    valid=batch['valid'][:,begin:end];context=batch['context'][:,begin:end]
    mask=action_mask(context);target=batch['target'][:,begin:end]
    error=2*F.binary_cross_entropy_with_logits(2*logits,(target+1)/2,reduction='none')
    groups=[];active=[]
    for a,b in ((0,2),(2,8),(8,20)):
        denom=mask[...,a:b].sum(-1)
        groups.append((error[...,a:b]*mask[...,a:b]).sum(-1)/denom.clamp_min(1))
        active.append((denom>0).float())
    imitation=sum(groups)/sum(active).clamp_min(1)
    frame_weights=valid.clone()
    # Teacher phase is loss metadata only. Balance navigation/throw within each complete sequence.
    all_phase=batch['basket_phase']
    for i in range(len(lengths)):
        if batch['task_id'][i]!=3:continue
        navigation=batch['valid'][i]*(all_phase[i]<.5)
        throwing=batch['valid'][i]*(all_phase[i]>=.5)
        if navigation.sum()>0 and throwing.sum()>0:
            frame_weights[i]=valid[i]*torch.where(all_phase[i,begin:end]<.5,
                lengths[i]/(2*navigation.sum()),lengths[i]/(2*throwing.sum()))
    per_episode=(imitation*frame_weights).sum(1)/lengths.clamp_min(1)
    catch=(1-context[...,0])*valid
    aux_target=batch['object_target'][:,begin:end]
    aux_valid=batch['object_valid'][:,begin:end]
    scale=prediction.new_tensor(config['auxiliary']['scales'])
    aux_error=F.smooth_l1_loss(prediction/scale,torch.where(aux_valid.bool(),aux_target,torch.zeros_like(aux_target))/scale,reduction='none')
    position=((aux_error[...,:3]*aux_valid[...,:3]).mean(-1)*catch).sum(1)/lengths.clamp_min(1)
    future_mask=batch['future_valid'][:,begin:end]*catch[...,None]
    future_target=batch['future_target'][:,begin:end]
    safe_target=torch.where(future_mask[...,None].bool(),future_target,torch.zeros_like(future_target))
    future_error=F.smooth_l1_loss(student.extra['future'].reshape(*future_mask.shape,3),safe_target,reduction='none').mean(-1)
    future=(future_error*future_mask).mean(-1).sum(1)/lengths.clamp_min(1)
    weights=batch['weights']
    result={'imitation_loss':(per_episode*weights).sum(),'position_loss':(position*weights).sum(),'future_loss':(future*weights).sum()}
    result['loss']=result['imitation_loss']+config['auxiliary']['position_weight']*result['position_loss']+.1*result['future_loss']
    # Fixed slots make keys stable even when a phase/group has no valid forecast labels.
    for i,task in enumerate(TASKS):
        select=batch['task_id']==i
        result['imitation/'+task]=(per_episode[select]*weights[select]).sum()*4
    for index,(task,obj) in enumerate([(t,o) for t in TASKS for o in config['tasks'][t]['objects']]):
        select=batch['group_id']==index
        result['imitation/'+task+'/'+obj]=per_episode[select].mean() if select.any() else per_episode.new_zeros(())
    basket=valid*batch['basket_valid'][:,begin:end]
    position_error=F.smooth_l1_loss(student.extra['basket_position'],batch['basket_target'][:,begin:end],reduction='none').mean(-1)
    per_episode=(position_error*basket).sum(1)/lengths.clamp_min(1)
    result['basket_position_loss']=(per_episode*weights).sum()
    result['loss']=result['loss']+config['basket_position_weight']*result['basket_position_loss']
    return result
