"""Stable fixed-base PPO for original/tennis/baseball with known physical inputs."""
import copy
import hashlib
import os
import time

import torch

from .ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
from .basket_mixed_policy import MixedNormalizer,import_throw,stratified_mean,MacroSuccess
from .basket_joint_stability import gaussian_kl,normalize_advantages,group_mean
from .utils import AverageScalarMeter
from gym_dcmm.envs.basket_mixed import NAMES,make_options
from gym_dcmm.utils.basket_mobile import terminal_records


class PPO_Basket_Mixed(PPO_Catch_OneStage):
    def __init__(self,env,output_dif,full_config):
        self.ball_mode=str(full_config.basket_mixed_ball)
        self.delivery_version=str(full_config.get('basket_delivery_version','legacy'))
        from gym_dcmm.utils.basket_delivery import make_options as delivery_options
        self.delivery_options=delivery_options(full_config.get('basket_delivery_options',{}))
        super().__init__(env,output_dif,full_config)
        if not self.fixed_basket or self.obs_shape!=(37,) or self.actions_num!=18:
            raise ValueError('Three-ball PPO requires fixed37/arm18')
        if self.lr_schedule!='fixed':raise ValueError('Three-ball initialization uses a fixed learning rate')
        self.running_mean_std=MixedNormalizer().to(self.device)
        self.input_normalization_frozen=bool(full_config.get('basket_mixed_source',''))
        self.mixed_options=make_options(full_config.basket_mixed_options)
        self.ball_protocol=self.env.call('mixed_protocol')[0]
        if any(version!=self.delivery_version for version in self.env.call('delivery_version')):
            raise ValueError('Environment and trainer delivery reward versions differ')
        if any(options!=self.delivery_options for options in self.env.call('delivery_options')):
            raise ValueError('Environment and trainer delivery reward options differ')
        self.source_metadata=dict(kind='scratch')
        if not full_config.test and full_config.basket_mixed_source:
            path=full_config.basket_mixed_source
            checkpoint=torch.load(path,map_location=self.device,weights_only=False)
            kind=import_throw(self.model,self.running_mean_std,checkpoint,self.finger_options,self.carry_options)
            with open(path,'rb') as stream:digest=hashlib.sha256(stream.read()).hexdigest()
            self.source_metadata=dict(kind=kind,name=os.path.basename(path),sha256=digest)
            print('[basket-three-ball] inherited trainable throw actor from '+path+' ('+kind+')',flush=True)
        self.reference=(copy.deepcopy(self.model).eval().requires_grad_(False)
                        if self.input_normalization_frozen else None)
        if not full_config.test and not full_config.basket_mixed_source:
            print('[basket-three-ball] SCRATCH: random actor/critic; fresh normalization; no old-policy loss',flush=True)
        self.per_ball=[AverageScalarMeter(200) for _ in NAMES]
        self.macro_success=MacroSuccess(self.per_ball)
        self.storage.storage_dict['ball_id']=torch.zeros((self.horizon_length,self.num_actors),dtype=torch.long,device=self.device)
        self.kl_limit=.01;self.reference_weight=.05 if self.reference is not None else 0.
        # Parent optimizer already references the same trainable model. Critic is fresh.
        print('[basket-three-ball] equal rollout loss per ball; fixed LR='+str(self.init_lr)+'; original-ball reference KL weight='+str(self.reference_weight),flush=True)

    def obs2tensor(self,obs):
        raw=super().obs2tensor(obs)
        properties=torch.as_tensor(obs['basket_ball_properties'],dtype=torch.float32,device=self.device)
        self.current_ball=torch.as_tensor(obs['basket_ball_id'],dtype=torch.long,device=self.device).reshape(-1)
        return torch.cat((raw,properties),dim=-1)

    def store_rollout_context(self,index):self.storage.update_data('ball_id',index,self.current_ball)

    def _update_mobile_metrics(self,infos,dones):
        for record in terminal_records(infos,dones):
            index=int(record['basket_ball_index'])
            self.per_ball[index].update(torch.tensor([float(record.get('safe_basket_success',False))]))
        for name,meter in zip(NAMES,self.per_ball):
            if len(meter):self.extra_info['basket_mixed/'+name+'_success']=meter.get_mean()
            self.extra_info['basket_mixed/'+name+'_episodes']=len(meter)

    def selection_success_meter(self):
        return (self.macro_success if self.ball_mode=='mixed' else self.per_ball[NAMES.index(self.ball_mode)])

    def training_status(self):
        text=''
        for name,meter in zip(NAMES,self.per_ball):
            value=f'{meter.get_mean():.2%}' if len(meter) else 'n/a'
            text+=' | '+name+' SR: '+value
        meter=self.selection_success_meter()
        macro=f'{meter.get_mean():.2%}' if len(meter) else 'n/a'
        return (text+(' | Macro SR: ' if self.ball_mode=='mixed' else ' | Task SR: ')+macro+f" | LR: {self.init_lr:.2e} | PPO KL: {self.extra_info.get('basket_mixed/kl_max',0.):.4f}"
            +f" | Actor updates: {self.extra_info.get('basket_mixed/actor_updates',0)}/{self.extra_info.get('basket_mixed/actor_attempts',0)}")

    def train_epoch(self):
        started=time.time();self.set_eval();self.play_steps()
        self.data_collect_time+=time.time()-started;started=time.time()
        self.set_train();self.running_mean_std.eval()
        data=self.storage.data_dict;labels=data['ball_id']
        groups=tuple(labels==i for i in range(3));counts=tuple(int(m.sum()) for m in groups)
        advantage=(data['returns']-data['values']).reshape(-1)
        data['advantages']=normalize_advantages(advantage,groups) if self.normalize_advantage else advantage
        actor=list(self.model.actor_mlp.parameters())+list(self.model.mu.parameters())+[self.model.sigma]
        critic=list(self.model.value_mlp.parameters())+list(self.model.value.parameters())
        a_losses,c_losses,b_losses,entropies,kls=[],[],[],[],[];stopped=False
        kl_by_ball=[0.,0.,0.]
        actor_updates=0
        for _ in range(self.mini_epochs_num):
            for index in range(len(self.storage)):
                old_value,old_logp,adv,old_mu,old_sigma,returns,actions,raw=self.storage[index]
                start,end=self.storage.last_range;batch_labels=labels[start:end]
                processed=self.running_mean_std(raw)
                result=self.model(dict(obs=processed,prev_actions=actions))
                with torch.no_grad():
                    kl=gaussian_kl(old_mu,old_sigma,result['mus'],result['sigmas'])
                    stopped=stopped or any(bool(group_mean(kl,batch_labels==i)>self.kl_limit) for i in range(3))
                ratio=torch.exp(old_logp-result['prev_neglogp'])
                surrogate=torch.maximum(-adv*ratio,-adv*ratio.clamp(1-self.e_clip,1+self.e_clip))
                a_loss=stratified_mean(surrogate,batch_labels,counts,len(labels))
                entropy=stratified_mean(result['entropy'],batch_labels,counts,len(labels))
                v_loss=(result['values']-returns).square()
                if self.clip_value_loss:
                    clipped=old_value+(result['values']-old_value).clamp(-self.e_clip,self.e_clip)
                    v_loss=torch.maximum(v_loss,(clipped-returns).square())
                c_loss=stratified_mean(v_loss.reshape(-1),batch_labels,counts,len(labels))
                bound=(result['mus'].abs()-1.1).clamp_min(0.).square().sum(-1)
                b_loss=stratified_mean(bound,batch_labels,counts,len(labels))
                original=batch_labels==0
                reference=result['mus'].sum()*0.
                if self.reference is not None and original.any():
                    with torch.no_grad():mu,std,_=self.reference._actor_critic(dict(obs=processed[original]))
                    reference=gaussian_kl(mu,std.exp(),result['mus'][original],result['sigmas'][original]).sum()
                    reference=reference/max(counts[0],1)*len(labels)/len(raw)
                loss=a_loss+.5*self.critic_coef*c_loss-self.entropy_coef*entropy+self.bounds_loss_coef*b_loss+self.reference_weight*reference
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                if stopped:
                    for parameter in actor:parameter.grad=None
                if self.truncate_grads:
                    for parameters in (actor,critic):torch.nn.utils.clip_grad_norm_(parameters,self.grad_norm)
                self.optimizer.step()
                if not stopped:actor_updates+=1
                a_losses.append(a_loss.detach());c_losses.append(c_loss.detach());b_losses.append(b_loss.detach());entropies.append(entropy.detach())
            with torch.no_grad():
                result=self.model(dict(obs=self.running_mean_std(data['obses']),prev_actions=data['actions']))
                divergence=gaussian_kl(data['mus'],data['sigmas'],result['mus'],result['sigmas'])
                kl_by_ball=[float(group_mean(divergence,mask)) for mask in groups]
                stopped=stopped or max(kl_by_ball)>self.kl_limit
                kls.append(divergence.new_tensor(max(kl_by_ball)))
        for name,count,kl in zip(NAMES,counts,kl_by_ball):
            self.extra_info['basket_mixed/'+name+'_samples']=count
            self.extra_info['basket_mixed/'+name+'_kl']=kl
        self.extra_info.update({'basket_mixed/kl_max':max(kl_by_ball),'basket_mixed/actor_stopped':float(stopped),
            'basket_mixed/actor_updates':actor_updates,
            'basket_mixed/actor_attempts':self.mini_epochs_num*len(self.storage)})
        if not self.input_normalization_frozen:
            self.running_mean_std.original.train()
            self.running_mean_std.original(data['obses'][:,:35])
            self.running_mean_std.original.eval()
        self.rl_train_time+=time.time()-started
        return a_losses,c_losses,b_losses,entropies,kls

    def save(self,name):
        torch.save(dict(model=self.model.state_dict(),running_mean_std=self.running_mean_std.state_dict(),
            value_mean_std=self.value_mean_std.state_dict(),reference=(self.reference.state_dict() if self.reference is not None else None),
            input_normalization_frozen=self.input_normalization_frozen,
            basket_mixed_protocol='mixed37_known_radius_mass_v1',basket_mixed_options=self.mixed_options,
            basket_ball_protocol=self.ball_protocol,basket_delivery_version=self.delivery_version,
            basket_delivery_options=self.delivery_options,basket_trained_ball=self.ball_mode,
            basket_fixed_protocol='aim_carry_fingers_v1',basket_finger_options=self.finger_options,
            basket_carry_options=self.carry_options,basket_arm_control='joint_velocity',action_layout='arm6_hand12',
            source_metadata=self.source_metadata,training_metrics=dict(agent_steps=int(self.agent_steps),
                episode_reward=float(self.episode_rewards.get_mean()),per_ball_success={n:m.get_mean() for n,m in zip(NAMES,self.per_ball)})),f'{name}.pth')

    def test(self):
        super().test()
        print('three_ball_metrics:',{name:dict(episodes=len(meter),success=meter.get_mean())
            for name,meter in zip(NAMES,self.per_ball)})

    def restore_test(self,path):
        checkpoint=torch.load(path,map_location=self.device,weights_only=False)
        if checkpoint.get('basket_mixed_protocol')!='mixed37_known_radius_mass_v1':
            raise ValueError('Use a newly trained three-ball checkpoint for testing; old weights are initialization only')
        if checkpoint.get('basket_delivery_version','legacy')!=self.delivery_version:
            raise ValueError('Use the checkpoint training reward preset (legacy or ballistic_v2)')
        if self.delivery_version!='legacy' and checkpoint.get('basket_delivery_options')!=self.delivery_options:
            raise ValueError('Delivery reward options differ from training')
        trained_ball=checkpoint.get('basket_trained_ball','mixed')
        if trained_ball!='mixed' and trained_ball!=self.ball_mode:
            raise ValueError('Single-ball checkpoint requires its matching ball preset')
        self._validate_mobile_checkpoint(checkpoint)
        if checkpoint.get('basket_mixed_options')!=self.mixed_options:raise ValueError('Use matching three-ball grip/options')
        if checkpoint.get('basket_ball_protocol')!=self.ball_protocol:raise ValueError('Three-ball physical parameters differ from training')
        self.model.load_state_dict(checkpoint['model'],strict=True)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'],strict=True)
        self.value_mean_std.load_state_dict(checkpoint['value_mean_std'],strict=True)
        self.input_normalization_frozen=bool(checkpoint.get('input_normalization_frozen',True))
        if checkpoint.get('reference') is not None:
            self.reference=copy.deepcopy(self.model).eval().requires_grad_(False)
            self.reference.load_state_dict(checkpoint['reference'],strict=True)
        else:self.reference=None
        self.reference_weight=.05 if self.reference is not None else 0.
        self.source_metadata=checkpoint.get('source_metadata',{})
