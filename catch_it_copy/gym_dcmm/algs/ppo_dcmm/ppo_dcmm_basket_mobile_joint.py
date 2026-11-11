"""Second-stage Basket PPO: both navigation and inherited throwing keep learning."""
import copy
import time
import hashlib
import os

import torch

from .ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
from .basket_mobile_joint import JointActorCritic, JointNormalizer


class PPO_Basket_MobileJoint(PPO_Catch_OneStage):
    def __init__(self, env, output_dif, full_config):
        if full_config.basket_training_stage != 'mobile_joint':
            raise ValueError('Joint Basket PPO requires mobile_joint stage')
        if not full_config.train.network.get('separate_value_mlp', True):
            raise ValueError('Joint Basket requires a separate critic')
        super().__init__(env,output_dif,full_config)
        self.conditioned=str(full_config.get('basket_mixed_ball','off'))=='mixed'
        self.navigation_dim=53 if self.conditioned else 51
        if self.obs_shape != (self.navigation_dim,) or self.storage.obs_dim != self.navigation_dim:
            raise ValueError('Joint Basket expects 51 policy inputs; rehearsal metadata must be excluded')
        self.model = JointActorCritic(list(self.network_config.mlp.units),self.conditioned).to(self.device)
        self.running_mean_std = JointNormalizer(self.conditioned).to(self.device)
        self.input_normalization_frozen = False
        from gym_dcmm.envs.basket_free_stop import options as free_options
        self.free_stop_options=free_options(full_config.basket_free_stop) if full_config.get('basket_free_stop') else {}
        if self.free_stop_options and any(v!=self.free_stop_options for v in self.env.call('free_stop_options')):
            raise ValueError('Workers disagree on free stopping region')
        self.single_ball=str(full_config.get('basket_mixed_ball','off'))
        self.single_ball_protocol=None
        if self.single_ball!='off':
            from gym_dcmm.utils.basket_delivery import make_options
            self.single_ball_protocol=self.env.call('mixed_protocol')[0]
            self.delivery_version=str(full_config.basket_delivery_version)
            self.delivery_options=make_options(full_config.basket_delivery_options)
            if any(p!=self.single_ball_protocol for p in self.env.call('mixed_protocol')):
                raise ValueError('Mobile workers disagree on ball physics')
            if any(v!=self.delivery_version for v in self.env.call('delivery_version')):
                raise ValueError('Mobile workers disagree on delivery reward')
            if any(o!=self.delivery_options for o in self.env.call('delivery_options')):
                raise ValueError('Mobile workers disagree on delivery options')
        if not full_config.test:
            source_path=full_config.basket_throw_checkpoint
            checkpoint=torch.load(source_path,map_location=self.device,weights_only=False)
            if self.free_stop_options:
                bounds=checkpoint.get('basket_range',{}).get('y_offset_range')
                selected=self.free_stop_options['y_offset_range']
                if bounds is None or selected[0]<bounds[0] or selected[1]>bounds[1]:
                    raise ValueError('Stopping region must stay within the source trained distance range')
            if self.single_ball!='off':
                from .basket_mixed_policy import single_ball_fixed35
                if checkpoint.get('basket_mixed_options')!=dict(full_config.basket_mixed_options):
                    raise ValueError('Source ball grip differs from mobile configuration')
                if self.conditioned:
                    if (checkpoint.get('basket_mixed_protocol')!='mixed37_known_radius_mass_v1'
                            or checkpoint.get('basket_trained_ball')!='mixed'
                            or checkpoint.get('basket_ball_protocol')!=self.single_ball_protocol
                            or checkpoint.get('basket_delivery_version')!=self.delivery_version
                            or checkpoint.get('basket_delivery_options')!=self.delivery_options):
                        raise ValueError('Mixed mobile source must match three-ball physics and ballistic reward')
                else:
                    checkpoint=single_ball_fixed35(checkpoint,self.single_ball,self.single_ball_protocol,
                        self.delivery_version,self.delivery_options)
            self.model.import_throw(checkpoint,self.running_mean_std,self.finger_options,self.carry_options)
            with open(source_path,'rb') as stream:digest=hashlib.sha256(stream.read()).hexdigest()
            self.basket_transfer_metadata=dict(version='joint_fixed37_throw_base2_v1' if self.conditioned else 'joint_fixed35_throw_base2_v1',
                source_name=os.path.basename(source_path),source_sha256=digest)
            print('[basket-mobile41-joint] arm18 initialized from '+source_path+'; base2 and arm18 both trainable',flush=True)
        base,throw,critic=self.model.actor_groups()
        self.throw_lr_scale=float(full_config.basket_throw_lr_scale)
        if not 0 < self.throw_lr_scale <= 1:
            raise ValueError('basket_throw_lr_scale must be in (0,1]')
        self.optimizer=torch.optim.Adam([
            dict(params=base,lr=self.init_lr,lr_scale=1.,name='base'),
            dict(params=throw,lr=self.init_lr*self.throw_lr_scale,lr_scale=self.throw_lr_scale,name='throw'),
            dict(params=critic,lr=self.init_lr,lr_scale=1.,name='critic')],eps=1e-5)
        print('[basket-mobile41-joint] PPO actions=20; throw/base learning-rate ratio='+str(self.throw_lr_scale),flush=True)
        from .basket_joint_stability import options
        self.joint_options=options(full_config.get('basket_joint_options',{}))
        if self.lr_schedule not in ('fixed','kl'):
            raise ValueError('Stable joint Basket supports fixed or capped kl learning rates')
        self.reference_actor=copy.deepcopy(self.model.throw_actor).eval().requires_grad_(False)
        self.reference_mu=copy.deepcopy(self.model.throw_mu).eval().requires_grad_(False)
        self.reference_sigma=self.model.throw_sigma.detach().clone()
        self.storage.storage_dict['rehearsal']=torch.zeros(
            (self.horizon_length,self.num_actors),dtype=torch.bool,device=self.device)
        self.initial_group_lrs={g['name']:g['lr'] for g in self.optimizer.param_groups}
        if self.conditioned:
            from .utils import AverageScalarMeter
            self.ball_mobile_meters={name:AverageScalarMeter(200) for name in ('original','tennis_ball','baseball')}
        print('[basket-joint-stable] schedule='+self.lr_schedule+
            '; LR ceilings: base='+str(self.init_lr)+' throw='+str(self.init_lr*self.throw_lr_scale)+
            '; summed KL limits='+str(self.joint_options)+'; reference actor is a loss target only',flush=True)

    def obs2tensor(self,obs):
        if 'basket_rehearsal' not in obs:
            raise ValueError('Joint Basket needs rehearsal metadata; sync the updated DcmmVecEnv.py')
        self.current_rehearsal=torch.as_tensor(obs['basket_rehearsal'],
            dtype=torch.bool,device=self.device).reshape(-1)
        raw=super().obs2tensor(obs)
        if self.conditioned:
            properties=torch.as_tensor(obs['basket_ball_properties'],dtype=torch.float32,device=self.device)
            raw=torch.cat((raw,properties),dim=-1)
        return raw

    def _update_mobile_metrics(self,infos,dones):
        super()._update_mobile_metrics(infos,dones)
        if self.conditioned:
            from gym_dcmm.utils.basket_mobile import terminal_records
            for record in terminal_records(infos,dones):
                if not record.get('basket_rehearsal',False):
                    self.ball_mobile_meters[record['basket_ball']].update(
                        torch.tensor([float(record.get('safe_basket_success',False))]))

    def store_rollout_context(self,index):
        self.storage.update_data('rehearsal',index,self.current_rehearsal)

    def reference_kl(self,processed,mu,sigma,replay):
        from .basket_joint_stability import gaussian_kl
        if not replay.any():return mu.sum()*0.
        with torch.no_grad():
            reference=torch.tanh(self.reference_mu(self.reference_actor(processed[replay,self.model.navigation_dim:self.model.navigation_dim+self.model.throw_dim])))
            std=self.reference_sigma.exp().expand_as(reference)
        return gaussian_kl(reference,std,mu[replay,2:],sigma[replay,2:]).mean()

    def training_status(self):
        def rate(key):
            meter=self.mobile_metrics[key]
            return f'{meter.get_mean():.2%}' if len(meter) else 'n/a'
        rates={g['name']:g['lr'] for g in self.optimizer.param_groups}
        ball_status=''
        if self.conditioned:
            for name,meter in self.ball_mobile_meters.items():
                ball_status+=' | '+name+' Mobile SR: '+(f'{meter.get_mean():.2%}' if len(meter) else 'n/a')
        return (ball_status+f" | Park SR: {rate('parking_success')} | Release SR: {rate('qualified_release')}"
            +f" | Base LR: {rates['base']:.2e} | Throw LR: {rates['throw']:.2e}"
            +f" | Base KL: {self.extra_info.get('basket_joint/base_kl_sum',0.):.4f}"
            +f" | Throw KL: {self.extra_info.get('basket_joint/throw_kl_sum',0.):.4f}")

    def train_epoch(self):
        from .basket_joint_stability import (balanced_loss,normalize_advantages,
            masks,gaussian_kl,group_mean,bounded_lr)
        for group in self.optimizer.param_groups:
            group['lr']=min(group['lr'],self.initial_group_lrs[group['name']])
        started=time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time+=time.time()-started
        started=time.time()
        self.set_train()
        # The rollout's behavior probabilities used these exact statistics.
        # Never mutate normalization inside the PPO epochs/minibatches.
        self.running_mean_std.eval()
        data=self.storage.data_dict
        all_groups=masks(data['obses'],data['rehearsal'])
        population_counts=tuple(int(group.sum()) for group in all_groups)
        loss_weights=dict(population_counts=population_counts,population_size=len(data['obses']))
        advantage=(data['returns']-data['values']).reshape(-1)
        if self.normalize_advantage:
            advantage=normalize_advantages(advantage,all_groups)
        data['advantages']=advantage
        stop_base=stop_throw=False
        losses_a,losses_c,losses_b,entropies,kls=[],[],[],[],[]
        base_params,throw_params,critic_params=self.model.actor_groups()
        for _ in range(self.mini_epochs_num):
            for index in range(len(self.storage)):
                old_value,old_logp,adv,old_mu,old_sigma,returns,actions,raw=self.storage[index]
                start,end=self.storage.last_range
                groups=masks(raw,data['rehearsal'][start:end])
                processed=self.running_mean_std(raw)
                result=self.model(dict(obs=processed,prev_actions=actions))
                with torch.no_grad():
                    base_kl=gaussian_kl(old_mu[:,:2],old_sigma[:,:2],
                        result['mus'][:,:2],result['sigmas'][:,:2])
                    throw_kl=gaussian_kl(old_mu[:,2:],old_sigma[:,2:],
                        result['mus'][:,2:],result['sigmas'][:,2:])
                    stop_base=stop_base or bool(group_mean(base_kl,groups[0])>self.joint_options['base_kl_limit'])
                    stop_throw=stop_throw or bool(group_mean(throw_kl,groups[1]|groups[2])>self.joint_options['throw_kl_limit'])
                ratio=torch.exp(old_logp-result['prev_neglogp'])
                surrogate=torch.maximum(-adv*ratio,-adv*ratio.clamp(1.-self.e_clip,1.+self.e_clip))
                actor_loss=balanced_loss(surrogate,groups,self.joint_options['replay_loss_weight'],**loss_weights)
                entropy=balanced_loss(result['entropy'],groups,self.joint_options['replay_loss_weight'],**loss_weights)
                value=result['values']
                critic=(value-returns).square()
                if self.clip_value_loss:
                    clipped=old_value+(value-old_value).clamp(-self.e_clip,self.e_clip)
                    critic=torch.maximum(critic,(clipped-returns).square())
                critic_loss=critic.mean()
                bound=(result['mus'].abs()-1.1).clamp_min(0.).square().sum(-1)
                bound_loss=balanced_loss(bound,groups,self.joint_options['replay_loss_weight'],**loss_weights)
                reference=self.reference_kl(processed,result['mus'],result['sigmas'],groups[2])
                reference=reference*int(groups[2].sum())/max(population_counts[2],1)*len(data['obses'])/len(raw)
                loss=(actor_loss+.5*self.critic_coef*critic_loss-self.entropy_coef*entropy
                    +self.bounds_loss_coef*bound_loss+self.joint_options['reference_weight']*reference)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for active,parameters in (
                        (not stop_base and bool(groups[0].any()),base_params),
                        (not stop_throw and bool((groups[1]|groups[2]).any()),throw_params)):
                    if not active:
                        for parameter in parameters:parameter.grad=None
                if self.truncate_grads:
                    for parameters in (base_params,throw_params,critic_params):
                        torch.nn.utils.clip_grad_norm_(parameters,self.grad_norm)
                self.optimizer.step()
                losses_a.append(actor_loss.detach());losses_c.append(critic_loss.detach())
                losses_b.append(bound_loss.detach());entropies.append(entropy.detach())
                # Never overwrite old_mu/old_sigma: the anchor is the collected policy,
                # not the distribution from the previous optimization pass.
            with torch.no_grad():
                processed=self.running_mean_std(data['obses'])
                result=self.model(dict(obs=processed,prev_actions=data['actions']))
                base_kl=gaussian_kl(data['mus'][:,:2],data['sigmas'][:,:2],result['mus'][:,:2],result['sigmas'][:,:2])
                throw_kl=gaussian_kl(data['mus'][:,2:],data['sigmas'][:,2:],result['mus'][:,2:],result['sigmas'][:,2:])
                bkl=float(group_mean(base_kl,all_groups[0]))
                tkl=float(group_mean(throw_kl,all_groups[1]|all_groups[2]))
                stop_base=stop_base or bkl>self.joint_options['base_kl_limit']
                stop_throw=stop_throw or tkl>self.joint_options['throw_kl_limit']
                ref=float(self.reference_kl(processed,result['mus'],result['sigmas'],all_groups[2]))
                kls.append(result['mus'].new_tensor(max(bkl,tkl)))
        for group in self.optimizer.param_groups:
            name=group['name'];initial=self.initial_group_lrs[name]
            if self.lr_schedule=='kl' and name in ('base','throw'):
                present=all_groups[0].any() if name=='base' else (all_groups[1]|all_groups[2]).any()
                if present:
                    group['lr']=bounded_lr(group['lr'],initial,bkl if name=='base' else tkl,
                        self.joint_options[name+'_kl_limit'])
            group['lr']=min(group['lr'],initial)
            self.extra_info['basket_joint/'+name+'_lr']=group['lr']
        self.last_lr=self.optimizer.param_groups[0]['lr']
        self.extra_info.update({
            'basket_joint/base_kl_sum':bkl,'basket_joint/throw_kl_sum':tkl,
            'basket_joint/replay_reference_kl':ref,
            'basket_joint/base_update_stopped':float(stop_base),
            'basket_joint/throw_update_stopped':float(stop_throw),
            'basket_joint/navigation_samples':int(all_groups[0].sum()),
            'basket_joint/mobile_throw_samples':int(all_groups[1].sum()),
            'basket_joint/replay_throw_samples':int(all_groups[2].sum())})
        # Update only new base statistics, once per collected rollout, after PPO.
        self.running_mean_std.navigation.train()
        self.running_mean_std.navigation(data['obses'])
        self.running_mean_std.eval()
        self.rl_train_time+=time.time()-started
        return losses_a,losses_c,losses_b,entropies,kls


    def save(self,name):
        weights=dict(model=self.model.state_dict(),running_mean_std=self.running_mean_std.state_dict(),
            value_mean_std=self.value_mean_std.state_dict(),
            joint_actor_version='phase_conditioned_base2_arm18_v1',
            action_layout='joint_base2_arm18_phase_v1',basket_training_stage='mobile_joint',
            basket_reward_version=self.basket_reward_version,
            basket_observation_version='mobile53_properties_v1' if self.conditioned else 'mobile51_v1',
            basket_arm_control=self.basket_arm_control,basket_mobile_options=self.mobile_options,
            basket_fixed_protocol='aim_carry_fingers_v1',basket_finger_options=self.finger_options,
            basket_carry_options=self.carry_options,basket_transfer=self.basket_transfer_metadata,
            basket_throw_lr_scale=self.throw_lr_scale,input_normalization_frozen=False,
            joint_stability_version='balanced_reference_kl_v2',basket_joint_options=self.joint_options,
            reference_actor=self.reference_actor.state_dict(),reference_mu=self.reference_mu.state_dict(),
            reference_sigma=self.reference_sigma,
            training_metrics=dict(agent_steps=int(self.agent_steps),epoch=int(self.epoch_num),
                episode_reward=float(self.episode_rewards.get_mean()),
                episode_success=float(self.episode_success.get_mean()),
                grouped_success={k:float(v.get_mean()) for k,v in self.mobile_group_metrics.items()},
                grouped_counts={k:int(len(v)) for k,v in self.mobile_group_metrics.items()}))
        if self.single_ball!='off':
            weights.update(basket_trained_ball=self.single_ball,basket_ball_protocol=self.single_ball_protocol,
                basket_delivery_version=self.delivery_version,basket_delivery_options=self.delivery_options,
                basket_conditioned_mobile=self.conditioned)
        weights['basket_free_stop']=self.free_stop_options
        torch.save(weights,f'{name}.pth')

    def restore_test(self,fn):
        checkpoint=torch.load(fn,map_location=self.device,weights_only=False)
        if bool(checkpoint.get('basket_conditioned_mobile',False))!=self.conditioned:
            raise ValueError('Checkpoint and mobile physical-input layout differ')
        if self.single_ball!='off':
            if (checkpoint.get('basket_trained_ball')!=self.single_ball
                    or checkpoint.get('basket_ball_protocol')!=self.single_ball_protocol
                    or checkpoint.get('basket_delivery_version')!=self.delivery_version
                    or checkpoint.get('basket_delivery_options')!=self.delivery_options):
                raise ValueError('Use matching single-ball mobile checkpoint and reward preset')
        elif checkpoint.get('basket_trained_ball') in ('tennis_ball','baseball'):
            raise ValueError('Single-ball mobile checkpoint requires its matching ball preset')
        if checkpoint.get('basket_free_stop',{})!=self.free_stop_options:
            raise ValueError('Checkpoint stopping geometry differs; use its matching configuration')
        self._validate_mobile_checkpoint(checkpoint)
        if checkpoint.get('joint_actor_version') != 'phase_conditioned_base2_arm18_v1':
            raise ValueError('Use the new jointly trained BasketMobile41 checkpoint, not the frozen version')
        self.model.load_state_dict(checkpoint['model'],strict=True)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'],strict=True)
        self.value_mean_std.load_state_dict(checkpoint['value_mean_std'],strict=True)
        self.basket_transfer_metadata=checkpoint.get('basket_transfer',{})
        if 'reference_actor' in checkpoint:
            self.reference_actor.load_state_dict(checkpoint['reference_actor'],strict=True)
            self.reference_mu.load_state_dict(checkpoint['reference_mu'],strict=True)
            self.reference_sigma=checkpoint['reference_sigma'].detach().clone().to(self.device)
