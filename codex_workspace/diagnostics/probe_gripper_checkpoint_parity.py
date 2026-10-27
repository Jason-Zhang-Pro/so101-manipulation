from pathlib import Path
import json,gc,hashlib
import torch,numpy as np
from safetensors import safe_open
from so101_capstone.policies.act_checkpoint_loader import load_act_checkpoint
from so101_capstone.policies.act_policy_adapter import ActPolicyAdapter
from so101_capstone.environments.policy_observation import PolicyObservation
from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS,gazebo_positions_to_policy,policy_positions_to_gazebo
from so101_capstone.training.act_ppo import predict_action_chunk_with_grad,actor_observation_from_batch,actor_observation_tensors,policy_action_bounds,_unnormalize_actions_with_grad
from so101_capstone.training.act_ppo_action_distribution import sample_policy_actions

torch.set_num_threads(2)
root=Path.cwd(); obs_path=root/'output_simulation/evaluations/paired_act_20260927_initial_vs_update21_r2/initial/idle_empty_0/observations.npz'
paths=[('initial',root/'codex_workspace/checkpoints/act_0005000_te_off_zdtwineg'),('update25',root/'output_simulation/finetuned/act_ppo_20260928_200236_from_checkpoints/checkpoints/update_000025/pretrained_model'),('update40',root/'output_simulation/finetuned/act_ppo_20260929_160417_resume_update25/checkpoints/update_000040/pretrained_model')]
out={'probe':'CPU offline action contract parity using saved historical images; not rollout or stage evaluation','observation_path':str(obs_path),'checkpoints':{},'calibration':[vars(x) for x in SIM_JOINT_CALIBRATIONS]}
reference_stats=None
with np.load(obs_path) as z:obs={p:PolicyObservation(joint_positions=z[p+'_q'],joint_velocities=np.zeros(6),top_image=z[p+'_top'],gripper_image=z[p+'_gripper']) for p in ['initial','final']}
for label,p in paths:
 cfg=json.loads((p/'config.json').read_text()); all_stats={}
 with safe_open(str(p/'model.safetensors'),framework='pt',device='cpu') as f:
  for k in f.keys():
   if k.startswith(('normalize_inputs','normalize_targets','unnormalize_outputs')):all_stats[k]=f.get_tensor(k)
 res={'path':str(p),'normalization_mapping':cfg['normalization_mapping'],'chunk_size':cfg['chunk_size'],'n_action_steps':cfg['n_action_steps'],'temporal_ensemble_coeff':cfg['temporal_ensemble_coeff'],'stats':{k:v.tolist() for k,v in all_stats.items()},'all_stats_identical_to_initial':reference_stats is None or all(k in reference_stats and v.equal(reference_stats[k]) for k,v in all_stats.items()),'probes':{}}
 if reference_stats is None:reference_stats=all_stats
 loaded=load_act_checkpoint(p,device='cpu',strict=True); policy=loaded.policy; adapter=ActPolicyAdapter(policy,device='cpu')
 res['normalization_parameter_requires_grad']={k:v.requires_grad for k,v in policy.named_parameters() if k.startswith(('normalize_inputs','normalize_targets','unnormalize_outputs'))}
 for part,ob in obs.items():
  batch=adapter.build_batch(ob); replay=actor_observation_tensors(actor_observation_from_batch(batch),device='cpu')
  with torch.no_grad():
   native=policy.predict_action_chunk(batch);ppo=predict_action_chunk_with_grad(policy,batch); replay_ppo=predict_action_chunk_with_grad(policy,replay)
   policy.reset();selected=policy.select_action(batch)
   lower,upper=policy_action_bounds(device='cpu');det,_,_=sample_policy_actions(ppo[:,0],torch.full((6,),-3.),lower=lower,upper=upper,deterministic=True)
  native_cmd=adapter.map_policy_positions(ob,selected[0].numpy()).controller_command(); ppo_cmd=adapter.map_policy_positions(ob,det[0].numpy()).controller_command()
  normalized_logits=torch.zeros((1,15,6),requires_grad=True); unnorm=_unnormalize_actions_with_grad(policy,normalized_logits);grad=torch.autograd.grad(unnorm[0,0].sum(),normalized_logits)[0][0,0]
  res['probes'][part]={'raw_joint_radians':ob.joint_positions.tolist(),'actor_policy_state':batch['observation.state'][0].tolist(),'ppo_replay_input_max_abs_deltas':{k:float((batch[k]-replay[k]).abs().max()) for k in batch},'native_vs_ppo_chunk_max_abs_delta':float((native-ppo).abs().max()),'native_vs_ppo_replay_chunk_max_abs_delta':float((native-replay_ppo).abs().max()),'native_selected_vs_chunk_first_max_abs_delta':float((selected-native[:,0]).abs().max()),'native_first_action':native[0,0].tolist(),'ppo_first_action':ppo[0,0].tolist(),'ppo_deterministic_policy_action':det[0].tolist(),'native_command_rad':native_cmd.tolist(),'ppo_command_rad':ppo_cmd.tolist(),'command_max_abs_delta_rad':float(np.max(np.abs(native_cmd-ppo_cmd))),'unnormalization_derivative_per_joint':grad.tolist()}
 out['checkpoints'][label]=res
 print(label,'chunk and command parity',[(k,v['native_vs_ppo_chunk_max_abs_delta'],v['command_max_abs_delta_rad']) for k,v in res['probes'].items()],flush=True)
 del loaded,policy,adapter;gc.collect()
cal=SIM_JOINT_CALIBRATIONS[-1]; home=json.loads((root/'config/pick_place_reward.json').read_text())['home_joint_positions_rad'][-1]
out['gripper_examples']={'home_q_rad':home,'home_policy':cal.gazebo_to_policy(home),'policy_samples':[{'policy':v,'command_rad':cal.policy_to_gazebo(v),'opening_from_home_rad':cal.policy_to_gazebo(v)-home} for v in [0,4.490096810932703,20,28.186521530151367,50,80,100]],'s1_opening_policy_interval':[cal.gazebo_to_policy(home+v) for v in [.45,.9]]}
path=Path('/tmp/so101_gripper_checkpoint_audit.json');path.write_text(json.dumps(out,indent=2));print('saved',path)
