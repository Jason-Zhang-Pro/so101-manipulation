"""Offline PBRS/GAE identity and read-only analysis of recorded critic outputs."""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import numpy as np
from so101_capstone.training.act_ppo import ActPpoConfig, ActPpoRolloutBuffer

RUN = Path('output_simulation/finetuned/act_ppo_20260927_153051_from_0005000')
OUT = Path('codex_workspace/reports/2026-09-27_act_ppo_pbrs_advantage_probe.json')


def gae(rewards, values, next_values, gamma, lam):
    buffer = ActPpoRolloutBuffer(ActPpoConfig(gamma=gamma, gae_lambda=lam))
    buffer.rewards = list(rewards)
    buffer.values = list(values)
    buffer.next_values = list(next_values)
    buffer.terminated = [False] * len(rewards)
    buffer.episode_end = [False] * (len(rewards)-1) + [True]
    return buffer._advantages_and_returns()[0].astype(float)


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), p05=float(np.quantile(a,.05)),
                p95=float(np.quantile(a,.95)), max=float(a.max()))


def phi_from_context(context):
    c = np.asarray(context, dtype=float)
    phase, mode = int(c[:7].argmax()), int(c[7:10].argmax())
    approach, contact, lift, transport, support, separation, home = c[-19:-12]
    if mode != 0 or phase == 0:
        return 2*home
    if phase in (1,2):
        return (1-contact)*2*approach + contact*(2+2*(.3+.7*lift))
    if phase == 3:
        return 4+2*transport
    # This audit run never reaches later phases. Refuse speculative decoding.
    raise ValueError(f'Unexpected active phase {phase}; inspect context schema before extending audit')


def analyze_rows(rows):
    vals=np.asarray([r['values'] for r in rows],dtype=float)
    phi=np.asarray([phi_from_context(r['critic_task_context']) for r in rows])
    delta=np.asarray([r['rewards']+.997*r['next_values']-r['values'] for r in rows])
    result={'value':stats(vals),'phi':stats(phi),'value_plus_phi':stats(vals+phi),
            'raw_td_delta':stats(delta),'raw_td_rms':float(np.sqrt(np.mean(delta**2))),
            'phase_counts':dict(Counter(int(np.argmax(r['critic_task_context'][:7])) for r in rows))}
    if phi.std()>1e-12 and vals.std()>1e-12:
        result['corr_value_phi']=float(np.corrcoef(vals,phi)[0,1])
        result['fit_value_vs_phi_slope']=float(np.polyfit(phi,vals,1)[0])
    return result


def main():
    config=json.loads((RUN/'config.json').read_text())
    gamma,lam=config['ppo']['gamma'],config['ppo']['gae_lambda']
    n=600
    phi=.2+1.2*np.linspace(0,1,n+1)+.15*np.sin(np.arange(n+1)/11)
    base=np.full(n,-.3/30)
    shaped=base+gamma*phi[1:]-phi[:-1]
    vbase=float(base[0]/(1-gamma))
    vexact=vbase-phi
    delta=shaped+gamma*vexact[1:]-vexact[:-1]
    adv=gae(shaped,vexact[:-1],vexact[1:],gamma,lam)
    transient=gae(shaped,np.zeros(n),np.zeros(n),gamma,lam)
    # Exact two-state MDP: far -> {far,near}, near absorbs, pi(reach)=0.5.
    # Far/perfect-reach potentials 0/2; neither has grasp/carry/place completion.
    prob=.5
    q_pbrs=np.array([base[0]+gamma*vbase,
                    base[0]+2*gamma+gamma*(vbase-2)])
    # Actual candidate: far progress=0, perfect reach progress=2 out of 10.
    # The near state still pays for unfinished downstream work.
    # This changes the objective. Exact evaluation under the same fixed policy.
    near_step_cost=-.01-.3/30*(1-2/10)
    vnear=near_step_cost/(1-gamma)
    vfar=((1-prob)*(-.02)+prob*(near_step_cost+gamma*vnear))/(1-(1-prob)*gamma)
    q_dense=np.array([-.02+gamma*vfar,near_step_cost+gamma*vnear])
    report={'run':str(RUN),'gamma':gamma,'gae_lambda':lam,
      'identity':{'constant_base_reward':float(base[0]),'exact_base_value':vbase,
        'trajectory_length':n,'truncation_bootstrapped':True,
        'max_abs_td_delta_exact_critic':float(abs(delta).max()),
        'max_abs_raw_gae_exact_critic':float(abs(adv).max()),
        'raw_gae_zero_critic':stats(transient),
        'two_state_pbrs_action_advantages':(q_pbrs-vbase).tolist(),
        'two_state_dense_action_advantages':(q_dense-vfar).tolist(),
        'dense_candidate_per_second':.3,
        'two_state_dense_near_still_unfinished_reward_per_step':near_step_cost,
        'approach_active_plus_dense_infinite_discounted_max_cost':(.6/30)/(1-gamma)},
      'limitations':['Synthetic exact-critic identity is not evidence that the run critic has converged.',
          'State-only PBRS can accelerate finite-time learning through imperfect value estimates.',
          'Dense remaining-task cost changes the objective and requires physical/loop validation.',
          'Logged context scores are float32; reconstructed potential has small rounding error.']}
    episodes=[json.loads(p.read_text()) for p in sorted((RUN/'episodes').glob('*.json'))]
    pick=[e for e in episodes if e['task_mode']=='pick_place']
    report['pick_place_summary']={'count':len(pick),
      'step_counts':dict(Counter(e['step_count'] for e in pick)),
      'outcomes':dict(Counter(e['final_info']['outcome'] for e in pick)),
      'component_totals':{k:sum(e['reward_components_total'][k] for e in pick)
                          for k in pick[0]['reward_components_total']}}
    per_update={}
    all_active=[]
    residuals=[]
    for episode in pick:
        rows=[json.loads(line) for line in (RUN/episode['step_records']['path']).open()]
        for i,row in enumerate(rows):
            phase=int(np.argmax(row['critic_task_context'][:7]))
            if phase not in (1,2,3): continue
            all_active.append(row)
            # All episodes are 600 transitions; this run updates each 3 episodes.
            per_update.setdefault(episode['episode_index']//3+1,[]).append(row)
            current_phi=phi_from_context(row['critic_task_context'])
            next_phi=(phi_from_context(rows[i+1]['critic_task_context']) if i+1<len(rows)
                      else episode['final_info']['potential_after'])
            residuals.append(row['rewards']-gamma*next_phi+current_phi)
    report['all_target_active']=analyze_rows(all_active)
    report['base_reward_after_removing_pbrs']=stats(residuals)
    report['target_active_per_update']={str(k):analyze_rows(v) for k,v in per_update.items()}
    updates=[json.loads(x) for x in (RUN/'metrics.jsonl').open() if json.loads(x).get('type')=='ppo_update']
    report['value_loss_per_update']={str(u['update_index']):u['value_loss'] for u in updates}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(OUT),'identity':report['identity'],
       'latest_target_active':report['target_active_per_update'][str(max(per_update))],
       'base_reward':report['base_reward_after_removing_pbrs']},indent=2))


if __name__=='__main__':main()
