"""Persistent tabular learning from upstream MiniGrid's partial symbolic images."""
import hashlib
import json
import random
from importlib.metadata import version
ENVIRONMENTS=('MiniGrid-Empty-Random-5x5-v0','MiniGrid-Empty-Random-6x6-v0')


def dependencies():
    try:
        import gymnasium as gym
        import minigrid
        return gym
    except ImportError as exc:raise ValueError('Install requirements-cognition.txt for MiniGrid learning') from exc


def observation_key(observation):
    return hashlib.sha256(observation['image'].tobytes()+bytes([int(observation['direction'])])).hexdigest()[:32]


def greedy(q,observation,rng):
    row=q.get(observation_key(observation),[0.0]*3)
    best=max(row)
    return rng.choice([i for i,v in enumerate(row) if v==best])


def validate(environment,seeds):
    if environment not in ENVIRONMENTS:raise ValueError('Only supported random-start empty rooms can use this learner')
    if not isinstance(seeds,list) or not 1<=len(seeds)<=100 or len(set(seeds))!=len(seeds) or any(type(s) is not int or not 0<=s<=1000000 for s in seeds):
        raise ValueError('Provide 1–100 distinct integer seeds in [0,1000000]')


def trials(model,environment,seeds):
    gym=dependencies();env=gym.make(environment,render_mode=None);result=[]
    try:
        for seed in seeds:
            observation,_=env.reset(seed=seed);rng=random.Random(seed);reward=0.0
            for step in range(100):
                observation,reward,terminated,truncated,_=env.step(greedy(model['q'],observation,rng))
                if terminated or truncated:break
            result.append({'seed':seed,'success':bool(reward>0),'steps':step+1,'native_reward':float(reward)})
        return result
    finally:env.close()


def train(state,environment='MiniGrid-Empty-Random-5x5-v0',episodes=600,seed=0):
    validate(environment,[seed])
    if type(episodes) is not int or not 1<=episodes<=1500:raise ValueError('episodes must be 1–1500')
    gym=dependencies();key='minigrid:'+environment
    model=state.setdefault('procedural_memory',{}).setdefault(key,{'q':{},'episodes':0,'training_seeds':[],
                                                                 'observation_encoding':'sha256_partial_image_and_compass_v1'})
    evaluation_seeds=list(range(900000,900020))
    training_seeds=list(range(seed,seed+episodes))
    if any(s in set(model['training_seeds']+training_seeds) for s in evaluation_seeds):
        raise ValueError('Training seeds overlap reserved evaluation seeds')
    before=trials(model,environment,evaluation_seeds)
    q=model['q'];rng=random.Random(seed);env=gym.make(environment,render_mode=None)
    parameters=state['brain']['parameters'];alpha=0.15+0.35*parameters['learning_rate'];gamma=0.9+0.09*parameters['discount']
    wins=0
    try:
        for i,episode_seed in enumerate(training_seeds):
            observation,_=env.reset(seed=episode_seed)
            epsilon=max(0.05,0.8*(1-i/episodes))
            for _ in range(100):
                current=observation_key(observation)
                row=q.setdefault(current,[0.0]*3)
                action=rng.randrange(3) if rng.random()<epsilon else greedy(q,observation,rng)
                target,reward,terminated,truncated,_=env.step(action)
                signal=float(reward) if reward>0 else -0.01
                next_row=q.get(observation_key(target),[0.0]*3)
                row[action]+=alpha*(signal+(0 if terminated or truncated else gamma*max(next_row))-row[action])
                observation=target
                if terminated or truncated:
                    wins+=int(reward>0);break
        model['episodes']+=episodes
        model['training_seeds']=sorted(set(model['training_seeds']+training_seeds))
        after=trials(model,environment,evaluation_seeds)
        return {'policy_id':key,'environment':environment,'episodes':episodes,'total_episodes':model['episodes'],
                'training_success_rate':wins/episodes,'before_success_rate':sum(t['success'] for t in before)/len(before),
                'after_success_rate':sum(t['success'] for t in after)/len(after),'evaluation_trials':after,
                'observation_states':len(q),'minigrid_version':version('minigrid'),
                'algorithm':'tabular Q-learning over partial symbolic images and compass',
                'training_reward':'native goal reward; unsuccessful steps -0.01','evaluation_reward':'native MiniGrid reward',
                'scope':'Untrained seeds/random starts in the same room layout; not unseen-layout or human generalization'}
    finally:env.close()


def evaluate(state,environment='MiniGrid-Empty-Random-5x5-v0',seeds=None):
    seeds=list(range(910000,910020)) if seeds is None else seeds
    validate(environment,seeds)
    key='minigrid:'+environment;model=state.get('procedural_memory',{}).get(key)
    if model is None:raise LookupError('MiniGrid policy not trained')
    if set(seeds)&set(model['training_seeds']):raise ValueError('Evaluation seeds overlap recorded training seeds')
    before=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
    evaluated=trials(model,environment,seeds)
    return {'policy_id':key,'success_rate':sum(t['success'] for t in evaluated)/len(evaluated),'trials':evaluated,
            'frozen_weights':before==hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest(),
            'held_out_seeds':True,'scope':'Same room layout, randomized starts and orientations; human fidelity unmeasured'}
