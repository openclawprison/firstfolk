"""Actual MiniGrid tasks with an engineered controller using partial observations."""
import random
from collections import deque
from importlib.metadata import version

ENVS=('MiniGrid-Empty-5x5-v0','MiniGrid-Empty-8x8-v0',
      'MiniGrid-DoorKey-5x5-v0','MiniGrid-DoorKey-6x6-v0','MiniGrid-DoorKey-8x8-v0')
VECTORS=((1,0),(0,1),(-1,0),(0,-1))


class Controller:
    """Relative odometry, observed spatial memory and object-interaction rules.

    No environment object or true agent position is passed to this controller.
    Turn/forward semantics and key-door rules are engineered, not learned.
    """
    def __init__(self):
        self.position=(0,0);self.known={};self.visits={};self.steps=0

    def observe(self,observation):
        image=observation['image'];size=image.shape[0]
        direction=int(observation['direction'])
        forward=VECTORS[direction];right=VECTORS[(direction+1)%4]
        for i in range(size):
            for j in range(size):
                obj,color,status=map(int,image[i,j])
                if obj==0:continue
                distance=size-1-j;side=i-size//2
                p=(self.position[0]+forward[0]*distance+right[0]*side,
                   self.position[1]+forward[1]*distance+right[1]*side)
                self.known[p]=(obj,color,status)
        held=tuple(map(int,image[size//2,size-1]))
        self.known[self.position]=(1,0,0)
        return direction,held

    def action(self,observation):
        direction,held=self.observe(observation)
        carrying=held[0]==5
        forward=VECTORS[direction]
        front=(self.position[0]+forward[0],self.position[1]+forward[1])
        item=self.known.get(front,(0,0,0))
        if item[0]==5 and not carrying:return 3,'pick up observed key'
        if item[0]==4 and item[2]!=0 and (item[2]==1 or (carrying and held[1]==item[1])):
            return 5,'open observed compatible door'
        def free(p):
            obj,color,status=self.known.get(p,(0,0,0))
            return obj in (1,3,5,8) or (obj==4 and (status in (0,1) or (carrying and color==held[1])))
        queue=deque([self.position]);routes={self.position:[]}
        while queue:
            p=queue.popleft()
            for d,(dx,dy) in enumerate(VECTORS):
                target=(p[0]+dx,p[1]+dy)
                if target not in routes and free(target):
                    routes[target]=routes[p]+[d];queue.append(target)
        goals=[p for p in routes if self.known[p][0]==8]
        keys=[p for p in routes if self.known[p][0]==5] if not carrying else []
        targets=goals or keys
        if targets:
            target=min(targets,key=lambda p:len(routes[p]))
            path=routes[target];reason='observed goal' if goals else 'observed key'
        else:
            frontiers=[p for p in routes if any((p[0]+dx,p[1]+dy) not in self.known for dx,dy in VECTORS)]
            if not frontiers:return 0,'rotate to observe unexplored view'
            target=min(frontiers,key=lambda p:(self.visits.get(p,0),len(routes[p]),p))
            path=routes[target];reason='observed frontier'
        self.visits[self.position]=self.visits.get(self.position,0)+1
        if not path:return 0,'rotate at frontier'
        desired=path[0]
        if desired!=direction:return (1 if (desired-direction)%4==1 else 0),'orient toward '+reason
        # Odometry uses the visible front-cell state, not hidden environment state.
        if item[0] in (1,3,8) or (item[0]==4 and item[2]==0):self.position=front
        return 2,'advance toward '+reason


def episode(environment,seed=0,max_steps=200,baseline=False):
    try:
        import gymnasium as gym
        import minigrid
    except ImportError as exc:raise ValueError('Install requirements-cognition.txt for MiniGrid') from exc
    if environment not in ENVS:raise ValueError('Unsupported MiniGrid environment')
    if type(seed) is not int or type(max_steps) is not int or not 1<=max_steps<=1000 or type(baseline) is not bool:
        raise ValueError('Invalid seed, step budget or baseline flag')
    env=gym.make(environment,render_mode=None)
    rng=random.Random(seed);controller=Controller();trace=[]
    try:
        observation,_=env.reset(seed=seed)
        mission=observation['mission']
        reward=0.0;terminated=truncated=False
        counts={'pickup':0,'toggle':0}
        for step in range(max_steps):
            if baseline:action,reason=rng.randrange(7),'random baseline'
            else:action,reason=controller.action(observation)
            if action==3:counts['pickup']+=1
            if action==5:counts['toggle']+=1
            observation,reward,terminated,truncated,_=env.step(action)
            if len(trace)<25:trace.append({'action':int(action),'reason':reason})
            if terminated or truncated:break
        return {'environment':environment,'seed':seed,'success':bool(reward>0 and terminated),
                'steps':step+1,'native_reward':float(reward),'mission':mission,
                'interaction_attempts':counts,'known_cells':len(controller.known),'trace':trace,
                'observation_scope':'partial symbolic image, compass direction and mission only',
                'controller':'random' if baseline else 'engineered symbolic mapping and planning',
                'minigrid_version':version('minigrid'),'learned_policy':False}
    finally:env.close()


def evaluate(state,environment='MiniGrid-DoorKey-6x6-v0',seeds=None,max_steps=200):
    seeds=list(range(20)) if seeds is None else seeds
    if not isinstance(seeds,list) or not 1<=len(seeds)<=50 or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Provide 1–50 distinct integer seeds')
    trials=[{'agent':episode(environment,s,max_steps),'baseline':episode(environment,s,max_steps,True)} for s in seeds]
    result={'trials':trials,'success_rate':sum(t['agent']['success'] for t in trials)/len(trials),
            'random_baseline_success_rate':sum(t['baseline']['success'] for t in trials)/len(trials),
            'environment':environment,'seed_count':len(seeds),'training_used':False,
            'scope':'Task success using partial observations and engineered planning; not human fidelity or learned embodied intelligence'}
    state.setdefault('embodied_evaluations',[]).append({k:v for k,v in result.items() if k!='trials'})
    state['embodied_evaluations']=state['embodied_evaluations'][-20:]
    return result
